import torch

import comfy.adapter
import comfy.model_patcher
import comfy.supported_models_base
import comfy.utils as utils
from comfy import model_detection
from comfy import model_management
from .ldm.models.autoencoder import AutoencoderKL


def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        params['device'] = offload_device
        if model_management.should_use_fp16(load_device, prioritize_performance=False):
            params['dtype'] = torch.float16
        else:
            params['dtype'] = torch.float32

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = comfy.model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def load_sd(self, sd):
        return self.cond_stage_model.load_sd(sd)

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

class VAE:
    def __init__(self, sd=None, device=None, config=None):
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
            sd = diffusers_convert.convert_vae_state_dict(sd)

        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Missing VAE keys", m)

        if len(u) > 0:
            print("Leftover VAE keys", u)

        if device is None:
            device = model_management.vae_device()
        self.device = device
        self.offload_device = model_management.vae_offload_device()
        self.vae_dtype = model_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
        steps = samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = utils.ProgressBar(steps)

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
        output = torch.clamp((
            (utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = 8, pbar = pbar) +
            utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = 8, pbar = pbar) +
             utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = 8, pbar = pbar))
            / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def decode(self, samples_in):
        self.first_stage_model = self.first_stage_model.to(self.device)
        try:
            memory_used = (2562 * samples_in.shape[2] * samples_in.shape[3] * 64) * 1.7
            model_management.free_memory(memory_used, self.device)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(samples).cpu().float() + 1.0) / 2.0, min=0.0, max=1.0)
        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd = utils.load_torch_file(ckpt_path)
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae = VAE(sd=vae_sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip = CLIP(clip_target, embedding_directory=embedding_directory)
            w.cond_stage_model = clip.cond_stage_model
            sd = model_config.process_clip_state_dict(sd)
            load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device(), current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)