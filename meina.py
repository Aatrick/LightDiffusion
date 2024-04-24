import os
import random
import sys
from typing import Sequence, Mapping, Any, Union

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.imp as sample
import comfy.sd as sd
import comfy.imp as utils
import trace

################################################ Folder_paths #########################################################


supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors'])

folder_names_and_paths = {}

base_path = os.path.dirname(os.path.realpath(__file__))
folder_names_and_paths["checkpoints"] = ([os.path.join(base_path)], supported_pt_extensions)
folder_names_and_paths["custom_nodes"] = ([os.path.join(base_path, "custom_nodes")], [])

output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")

filename_list_cache = {}


def get_output_directory():
    global output_directory
    return output_directory


def get_full_path(folder_name, filename):
    global folder_names_and_paths
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path

    return None


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1:].split('_')[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
        print("Saving image outside the output folder is not allowed.")
        return {}

    try:
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                             map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


################################################ latent_preview #######################################################


MAX_PREVIEW_RESOLUTION = 512


class LatentPreviewer:
    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        latent_image = x0[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors

        latents_ubyte = (((latent_image + 1) / 2)
                         .clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


def get_previewer(device, latent_format):
    previewer = None
    if previewer is None:
        if latent_format.latent_rgb_factors is not None:
            previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
    return previewer


def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback


####################################################### Nodes ###################################################################


class EmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent},)


class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class SaveImage:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix,
                                                                                                self.output_dir,
                                                                                                images[0].shape[1],
                                                                                                images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = prepare_callback(model, steps)
    disable_pbar = not utils.PROGRESS_BAR_ENABLED
    samples = sample.sample1(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                            denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                            force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                            disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler:
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise)


class CheckpointLoaderSimple:
    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = get_full_path("checkpoints", ckpt_name)
        out = sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        return out[:3]


class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


with open('prompt.txt', 'r') as file:
    lines = file.readlines()

prompt = lines[0].split(':')[1].strip()
w = int(lines[1].split(':')[1].strip())
h = int(lines[2].split(':')[1].strip())


def gen(prompt, w, h):
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
            ckpt_name="meinamix_meinaV11.safetensors"
        )
        cliptextencode = CLIPTextEncode()
        cliptextencode_242 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(checkpointloadersimple_241, 1),
        )
        cliptextencode_243 = cliptextencode.encode(
            text="(worst_quality:1.6 low_quality:1.6) monochrome (zombie sketch interlocked_fingers comic) (hands) text signature logo",
            clip=get_value_at_index(checkpointloadersimple_241, 1),
        )
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_244 = emptylatentimage.generate(
            width=w, height=h, batch_size=1
        )
        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        ksampler_239 = ksampler.sample(
            seed=random.randint(1, 2 ** 64),
            steps=300,
            cfg=7,
            sampler_name="dpm_adaptive",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(checkpointloadersimple_241, 0),
            positive=get_value_at_index(cliptextencode_242, 0),
            negative=get_value_at_index(cliptextencode_243, 0),
            latent_image=get_value_at_index(emptylatentimage_244, 0),
        )
        vaedecode_240 = vaedecode.decode(
            samples=get_value_at_index(ksampler_239, 0),
            vae=get_value_at_index(checkpointloadersimple_241, 2),
        )
        saveimage_248 = saveimage.save_images(
            filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_240, 0)
        )


#tracer = trace.Trace(countfuncs=1)
#tracer.runfunc(gen, prompt, w, h)
#r = tracer.results()
#print(r)

gen(prompt, w, h)