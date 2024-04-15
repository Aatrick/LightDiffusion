import os

import torch
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig, modeling_utils

import comfy.model_management
import comfy.model_patcher
import comfy.ops
from comfy.utils import transformers_convert


class ClipVisionModel():
    def __init__(self, json_config):
        config = CLIPVisionConfig.from_json_file(json_config)
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32
        if comfy.model_management.should_use_fp16(self.load_device, prioritize_performance=False):
            self.dtype = torch.float16

        with comfy.ops.use_comfy_ops(offload_device, self.dtype):
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.dtype)

        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        print("missing clip vision:", m)
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip