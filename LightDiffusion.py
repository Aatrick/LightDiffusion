from __future__ import annotations

import glob
import os
import random
import sys
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from typing import Literal

import customtkinter as ctk
import safetensors.torch
from PIL import ImageSequence, UnidentifiedImageError, ImageFile
from PIL import ImageTk
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import os
import time
import packaging.version
import torch

if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True

supported_pt_extensions = set([".ckpt", ".pt", ".bin", ".pth", ".safetensors", ".pkl"])

folder_names_and_paths = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "_internal")
folder_names_and_paths["checkpoints"] = (
    [os.path.join(models_dir, "checkpoints")],
    supported_pt_extensions,
)

folder_names_and_paths["loras"] = (
    [os.path.join(models_dir, "loras")],
    supported_pt_extensions,
)

folder_names_and_paths["ERSGAN"] = (
    [os.path.join(models_dir, "ERSGAN")],
    supported_pt_extensions,
)

output_directory = ".\\_internal\\output"

filename_list_cache = {}


args_parsing = False


def enable_args_parsing(enable=True):
    global args_parsing
    args_parsing = enable


class LatentFormat:
    scale_factor = 1.0
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.13025
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]
        self.taesd_decoder_name = "taesdxl_decoder"


class SDXL_Playground_2_5(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(
            1, 4, 1, 1
        )
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(
            1, 4, 1, 1
        )

        self.latent_rgb_factors = [
            #   R        G        B
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.latent_rgb_factors = [
            [-0.2340, -0.3863, -0.3257],
            [0.0994, 0.0885, -0.0908],
            [-0.2833, -0.2349, -0.3741],
            [0.2523, -0.0055, -0.1651],
        ]


class SC_Prior(LatentFormat):
    def __init__(self):
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.0326, -0.0204, -0.0127],
            [-0.1592, -0.0427, 0.0216],
            [0.0873, 0.0638, -0.0020],
            [-0.0602, 0.0442, 0.1304],
            [0.0800, -0.0313, -0.1796],
            [-0.0810, -0.0638, -0.1581],
            [0.1791, 0.1180, 0.0967],
            [0.0740, 0.1416, 0.0432],
            [-0.1745, -0.1888, -0.1373],
            [0.2412, 0.1577, 0.0928],
            [0.1908, 0.0998, 0.0682],
            [0.0209, 0.0365, -0.0092],
            [0.0448, -0.0650, -0.1728],
            [-0.1658, -0.1045, -0.1308],
            [0.0542, 0.1545, 0.1325],
            [-0.0352, -0.1672, -0.2541],
        ]


class SC_B(LatentFormat):
    def __init__(self):
        self.scale_factor = 1.0 / 0.43
        self.latent_rgb_factors = [
            [0.1121, 0.2006, 0.1023],
            [-0.2093, -0.0222, -0.0195],
            [-0.3087, -0.1535, 0.0366],
            [0.0290, -0.1574, -0.4078],
        ]


import re

# conversion code from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2 * j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3 - i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i + 1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                logging.debug(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    (
        "token_embedding.weight",
        "transformer.text_model.embeddings.token_embedding.weight",
    ),
    (
        "positional_embedding",
        "transformer.text_model.embeddings.position_embedding.weight",
    ),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


# This function exists because at the time of writing torch.cat can't do fp8 with cuda
def cat_tensors(tensors):
    x = 0
    for t in tensors:
        x += t.shape[0]

    shape = [x] + list(tensors[0].shape)[1:]
    out = torch.empty(shape, device=tensors[0].device, dtype=tensors[0].dtype)

    x = 0
    for t in tensors:
        out[x : x + t.shape[0]] = t
        x += t.shape[0]

    return out


def convert_text_enc_state_dict_v20(text_enc_dict, prefix=""):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if not k.startswith(prefix):
            continue
        if (
            k.endswith(".self_attn.q_proj.weight")
            or k.endswith(".self_attn.k_proj.weight")
            or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
            k.endswith(".self_attn.q_proj.bias")
            or k.endswith(".self_attn.k_proj.bias")
            or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        text_proj = "transformer.text_projection.weight"
        if k.endswith(text_proj):
            new_state_dict[k.replace(text_proj, "text_projection")] = v.transpose(
                0, 1
            ).contiguous()
        else:
            relabelled_key = textenc_pattern.sub(
                lambda m: protected[re.escape(m.group(0))], k
            )
            new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception(
                "CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing"
            )
        relabelled_key = textenc_pattern.sub(
            lambda m: protected[re.escape(m.group(0))], k_pre
        )
        new_state_dict[relabelled_key + ".in_proj_weight"] = cat_tensors(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception(
                "CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing"
            )
        relabelled_key = textenc_pattern.sub(
            lambda m: protected[re.escape(m.group(0))], k_pre
        )
        new_state_dict[relabelled_key + ".in_proj_bias"] = cat_tensors(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


import pickle

load = pickle.load


class Empty:
    pass


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)


# taken from https://github.com/TencentARC/T2I-Adapter
from collections import OrderedDict

import torch.nn as nn


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if not self.use_conv:
            padding = [x.shape[2] % 2, x.shape[3] % 2]
            self.op.padding = padding

        x = self.op(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(
        self,
        channels=[320, 640, 1280, 1280],
        nums_rb=3,
        cin=64,
        ksize=3,
        sk=False,
        use_conv=True,
        xl=True,
    ):
        super(Adapter, self).__init__()
        self.unshuffle_amount = 8
        resblock_no_downsample = []
        resblock_downsample = [3, 2, 1]
        self.xl = xl
        if self.xl:
            self.unshuffle_amount = 16
            resblock_no_downsample = [1]
            resblock_downsample = [2]

        self.input_channels = cin // (self.unshuffle_amount * self.unshuffle_amount)
        self.unshuffle = nn.PixelUnshuffle(self.unshuffle_amount)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i in resblock_downsample) and (j == 0):
                    self.body.append(
                        ResnetBlock(
                            channels[i - 1],
                            channels[i],
                            down=True,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
                elif (i in resblock_no_downsample) and (j == 0):
                    self.body.append(
                        ResnetBlock(
                            channels[i - 1],
                            channels[i],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock(
                            channels[i],
                            channels[i],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            if self.xl:
                features.append(None)
                if i == 0:
                    features.append(None)
                    features.append(None)
                if i == 2:
                    features.append(None)
            else:
                features.append(None)
                features.append(None)
            features.append(x)

        return features


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width**-0.5
        self.transformer_layes = nn.Sequential(
            *[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)]
        )
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]),
            device=x.device,
        )
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token :, :])
        x = x @ self.proj

        return x


class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super(Adapter_light, self).__init__()
        self.unshuffle_amount = 8
        self.unshuffle = nn.PixelUnshuffle(self.unshuffle_amount)
        self.input_channels = cin // (self.unshuffle_amount * self.unshuffle_amount)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        self.xl = False

        for i in range(len(channels)):
            if i == 0:
                self.body.append(
                    extractor(
                        in_c=cin,
                        inter_c=channels[i] // 4,
                        out_c=channels[i],
                        nums_rb=nums_rb,
                        down=False,
                    )
                )
            else:
                self.body.append(
                    extractor(
                        in_c=channels[i - 1],
                        inter_c=channels[i] // 4,
                        out_c=channels[i],
                        nums_rb=nums_rb,
                        down=True,
                    )
                )
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(None)
            features.append(None)
            features.append(x)

        return features


import importlib

import torch
from PIL import ImageFont
from torch import optim


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(
        self,
        params,
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-2,
        amsgrad=False,
        ema_decay=0.9999,  # ema decay to match previous code
        ema_power=1.0,
        param_names=(),
    ):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            ema_decay=ema_decay,
            ema_power=ema_power,
            param_names=param_names,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            ema_decay = group["ema_decay"]
            ema_power = group["ema_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    # Exponential moving average of parameter values
                    state["param_exp_avg"] = p.detach().float().clone()

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                ema_params_with_grad.append(state["param_exp_avg"])

                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # update the steps for each param group update
                state["step"] += 1
                # record the step after step update
                state_steps.append(state["step"])

            optim._functional.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

            cur_ema_decay = min(ema_decay, 1 - state["step"] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(
                    param.float(), alpha=1 - cur_ema_decay
                )

        return loss


from torch import nn


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            (
                torch.tensor(0, dtype=torch.int)
                if use_num_upates
                else torch.tensor(-1, dtype=torch.int)
            ),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - m_param[key])
                    )
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from torch import nn


# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(
                c_latent, affine=False
            ),  # then normalize them to have mean 0 and std 1
        )
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean.view([3, 1, 1])) / self.std.view([3, 1, 1])
        o = self.mapper(self.backbone(x))
        return o


# Fast Decoder for Stage C latents. E.g. 16 x 24 x 24 -> 3 x 192 x 192
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),
            nn.ConvTranspose2d(
                c_hidden, c_hidden // 2, kernel_size=2, stride=2
            ),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),
            nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),
            nn.ConvTranspose2d(
                c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2
            ),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.ConvTranspose2d(
                c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2
            ),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return (self.blocks(x) - 0.5) * 2.0


class StageC_coder(nn.Module):
    def __init__(self):
        super().__init__()
        self.previewer = Previewer()
        self.encoder = EfficientNetEncoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.previewer(x)


import hashlib
import shutil
import urllib
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import optim
from torch.utils import data


def hf_datasets_augs_helper(examples, transform, image_key, mode="RGB"):
    """Apply passed in transforms for HuggingFace Datasets."""
    images = [transform(image.convert(mode)) for image in examples[image_key]]
    return {image_key: images}


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == "mps" else expanded


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """Downloads a file if it does not exist, optionally checking its SHA-256 hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, "wb") as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest != file_digest:
            raise OSError(f"hash of {path} (url: {url}) failed to validate")
    return path


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for _internal you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for _internal
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(
        self,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        max_value=1.0,
        start_at=0,
        last_epoch=0,
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0.0 if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


class InverseLR(optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        inv_gamma=1.0,
        power=1.0,
        warmup=0.0,
        min_lr=0.0,
        last_epoch=-1,
        verbose=False,
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0.0 <= warmup < 1:
            raise ValueError("Invalid value for warmup")
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [
            warmup * max(self.min_lr, base_lr * lr_mult) for base_lr in self.base_lrs
        ]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """Implements an exponential learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr. Decays the learning rate
    continuously by decay (default 0.5) every num_steps steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (float): The number of steps to decay the learning rate by decay in.
        decay (float): The factor by which to decay the learning rate every num_steps
            steps. Default: 0.5.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        num_steps,
        decay=0.5,
        warmup=0.0,
        min_lr=0.0,
        last_epoch=-1,
        verbose=False,
    ):
        self.num_steps = num_steps
        self.decay = decay
        if not 0.0 <= warmup < 1:
            raise ValueError("Invalid value for warmup")
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch
        return [
            warmup * max(self.min_lr, base_lr * lr_mult) for base_lr in self.base_lrs
        ]


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (
        torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value)
        + min_value
    ).exp()


def rand_v_diffusion(
    shape,
    sigma_data=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_split_log_normal(
    shape, loc, scale_1, scale_2, device="cpu", dtype=torch.float32
):
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


class FolderOfImages(data.Dataset):
    """Recursively finds all images in a directory. It does not support
    classes/targets."""

    IMG_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.suffix.lower() in self.IMG_EXTENSIONS
        )

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        image = self.transform(image)
        return (image,)


class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = Path(filename)
        self.columns = columns
        if self.filename.exists():
            self.file = open(self.filename, "a")
        else:
            self.file = open(self.filename, "w")
            self.write(*self.columns)

    def write(self, *args):
        print(*args, sep=",", file=self.file, flush=True)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """A context manager that sets whether TF32 is allowed on cuDNN or matmul."""
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old


# code taken from: https://github.com/wl-zhao/UniPC and modified


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion _internal by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion _internal, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================
        """

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape(
                (1, -1)
            )
            self.log_alpha_array = log_alphas.reshape(
                (
                    1,
                    -1,
                )
            )
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(
                math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            self.schedule = schedule
            if schedule == "cosine":
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device),
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: torch.log(
                torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = (
                2.0
                * (self.beta_1 - self.beta_0)
                * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            )
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(
                torch.zeros((1,)).to(lamb.device), -2.0 * lamb
            )
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = (
                lambda log_alpha_t: torch.arccos(
                    torch.exp(log_alpha_t + self.cosine_log_alpha_0)
                )
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion _internal."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion _internal beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            ``
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).


    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        output = model(x, t_input, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous
            ), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(
                sigma_t, dims
            )
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous
            ), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return (
                noise
                - guidance_scale
                * expand_dims(sigma_t, dims=cond_grad.dim())
                * cond_grad
            )
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0=True,
        thresholding=False,
        max_val=1.0,
        variant="bh1",
    ):
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s).to(s.device)
            ),
            dims,
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(
            t
        ), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(
                torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims
            )
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(
                lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1
            ).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1)
                .pow(t_order)
                .to(device)
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self, steps, order, skip_type, t_T, t_0, device
    ):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(
        self, x, model_prev_list, t_prev_list, t, order, **kwargs
    ):
        if len(t.shape) == 0:
            t = t.view(-1)
        if "bh" in self.variant:
            return self.multistep_uni_pc_bh_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )
        else:
            assert self.variant == "vary_coeff"
            return self.multistep_uni_pc_vary_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )

    def multistep_uni_pc_vary_update(
        self, x, model_prev_list, t_prev_list, t, order, use_corrector=True
    ):
        print(
            f"using unified predictor-corrector with order {order} (solver type: vary coeff)"
        )
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            print("using corrector")
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= k + 1

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]
                    )
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
                t_prev_0
            ), ns.marginal_log_mean_coeff(t)
            x_t_ = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (
                sigma_t * h_phi_1
            ) * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]
                    )
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(
        self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True
    ):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0
        ), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0] if self.predict_x0 else h[0]
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * h_phi_1, dims) * model_prev_0
            )

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * (
                    corr_res + rhos_c[-1] * D1_t
                )
        else:
            x_t_ = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * (
                    corr_res + rhos_c[-1] * D1_t
                )
        return x_t, model_t

    def sample(
        self,
        x,
        timesteps,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
        corrector=False,
        callback=None,
        disable_pbar=False,
    ):
        # t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        # t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        steps = len(timesteps) - 1
        if method == "multistep":
            assert steps >= order
            # timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # with torch.no_grad():
            for step_index in trange(steps, disable=disable_pbar):
                if step_index == 0:
                    vec_t = timesteps[0].expand((x.shape[0]))
                    model_prev_list = [self.model_fn(x, vec_t)]
                    t_prev_list = [vec_t]
                elif step_index < order:
                    init_order = step_index
                    # Init the first `order` values by lower order multistep DPM-Solver.
                    # for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        init_order,
                        use_corrector=True,
                    )
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                else:
                    extra_final_step = 0
                    if step_index == (steps - 1):
                        extra_final_step = 1
                    for step in range(step_index, step_index + 1 + extra_final_step):
                        vec_t = timesteps[step].expand(x.shape[0])
                        if lower_order_final:
                            step_order = min(order, steps + 1 - step)
                        else:
                            step_order = order
                        # print('this step order:', step_order)
                        if step == steps:
                            # print('do not run corrector at the last step')
                            use_corrector = False
                        else:
                            use_corrector = True
                        x, model_x = self.multistep_uni_pc_update(
                            x,
                            model_prev_list,
                            t_prev_list,
                            vec_t,
                            step_order,
                            use_corrector=use_corrector,
                        )
                        for i in range(order - 1):
                            t_prev_list[i] = t_prev_list[i + 1]
                            model_prev_list[i] = model_prev_list[i + 1]
                        t_prev_list[-1] = vec_t
                        # We do not need to evaluate the final model value.
                        if step < steps:
                            if model_x is None:
                                model_x = self.model_fn(x, vec_t)
                            model_prev_list[-1] = model_x
                if callback is not None:
                    callback({"x": x, "i": step_index, "denoised": model_prev_list[-1]})
        else:
            raise NotImplementedError()
        # if denoise_to_zero:
        #     x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1
    )
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)
    ).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


class SigmaConvert:
    schedule = ""

    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def predict_eps_sigma(model, input, sigma_in, **kwargs):
    sigma = sigma_in.view(sigma_in.shape[:1] + (1,) * (input.ndim - 1))
    input = input * ((sigma**2 + 1.0) ** 0.5)
    return (input - model(input, sigma_in, **kwargs)) / sigma


def sample_unipc(
    model, noise, sigmas, extra_args=None, callback=None, disable=False, variant="bh1"
):
    timesteps = sigmas.clone()
    if sigmas[-1] == 0:
        timesteps = sigmas[:]
        timesteps[-1] = 0.001
    else:
        timesteps = sigmas.clone()
    ns = SigmaConvert()

    noise = noise / torch.sqrt(1.0 + timesteps[0] ** 2.0)
    model_type = "noise"

    model_fn = model_wrapper(
        lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
        ns,
        model_type=model_type,
        guidance_type="uncond",
        model_kwargs=extra_args,
    )

    order = min(3, len(timesteps) - 2)
    uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=variant)
    x = uni_pc.sample(
        noise,
        timesteps=timesteps,
        skip_type="time_uniform",
        method="multistep",
        order=order,
        lower_order_final=True,
        callback=callback,
        disable_pbar=disable,
    )
    x /= ns.marginal_alpha(timesteps[-1])
    return x


def sample_unipc_bh2(
    model, noise, sigmas, extra_args=None, callback=None, disable=False
):
    return sample_unipc(
        model, noise, sigmas, extra_args, callback, disable, variant="bh2"
    )


import struct

import safetensors.torch
import torch


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:
                logging.warning(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
                )
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=Unpickler)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(
                    prefix_from, resblock, x, y
                )
                k_to = "{}encoder.layers.{}.{}.{}".format(
                    prefix_to, resblock, resblock_to_replace[x], y
                )
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(
                prefix_from, resblock, y
            )
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(
                        prefix_to, resblock, p[x], y
                    )
                    sd[k_to] = weights[shape_from * x : shape_from * (x + 1)]

    return sd


def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = (
            sd.pop(tp).transpose(0, 1).contiguous()
        )
    return sd


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
}


def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = (
                "input_blocks.{}.0.op.{}".format(n, k)
            )

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = (
            "middle_block.1.{}".format(b)
        )
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map[
                "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)
            ] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map[
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])
            ] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(
                            n, t, b
                        )
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map[
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k)
                    ] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat(
            [math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1)
        )[:batch_size]
    return tensor


def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty(
        [batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device
    )
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output


def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict


def safetensors_header(safetensors_path, max_size=100 * 1024 * 1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack("<Q", header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj, attr, value):
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        """slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC"""

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
            1
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
            1
        ) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1)
        )
        coords_1 = torch.nn.functional.interpolate(
            coords_1, size=(1, length_new), mode="bilinear"
        )
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape(
                (1, 1, 1, -1)
            )
            + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2, size=(1, length_new), mode="bilinear"
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def lanczos(samples, width, height):
    images = [
        Image.fromarray(
            np.clip(255.0 * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)
        )
        for image in samples
    ]
    images = [
        image.resize((width, height), resample=Image.Resampling.LANCZOS)
        for image in images
    ]
    images = [
        torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0)
        for image in images
    ]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y : old_height - y, x : old_width - x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, width, height)
    elif upscale_method == "lanczos":
        return lanczos(s, width, height)
    else:
        return torch.nn.functional.interpolate(
            s, size=(height, width), mode=upscale_method
        )


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    output = torch.empty(
        (
            samples.shape[0],
            out_channels,
            round(samples.shape[2] * upscale_amount),
            round(samples.shape[3] * upscale_amount),
        ),
        device=output_device,
    )
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device=output_device,
        )
        out_div = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device=output_device,
        )
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                x = max(0, min(s.shape[-1] - overlap, x))
                y = max(0, min(s.shape[-2] - overlap, y))
                s_in = s[:, :, y : y + tile_y, x : x + tile_x]

                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                        1.0 / feather
                    ) * (t + 1)
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                        1.0 / feather
                    ) * (t + 1)
                out[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += (
                    ps * mask
                )
                out_div[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b : b + 1] = out / out_div
    return output


PROGRESS_BAR_ENABLED = True


def set_progress_bar_enabled(enabled):
    global PROGRESS_BAR_ENABLED
    PROGRESS_BAR_ENABLED = enabled


PROGRESS_BAR_HOOK = None


def set_progress_bar_global_hook(function):
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function


class ProgressBar:
    def __init__(self, total):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview)

    def update(self, value):
        self.update_absolute(self.current + value)


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name = "{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (
                "lora",
                (lora[A_name], lora[B_name], alpha, mid, dora_scale),
            )
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)

        ######## loha
        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            patch_dict[to_load[x]] = (
                "loha",
                (
                    lora[hada_w1_a_name],
                    lora[hada_w1_b_name],
                    alpha,
                    lora[hada_w2_a_name],
                    lora[hada_w2_b_name],
                    hada_t1,
                    hada_t2,
                    dora_scale,
                ),
            )
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)

        ######## lokr
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (
            (lokr_w1 is not None)
            or (lokr_w2 is not None)
            or (lokr_w1_a is not None)
            or (lokr_w2_a is not None)
        ):
            patch_dict[to_load[x]] = (
                "lokr",
                (
                    lokr_w1,
                    lokr_w2,
                    alpha,
                    lokr_w1_a,
                    lokr_w1_b,
                    lokr_w2_a,
                    lokr_w2_b,
                    lokr_t2,
                    dora_scale,
                ),
            )

        # glora
        a1_name = "{}.a1.weight".format(x)
        a2_name = "{}.a2.weight".format(x)
        b1_name = "{}.b1.weight".format(x)
        b2_name = "{}.b2.weight".format(x)
        if a1_name in lora:
            patch_dict[to_load[x]] = (
                "glora",
                (
                    lora[a1_name],
                    lora[a2_name],
                    lora[b1_name],
                    lora[b2_name],
                    alpha,
                    dora_scale,
                ),
            )
            loaded_keys.add(a1_name)
            loaded_keys.add(a2_name)
            loaded_keys.add(b1_name)
            loaded_keys.add(b2_name)

        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                loaded_keys.add(b_norm_name)
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                    "diff",
                    (b_norm,),
                )

        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                "diff",
                (diff_bias,),
            )
            loaded_keys.add(diff_bias_name)

    for x in lora.keys():
        if x not in loaded_keys:
            logging.warning("lora key not loaded: {}".format(x))
    return patch_dict


def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(
                    b, LORA_CLIP_MAP[c]
                )
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                    b, c
                )  # diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(
                    b, LORA_CLIP_MAP[c]
                )  # SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                    b, c
                )  # diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(
                        b, LORA_CLIP_MAP[c]
                    )  # SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(
                        b, c
                    )  # diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(
                        b, LORA_CLIP_MAP[c]
                    )
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                        b, c
                    )  # diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(
                        b, LORA_CLIP_MAP[c]
                    )  # cascade lora:
                    key_map[lora_key] = k

    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_prior_te_text_projection"] = k  # cascade lora?
        # key_map["text_encoder.text_projection"] = k
        # key_map["lora_te_text_projection"] = k

    return key_map


def model_lora_keys_unet(model, key_map={}):
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = (
                k  # cascade lora:
            )

    diffusers_keys = unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(
                    p, k[: -len(".weight")].replace(".to_", ".processor.to_")
                )
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key
    return key_map


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class CONDNoiseShape(CONDRegular):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if (
                diff > 4
            ):  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(
                    1, crossattn_max_len // c.shape[1], 1
                )  # padding with repeat doesn't change result
            out.append(c)
        return torch.cat(out)


class CONDConstant(CONDRegular):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


import argparse
import enum


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--listen",
    type=str,
    default="127.0.0.1",
    metavar="IP",
    nargs="?",
    const="0.0.0.0",
    help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)",
)
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument(
    "--tls-keyfile",
    type=str,
    help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function",
)
parser.add_argument(
    "--tls-certfile",
    type=str,
    help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function",
)
parser.add_argument(
    "--enable-cors-header",
    type=str,
    default=None,
    metavar="ORIGIN",
    nargs="?",
    const="*",
    help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.",
)
parser.add_argument(
    "--max-upload-size",
    type=float,
    default=100,
    help="Set the maximum upload size in MB.",
)

parser.add_argument(
    "--extra-model-paths-config",
    type=str,
    default=None,
    metavar="PATH",
    nargs="+",
    action="append",
    help="Load one or more extra_model_paths.yaml files.",
)
parser.add_argument(
    "--output-directory",
    type=str,
    default=None,
    help="Set the ComfyUI output directory.",
)
parser.add_argument(
    "--temp-directory",
    type=str,
    default=None,
    help="Set the ComfyUI temp directory (default is in the ComfyUI directory).",
)
parser.add_argument(
    "--input-directory", type=str, default=None, help="Set the ComfyUI input directory."
)
parser.add_argument(
    "--auto-launch",
    action="store_true",
    help="Automatically launch ComfyUI in the default browser.",
)
parser.add_argument(
    "--disable-auto-launch",
    action="store_true",
    help="Disable auto launching the browser.",
)
parser.add_argument(
    "--cuda-device",
    type=int,
    default=None,
    metavar="DEVICE_ID",
    help="Set the id of the cuda device this instance will use.",
)
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument(
    "--cuda-malloc",
    action="store_true",
    help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).",
)
cm_group.add_argument(
    "--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync."
)

parser.add_argument(
    "--dont-upcast-attention",
    action="store_true",
    help="Disable upcasting of attention. Can boost speed but increase the chances of black images.",
)

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument(
    "--force-fp32",
    action="store_true",
    help="Force fp32 (If this makes your GPU work better please report it).",
)
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument(
    "--bf16-unet",
    action="store_true",
    help="Run the UNET in bf16. This should only be used for testing stuff.",
)
fpunet_group.add_argument(
    "--fp16-unet", action="store_true", help="Store unet weights in fp16."
)
fpunet_group.add_argument(
    "--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn."
)
fpunet_group.add_argument(
    "--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2."
)

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument(
    "--fp16-vae",
    action="store_true",
    help="Run the VAE in fp16, might cause black images.",
)
fpvae_group.add_argument(
    "--fp32-vae", action="store_true", help="Run the VAE in full precision fp32."
)
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument(
    "--fp8_e4m3fn-text-enc",
    action="store_true",
    help="Store text encoder weights in fp8 (e4m3fn variant).",
)
fpte_group.add_argument(
    "--fp8_e5m2-text-enc",
    action="store_true",
    help="Store text encoder weights in fp8 (e5m2 variant).",
)
fpte_group.add_argument(
    "--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16."
)
fpte_group.add_argument(
    "--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32."
)

parser.add_argument(
    "--directml",
    type=int,
    nargs="?",
    metavar="DIRECTML_DEVICE",
    const=-1,
    help="Use torch-directml.",
)

parser.add_argument(
    "--disable-ipex-optimize",
    action="store_true",
    help="Disables ipex.optimize when loading _internal with Intel GPUs.",
)


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


parser.add_argument(
    "--preview-method",
    type=LatentPreviewMethod,
    default=LatentPreviewMethod.NoPreviews,
    help="Default preview method for sampler nodes.",
    action=EnumAction,
)

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument(
    "--use-split-cross-attention",
    action="store_true",
    help="Use the split cross attention optimization. Ignored when xformers is used.",
)
attn_group.add_argument(
    "--use-quad-cross-attention",
    action="store_true",
    help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.",
)
attn_group.add_argument(
    "--use-pytorch-cross-attention",
    action="store_true",
    help="Use the new pytorch 2.0 cross attention function.",
)

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument(
    "--gpu-only",
    action="store_true",
    help="Store and run everything (text encoders/CLIP _internal, etc... on the GPU).",
)
vram_group.add_argument(
    "--highvram",
    action="store_true",
    help="By default _internal will be unloaded to CPU memory after being used. This option keeps them in GPU memory.",
)
vram_group.add_argument(
    "--normalvram",
    action="store_true",
    help="Used to force normal vram use if lowvram gets automatically enabled.",
)
vram_group.add_argument(
    "--lowvram", action="store_true", help="Split the unet in parts to use less vram."
)
vram_group.add_argument(
    "--novram", action="store_true", help="When lowvram isn't enough."
)
vram_group.add_argument(
    "--cpu", action="store_true", help="To use the CPU for everything (slow)."
)

parser.add_argument(
    "--disable-smart-memory",
    action="store_true",
    help="Force ComfyUI to agressively offload to regular ram instead of keeping _internal in vram when it can.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.",
)

parser.add_argument(
    "--dont-print-server", action="store_true", help="Don't print server output."
)
parser.add_argument(
    "--quick-test-for-ci", action="store_true", help="Quick test for CI."
)
parser.add_argument(
    "--windows-standalone-build",
    action="store_true",
    help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disable saving prompt metadata in files.",
)

parser.add_argument(
    "--multi-user", action="store_true", help="Enables per-user storage."
)

parser.add_argument("--verbose", action="store_true", help="Enables more debug prints.")

if args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.windows_standalone_build:
    args.auto_launch = True

if args.disable_auto_launch:
    args.auto_launch = False

import logging

logging_level = logging.INFO
if args.verbose:
    logging_level = logging.DEBUG

logging.basicConfig(format="%(message)s", level=logging_level)

# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import torch
import torch.nn as nn
from einops import repeat


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor, device) -> torch.Tensor:
        # skip_time_mix = rearrange(repeat(skip_time_mix, 'b -> (b t) () () ()', t=t), '(b t) 1 ... -> b 1 t ...', t=t)
        if self.merge_strategy == "fixed":
            # make shape compatible
            # alpha = repeat(self.mix_factor, '1 -> b () t  () ()', t=t, b=bs)
            alpha = self.mix_factor.to(device)
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor.to(device))
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                alpha = rearrange(
                    torch.sigmoid(self.mix_factor.to(device)), "... -> ... 1"
                )
            else:
                alpha = torch.where(
                    image_only_indicator.bool(),
                    torch.ones(1, 1, device=image_only_indicator.device),
                    rearrange(
                        torch.sigmoid(self.mix_factor.to(image_only_indicator.device)),
                        "... -> ... 1",
                    ),
                )
            alpha = rearrange(alpha, self.rearrange_pattern)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial,
        x_temporal,
        image_only_indicator=None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.device)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)

    elif schedule == "squaredcos_cap_v2":  # used for karlo prior
        # return early
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def make_ddim_timesteps(
    ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True
):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(
            f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
        )
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


from functools import partial

import torch.nn as nn


class AbstractLowScaleModel(nn.Module):
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        super(AbstractLowScaleModel, self).__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

    def q_sample(self, x_start, t, noise=None, seed=None):
        if noise is None:
            if seed is None:
                noise = torch.randn_like(x_start)
            else:
                noise = torch.randn(
                    x_start.size(),
                    dtype=x_start.dtype,
                    layout=x_start.layout,
                    generator=torch.manual_seed(seed),
                ).to(x_start.device)
        return (
            extract_into_tensor(
                self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * x_start
            + extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * noise
        )

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x


class SimpleImageConcat(AbstractLowScaleModel):
    # no noise level conditioning
    def __init__(self):
        super(SimpleImageConcat, self).__init__(noise_schedule_config=None)
        self.max_noise_level = 0

    def forward(self, x):
        # fix to constant noise level
        return x, torch.zeros(x.shape[0], device=x.device).long()


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(
                0, self.max_noise_level, (x.shape[0],), device=x.device
            ).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        z = self.q_sample(x, noise_level, seed=seed)
        return z, noise_level


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from torch import nn
from torch.autograd import Function


class vector_quantize(Function):
    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook**2, dim=1)
            x_sqr = torch.sum(x**2, dim=1, keepdim=True)

            dist = torch.addmm(
                codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0
            )
            _, indices = dist.min(dim=1)

            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)

            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)


class VectorQuantize(nn.Module):
    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neigbour embeddings to each of the inputs,
        with the same size as the input, vq and commitment components for the loss as a touple
        in the second output and the indices of the quantized vectors in the third:
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()

        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.vq = vector_quantize.apply

        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer("ema_element_count", torch.ones(k))
            self.register_buffer(
                "ema_weight_sum", torch.zeros_like(self.codebook.weight)
            )

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return (x + epsilon) / (n + x.size(0) * epsilon) * n

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)

        self.ema_element_count = (self.ema_decay * self.ema_element_count) + (
            (1 - self.ema_decay) * elem_count
        )
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)
        self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + (
            (1 - self.ema_decay) * weight_sum
        )

        self.codebook.weight.data = (
            self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)
        )

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        # pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()

        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


class ResBlock(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1), nn.Conv2d(c, c, kernel_size=3, groups=c)
        )

        # channelwise
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

        # Init weights
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas

        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        try:
            x = x + self.depthwise(x_temp) * mods[2]
        except:  # operation not implemented for bf16
            x_temp = self.depthwise[0](x_temp.float()).to(x.dtype)
            x = x + self.depthwise[1](x_temp) * mods[2]

        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = (
            x
            + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]
        )

        return x


class StageA(nn.Module):
    def __init__(
        self,
        levels=2,
        bottleneck_blocks=12,
        c_hidden=384,
        c_latent=4,
        codebook_size=8192,
    ):
        super().__init__()
        self.c_latent = c_latent
        c_levels = [c_hidden // (2**i) for i in reversed(range(levels))]

        # Encoder blocks
        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(2), nn.Conv2d(3 * 4, c_levels[0], kernel_size=1)
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(
                    nn.Conv2d(
                        c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1
                    )
                )
            block = ResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(
            nn.Sequential(
                nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
                nn.BatchNorm2d(
                    c_latent
                ),  # then normalize them to have mean 0 and std 1
            )
        )
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]

        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)

        # Decoder blocks
        up_blocks = [nn.Sequential(nn.Conv2d(c_latent, c_levels[-1], kernel_size=1))]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    nn.ConvTranspose2d(
                        c_levels[levels - 1 - i],
                        c_levels[levels - 2 - i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], 3 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return qe, x, indices, vq_loss + commit_loss * 0.25
        else:
            return x

    def decode(self, x):
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


class Discriminator(nn.Module):
    def __init__(self, c_in=3, c_cond=0, c_hidden=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(c_in, c_hidden // (2**d), kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = c_hidden // (2 ** max((d - i), 0))
            c_out = c_hidden // (2 ** max((d - 1 - i), 0))
            layers.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
                )
            )
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d(
            (c_hidden + c_cond) if c_cond > 0 else c_hidden, 1, kernel_size=1
        )
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(
                cond.size(0),
                cond.size(1),
                1,
                1,
            ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x


import torch
import torchsde
from scipy import integrate
from torch import nn
from tqdm.auto import trange, tqdm


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(
        ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)
    )
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [
                torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs)
                for s in seed
            ]
        else:
            self.trees = [
                torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed
            ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack(
                [
                    tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device)
                    for tree in self.trees
                ]
            ) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(
        self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False
    ):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(
            torch.as_tensor(sigma_max)
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(
            torch.as_tensor(sigma_next)
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        cur_order = min(i + 1, order)
        coeffs = [
            linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
            for j in range(cur_order)
        ]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""

    def __init__(
        self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8
    ):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = (
            self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        )
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (
            x - self.model(x, sigma, *args, **self.extra_args, **kwargs)
        ) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        )
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (
            x
            - self.sigma(s2) * (r2 * h).expm1() * eps
            - self.sigma(s2)
            * (r2 / r1)
            * ((r2 * h).expm1() / (r2 * h) - 1)
            * (eps_r1 - eps)
        )
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        )
        return x_3, eps_cache

    def dpm_solver_fast(
        self, x, t_start, t_end, nfe, eta=0.0, s_noise=1.0, noise_sampler=None
    ):
        noise_sampler = (
            default_noise_sampler(x) if noise_sampler is None else noise_sampler
        )
        if not t_end > t_start and eta:
            raise ValueError("eta must be 0 for reverse sampling")

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.0

            eps, eps_cache = self.eps(eps_cache, "eps", x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback(
                    {"x": x, "i": i, "t": ts[i], "t_up": t, "denoised": denoised}
                )

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(
                    x, t, t_next_, eps_cache=eps_cache
                )
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(
                    x, t, t_next_, eps_cache=eps_cache
                )
            else:
                x, eps_cache = self.dpm_solver_3_step(
                    x, t, t_next_, eps_cache=eps_cache
                )

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(
        self,
        x,
        t_start,
        t_end,
        order=3,
        rtol=0.05,
        atol=0.0078,
        h_init=0.05,
        pcoeff=0.0,
        icoeff=1.0,
        dcoeff=0.0,
        accept_safety=0.81,
        eta=0.0,
        s_noise=1.0,
        noise_sampler=None,
    ):
        noise_sampler = (
            default_noise_sampler(x) if noise_sampler is None else noise_sampler
        )
        if order not in {2, 3}:
            raise ValueError("order should be 2 or 3")
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError("eta must be 0 for reverse sampling")
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(
            h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety
        )
        info = {"steps": 0, "nfe": 0, "n_accept": 0, "n_reject": 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = (
                torch.minimum(t_end, s + pid.h)
                if forward
                else torch.maximum(t_end, s + pid.h)
            )
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.0

            eps, eps_cache = self.eps(eps_cache, "eps", x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(
                    x, s, t_, eps_cache=eps_cache
                )
            else:
                x_low, eps_cache = self.dpm_solver_2_step(
                    x, s, t_, r1=1 / 3, eps_cache=eps_cache
                )
                x_high, eps_cache = self.dpm_solver_3_step(
                    x, s, t_, eps_cache=eps_cache
                )
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info["n_accept"] += 1
            else:
                info["n_reject"] += 1
            info["nfe"] += order
            info["steps"] += 1

            if self.info_callback is not None:
                self.info_callback(
                    {
                        "x": x,
                        "i": info["steps"] - 1,
                        "t": s,
                        "t_up": s,
                        "denoised": denoised,
                        "error": error,
                        "h": pid.h,
                        **info,
                    }
                )

        return x, info


@torch.no_grad()
def sample_dpm_fast(
    model,
    x,
    sigma_min,
    sigma_max,
    n,
    extra_args=None,
    callback=None,
    disable=None,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        return dpm_solver.dpm_solver_fast(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            n,
            eta,
            s_noise,
            noise_sampler,
        )


@torch.no_grad()
def sample_dpm_adaptive(
    model,
    x,
    sigma_min,
    sigma_max,
    extra_args=None,
    callback=None,
    disable=None,
    order=3,
    rtol=0.05,
    atol=0.0078,
    h_init=0.05,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    accept_safety=0.81,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
    return_info=False,
):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        x, info = dpm_solver.dpm_solver_adaptive(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            order,
            rtol,
            atol,
            h_init,
            pcoeff,
            icoeff,
            dcoeff,
            accept_safety,
            eta,
            s_noise,
            noise_sampler,
        )
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=1 / 2,
):
    """DPM-Solver++ (stochastic)."""
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (
                t - t_next_
            ).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
):
    """DPM-Solver++(2M) SDE."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = (
                sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x
                + (-h - eta_h).expm1().neg() * denoised
            )

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                        1 / r
                    ) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                        denoised - old_denoised
                    )

            if eta:
                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * s_noise
                )

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """DPM-Solver++(3M) SDE."""

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * h * eta).expm1().neg().sqrt()
                    * s_noise
                )

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False
        )
        if noise_sampler is None
        else noise_sampler
    )
    return sample_dpmpp_3m_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
    )


@torch.no_grad()
def sample_dpmpp_2m_sde_gpu(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False
        )
        if noise_sampler is None
        else noise_sampler
    )
    return sample_dpmpp_2m_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        solver_type=solver_type,
    )


@torch.no_grad()
def sample_dpmpp_sde_gpu(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=1 / 2,
):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False
        )
        if noise_sampler is None
        else noise_sampler
    )
    return sample_dpmpp_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        r=r,
    )


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = alpha_cumprod / alpha_cumprod_prev

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += (
            (1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        ).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


def generic_step_sampler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    step_function=None,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        x = step_function(
            x / torch.sqrt(1.0 + sigmas[i] ** 2.0),
            sigmas[i],
            sigmas[i + 1],
            (x - denoised) / sigmas[i],
            noise_sampler,
        )
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(
    model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None
):
    return generic_step_sampler(
        model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step
    )


@torch.no_grad()
def sample_lcm(
    model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(
                sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x
            )
    return x


@torch.no_grad()
def sample_heunpp2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    # From MIT licensed: https://github.com/Carzit/sd-webui-samplers-scheduler/
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == s_end:

            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)

            w = 2 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w1 = 1 - w2

            d_prime = d * w1 + d_2 * w2

            x = x + d_prime * dt

        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]

            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)

            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3

            d_prime = w1 * d + w2 * d_2 + w3 * d_3
            x = x + d_prime * dt
    return x


import torch


class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return (
            model_input * self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            - model_output
            * sigma
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )


class EDM(V_PREDICTION):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return (
            model_input * self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            + model_output
            * sigma
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        self._register_schedule(
            given_betas=None,
            beta_schedule=beta_schedule,
            timesteps=1000,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
        )
        self.sigma_data = 1.0

    def _register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        # self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class ModelSamplingContinuousEDM(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        sigma_min = sampling_settings.get("sigma_min", 0.002)
        sigma_max = sampling_settings.get("sigma_max", 120.0)
        sigma_data = sampling_settings.get("sigma_data", 1.0)
        self.set_parameters(sigma_min, sigma_max, sigma_data)

    def set_parameters(self, sigma_min, sigma_max, sigma_data):
        self.sigma_data = sigma_data
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()

        self.register_buffer("sigmas", sigmas)  # for compatibility with some schedulers
        self.register_buffer("log_sigmas", sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp(
            (math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min
        )


class StableCascadeSampling(ModelSamplingDiscrete):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(sampling_settings.get("shift", 1.0))

    def set_parameters(self, shift=1.0, cosine_s=8e-3):
        self.shift = shift
        self.cosine_s = torch.tensor(cosine_s)
        self._init_alpha_cumprod = (
            torch.cos(self.cosine_s / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
        )

        # This part is just for compatibility with some schedulers in the codebase
        self.num_timesteps = 10000
        sigmas = torch.empty((self.num_timesteps), dtype=torch.float32)
        for x in range(self.num_timesteps):
            t = (x + 1) / self.num_timesteps
            sigmas[x] = self.sigma(t)

        self.set_sigmas(sigmas)

    def sigma(self, timestep):
        alpha_cumprod = (
            torch.cos((timestep + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5)
            ** 2
            / self._init_alpha_cumprod
        )

        if self.shift != 1.0:
            var = alpha_cumprod
            logSNR = (var / (1 - var)).log()
            logSNR += 2 * torch.log(1.0 / torch.tensor(self.shift))
            alpha_cumprod = logSNR.sigmoid()

        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        return ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5

    def timestep(self, sigma):
        var = 1 / ((sigma * sigma) + 1)
        var = var.clamp(0, 1.0)
        s, min_var = self.cosine_s.to(var.device), self._init_alpha_cumprod.to(
            var.device
        )
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0

        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent))


import logging
import sys
from enum import Enum

import psutil
import torch


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move _internal to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but _internal still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

if args.deterministic:
    logging.info("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    import torch_directml

    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    logging.info(
        "Using directml with device: {}".format(
            torch_directml.device_name(device_index)
        )
    )
    # torch_directml.disable_tiled_resources(True)
    lowvram_available = False

try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

if args.cpu:
    cpu_state = CPUState.CPU


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info(
    "Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram)
)
if not args.normalvram and not args.cpu:
    if lowvram_available and total_vram <= 4096:
        logging.warning(
            "Trying to enable lowvram mode because your GPU seems to have 4GB or less. If you don't want this use: --normalvram"
        )
        set_vram_to = VRAMState.LOW_VRAM

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops

        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            logging.info("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                logging.warning(
                    "\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images."
                )
                logging.warning(
                    "Please downgrade or upgrade xformers to a different version.\n"
                )
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if (
                ENABLE_PYTORCH_ATTENTION == False
                and args.use_split_cross_attention == False
                and args.use_quad_cross_attention == False
            ):
                ENABLE_PYTORCH_ATTENTION = True
            if (
                torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_properties(torch.cuda.current_device()).major
                >= 8
            ):
                VAE_DTYPE = torch.bfloat16
    if is_intel_xpu():
        if (
            args.use_split_cross_attention == False
            and args.use_quad_cross_attention == False
        ):
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if args.cpu_vae:
    VAE_DTYPE = torch.float32

if args.fp16_vae:
    VAE_DTYPE = torch.float16
elif args.bf16_vae:
    VAE_DTYPE = torch.bfloat16
elif args.fp32_vae:
    VAE_DTYPE = torch.float32

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.force_fp32:
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    logging.info("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")


def get_torch_device_name(device):
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(
                device, torch.cuda.get_device_name(device), allocator_backend
            )
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")

logging.info("VAE dtype: {}".format(VAE_DTYPE))

current_loaded_models = []


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        try:
            if lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(
                    device_to=patch_model_to,
                    lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights,
                )
            else:
                self.real_model = self.model.patch_model(
                    device_to=patch_model_to, patch_weights=load_weights
                )
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if is_intel_xpu() and not args.disable_ipex_optimize:
            self.real_model = ipex.optimize(
                self.real_model.eval(), graph_mode=True, concat_linear=True
            )

        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter > 0:
            return True
        return False

    def model_unload(self, unpatch_weights=True):
        self.model.unpatch_model(
            self.model.offload_device, unpatch_weights=unpatch_weights
        )
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def __eq__(self, other):
        return self.model is other.model


def minimum_inference_memory():
    return 1024 * 1024 * 1024


def unload_model_clones(model, unload_weights_only=True, force_unload=True):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0
    for i in to_unload:
        if model.clone_has_same_weights(current_loaded_models[i].model):
            same_weights += 1

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight == False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight


def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append(
                    (sys.getrefcount(shift_model.model), shift_model.model_memory(), i)
                )

    for x in sorted(can_unload):
        i = x[-1]
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(
                device, torch_free_too=True
            )
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def load_models_gpu(models, memory_required=0, force_patch_weights=False):
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models = set(models)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        loaded = None

        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            if loaded.should_reload_model(
                force_patch_weights=force_patch_weights
            ):
                current_loaded_models.pop(loaded_model_index).model_unload(
                    unpatch_weights=True
                )
                loaded = None
            else:
                models_already_loaded.append(loaded)

        if loaded is None:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return

    logging.info(
        f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}"
    )

    total_memory_required = {}
    for loaded_model in models_to_load:
        if (
            unload_model_clones(
                loaded_model.model, unload_weights_only=True, force_unload=False
            )
            == True
        ):  # unload clones where the weights are different
            total_memory_required[loaded_model.device] = total_memory_required.get(
                loaded_model.device, 0
            ) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(
                total_memory_required[device] * 1.3 + extra_mem,
                device,
                models_already_loaded,
            )

    for loaded_model in models_to_load:
        weights_unloaded = unload_model_clones(
            loaded_model.model, unload_weights_only=False, force_unload=False
        )  # unload the rest of the clones where the weights can stay loaded
        if weights_unloaded is not None:
            loaded_model.weights_loaded = not weights_unloaded

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (
            vram_set_state == VRAMState.LOW_VRAM
            or vram_set_state == VRAMState.NORMAL_VRAM
        ):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(
                max(64 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3)
            )
            if model_size > (
                current_free_mem - inference_memory
            ):  # only switch to lowvram if really necessary
                vram_set_state = VRAMState.LOW_VRAM
            else:
                lowvram_model_memory = 0

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 64 * 1024 * 1024

        cur_loaded_model = loaded_model.model_load(
            lowvram_model_memory, force_patch_weights=force_patch_weights
        )
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models(keep_clone_weights_loaded=False):
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            elif (
                sys.getrefcount(current_loaded_models[i].real_model) <= 3
            ):  # references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_dtype(
    device=None,
    model_params=0,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32


# None means no manual cast
def unet_manual_cast(
    weight_dtype,
    inference_device,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    elif bf16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32


def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    elif args.fp16_text_enc:
        return torch.float16
    elif args.fp32_text_enc:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_dtype():
    global VAE_DTYPE
    return VAE_DTYPE


def get_autocast_device(dev):
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    return False
    # return True


def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
    return False


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = (
                torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            )
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, "type"):
        if device.type == type:
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, "cpu")


def is_device_mps(device):
    return is_device_type(device, "mps")


def is_device_cuda(device):
    return is_device_type(device, "cuda")


def should_use_fp16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    fp16_works = False
    # FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    # when the model doesn't actually fit on the card
    nvidia_10_series = [
        "1080",
        "1070",
        "titan x",
        "p3000",
        "p3200",
        "p4000",
        "p4200",
        "p5000",
        "p5200",
        "p6000",
        "1060",
        "1050",
        "p40",
        "p100",
        "p6",
        "p4",
    ]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = [
        "1660",
        "1650",
        "1630",
        "T500",
        "T550",
        "T600",
        "MX550",
        "MX450",
        "CMP 30HX",
        "T2000",
        "T1000",
        "T1200",
    ]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
    if device is not None:
        if is_device_cpu(device):
            return False

    if device is not None:
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if (
            force or is_nvidia()
        ):  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models():
    free_memory(1e30, get_torch_device())


def resolve_lowvram_weight(weight, model, key):
    return weight



import threading


class InterruptProcessingException(Exception):
    pass


interrupt_processing_mutex = threading.RLock()

interrupt_processing = False


def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value


def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing


def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()


import torch


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(
        noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])),
        size=(shape[2], shape[3]),
        mode="bilinear",
    )
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models


def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def get_additional_models(conds, dtype):
    """loads additional _internal in conditioning"""
    cnets = []
    gligen = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def cleanup_additional_models(models):
    """cleanup additional _internal that were loaded"""
    for m in models:
        if hasattr(m, "cleanup"):
            m.cleanup()


def prepare_sampling(model, noise_shape, conds):
    device = model.load_device
    real_model = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    load_models_gpu(
        [model] + models,
        model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]))
        + inference_memory,
    )
    real_model = model.model

    return real_model, conds, models


def cleanup_models(conds, models):
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))


def cast_bias_weight(s, input):
    bias = None
    non_blocking = device_supports_non_blocking(input.device)
    if s.bias is not None:
        bias = s.bias.to(
            device=input.device, dtype=input.dtype, non_blocking=non_blocking
        )
        if s.bias_function is not None:
            bias = s.bias_function(bias)
    weight = s.weight.to(
        device=input.device, dtype=input.dtype, non_blocking=non_blocking
    )
    if s.weight_function is not None:
        weight = s.weight_function(weight)
    return weight, bias


class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = None
    bias_function = None


class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True


import collections


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if "timestep_start" in conds:
        timestep_start = conds["timestep_start"]
        if timestep_in[0] > timestep_start:
            return None
    if "timestep_end" in conds:
        timestep_end = conds["timestep_end"]
        if timestep_in[0] < timestep_end:
            return None
    if "area" in conds:
        area = conds["area"]
    if "strength" in conds:
        strength = conds["strength"]

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
    if "mask" in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds["mask"]
        assert mask.shape[1] == x_in.shape[2]
        assert mask.shape[2] == x_in.shape[3]
        mask = (
            mask[
                : input_x.shape[0],
                area[2] : area[0] + area[2],
                area[3] : area[1] + area[3],
            ]
            * mask_strength
        )
        mask = mask.unsqueeze(1).repeat(
            input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1
        )
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if "mask" not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:, :, t : 1 + t, :] *= (1.0 / rr) * (t + 1)
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:, :, area[0] - 1 - t : area[0] - t, :] *= (1.0 / rr) * (t + 1)
        if area[3] != 0:
            for t in range(rr):
                mult[:, :, :, t : 1 + t] *= (1.0 / rr) * (t + 1)
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:, :, :, area[1] - 1 - t : area[1] - t] *= (1.0 / rr) * (t + 1)

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(
            batch_size=x_in.shape[0], device=x_in.device, area=area
        )

    control = conds.get("control", None)

    patches = None
    if "gligen" in conds:
        gligen = conds["gligen"]
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(
                input_x.shape, gligen[2], input_x.device
            )
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches["middle_patch"] = [gligen_patch]

    cond_obj = collections.namedtuple(
        "cond_obj", ["input_x", "mult", "conditioning", "area", "control", "patches"]
    )
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def calc_cond_batch(model, conds, x_in, timestep, model_options):
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c["control"] = control.get_control(
                input_x, timestep_, c, len(cond_or_uncond)
            )

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c["transformer_options"] = transformer_options

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            out_conds[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += (
                output[o] * mult[o]
            )
            out_counts[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds


def calc_cond_uncond_batch(
    model, cond, uncond, x_in, timestep, model_options
):
    logging.warning(
        "WARNING: The comfy.samplers.calc_cond_uncond_batch function is deprecated please use the calc_cond_batch one instead."
    )
    return tuple(calc_cond_batch(model, [cond, uncond], x_in, timestep, model_options))


def cfg_function(
    model,
    cond_pred,
    uncond_pred,
    cond_scale,
    x,
    timestep,
    model_options={},
    cond=None,
    uncond=None,
):
    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options,
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "model": model,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x,
        }
        cfg_result = fn(args)

    return cfg_result


# The main sampling function shared by all the samplers
# Returns denoised
def sampling_function(
    model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None
):
    if (
        math.isclose(cond_scale, 1.0)
        and model_options.get("disable_cfg1_optimization", False) == False
    ):
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)
    return cfg_function(
        model,
        out[0],
        out[1],
        cond_scale,
        x,
        timestep,
        model_options=model_options,
        cond=cond,
        uncond=uncond_,
    )


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas

    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](
                    sigma,
                    denoise_mask,
                    extra_options={"model": self.inner_model, "sigmas": self.sigmas},
                )
            latent_mask = 1.0 - denoise_mask
            x = (
                x * denoise_mask
                + self.inner_model.inner_model.model_sampling.noise_scaling(
                    sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)),
                    self.noise,
                    self.latent_image,
                )
                * latent_mask
            )
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out


def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def ddim_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = max(len(s.sigmas) // steps, 1)
    x = 1
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty


def resolve_areas_and_cond_masks(conditions, h, w, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if "area" in c:
            area = c["area"]
            if area[0] == "percentage":
                modified = c.copy()
                area = (
                    max(1, round(area[1] * h)),
                    max(1, round(area[2] * w)),
                    round(area[3] * h),
                    round(area[4] * w),
                )
                modified["area"] = area
                c = modified
                conditions[i] = c

        if "mask" in c:
            mask = c["mask"]
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(1)

            if modified.get("set_area_to_bounds", False):
                bounds = torch.max(torch.abs(mask), dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified["area"] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (
                        box[3] - box[1] + 1,
                        box[2] - box[0] + 1,
                        box[1],
                        box[0],
                    )
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified["area"] = area

            modified["mask"] = mask
            conditions[i] = modified


def create_cond_with_same_area_if_none(conds, c):
    if "area" not in c:
        return

    c_area = c["area"]
    smallest = None
    for x in conds:
        if "area" in x:
            a = x["area"]
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif "area" not in smallest:
                            smallest = x
                        else:
                            if smallest["area"][0] * smallest["area"][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if "area" in smallest:
        if smallest["area"] == c_area:
            return

    out = c.copy()
    out["model_conds"] = smallest[
        "model_conds"
    ].copy()
    conds += [out]


def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if "start_percent" in x:
            timestep_start = s.percent_to_sigma(x["start_percent"])
        if "end_percent" in x:
            timestep_end = s.percent_to_sigma(x["end_percent"])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if timestep_start is not None:
                n["timestep_start"] = timestep_start
            if timestep_end is not None:
                n["timestep_end"] = timestep_end
            conds[t] = n


def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if "control" in x:
            x["control"].pre_run(model, percent_to_timestep_function)


def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if "area" not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if "area" not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        params["width"] = params.get("width", noise.shape[3] * 8)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        for k in out:
            model_conds[k] = out[k]
        x["model_conds"] = model_conds
        conds[t] = x
    return conds


class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


KSAMPLER_NAMES = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
]


class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get(
            "random", False
        ):
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = (
                torch.randn(noise.shape, generator=generator, device="cpu")
                .to(noise.dtype)
                .to(noise.device)
            )
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":

        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return sample_dpm_fast(
                model,
                noise,
                sigma_min,
                sigmas[0],
                total_steps,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
            )

        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":

        def dpm_adaptive_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpm_adaptive(
                model,
                noise,
                sigma_min,
                sigmas[0],
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpm_adaptive_function
    elif sampler_name == "dpmpp_2m_sde":

        def dpmpp_sde_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpmpp_2m_sde(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpmpp_sde_function
    elif sampler_name == "euler_ancestral":

        def euler_ancestral_function(
            model, noise, sigmas, extra_args, callback, disable
        ):
            if len(sigmas) <= 1:
                return noise

            return sample_euler_ancestral(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = euler_ancestral_function

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(
    model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None
):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks(conds[k], noise.shape[2], noise.shape[3], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(
                model.extra_conds,
                conds[k],
                noise,
                device,
                k,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed,
            )

    # make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(
                    list(
                        filter(
                            lambda c: c.get("control_apply_to_uncond", False) == True,
                            positive,
                        )
                    ),
                    conds[k],
                    "control",
                    lambda cond_cnets, x: cond_cnets[x],
                )
                apply_empty_x_to_equal_area(
                    positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x]
                )

    return conds


class CFGGuider:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(
            self.inner_model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = prepare_sampling(
            self.model_patcher, noise.shape, self.conds
        )
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        output = self.inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
        )

        cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output


def sample(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options={},
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
    )


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


def calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(
            n=steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    elif scheduler_name == "exponential":
        sigmas = get_sigmas_exponential(
            n=steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model_sampling, steps)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model_sampling, steps)
    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model_sampling, steps)
    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model_sampling, steps, sgm=True)
    else:
        logging.error("error invalid scheduler {}".format(scheduler_name))
    return sigmas


def sampler_object(name):
    if name == "uni_pc":
        sampler = KSAMPLER(sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler


class KSampler1:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(
        ("dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2")
    )

    def __init__(
        self,
        model,
        steps,
        device,
        sampler=None,
        scheduler=None,
        denoise=None,
        model_options={},
    ):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps / denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1) :]

    def sample(
        self,
        noise,
        positive,
        negative,
        cfg,
        latent_image=None,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        denoise_mask=None,
        sigmas=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[: last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(
            self.model,
            noise,
            positive,
            negative,
            cfg,
            self.device,
            sampler,
            sigmas,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )


def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn(
            [1] + list(latent_image.size())[1:],
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def prepare_sampling1(model, noise_shape, positive, negative, noise_mask):
    logging.warning(
        "Warning: comfy.sample.prepare_sampling isn't used anymore and can be removed"
    )
    return model, positive, negative, noise_mask, []


def cleanup_additional_models1(models):
    logging.warning(
        "Warning: comfy.sample.cleanup_additional_models isn't used anymore and can be removed"
    )


def sample1(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    noise_mask=None,
    sigmas=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    sampler = KSampler1(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
    )

    samples = sampler.sample(
        noise,
        positive,
        negative,
        cfg=cfg,
        latent_image=latent_image,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        denoise_mask=noise_mask,
        sigmas=sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(intermediate_device())
    return samples


def sample_custom(
    model,
    noise,
    cfg,
    sampler,
    sigmas,
    positive,
    negative,
    latent_image,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    samples = sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        model.load_device,
        sampler,
        sigmas,
        model_options=model.model_options,
        latent_image=latent_image,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(intermediate_device())
    return samples


import inspect
import uuid


def apply_weight_decompose(dora_scale, weight):
    weight_norm = (
        weight.transpose(0, 1)
        .reshape(weight.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[1], *[1] * (weight.dim() - 1))
        .transpose(0, 1)
    )

    return weight * (dora_scale / weight_norm)


def set_model_options_patch_replace(
    model_options, patch, name, block_name, number, transformer_index=None
):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


class ModelPatcher:
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        current_device=None,
        weight_inplace_update=False,
    ):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone):
        if not self.is_clone(clone):
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning(
                    "WARNING: something went wrong, same patch uuid but different length of patches."
                )
            else:
                return True

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(
        self, sampler_cfg_function, disable_cfg1_optimization=False
    ):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = (
                lambda args: sampler_cfg_function(
                    args["cond"], args["uncond"], args["cond_scale"]
                )
            )  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(
        self, post_cfg_function, disable_cfg1_optimization=False
    ):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get(
            "sampler_post_cfg_function", []
        ) + [post_cfg_function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(
        self, patch, name, block_name, number, transformer_index=None
    ):
        self.model_options = set_model_options_patch_replace(
            self.model_options,
            patch,
            name,
            block_name,
            number,
            transformer_index=transformer_index,
        )

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(
        self, patch, block_name, number, transformer_index=None
    ):
        self.set_model_patch_replace(
            patch, "attn1", block_name, number, transformer_index
        )

    def set_model_attn2_replace(
        self, patch, block_name, number, transformer_index=None
    ):
        self.set_model_patch_replace(
            patch, "attn2", block_name, number, transformer_index
        )

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)

    def get_key_patches(self, filter_prefix=None):
        unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_weight_to_device(self, key, device_to=None):
        if key not in self.patches:
            return

        weight = get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(
                device=self.offload_device, copy=inplace_update
            )

        if device_to is not None:
            temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(
            weight.dtype
        )
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)

    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning(
                        "could not patch. key doesn't exist in model: {}".format(key)
                    )
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(
        self, device_to=None, lowvram_model_memory=0, force_patch_weights=False
    ):
        self.patch_model(device_to, patch_weights=False)

        logging.info(
            "loading in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024))
        )

        class LowVramPatch:
            def __init__(self, key, model_patcher):
                self.key = key
                self.model_patcher = model_patcher

            def __call__(self, weight):
                return self.model_patcher.calculate_weight(
                    self.model_patcher.patches[self.key], weight, self.key
                )

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning(
                            "WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(
                                key, w1.shape, weight.shape
                            )
                        )
                    else:
                        weight += alpha * cast_to_device(
                            w1, weight.device, weight.dtype
                        )
            elif patch_type == "lora":  # lora/locon
                mat1 = cast_to_device(v[0], weight.device, torch.float32)
                mat2 = cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    # locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [
                        mat2.shape[1],
                        mat2.shape[0],
                        mat3.shape[2],
                        mat3.shape[3],
                    ]
                    mat2 = (
                        torch.mm(
                            mat2.transpose(0, 1).flatten(start_dim=1),
                            mat3.transpose(0, 1).flatten(start_dim=1),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )
                try:
                    weight += (
                        (
                            alpha
                            * torch.mm(
                                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
                            )
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            cast_to_device(dora_scale, weight.device, torch.float32),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(
                        cast_to_device(w1_a, weight.device, torch.float32),
                        cast_to_device(w1_b, weight.device, torch.float32),
                    )
                else:
                    w1 = cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(
                            cast_to_device(w2_a, weight.device, torch.float32),
                            cast_to_device(w2_b, weight.device, torch.float32),
                        )
                    else:
                        w2 = torch.einsum(
                            "i j k l, j r, i p -> p r k l",
                            cast_to_device(t2, weight.device, torch.float32),
                            cast_to_device(w2_b, weight.device, torch.float32),
                            cast_to_device(w2_a, weight.device, torch.float32),
                        )
                else:
                    w2 = cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(
                        weight.dtype
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            cast_to_device(dora_scale, weight.device, torch.float32),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        cast_to_device(t1, weight.device, torch.float32),
                        cast_to_device(w1b, weight.device, torch.float32),
                        cast_to_device(w1a, weight.device, torch.float32),
                    )

                    m2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        cast_to_device(t2, weight.device, torch.float32),
                        cast_to_device(w2b, weight.device, torch.float32),
                        cast_to_device(w2a, weight.device, torch.float32),
                    )
                else:
                    m1 = torch.mm(
                        cast_to_device(w1a, weight.device, torch.float32),
                        cast_to_device(w1b, weight.device, torch.float32),
                    )
                    m2 = torch.mm(
                        cast_to_device(w2a, weight.device, torch.float32),
                        cast_to_device(w2b, weight.device, torch.float32),
                    )

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            cast_to_device(dora_scale, weight.device, torch.float32),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]

                dora_scale = v[5]

                a1 = cast_to_device(
                    v[0].flatten(start_dim=1), weight.device, torch.float32
                )
                a2 = cast_to_device(
                    v[1].flatten(start_dim=1), weight.device, torch.float32
                )
                b1 = cast_to_device(
                    v[2].flatten(start_dim=1), weight.device, torch.float32
                )
                b2 = cast_to_device(
                    v[3].flatten(start_dim=1), weight.device, torch.float32
                )

                try:
                    weight += (
                        (
                            (
                                torch.mm(b2, b1)
                                + torch.mm(
                                    torch.mm(weight.flatten(start_dim=1), a2), a1
                                )
                            )
                            * alpha
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            cast_to_device(dora_scale, weight.device, torch.float32),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            else:
                logging.warning(
                    "patch type not recognized {} {}".format(patch_type, key)
                )

        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self.model_lowvram:
                for m in self.model.modules():
                    if hasattr(m, "prev_comfy_cast_weights"):
                        m.comfy_cast_weights = m.prev_comfy_cast_weights
                        del m.prev_comfy_cast_weights
                    m.weight_function = None
                    m.bias_function = None

                self.model_lowvram = False
                self.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            if self.weight_inplace_update:
                for k in keys:
                    copy_to_param(self.model, k, self.backup[k])
            else:
                for k in keys:
                    set_attr_param(self.model, k, self.backup[k])

            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()


def conv(n_in, n_out, **kwargs):
    return disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            disable_weight_init.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder():
    return nn.Sequential(
        conv(3, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 4),
    )


def Decoder():
    return nn.Sequential(
        Clamp(),
        conv(4, 64),
        nn.ReLU(),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, 3),
    )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.taesd_encoder = Encoder()
        self.taesd_decoder = Decoder()
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(
                load_torch_file(encoder_path, safe_load=True)
            )
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(
                load_torch_file(decoder_path, safe_load=True)
            )

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        x_sample = self.taesd_decoder(x * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        return self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale


# import pytorch_lightning as pl
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import torch


class DiagonalGaussianRegularizer(torch.nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AbstractAutoencoder(torch.nn.Module):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP _internal, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
        **kwargs,
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logpy.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logpy.info(f"{context}: Restored training weights")

    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        regularizer_config: Dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        self.regularization: AbstractRegularizer = instantiate_from_config(
            regularizer_config
        )

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        super().__init__(
            encoder_config={
                "target": "LightDiffusion.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "LightDiffusion.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = disable_weight_init.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = disable_weight_init.Conv2d(
            embed_dim, ddconfig["z_channels"], 1
        )
        self.embed_dim = embed_dim

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={"target": ("LightDiffusion.DiagonalGaussianRegularizer")},
            **kwargs,
        )


# pytorch_diffusion + derived encoder decoder

import torch.nn as nn

ops = disable_weight_init

if xformers_enabled_vae():
    import xformers
    import xformers.ops


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        try:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        except:  # operation not implemented for bf16
            b, c, h, w = x.shape
            out = torch.empty(
                (b, c, h * 2, w * 2), dtype=x.dtype, layout=x.layout, device=x.device
            )
            split = 8
            l = out.shape[1] // split
            for i in range(0, out.shape[1], l):
                out[:, i : i + l] = torch.nn.functional.interpolate(
                    x[:, i : i + l].to(torch.float32), scale_factor=2.0, mode="nearest"
                ).to(x.dtype)
            del x
            x = out

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = ops.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = ops.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ops.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = ops.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


def slice_attention(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = int(q.shape[-1]) ** (-0.5)

    mem_free_total = get_free_memory(q.device)

    gb = 1024**3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = (
                q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            )
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except OOM_EXCEPTION as e:
            soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            logging.warning(
                "out of memory error, increasing steps and trying again {}".format(
                    steps
                )
            )

    return r1


def normal_attention(q, k, v):
    # compute attention
    b, c, h, w = q.shape

    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    v = v.reshape(b, c, h * w)

    r1 = slice_attention(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def xformers_attention(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(B, C, H, W)
    except NotImplementedError as e:
        out = slice_attention(
            q.view(B, -1, C),
            k.view(B, -1, C).transpose(1, 2),
            v.view(B, -1, C).transpose(1, 2),
        ).reshape(B, C, H, W)
    return out


def pytorch_attention(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(2, 3).reshape(B, C, H, W)
    except OOM_EXCEPTION as e:
        logging.warning(
            "scaled_dot_product_attention OOMed: switched to slice attention"
        )
        out = slice_attention(
            q.view(B, -1, C),
            k.view(B, -1, C).transpose(1, 2),
            v.view(B, -1, C).transpose(1, 2),
        ).reshape(B, C, H, W)
    return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        if xformers_enabled_vae():
            logging.info("Using xformers attention in VAE")
            self.optimized_attention = xformers_attention
        elif pytorch_attention_enabled():
            logging.info("Using pytorch attention in VAE")
            self.optimized_attention = pytorch_attention
        else:
            logging.info("Using split attention in VAE")
            self.optimized_attention = normal_attention

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    return AttnBlock(in_channels)


class Model(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList(
                [
                    ops.Linear(self.ch, self.temb_ch),
                    ops.Linear(self.temb_ch, self.temb_ch),
                ]
            )

        # downsampling
        self.conv_in = ops.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = ops.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        conv_out_op=ops.Conv2d,
        resnet_op=ResnetBlock,
        attn_op=AttnBlock,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = ops.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resnet_op(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(attn_op(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


import logging
from typing import Optional

from torch import nn, einsum

if xformers_enabled():
    import xformers
    import xformers.ops

ops = disable_weight_init

# CrossAttn precision handling
if args.dont_upcast_attention:
    logging.info("disabling upcasting of attention")
    _ATTN_PRECISION = "fp16"
else:
    _ATTN_PRECISION = "fp32"


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU()
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


def attention_basic(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head**-0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        sim = einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = rearrange(
                mask, "b ... -> b (...)"
            )
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = (
                mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
                .expand(b, heads, -1, -1)
                .reshape(-1, mask.shape[-2], mask.shape[-1])
            )
            sim.add_(mask)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


def attention_split(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head**-0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = get_free_memory(q.device)

    if _ATTN_PRECISION == "fp32":
        element_size = 4
    else:
        element_size = q.element_size()

    gb = 1024**3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(
            f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). "
            f"Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free"
        )

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = (
            mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
            .expand(b, heads, -1, -1)
            .reshape(-1, mask.shape[-2], mask.shape[-1])
        )

    # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
    first_op_done = False
    cleared_cache = False
    while True:
        try:
            slice_size = (
                q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            )
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                if _ATTN_PRECISION == "fp32":
                    with torch.autocast(enabled=False, device_type="cuda"):
                        s1 = (
                            einsum(
                                "b i d, b j d -> b i j", q[:, i:end].float(), k.float()
                            )
                            * scale
                        )
                else:
                    s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k) * scale

                if mask is not None:
                    if len(mask.shape) == 2:
                        s1 += mask[i:end]
                    else:
                        s1 += mask[:, i:end]

                s2 = s1.softmax(dim=-1).to(v.dtype)
                del s1
                first_op_done = True

                r1[:, i:end] = einsum("b i j, b j d -> b i d", s2, v)
                del s2
            break
        except OOM_EXCEPTION as e:
            if first_op_done == False:
                soft_empty_cache(True)
                if cleared_cache == False:
                    cleared_cache = True
                    logging.warning(
                        "out of memory error, emptying cache and trying again"
                    )
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                logging.warning(
                    "out of memory error, increasing steps and trying again {}".format(
                        steps
                    )
                )
            else:
                raise e

    del q, k, v

    r1 = (
        r1.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return r1


BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
except:
    pass


def attention_xformers(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    if BROKEN_XFORMERS:
        if b * heads > 65535:
            return attention_pytorch(q, k, v, heads, mask)

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty(
            [q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device
        )
        mask_out[:, :, : mask.shape[-1]] = mask
        mask = mask_out[:, :, : mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


optimized_attention = attention_basic

if xformers_enabled():
    logging.info("Using xformers cross attention")
    optimized_attention = attention_xformers
elif pytorch_attention_enabled():
    logging.info("Using pytorch cross attention")
    optimized_attention = attention_pytorch
else:

    logging.info("Using split optimization for cross attention")
    optimized_attention = attention_split

optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
    if small_input:
        if pytorch_attention_enabled():
            return attention_pytorch
        else:
            return attention_basic

    if device == torch.device("cpu"):
        return attention_split

    if mask:
        return optimized_attention_masked

    return optimized_attention


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        ff_in=False,
        inner_dim=None,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(
                dim,
                dim_out=inner_dim,
                dropout=dropout,
                glu=gated_ff,
                dtype=dtype,
                device=device,
                operations=operations,
            )

        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(
            inner_dim,
            dim_out=dim,
            dropout=dropout,
            glu=gated_ff,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(
                query_dim=inner_dim,
                context_dim=context_dim_attn2,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                device=device,
                operations=operations,
            )  # is self-attn if context is none
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(
            self._forward,
            (x, context, transformer_options),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(
                    n, context_attn1, value_attn1, extra_options
                )

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](
                n, context_attn1, value_attn1, extra_options
            )
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(
                        n, context_attn2, value_attn2, extra_options
                    )

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](
                    n, context_attn2, value_attn2, extra_options
                )
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        if not use_linear:
            self.proj_in = operations.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    # timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            operations.Linear(
                self.in_channels, time_embed_dim, dtype=dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, self.in_channels, dtype=dtype, device=device
            ),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        transformer_options={},
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        ).to(x.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            transformer_options["block_index"] = it_
            x = block(
                x,
                context=spatial_context,
                transformer_options=transformer_options,
            )

            x_mix = x
            x_mix = x_mix + emb

            B, S, C = x_mix.shape
            x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
            x_mix = mix_block(x_mix, context=time_context)
            x_mix = rearrange(
                x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            )

            x = self.time_mixer(
                x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator
            )

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


import torch


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.k_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.v_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

        self.out_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}


class CLIPMLP(torch.nn.Module):
    def __init__(
        self, embed_dim, intermediate_size, activation, dtype, device, operations
    ):
        super().__init__()
        self.fc1 = operations.Linear(
            embed_dim, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(
            intermediate_size, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(
            embed_dim,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                CLIPLayer(
                    embed_dim,
                    heads,
                    intermediate_size,
                    intermediate_activation,
                    dtype,
                    device,
                    operations,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(
            x.device, mask=mask is not None, small_input=True
        )

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(
        self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )
        self.final_layer_norm = operations.LayerNorm(
            embed_dim, dtype=dtype, device=device
        )

    def forward(
        self,
        input_tokens,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
    ):
        x = self.embeddings(input_tokens)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_channels=3,
        patch_size=14,
        image_size=224,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.class_embedding = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=dtype, device=device)
        )

        self.patch_embedding = operations.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return torch.cat(
            [
                self.class_embedding.to(embeds.device).expand(
                    pixel_values.shape[0], 1, -1
                ),
                embeds,
            ],
            dim=1,
        ) + self.position_embedding.weight.to(embeds.device)


class CLIPVision(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]

        self.embeddings = CLIPVisionEmbeddings(
            embed_dim,
            config_dict["num_channels"],
            config_dict["patch_size"],
            config_dict["image_size"],
            dtype=torch.float32,
            device=device,
            operations=operations,
        )
        self.pre_layrnorm = operations.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output


class CLIPVisionModelProjection(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.vision_model = CLIPVision(config_dict, dtype, device, operations)
        self.visual_projection = operations.Linear(
            config_dict["hidden_size"], config_dict["projection_dim"], bias=False
        )

    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        return (x[0], x[1], out)


class Output:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)


def clip_preprocess(image, size=224):
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype
    )
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype
    )
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        scale = size / min(image.shape[2], image.shape[3])
        image = torch.nn.functional.interpolate(
            image,
            size=(round(scale * image.shape[2]), round(scale * image.shape[3])),
            mode="bicubic",
            antialias=True,
        )
        h = (image.shape[2] - size) // 2
        w = (image.shape[3] - size) // 2
        image = image[:, :, h : h + size, w : w + size]
    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


class ClipVisionModel:
    def __init__(self, json_config):
        with open(json_config) as f:
            config = json.load(f)

        self.load_device = text_encoder_device()
        offload_device = text_encoder_offload_device()
        self.dtype = text_encoder_dtype(self.load_device)
        self.model = CLIPVisionModelProjection(
            config, self.dtype, offload_device, manual_cast
        )
        self.model.eval()

        self.patcher = ModelPatcher(
            self.model, load_device=self.load_device, offload_device=offload_device
        )

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image):
        load_model_gpu(self.patcher)
        pixel_values = clip_preprocess(image.to(self.load_device)).float()
        out = self.model(pixel_values=pixel_values, intermediate_output=-2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(intermediate_device())
        outputs["image_embeds"] = out[2].to(intermediate_device())
        outputs["penultimate_hidden_states"] = out[1].to(intermediate_device())
        return outputs


def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(
                prefix
            ): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(
                prefix
            ): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(
                prefix
            ): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd["visual_projection.weight"] = sd.pop("{}proj".format(prefix)).transpose(
                0, 1
            )

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd


def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = ".\\_internal\\clip\\clip_vision_config_g.json"
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = ".\\_internal\\clip\\clip_vision_config_h.json"
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        json_config = ".\\_internal\\clip\\clip_vision_config_vitl.json"
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip


def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)


from inspect import isfunction

import torch
from torch import nn

ops = manual_cast


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = ops.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward2(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(ops.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), ops.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            operations=ops,
        )
        self.ff = FeedForward2(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(
            self.norm1(x), objs, objs
        )
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj
        # feature
        self.linear = ops.Linear(context_dim, query_dim)

        self.attn = CrossAttention(
            query_dim=query_dim,
            context_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            operations=ops,
        )
        self.ff = FeedForward2(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):
        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = (
            x
            + self.scale
            * torch.tanh(self.alpha_attn)
            * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, 0:N_visual, :]
        )
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj
        # feature
        self.linear = ops.Linear(context_dim, query_dim)

        self.attn = CrossAttention(
            query_dim=query_dim, context_dim=query_dim, dim_head=d_head, operations=ops
        )
        self.ff = FeedForward2(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):
        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)

        # sanity check
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual
        out = self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, N_visual:, :]
        out = out.permute(0, 2, 1).reshape(B, -1, size_g, size_g)
        out = torch.nn.functional.interpolate(out, (size_v, size_v), mode="bicubic")
        residual = out.reshape(B, -1, N_visual).permute(0, 2, 1)

        # add residual to visual feature
        x = x + self.scale * torch.tanh(self.alpha_attn) * residual
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class FourierEmbedder:
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            ops.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            ops.Linear(512, 512),
            nn.SiLU(),
            ops.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(
            torch.zeros([self.position_dim])
        )

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)
        positive_embeddings = positive_embeddings

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.to(
            device=boxes.device, dtype=boxes.dtype
        ).view(1, 1, -1)
        xyxy_null = self.null_position_feature.to(
            device=boxes.device, dtype=boxes.dtype
        ).view(1, 1, -1)

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs


class Gligen(nn.Module):
    def __init__(self, modules, position_net, key_dim):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.position_net = position_net
        self.key_dim = key_dim
        self.max_objs = 30
        self.current_device = torch.device("cpu")

    def _set_position(self, boxes, masks, positive_embeddings):
        objs = self.position_net(boxes, masks, positive_embeddings)

        def func(x, extra_options):
            key = extra_options["transformer_index"]
            module = self.module_list[key]
            return module(x, objs.to(device=x.device, dtype=x.dtype))

        return func

    def set_position(self, latent_image_shape, position_params, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device="cpu")
        boxes = []
        positive_embeddings = []
        for p in position_params:
            x1 = (p[4]) / w
            y1 = (p[3]) / h
            x2 = (p[4] + p[2]) / w
            y2 = (p[3] + p[1]) / h
            masks[len(boxes)] = 1.0
            boxes += [torch.tensor((x1, y1, x2, y2)).unsqueeze(0)]
            positive_embeddings += [p[0]]
        append_boxes = []
        append_conds = []
        if len(boxes) < self.max_objs:
            append_boxes = [torch.zeros([self.max_objs - len(boxes), 4], device="cpu")]
            append_conds = [
                torch.zeros([self.max_objs - len(boxes), self.key_dim], device="cpu")
            ]

        box_out = torch.cat(boxes + append_boxes).unsqueeze(0).repeat(batch, 1, 1)
        masks = masks.unsqueeze(0).repeat(batch, 1)
        conds = (
            torch.cat(positive_embeddings + append_conds)
            .unsqueeze(0)
            .repeat(batch, 1, 1)
        )
        return self._set_position(
            box_out.to(device), masks.to(device), conds.to(device)
        )

    def set_empty(self, latent_image_shape, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device="cpu").repeat(batch, 1)
        box_out = torch.zeros([self.max_objs, 4], device="cpu").repeat(batch, 1, 1)
        conds = torch.zeros([self.max_objs, self.key_dim], device="cpu").repeat(
            batch, 1, 1
        )
        return self._set_position(
            box_out.to(device), masks.to(device), conds.to(device)
        )


def load_gligen(sd):
    sd_k = sd.keys()
    output_list = []
    key_dim = 768
    for a in ["input_blocks", "middle_block", "output_blocks"]:
        for b in range(20):
            k_temp = filter(
                lambda k: "{}.{}.".format(a, b) in k and ".fuser." in k, sd_k
            )
            k_temp = map(lambda k: (k, k.split(".fuser.")[-1]), k_temp)

            n_sd = {}
            for k in k_temp:
                n_sd[k[1]] = sd[k[0]]
            if len(n_sd) > 0:
                query_dim = n_sd["linear.weight"].shape[0]
                key_dim = n_sd["linear.weight"].shape[1]

                if key_dim == 768:  # SD1.x
                    n_heads = 8
                    d_head = query_dim // n_heads
                else:
                    d_head = 64
                    n_heads = query_dim // d_head

                gated = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
                gated.load_state_dict(n_sd, strict=False)
                output_list.append(gated)

    if "position_net.null_positive_feature" in sd_k:
        in_dim = sd["position_net.null_positive_feature"].shape[0]
        out_dim = sd["position_net.linears.4.weight"].shape[0]

        class WeightsLoader(torch.nn.Module):
            pass

        w = WeightsLoader()
        w.position_net = PositionNet(in_dim, out_dim)
        w.load_state_dict(sd, strict=False)

    gligen = Gligen(output_list, w.position_net, key_dim)
    return gligen


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn


class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None


class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None


class OptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead

        self.to_q = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        out = optimized_attention(q, k, v, self.heads)

        return self.out_proj(out)


class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(
            c, nhead, dtype=dtype, device=device, operations=operations
        )
        # self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True, dtype=dtype, device=device)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        # x = self.attn(x, kv, kv, need_weights=False)[0]
        x = self.attn(x, kv, kv)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    return LayerNorm2d


class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"

    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim, dtype=dtype, device=device))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return (
            self.gamma.to(device=x.device, dtype=x.dtype) * (x * Nx)
            + self.beta.to(device=x.device, dtype=x.dtype)
            + x
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        c,
        c_skip=0,
        kernel_size=3,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=None,
    ):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = operations.Conv2d(
            c,
            c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=c,
            dtype=dtype,
            device=device,
        )
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d_op(operations)(
            c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.channelwise = nn.Sequential(
            operations.Linear(c + c_skip, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device),
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


class AttnBlock2(nn.Module):
    def __init__(
        self,
        c,
        c_cond,
        nhead,
        self_attn=True,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d_op(operations)(
            c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.attention = Attention2D(
            c, nhead, dropout, dtype=dtype, device=device, operations=operations
        )
        self.kv_mapper = nn.Sequential(
            nn.SiLU(), operations.Linear(c_cond, c, dtype=dtype, device=device)
        )

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = LayerNorm2d_op(operations)(
            c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.channelwise = nn.Sequential(
            operations.Linear(c, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device),
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(
        self, c, c_timestep, conds=["sca"], dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.mapper = operations.Linear(c_timestep, c * 2, dtype=dtype, device=device)
        self.conds = conds
        for cname in conds:
            setattr(
                self,
                f"mapper_{cname}",
                operations.Linear(c_timestep, c * 2, dtype=dtype, device=device),
            )

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(
                2, dim=1
            )
            a, b = a + ac, b + bc
        return x * (1 + a) + b


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torchvision
from torch import nn


class CNetResBlock(nn.Module):
    def __init__(self, c, dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.Sequential(
            LayerNorm2d_op(operations)(c, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c, c, kernel_size=3, padding=1),
            LayerNorm2d_op(operations)(c, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.blocks(x)


class ControlNet(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_proj=2048,
        proj_blocks=None,
        bottleneck_mode=None,
        dtype=None,
        device=None,
        operations=nn,
    ):
        super().__init__()
        if bottleneck_mode is None:
            bottleneck_mode = "effnet"
        self.proj_blocks = proj_blocks
        if bottleneck_mode == "effnet":
            embd_channels = 1280
            self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
            if c_in != 3:
                in_weights = self.backbone[0][0].weight.data
                self.backbone[0][0] = operations.Conv2d(
                    c_in,
                    24,
                    kernel_size=3,
                    stride=2,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
                if c_in > 3:
                    # nn.init.constant_(self.backbone[0][0].weight, 0)
                    self.backbone[0][0].weight.data[:, :3] = in_weights[:, :3].clone()
                else:
                    self.backbone[0][0].weight.data = in_weights[:, :c_in].clone()
        elif bottleneck_mode == "simple":
            embd_channels = c_in
            self.backbone = nn.Sequential(
                operations.Conv2d(
                    embd_channels,
                    embd_channels * 4,
                    kernel_size=3,
                    padding=1,
                    dtype=dtype,
                    device=device,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                operations.Conv2d(
                    embd_channels * 4,
                    embd_channels,
                    kernel_size=3,
                    padding=1,
                    dtype=dtype,
                    device=device,
                ),
            )
        elif bottleneck_mode == "large":
            self.backbone = nn.Sequential(
                operations.Conv2d(
                    c_in, 4096 * 4, kernel_size=1, dtype=dtype, device=device
                ),
                nn.LeakyReLU(0.2, inplace=True),
                operations.Conv2d(
                    4096 * 4, 1024, kernel_size=1, dtype=dtype, device=device
                ),
                *[
                    CNetResBlock(
                        1024, dtype=dtype, device=device, operations=operations
                    )
                    for _ in range(8)
                ],
                operations.Conv2d(
                    1024, 1280, kernel_size=1, dtype=dtype, device=device
                ),
            )
            embd_channels = 1280
        else:
            raise ValueError(f"Unknown bottleneck mode: {bottleneck_mode}")
        self.projections = nn.ModuleList()
        for _ in range(len(proj_blocks)):
            self.projections.append(
                nn.Sequential(
                    operations.Conv2d(
                        embd_channels,
                        embd_channels,
                        kernel_size=1,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                    operations.Conv2d(
                        embd_channels,
                        c_proj,
                        kernel_size=1,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
            # nn.init.constant_(self.projections[-1][-1].weight, 0)  # zero output projection
        self.xl = False
        self.input_channels = c_in
        self.unshuffle_amount = 8

    def forward(self, x):
        x = self.backbone(x)
        proj_outputs = [None for _ in range(max(self.proj_blocks) + 1)]
        for i, idx in enumerate(self.proj_blocks):
            proj_outputs[idx] = self.projections[i](x)
        return proj_outputs


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from torch import nn


class StageB(nn.Module):
    def __init__(
        self,
        c_in=4,
        c_out=4,
        c_r=64,
        patch_size=2,
        c_cond=1280,
        c_hidden=[320, 640, 1280, 1280],
        nhead=[-1, -1, 20, 20],
        blocks=[[2, 6, 28, 6], [6, 28, 6, 2]],
        block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]],
        level_config=["CT", "CT", "CTA", "CTA"],
        c_clip=1280,
        c_clip_seq=4,
        c_effnet=16,
        c_pixels=3,
        kernel_size=3,
        dropout=[0, 0, 0.0, 0.0],
        self_attn=True,
        t_conds=["sca"],
        stable_cascade_stage=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.effnet_mapper = nn.Sequential(
            operations.Conv2d(
                c_effnet, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device
            ),
            nn.GELU(),
            operations.Conv2d(
                c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device
            ),
            LayerNorm2d_op(operations)(
                c_hidden[0],
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            ),
        )
        self.pixels_mapper = nn.Sequential(
            operations.Conv2d(
                c_pixels, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device
            ),
            nn.GELU(),
            operations.Conv2d(
                c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device
            ),
            LayerNorm2d_op(operations)(
                c_hidden[0],
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            ),
        )
        self.clip_mapper = operations.Linear(
            c_clip, c_cond * c_clip_seq, dtype=dtype, device=device
        )
        self.clip_norm = operations.LayerNorm(
            c_cond, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            operations.Conv2d(
                c_in * (patch_size**2),
                c_hidden[0],
                kernel_size=1,
                dtype=dtype,
                device=device,
            ),
            LayerNorm2d_op(operations)(
                c_hidden[0],
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            ),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlock(
                    c_hidden,
                    c_skip,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "A":
                return AttnBlock2(
                    c_hidden,
                    c_cond,
                    nhead,
                    self_attn=self_attn,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "F":
                return FeedForwardBlock(
                    c_hidden,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "T":
                return TimestepBlock(
                    c_hidden,
                    c_r,
                    conds=t_conds,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            else:
                raise Exception(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        LayerNorm2d_op(operations)(
                            c_hidden[i - 1],
                            elementwise_affine=False,
                            eps=1e-6,
                            dtype=dtype,
                            device=device,
                        ),
                        operations.Conv2d(
                            c_hidden[i - 1],
                            c_hidden[i],
                            kernel_size=2,
                            stride=2,
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(
                        operations.Conv2d(
                            c_hidden[i],
                            c_hidden[i],
                            kernel_size=1,
                            dtype=dtype,
                            device=device,
                        )
                    )
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        LayerNorm2d_op(operations)(
                            c_hidden[i],
                            elementwise_affine=False,
                            eps=1e-6,
                            dtype=dtype,
                            device=device,
                        ),
                        operations.ConvTranspose2d(
                            c_hidden[i],
                            c_hidden[i - 1],
                            kernel_size=2,
                            stride=2,
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(
                        operations.Conv2d(
                            c_hidden[i],
                            c_hidden[i],
                            kernel_size=1,
                            dtype=dtype,
                            device=device,
                        )
                    )
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d_op(operations)(
                c_hidden[0],
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            ),
            operations.Conv2d(
                c_hidden[0],
                c_out * (patch_size**2),
                kernel_size=1,
                dtype=dtype,
                device=device,
            ),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---

    #     self.apply(self._init_weights)  # General init
    #     nn.init.normal_(self.clip_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)  # conditionings
    #     torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
    #     nn.init.constant_(self.clf[1].weight, 0)  # outputs
    #
    #     # blocks
    #     for level_block in self.down_blocks + self.up_blocks:
    #         for block in level_block:
    #             if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
    #                 block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
    #             elif isinstance(block, TimestepBlock):
    #                 for layer in block.modules():
    #                     if isinstance(layer, nn.Linear):
    #                         nn.init.constant_(layer.weight, 0)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def gen_c_embeddings(self, clip):
        if len(clip.shape) == 2:
            clip = clip.unsqueeze(1)
        clip = self.clip_mapper(clip).view(
            clip.size(0), clip.size(1) * self.c_clip_seq, -1
        )
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(
            self.down_blocks, self.down_downscalers, self.down_repeat_mappers
        )
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (
                            x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)
                        ):
                            x = torch.nn.functional.interpolate(
                                x, skip.shape[-2:], mode="bilinear", align_corners=True
                            )
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, effnet, clip, pixels=None, **kwargs):
        if pixels is None:
            pixels = x.new_zeros(x.size(0), 3, 8, 8)

        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat(
                [r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1
            )
        clip = self.gen_c_embeddings(clip)

        # Model Blocks
        x = self.embedding(x)
        x = x + self.effnet_mapper(
            nn.functional.interpolate(
                effnet, size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        )
        x = x + nn.functional.interpolate(
            self.pixels_mapper(pixels),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(
                self_params.device
            ) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(
                self_buffers.device
            ) * (1 - beta)


"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from torch import nn


# from .controlnet import ControlNetDeliverer


class UpDownBlock2d(nn.Module):
    def __init__(
        self, c_in, c_out, mode, enabled=True, dtype=None, device=None, operations=None
    ):
        super().__init__()
        assert mode in ["up", "down"]
        interpolation = (
            nn.Upsample(
                scale_factor=2 if mode == "up" else 0.5,
                mode="bilinear",
                align_corners=True,
            )
            if enabled
            else nn.Identity()
        )
        mapping = operations.Conv2d(
            c_in, c_out, kernel_size=1, dtype=dtype, device=device
        )
        self.blocks = nn.ModuleList(
            [interpolation, mapping] if mode == "up" else [mapping, interpolation]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StageC(nn.Module):
    def __init__(
        self,
        c_in=16,
        c_out=16,
        c_r=64,
        patch_size=1,
        c_cond=2048,
        c_hidden=[2048, 2048],
        nhead=[32, 32],
        blocks=[[8, 24], [24, 8]],
        block_repeat=[[1, 1], [1, 1]],
        level_config=["CTA", "CTA"],
        c_clip_text=1280,
        c_clip_text_pooled=1280,
        c_clip_img=768,
        c_clip_seq=4,
        kernel_size=3,
        dropout=[0.0, 0.0],
        self_attn=True,
        t_conds=["sca", "crp"],
        switch_level=[False],
        stable_cascade_stage=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.clip_txt_mapper = operations.Linear(
            c_clip_text, c_cond, dtype=dtype, device=device
        )
        self.clip_txt_pooled_mapper = operations.Linear(
            c_clip_text_pooled, c_cond * c_clip_seq, dtype=dtype, device=device
        )
        self.clip_img_mapper = operations.Linear(
            c_clip_img, c_cond * c_clip_seq, dtype=dtype, device=device
        )
        self.clip_norm = operations.LayerNorm(
            c_cond, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            operations.Conv2d(
                c_in * (patch_size**2),
                c_hidden[0],
                kernel_size=1,
                dtype=dtype,
                device=device,
            ),
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlock(
                    c_hidden,
                    c_skip,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "A":
                return AttnBlock2(
                    c_hidden,
                    c_cond,
                    nhead,
                    self_attn=self_attn,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "F":
                return FeedForwardBlock(
                    c_hidden,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            elif block_type == "T":
                return TimestepBlock(
                    c_hidden,
                    c_r,
                    conds=t_conds,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            else:
                raise Exception(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        LayerNorm2d_op(operations)(
                            c_hidden[i - 1], elementwise_affine=False, eps=1e-6
                        ),
                        UpDownBlock2d(
                            c_hidden[i - 1],
                            c_hidden[i],
                            mode="down",
                            enabled=switch_level[i - 1],
                            dtype=dtype,
                            device=device,
                            operations=operations,
                        ),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(
                        operations.Conv2d(
                            c_hidden[i],
                            c_hidden[i],
                            kernel_size=1,
                            dtype=dtype,
                            device=device,
                        )
                    )
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        LayerNorm2d_op(operations)(
                            c_hidden[i], elementwise_affine=False, eps=1e-6
                        ),
                        UpDownBlock2d(
                            c_hidden[i],
                            c_hidden[i - 1],
                            mode="up",
                            enabled=switch_level[i - 1],
                            dtype=dtype,
                            device=device,
                            operations=operations,
                        ),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(
                        operations.Conv2d(
                            c_hidden[i],
                            c_hidden[i],
                            kernel_size=1,
                            dtype=dtype,
                            device=device,
                        )
                    )
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d_op(operations)(
                c_hidden[0],
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            ),
            operations.Conv2d(
                c_hidden[0],
                c_out * (patch_size**2),
                kernel_size=1,
                dtype=dtype,
                device=device,
            ),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---

    #     self.apply(self._init_weights)  # General init
    #     nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
    #     torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
    #     nn.init.constant_(self.clf[1].weight, 0)  # outputs
    #
    #     # blocks
    #     for level_block in self.down_blocks + self.up_blocks:
    #         for block in level_block:
    #             if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
    #                 block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
    #             elif isinstance(block, TimestepBlock):
    #                 for layer in block.modules():
    #                     if isinstance(layer, nn.Linear):
    #                         nn.init.constant_(layer.weight, 0)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pooled = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1
        )
        clip_img = self.clip_img_mapper(clip_img).view(
            clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1
        )
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None):
        level_outputs = []
        block_group = zip(
            self.down_blocks, self.down_downscalers, self.down_repeat_mappers
        )
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(
                                    next_cnet,
                                    size=x.shape[-2:],
                                    mode="bilinear",
                                    align_corners=True,
                                ).to(x.dtype)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (
                            x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)
                        ):
                            x = torch.nn.functional.interpolate(
                                x, skip.shape[-2:], mode="bilinear", align_corners=True
                            )
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(
                                    next_cnet,
                                    size=x.shape[-2:],
                                    mode="bilinear",
                                    align_corners=True,
                                ).to(x.dtype)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(
        self, x, r, clip_text, clip_text_pooled, clip_img, control=None, **kwargs
    ):
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat(
                [r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1
            )
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        if control is not None:
            cnet = control.get("input")
        else:
            cnet = None

        # Model Blocks
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip, cnet)
        x = self._up_decode(level_outputs, r_embed, clip, cnet)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(
                self_params.device
            ) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(
                self_buffers.device
            ) * (1 - beta)


import json
import traceback
import zipfile

import torch
from transformers import CLIPTokenizer


def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        out, pooled = self.encode(to_encode)
        if pooled is not None:
            first_pooled = pooled[0:1].to(intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if len(output) == 0:
            return out[-1:].to(intermediate_device()), first_pooled
        return torch.cat(output, dim=-2).to(intermediate_device()), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cpu",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        textmodel_json_config=None,
        dtype=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        enable_attention_masks=False,
        return_projected_pooled=True,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = ".\\_internal\\clip\\sd1_clip_config.json"

        with open(textmodel_json_config) as f:
            config = json.load(f)

        self.transformer = model_class(config, dtype, device, manual_cast)
        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:  # EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        logging.warning(
                            "WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
                                y.shape[0], current_embeds.weight.shape[1]
                            )
                        )
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(
                next_new_token + 1,
                current_embeds.weight.shape[1],
                device=current_embeds.weight.device,
                dtype=current_embeds.weight.dtype,
            )
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))
            ]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if self.enable_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == max_token:
                        break

        outputs = self.transformer(
            tokens,
            attention_mask,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def safe_load_embed_zip(embed_path):
    with zipfile.ZipFile(embed_path) as myzip:
        names = list(filter(lambda a: "data/" in a, myzip.namelist()))
        names.reverse()
        for n in names:
            with myzip.open(n) as myfile:
                data = myfile.read()
                number = len(data) // 4
                length_embed = 1024  # sd2.x
                if number < 768:
                    continue
                if number % 768 == 0:
                    length_embed = 768  # sd1.x
                num_embeds = number // length_embed
                embed = torch.frombuffer(data, dtype=torch.float)
                out = embed.reshape((num_embeds, length_embed)).clone()
                del embed
                return out


def expand_directory_list(directories):
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = [".safetensors", ".pt", ".bin"]
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch

            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            if "weights_only" in torch.load.__code__.co_varnames:
                try:
                    embed = torch.load(
                        embed_path, weights_only=True, map_location="cpu"
                    )
                except:
                    embed_out = safe_load_embed_zip(embed_path)
            else:
                embed = torch.load(embed_path, map_location="cpu")
    except Exception as e:
        logging.warning(
            "{}\n\nerror loading embedding, skipping loading: {}".format(
                traceback.format_exc(), embedding_name
            )
        )
        return None

    if embed_out is None:
        if "string_to_param" in embed:
            values = embed["string_to_param"].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            values = embed.values()
            embed_out = next(iter(values))
    return embed_out


class SDTokenizer:
    def __init__(
        self,
        tokenizer_path=None,
        max_length=77,
        pad_with_end=True,
        embedding_directory=None,
        embedding_size=768,
        embedding_key="clip_l",
        tokenizer_class=CLIPTokenizer,
        has_start_token=True,
        pad_to_max_length=True,
        min_length=None,
    ):
        if tokenizer_path is None:
            tokenizer_path = "_internal\\sd1_tokenizer\\"
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name: str):
        """
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        """
        embed = load_embed(
            embedding_name,
            self.embedding_directory,
            self.embedding_size,
            self.embedding_key,
        )
        if embed is None:
            stripped = embedding_name.strip(",")
            if len(stripped) < len(embedding_name):
                embed = load_embed(
                    stripped,
                    self.embedding_directory,
                    self.embedding_size,
                    self.embedding_key,
                )
                return (embed, embedding_name[len(stripped) :])
        return (embed, "")

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        """
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = (
                unescape_important(weighted_segment).replace("\n", " ").split(" ")
            )
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if (
                    word.startswith(self.embedding_identifier)
                    and self.embedding_directory is not None
                ):
                    embedding_name = word[len(self.embedding_identifier) :].strip("\n")
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(
                            f"warning, embedding:{embedding_name} does not exist, ignoring"
                        )
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append(
                                [(embed[x], weight) for x in range(embed.shape[0])]
                            )
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                # parse word
                tokens.append(
                    [
                        (t, weight)
                        for t in self.tokenizer(word)["input_ids"][
                            self.tokens_start : -1
                        ]
                    ]
                )

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend(
                            [(t, w, i + 1) for t, w in t_group[:remaining_length]]
                        )
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SD1Tokenizer:
    def __init__(self, embedding_directory=None, clip_name="l", tokenizer=SDTokenizer):
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(
            text, return_word_ids
        )
        return out

    def untokenize(self, token_weight_pair):
        return getattr(self, self.clip).untokenize(token_weight_pair)


class SD1ClipModel(torch.nn.Module):
    def __init__(
        self, device="cpu", dtype=None, clip_name="l", clip_model=SDClipModel, **kwargs
    ):
        super().__init__()
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def set_clip_options(self, options):
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self):
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out, pooled = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out, pooled

    def load_sd(self, sd):
        return getattr(self, self.clip).load_sd(sd)


class SD2ClipHModel(SDClipModel):
    def __init__(
        self,
        arch="ViT-H-14",
        device="cpu",
        max_length=77,
        freeze=True,
        layer="penultimate",
        layer_idx=None,
        dtype=None,
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2

        textmodel_json_config = ".\\_internal\\clip\\sd2_clip_config.json"
        super().__init__(
            device=device,
            freeze=freeze,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
        )


class SD2ClipHTokenizer(SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=1024,
        )


class SD2Tokenizer(SD1Tokenizer):
    def __init__(self, embedding_directory=None):
        super().__init__(
            embedding_directory=embedding_directory,
            clip_name="h",
            tokenizer=SD2ClipHTokenizer,
        )


class SD2ClipModel(SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, **kwargs):
        super().__init__(
            device=device,
            dtype=dtype,
            clip_name="h",
            clip_model=SD2ClipHModel,
            **kwargs,
        )


import torch


class SDXLClipG(SDClipModel):
    def __init__(
        self,
        device="cpu",
        max_length=77,
        freeze=True,
        layer="penultimate",
        layer_idx=None,
        dtype=None,
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2

        textmodel_json_config = ".\\_internal\\clip\\clip_config_bigg.json"
        super().__init__(
            device=device,
            freeze=freeze,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )

    def load_sd(self, sd):
        return super().load_sd(sd)


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=1280,
            embedding_key="clip_g",
        )


class SDXLTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = SDTokenizer(embedding_directory=embedding_directory)
        self.clip_g = SDXLClipGTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)


class SDXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        self.clip_l = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device=device,
            dtype=dtype,
            layer_norm_hidden_state=False,
        )
        self.clip_g = SDXLClipG(device=device, dtype=dtype)

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)


class SDXLRefinerClipModel(SD1ClipModel):
    def __init__(self, device="cpu", dtype=None):
        super().__init__(
            device=device, dtype=dtype, clip_name="g", clip_model=SDXLClipG
        )


class StableCascadeClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(
            tokenizer_path,
            pad_with_end=True,
            embedding_directory=embedding_directory,
            embedding_size=1280,
            embedding_key="clip_g",
        )


class StableCascadeTokenizer(SD1Tokenizer):
    def __init__(self, embedding_directory=None):
        super().__init__(
            embedding_directory=embedding_directory,
            clip_name="g",
            tokenizer=StableCascadeClipGTokenizer,
        )


class StableCascadeClipG(SDClipModel):
    def __init__(
        self,
        device="cpu",
        max_length=77,
        freeze=True,
        layer="hidden",
        layer_idx=-1,
        dtype=None,
    ):
        textmodel_json_config = ".\\_internal\\clip\\clip_config_bigg.json"
        super().__init__(
            device=device,
            freeze=freeze,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 49407},
            layer_norm_hidden_state=False,
            enable_attention_masks=True,
        )

    def load_sd(self, sd):
        return super().load_sd(sd)


class StableCascadeClipModel(SD1ClipModel):
    def __init__(self, device="cpu", dtype=None):
        super().__init__(
            device=device, dtype=dtype, clip_name="g", clip_model=StableCascadeClipG
        )


from abc import abstractmethod

import torch as th
import torch.nn as nn

oai_ops = disable_weight_init


class TimestepBlock1(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


# This is needed because accelerate makes a copy of transformer_options which breaks "transformer_index"
def forward_timestep_embed1(
    ts,
    x,
    emb,
    context=None,
    transformer_options={},
    output_shape=None,
    time_context=None,
    num_video_frames=None,
    image_only_indicator=None,
):
    for layer in ts:
        if isinstance(layer, VideoResBlock1):
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, TimestepBlock1):
            x = layer(x, emb)
        elif isinstance(layer, SpatialVideoTransformer):
            x = layer(
                x,
                context,
                time_context,
                num_video_frames,
                image_only_indicator,
                transformer_options,
            )
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample1):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class TimestepEmbedSequential1(nn.Sequential, TimestepBlock1):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, *args, **kwargs):
        return forward_timestep_embed1(self, *args, **kwargs)


class Upsample1(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = operations.conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
                dtype=dtype,
                device=device,
            )

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample1(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = operations.conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
                dtype=dtype,
                device=device,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock1(TimestepBlock1):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample1(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample1(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample1(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample1(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    emb_channels,
                    (
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                    dtype=dtype,
                    device=device,
                ),
            )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(
                dims,
                self.out_channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            )
        else:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 1, dtype=dtype, device=device
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= 1 + scale
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class VideoResBlock1(ResBlock1):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size=3,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        out_channels=None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.time_stack = ResBlock1(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> b 1 t 1 1",
        )

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        num_video_frames: int,
        image_only_indicator=None,
    ) -> th.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x = self.time_stack(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        )
        x = self.time_mixer(
            x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
        )
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class Timestep1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


def apply_control1(h, control, name):
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            try:
                h += ctrl
            except:
                logging.warning(
                    "warning control could not be applied {} {}".format(
                        h.shape, ctrl.shape
                    )
                )
    return h


class UNetModel1(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=th.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_temporal_resblock=False,
        use_temporal_attention=False,
        time_context_dim=None,
        extra_ff_mix_layer=False,
        use_spatial_context=False,
        merge_strategy=None,
        merge_factor=0.0,
        video_kernel_size=None,
        disable_temporal_crossattention=False,
        max_ddpm_temb_period=10000,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None

        self.default_num_video_frames = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(
                model_channels, time_embed_dim, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, time_embed_dim, dtype=self.dtype, device=device
            ),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(
                    num_classes, time_embed_dim, dtype=self.dtype, device=device
                )
            elif self.num_classes == "continuous":
                logging.debug("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(
                            adm_in_channels,
                            time_embed_dim,
                            dtype=self.dtype,
                            device=device,
                        ),
                        nn.SiLU(),
                        operations.Linear(
                            time_embed_dim,
                            time_embed_dim,
                            dtype=self.dtype,
                            device=device,
                        ),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential1(
                    operations.conv_nd(
                        dims,
                        in_channels,
                        model_channels,
                        3,
                        padding=1,
                        dtype=self.dtype,
                        device=device,
                    )
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disable_self_attn=False,
        ):
            if use_temporal_attention:
                return SpatialVideoTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=depth,
                    context_dim=context_dim,
                    time_context_dim=time_context_dim,
                    dropout=dropout,
                    ff_in=extra_ff_mix_layer,
                    use_spatial_context=use_spatial_context,
                    merge_strategy=merge_strategy,
                    merge_factor=merge_factor,
                    checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    max_time_embed_period=max_ddpm_temb_period,
                    dtype=self.dtype,
                    device=device,
                    operations=operations,
                )
            else:
                return SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    device=device,
                    operations=operations,
                )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_channels,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
            dtype=None,
            device=None,
            operations=oai_ops,
        ):
            if self.use_temporal_resblocks:
                return VideoResBlock1(
                    merge_factor=merge_factor,
                    merge_strategy=merge_strategy,
                    video_kernel_size=video_kernel_size,
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            else:
                return ResBlock1(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_channels,
                    use_checkpoint=use_checkpoint,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential1(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Downsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )
        ]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [
                    get_attention_layer(  # always uses a self-attn
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth_middle,
                        context_dim=context_dim,
                        disable_self_attn=disable_middle_self_attn,
                        use_checkpoint=use_checkpoint,
                    ),
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=None,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    ),
                ]
            self.middle_block = TimestepEmbedSequential1(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Upsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            zero_module(
                operations.conv_nd(
                    dims,
                    model_channels,
                    out_channels,
                    3,
                    padding=1,
                    dtype=self.dtype,
                    device=device,
                )
            ),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
                operations.conv_nd(
                    dims, model_channels, n_embed, 1, dtype=self.dtype, device=device
                ),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control1(h, control, "input")
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed1(
                self.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control1(h, control, "middle")

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control1(hsp, control, "output")

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep1(timestep_dim)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean.to(x.device)) * 1.0 / self.data_std.to(x.device)
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std.to(x.device)) + self.data_mean.to(x.device)
        return x

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(
                0, self.max_noise_level, (x.shape[0],), device=x.device
            ).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level, seed=seed)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level


from typing import Iterable, Union

import torch
from einops import rearrange, repeat

ae_ops = disable_weight_init


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


class VideoResBlock(ResnetBlock):
    def __init__(
        self,
        out_channels,
        *args,
        dropout=0.0,
        video_kernel_size=3,
        alpha=0.0,
        merge_strategy="learned",
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = ResBlock1(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=False,
            skip_t_emb=True,
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, bs):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError()

    def forward(self, x, temb, skip_video=False, timesteps=None):
        b, c, h, w = x.shape
        if timesteps is None:
            timesteps = b

        x = super().forward(x, temb)

        if not skip_video:
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = self.time_stack(x, temb)

            alpha = self.get_alpha(bs=b // timesteps).to(x.device)
            x = alpha * x + (1.0 - alpha) * x_mix

            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class AE3DConv(ae_ops.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, Iterable):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)

        self.time_mix_conv = ae_ops.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding,
        )

    def forward(self, input, timesteps=None, skip_video=False):
        if timesteps is None:
            timesteps = input.shape[0]
        x = super().forward(input)
        if skip_video:
            return x
        x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        x = self.time_mix_conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class AttnVideoBlock(AttnBlock):
    def __init__(
        self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"
    ):
        super().__init__(in_channels)
        # no context, single headed, as in base class
        self.time_mix_block = BasicTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=False,
            ff_in=True,
        )

        time_embed_dim = self.in_channels * 4
        self.video_time_embed = torch.nn.Sequential(
            ae_ops.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            ae_ops.Linear(time_embed_dim, self.in_channels),
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def forward(self, x, timesteps=None, skip_time_block=False):
        if skip_time_block:
            return super().forward(x)

        if timesteps is None:
            timesteps = x.shape[0]

        x_in = x
        x = self.attention(x)
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")

        x_mix = x
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)  # b, n_channels
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        alpha = self.get_alpha().to(x.device)
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        return x_in + x

    def get_alpha(
        self,
    ):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


def make_time_attn(
    in_channels,
    attn_type="vanilla",
    attn_kwargs=None,
    alpha: float = 0,
    merge_strategy: str = "learned",
):
    return partialclass(
        AttnVideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy
    )


class Conv2DWrapper(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class VideoDecoder(Decoder):
    available_time_modes = ["all", "conv-only", "attn-only"]

    def __init__(
        self,
        *args,
        video_kernel_size: Union[int, list] = 3,
        alpha: float = 0.0,
        merge_strategy: str = "learned",
        time_mode: str = "conv-only",
        **kwargs,
    ):
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.time_mode = time_mode
        assert (
            self.time_mode in self.available_time_modes
        ), f"time_mode parameter has to be in {self.available_time_modes}"

        if self.time_mode != "attn-only":
            kwargs["conv_out_op"] = partialclass(
                AE3DConv, video_kernel_size=self.video_kernel_size
            )
        if self.time_mode not in ["conv-only", "only-last-conv"]:
            kwargs["attn_op"] = partialclass(
                make_time_attn, alpha=self.alpha, merge_strategy=self.merge_strategy
            )
        if self.time_mode not in ["attn-only", "only-last-conv"]:
            kwargs["resnet_op"] = partialclass(
                VideoResBlock,
                video_kernel_size=self.video_kernel_size,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy,
            )

        super().__init__(*args, **kwargs)

    def get_last_layer(self, skip_time_mix=False, **kwargs):
        if self.time_mode == "attn-only":
            raise NotImplementedError("TODO")
        else:
            return (
                self.conv_out.time_mix_conv.weight
                if not skip_time_mix
                else self.conv_out.weight
            )


# taken from: https://github.com/lllyasviel/ControlNet
# and modified

import torch
import torch.nn as nn


class ControlledUnetModel1(UNetModel1):
    # implemented in the ldm unet
    pass


class ControlNet1(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=torch.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        device=None,
        operations=disable_weight_init,
        **kwargs,
    ):
        super().__init__()
        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )

        transformer_depth = transformer_depth[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(
                model_channels, time_embed_dim, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, time_embed_dim, dtype=self.dtype, device=device
            ),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(
                            adm_in_channels,
                            time_embed_dim,
                            dtype=self.dtype,
                            device=device,
                        ),
                        nn.SiLU(),
                        operations.Linear(
                            time_embed_dim,
                            time_embed_dim,
                            dtype=self.dtype,
                            device=device,
                        ),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential1(
                    operations.conv_nd(
                        dims,
                        in_channels,
                        model_channels,
                        3,
                        padding=1,
                        dtype=self.dtype,
                        device=device,
                    )
                )
            ]
        )
        self.zero_convs = nn.ModuleList(
            [
                self.make_zero_conv(
                    model_channels,
                    operations=operations,
                    dtype=self.dtype,
                    device=device,
                )
            ]
        )

        self.input_hint_block = TimestepEmbedSequential1(
            operations.conv_nd(
                dims, hint_channels, 16, 3, padding=1, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 16, 16, 3, padding=1, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 16, 32, 3, padding=1, stride=2, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 32, 32, 3, padding=1, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 32, 96, 3, padding=1, stride=2, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 96, 96, 3, padding=1, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 96, 256, 3, padding=1, stride=2, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.conv_nd(
                dims, 256, model_channels, 3, padding=1, dtype=self.dtype, device=device
            ),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock1(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                dtype=self.dtype,
                                device=device,
                                operations=operations,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential1(*layers))
                self.zero_convs.append(
                    self.make_zero_conv(
                        ch, operations=operations, dtype=self.dtype, device=device
                    )
                )
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential1(
                        ResBlock1(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Downsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(
                    self.make_zero_conv(
                        ch, operations=operations, dtype=self.dtype, device=device
                    )
                )
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            ResBlock1(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )
        ]
        if transformer_depth_middle >= 0:
            mid_block += [
                SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    device=device,
                    operations=operations,
                ),
                ResBlock1(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    device=device,
                    operations=operations,
                ),
            ]
        self.middle_block = TimestepEmbedSequential1(*mid_block)
        self.middle_block_out = self.make_zero_conv(
            ch, operations=operations, dtype=self.dtype, device=device
        )
        self._feature_size += ch

    def make_zero_conv(self, channels, operations=None, dtype=None, device=None):
        return TimestepEmbedSequential1(
            operations.conv_nd(
                self.dims, channels, channels, 1, padding=0, dtype=dtype, device=device
            )
        )

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        hs = []
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


from enum import Enum

import torch


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5


def model_sampling(model_config, model_type):
    s = ModelSamplingDiscrete

    if model_type == ModelType.EPS:
        c = EPS
    elif model_type == ModelType.V_PREDICTION:
        c = V_PREDICTION
    elif model_type == ModelType.V_PREDICTION_EDM:
        c = V_PREDICTION
        s = ModelSamplingContinuousEDM
    elif model_type == ModelType.STABLE_CASCADE:
        c = EPS
        s = StableCascadeSampling
    elif model_type == ModelType.EDM:
        c = EDM
        s = ModelSamplingContinuousEDM

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel1
    ):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype

        if not unet_config.get("disable_unet_model_creation", False):
            if self.manual_cast_dtype is not None:
                operations = manual_cast
            else:
                operations = disable_weight_init
            self.diffusion_model = unet_model(
                **unet_config, device=device, operations=operations
            )
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(
            xc,
            t,
            context=context,
            control=control,
            transformer_options=transformer_options,
            **extra_conds,
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if len(self.concat_keys) > 0:
            cond_concat = []
            denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
            concat_latent_image = kwargs.get("concat_latent_image", None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get("latent_image", None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)

            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = common_upscale(
                    concat_latent_image,
                    noise.shape[-1],
                    noise.shape[-2],
                    "bilinear",
                    "center",
                )

            concat_latent_image = resize_to_batch_size(
                concat_latent_image, noise.shape[0]
            )

            if denoise_mask is not None:
                if len(denoise_mask.shape) == len(noise.shape):
                    denoise_mask = denoise_mask[:, :1]

                denoise_mask = denoise_mask.reshape(
                    (-1, 1, denoise_mask.shape[-2], denoise_mask.shape[-1])
                )
                if denoise_mask.shape[-2:] != noise.shape[-2:]:
                    denoise_mask = common_upscale(
                        denoise_mask,
                        noise.shape[-1],
                        noise.shape[-2],
                        "bilinear",
                        "center",
                    )
                denoise_mask = resize_to_batch_size(
                    denoise_mask.round(), noise.shape[0]
                )

            for ck in self.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask.to(device))
                    elif ck == "masked_image":
                        cond_concat.append(
                            concat_latent_image.to(device)
                        )  # NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(self.blank_inpaint_image_like(noise))
            data = torch.cat(cond_concat, dim=1)
            out["c_concat"] = CONDNoiseShape(data)

        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out["crossattn_controlnet"] = CONDCrossAttn(cross_attn_cnet)

        c_concat = kwargs.get("noise_concat", None)
        if c_concat is not None:
            out["c_concat"] = CONDNoiseShape(c_concat)

        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(
        self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None
    ):
        extra_sds = []
        if clip_state_dict is not None:
            extra_sds.append(
                self.model_config.process_clip_state_dict_for_saving(clip_state_dict)
            )
        if vae_state_dict is not None:
            extra_sds.append(
                self.model_config.process_vae_state_dict_for_saving(vae_state_dict)
            )
        if clip_vision_state_dict is not None:
            extra_sds.append(
                self.model_config.process_clip_vision_state_dict_for_saving(
                    clip_vision_state_dict
                )
            )

        unet_state_dict = self.diffusion_model.state_dict()
        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(
            unet_state_dict
        )

        if self.get_dtype() == torch.float16:
            extra_sds = map(lambda sd: convert_sd_to(sd, torch.float16), extra_sds)

        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        for sd in extra_sds:
            unet_state_dict.update(sd)

        return unet_state_dict

    def set_inpaint(self):
        self.concat_keys = ("mask", "masked_image")

        def blank_inpaint_image_like(latent_image):
            blank_image = torch.ones_like(latent_image)
            # these are the values for "zero" in pixel space translated to latent space
            blank_image[:, 0] *= 0.8223
            blank_image[:, 1] *= -0.6876
            blank_image[:, 2] *= 0.6364
            blank_image[:, 3] *= 0.1380
            return blank_image

        self.blank_inpaint_image_like = blank_inpaint_image_like

    def memory_required(self, input_shape):
        if xformers_enabled() or pytorch_attention_flash_attention():
            dtype = self.get_dtype()
            if self.manual_cast_dtype is not None:
                dtype = self.manual_cast_dtype
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (area * dtype_size(dtype) / 50) * (1024 * 1024)
        else:
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)


def unclip_adm(
    unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0, seed=None
):
    adm_inputs = []
    weights = []
    noise_aug = []
    for unclip_cond in unclip_conditioning:
        for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
            weight = unclip_cond["strength"]
            noise_augment = unclip_cond["noise_augmentation"]
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(
                adm_cond.to(device),
                noise_level=torch.tensor([noise_level], device=device),
                seed=seed,
            )
            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            weights.append(weight)
            noise_aug.append(noise_augment)
            adm_inputs.append(adm_out)

    if len(noise_aug) > 1:
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(
            adm_out[:, : noise_augmentor.time_embed.dim],
            noise_level=torch.tensor([noise_level], device=device),
        )
        adm_out = torch.cat((c_adm, noise_level_emb), 1)

    return adm_out


class SD21UNCLIP(BaseModel):
    def __init__(
        self,
        model_config,
        noise_aug_config,
        model_type=ModelType.V_PREDICTION,
        device=None,
    ):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

    def encode_adm(self, **kwargs):
        unclip_conditioning = kwargs.get("unclip_conditioning", None)
        device = kwargs["device"]
        if unclip_conditioning is None:
            return torch.zeros((1, self.adm_channels))
        else:
            return unclip_adm(
                unclip_conditioning,
                device,
                self.noise_augmentor,
                kwargs.get("unclip_noise_augment_merge", 0.05),
                kwargs.get("seed", 0) - 10,
            )


def sdxl_pooled(args, noise_augmentor):
    if "unclip_conditioning" in args:
        return unclip_adm(
            args.get("unclip_conditioning", None),
            args["device"],
            noise_augmentor,
            seed=args.get("seed", 0) - 10,
        )[:, :1280]
    else:
        return args["pooled_output"]


class SDXLRefiner(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep1(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(
            **{
                "noise_schedule_config": {
                    "timesteps": 1000,
                    "beta_schedule": "squaredcos_cap_v2",
                },
                "timestep_dim": 1280,
            }
        )

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = (
            torch.flatten(torch.cat(out))
            .unsqueeze(dim=0)
            .repeat(clip_pooled.shape[0], 1)
        )
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SDXL(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep1(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(
            **{
                "noise_schedule_config": {
                    "timesteps": 1000,
                    "beta_schedule": "squaredcos_cap_v2",
                },
                "timestep_dim": 1280,
            }
        )

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = (
            torch.flatten(torch.cat(out))
            .unsqueeze(dim=0)
            .repeat(clip_pooled.shape[0], 1)
        )
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SVD_img2vid(BaseModel):
    def __init__(
        self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None
    ):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep1(256)

    def encode_adm(self, **kwargs):
        fps_id = kwargs.get("fps", 6) - 1
        motion_bucket_id = kwargs.get("motion_bucket_id", 127)
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.Tensor([fps_id])))
        out.append(self.embedder(torch.Tensor([motion_bucket_id])))
        out.append(self.embedder(torch.Tensor([augmentation])))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat

    def extra_conds(self, **kwargs):
        out = {}
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = CONDRegular(adm)

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = common_upscale(
                latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )

        latent_image = resize_to_batch_size(latent_image, noise.shape[0])

        out["c_concat"] = CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = CONDCrossAttn(cross_attn)

        if "time_conditioning" in kwargs:
            out["time_context"] = CONDCrossAttn(kwargs["time_conditioning"])

        out["num_video_frames"] = CONDConstant(noise.shape[0])
        return out


class SV3D_u(SVD_img2vid):
    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat


class SV3D_p(SVD_img2vid):
    def __init__(
        self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None
    ):
        super().__init__(model_config, model_type, device=device)
        self.embedder_512 = Timestep1(512)

    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)
        elevation = kwargs.get(
            "elevation", 0
        )  # elevation and azimuth are in degrees here
        azimuth = kwargs.get("azimuth", 0)
        noise = kwargs.get("noise", None)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))
        out.append(
            self.embedder_512(
                torch.deg2rad(
                    torch.fmod(torch.flatten(90 - torch.Tensor([elevation])), 360.0)
                )
            )
        )
        out.append(
            self.embedder_512(
                torch.deg2rad(torch.fmod(torch.flatten(torch.Tensor([azimuth])), 360.0))
            )
        )

        out = list(map(lambda a: resize_to_batch_size(a, noise.shape[0]), out))
        return torch.cat(out, dim=1)


class Stable_Zero123(BaseModel):
    def __init__(
        self,
        model_config,
        model_type=ModelType.EPS,
        device=None,
        cc_projection_weight=None,
        cc_projection_bias=None,
    ):
        super().__init__(model_config, model_type, device=device)
        self.cc_projection = manual_cast.Linear(
            cc_projection_weight.shape[1],
            cc_projection_weight.shape[0],
            dtype=self.get_dtype(),
            device=device,
        )
        self.cc_projection.weight.copy_(cc_projection_weight)
        self.cc_projection.bias.copy_(cc_projection_bias)

    def extra_conds(self, **kwargs):
        out = {}

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = common_upscale(
                latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )

        latent_image = resize_to_batch_size(latent_image, noise.shape[0])

        out["c_concat"] = CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if cross_attn.shape[-1] != 768:
                cross_attn = self.cc_projection(cross_attn)
            out["c_crossattn"] = CONDCrossAttn(cross_attn)
        return out


class SD_X4Upscaler(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = ImageConcatWithNoiseAugmentation(
            noise_schedule_config={"linear_start": 0.0001, "linear_end": 0.02},
            max_noise_level=350,
        )

    def extra_conds(self, **kwargs):
        out = {}

        image = kwargs.get("concat_image", None)
        noise = kwargs.get("noise", None)
        noise_augment = kwargs.get("noise_augmentation", 0.0)
        device = kwargs["device"]
        seed = kwargs["seed"] - 10

        noise_level = round((self.noise_augmentor.max_noise_level) * noise_augment)

        if image is None:
            image = torch.zeros_like(noise)[:, :3]

        if image.shape[1:] != noise.shape[1:]:
            image = common_upscale(
                image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )

        noise_level = torch.tensor([noise_level], device=device)
        if noise_augment > 0:
            image, noise_level = self.noise_augmentor(
                image.to(device), noise_level=noise_level, seed=seed
            )

        image = resize_to_batch_size(image, noise.shape[0])

        out["c_concat"] = CONDNoiseShape(image)
        out["y"] = CONDRegular(noise_level)
        return out


class IP2P:
    def extra_conds(self, **kwargs):
        out = {}

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        if image.shape[1:] != noise.shape[1:]:
            image = common_upscale(
                image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )

        image = resize_to_batch_size(image, noise.shape[0])

        out["c_concat"] = CONDNoiseShape(self.process_ip2p_image_in(image))
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = CONDRegular(adm)
        return out


class SD15_instructpix2pix(IP2P, BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.process_ip2p_image_in = lambda image: image


class SDXL_instructpix2pix(IP2P, SDXL):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        if model_type == ModelType.V_PREDICTION_EDM:
            self.process_ip2p_image_in = lambda image: latent_formats.SDXL().process_in(
                image
            )  # cosxl ip2p
        else:
            self.process_ip2p_image_in = lambda image: image  # diffusers ip2p


class StableCascade_C(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageC)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out["clip_text_pooled"] = CONDRegular(clip_text_pooled)

        if "unclip_conditioning" in kwargs:
            embeds = []
            for unclip_cond in kwargs["unclip_conditioning"]:
                weight = unclip_cond["strength"]
                embeds.append(
                    unclip_cond["clip_vision_output"].image_embeds.unsqueeze(0) * weight
                )
            clip_img = torch.cat(embeds, dim=1)
        else:
            clip_img = torch.zeros((1, 1, 768))
        out["clip_img"] = CONDRegular(clip_img)
        out["sca"] = CONDRegular(torch.zeros((1,)))
        out["crp"] = CONDRegular(torch.zeros((1,)))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["clip_text"] = CONDCrossAttn(cross_attn)
        return out


class StableCascade_B(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageB)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        noise = kwargs.get("noise", None)

        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out["clip"] = CONDRegular(clip_text_pooled)

        # size of prior doesn't really matter if zeros because it gets resized but I still want it to get batched
        prior = kwargs.get(
            "stable_cascade_prior",
            torch.zeros(
                (1, 16, (noise.shape[2] * 4) // 42, (noise.shape[3] * 4) // 42),
                dtype=noise.dtype,
                layout=noise.layout,
                device=noise.device,
            ),
        )

        out["effnet"] = CONDRegular(prior)
        out["sca"] = CONDRegular(torch.zeros((1,)))
        return out


import torch


class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}


class BASE:
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    manual_cast_dtype = None

    @classmethod
    def matches(s, unet_config, state_dict=None):
        for k in s.unet_config:
            if k not in unet_config or s.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in s.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict, prefix=""):
        return ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        if self.noise_aug_config is not None:
            out = SD21UNCLIP(
                self,
                self.noise_aug_config,
                model_type=self.model_type(state_dict, prefix),
                device=device,
            )
        else:
            out = BaseModel(
                self, model_type=self.model_type(state_dict, prefix), device=device
            )
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        state_dict = state_dict_prefix_replace(
            state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True
        )
        return state_dict

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.text_encoder_key_prefix[0]}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_clip_vision_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        if self.clip_vision_prefix is not None:
            replace_prefix[""] = self.clip_vision_prefix
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.vae_key_prefix[0]}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


import torch


class sm_SD15(BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = SD15

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                "cond_stage_model.transformer.text_model."
            ):
                y = x.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                state_dict[y] = state_dict.pop(x)

        if (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
            in state_dict
        ):
            ids = state_dict[
                "cond_stage_model.transformer.text_model.embeddings.position_ids"
            ]
            if ids.dtype == torch.float32:
                state_dict[
                    "cond_stage_model.transformer.text_model.embeddings.position_ids"
                ] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict:
                state_dict.pop(p)

        replace_prefix = {"clip_l.": "cond_stage_model."}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self):
        return ClipTarget(SD1Tokenizer, SD1ClipModel)


class sm_SD20(BASE):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    latent_format = SD15

    def model_type(self, state_dict, prefix=""):
        if (
            self.unet_config["in_channels"] == 4
        ):  # SD2.0 inpainting _internal are not v prediction
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict.get(k, None)
            if (
                out is not None and torch.std(out, unbiased=False) > 0.09
            ):  # not sure how well this will actually work. I guess we will find out.
                return ModelType.V_PREDICTION
        return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = (
            "clip_h."  # SD2 in sgm format
        )
        replace_prefix["cond_stage_model.model."] = "clip_h."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )
        state_dict = clip_text_transformers_convert(
            state_dict, "clip_h.", "clip_h.transformer."
        )
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self):
        return ClipTarget(SD2Tokenizer, SD2ClipModel)


class smSD21UnclipL(sm_SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 1536,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {
        "noise_schedule_config": {
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
        "timestep_dim": 768,
    }


class smSD21UnclipH(sm_SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2048,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {
        "noise_schedule_config": {
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
        "timestep_dim": 1024,
    }


class smSDXLRefiner(BASE):
    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = SDXL

    def get_model(self, state_dict, prefix="", device=None):
        return SDXLRefiner(self, device=device)

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_g."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )

        state_dict = clip_text_transformers_convert(
            state_dict, "clip_g.", "clip_g.transformer."
        )
        state_dict = state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        state_dict_g = convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix["clip_g"] = "conditioner.embedders.0.model"
        state_dict_g = state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self):
        return ClipTarget(SDXLTokenizer, SDXLRefinerClipModel)


class smSDXL(BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = SDXL

    def model_type(self, state_dict, prefix=""):
        if "edm_mean" in state_dict and "edm_std" in state_dict:  # Playground V2.5
            self.latent_format = SDXL_Playground_2_5()
            self.sampling_settings["sigma_data"] = 0.5
            self.sampling_settings["sigma_max"] = 80.0
            self.sampling_settings["sigma_min"] = 0.002
            return ModelType.EDM
        elif "edm_vpred.sigma_max" in state_dict:
            self.sampling_settings["sigma_max"] = float(
                state_dict["edm_vpred.sigma_max"].item()
            )
            if "edm_vpred.sigma_min" in state_dict:
                self.sampling_settings["sigma_min"] = float(
                    state_dict["edm_vpred.sigma_min"].item()
                )
            return ModelType.V_PREDICTION_EDM
        elif "v_pred" in state_dict:
            return ModelType.V_PREDICTION
        else:
            return ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        out = SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = (
            "clip_l.transformer.text_model"
        )
        replace_prefix["conditioner.embedders.1.model."] = "clip_g."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )

        state_dict = state_dict_key_replace(state_dict, keys_to_replace)
        state_dict = clip_text_transformers_convert(
            state_dict, "clip_g.", "clip_g.transformer."
        )
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        keys_to_replace = {}
        state_dict_g = convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = (
            torch.arange(77).expand((1, -1))
        )
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix["clip_g"] = "conditioner.embedders.1.model"
        replace_prefix["clip_l"] = "conditioner.embedders.0"
        state_dict_g = state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self):
        return ClipTarget(SDXLTokenizer, SDXLClipModel)


class smSSD1B(smSDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class smSegmind_Vega(smSDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class smKOALA_700M(smSDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 5],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class smKOALA_1B(smSDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 6],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class smSVD_img2vid(BASE):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 768,
        "use_temporal_attention": True,
        "use_temporal_resblock": True,
    }

    clip_vision_prefix = "conditioner.embedders.0.open_clip.model.visual."

    latent_format = SD15

    sampling_settings = {"sigma_max": 700.0, "sigma_min": 0.002}

    def get_model(self, state_dict, prefix="", device=None):
        out = SVD_img2vid(self, device=device)
        return out

    def clip_target(self):
        return None


class smSV3D_u(smSVD_img2vid):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 256,
        "use_temporal_attention": True,
        "use_temporal_resblock": True,
    }

    vae_key_prefix = ["conditioner.embedders.1.encoder."]

    def get_model(self, state_dict, prefix="", device=None):
        out = SV3D_u(self, device=device)
        return out


class smSV3D_p(smSV3D_u):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 1280,
        "use_temporal_attention": True,
        "use_temporal_resblock": True,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = SV3D_p(self, device=device)
        return out


class smStable_Zero123(BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    required_keys = {
        "cc_projection.weight": None,
        "cc_projection.bias": None,
    }

    clip_vision_prefix = "cond_stage_model.model.visual."

    latent_format = SD15

    def get_model(self, state_dict, prefix="", device=None):
        out = Stable_Zero123(
            self,
            device=device,
            cc_projection_weight=state_dict["cc_projection.weight"],
            cc_projection_bias=state_dict["cc_projection.bias"],
        )
        return out

    def clip_target(self):
        return None


class smSD_X4Upscaler(sm_SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 256,
        "in_channels": 7,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "disable_self_attentions": [True, True, True, False],
        "num_classes": 1000,
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = SD_X4

    sampling_settings = {
        "linear_start": 0.0001,
        "linear_end": 0.02,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = SD_X4Upscaler(self, device=device)
        return out


class smStable_Cascade_C(BASE):
    unet_config = {
        "stable_cascade_stage": "c",
    }

    unet_extra_config = {}

    latent_format = SC_Prior
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 2.0,
    }

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoder."]
    clip_vision_prefix = "clip_l_vision."

    def process_unet_state_dict(self, state_dict):
        key_list = list(state_dict.keys())
        for y in ["weight", "bias"]:
            suffix = "in_proj_{}".format(y)
            keys = filter(lambda a: a.endswith(suffix), key_list)
            for k_from in keys:
                weights = state_dict.pop(k_from)
                prefix = k_from[: -(len(suffix) + 1)]
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["to_q", "to_k", "to_v"]
                    k_to = "{}.{}.{}".format(prefix, p[x], y)
                    state_dict[k_to] = weights[shape_from * x : shape_from * (x + 1)]
        return state_dict

    def process_clip_state_dict(self, state_dict):
        state_dict = state_dict_prefix_replace(
            state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True
        )
        if "clip_g.text_projection" in state_dict:
            state_dict["clip_g.transformer.text_projection.weight"] = state_dict.pop(
                "clip_g.text_projection"
            ).transpose(0, 1)
        return state_dict

    def get_model(self, state_dict, prefix="", device=None):
        out = StableCascade_C(self, device=device)
        return out

    def clip_target(self):
        return ClipTarget(StableCascadeTokenizer, StableCascadeClipModel)


class smStable_Cascade_B(smStable_Cascade_C):
    unet_config = {
        "stable_cascade_stage": "b",
    }

    unet_extra_config = {}

    latent_format = SC_B
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 1.0,
    }

    clip_vision_prefix = None

    def get_model(self, state_dict, prefix="", device=None):
        out = StableCascade_B(self, device=device)
        return out


class smSD15_instructpix2Pix(sm_SD15):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return SD15_instructpix2pix(self, device=device)


class smSDXL_instructpix2pix(smSDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return SDXL_instructpix2pix(
            self, model_type=self.model_type(state_dict, prefix), device=device
        )


models = [
    smStable_Zero123,
    smSD15_instructpix2Pix,
    sm_SD15,
    sm_SD20,
    smSD21UnclipL,
    smSD21UnclipH,
    smSDXL_instructpix2pix,
    smSDXLRefiner,
    smSDXL,
    smSSD1B,
    smKOALA_700M,
    smKOALA_1B,
    smSegmind_Vega,
    smSD_X4Upscaler,
    smStable_Cascade_C,
    smStable_Cascade_B,
    smSV3D_u,
    smSV3D_p,
]

models += [smSVD_img2vid]


def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(
        list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
    )
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(
            state_dict_keys, transformer_prefix + "{}"
        )
        context_dim = state_dict[
            "{}0.attn2.to_k.weight".format(transformer_prefix)
        ].shape[1]
        use_linear_in_transformer = (
            len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
        )
        time_stack = (
            "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict
            or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
        )
        return (
            last_transformer_depth,
            context_dim,
            use_linear_in_transformer,
            time_stack,
        )
    return None


def detect_unet_config(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())

    if "{}clf.1.weight".format(key_prefix) in state_dict_keys:  # stable cascade
        unet_config = {}
        text_mapper_name = "{}clip_txt_mapper.weight".format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config["stable_cascade_stage"] = "c"
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536:  # stage c lite
                unet_config["c_cond"] = 1536
                unet_config["c_hidden"] = [1536, 1536]
                unet_config["nhead"] = [24, 24]
                unet_config["blocks"] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048:  # stage c full
                unet_config["c_cond"] = 2048
        elif "{}clip_mapper.weight".format(key_prefix) in state_dict_keys:
            unet_config["stable_cascade_stage"] = "b"
            w = state_dict["{}down_blocks.1.0.channelwise.0.weight".format(key_prefix)]
            if w.shape[-1] == 640:
                unet_config["c_hidden"] = [320, 640, 1280, 1280]
                unet_config["nhead"] = [-1, -1, 20, 20]
                unet_config["blocks"] = [[2, 6, 28, 6], [6, 28, 6, 2]]
                unet_config["block_repeat"] = [[1, 1, 1, 1], [3, 3, 2, 2]]
            elif w.shape[-1] == 576:  # stage b lite
                unet_config["c_hidden"] = [320, 576, 1152, 1152]
                unet_config["nhead"] = [-1, 9, 18, 18]
                unet_config["blocks"] = [[2, 4, 14, 4], [4, 14, 4, 2]]
                unet_config["block_repeat"] = [[1, 1, 1, 1], [2, 2, 2, 2]]

        return unet_config

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False,
    }

    y_input = "{}label_emb.0.0.weight".format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    model_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[0]
    in_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[1]

    out_key = "{}out.2.weight".format(key_prefix)
    if out_key in state_dict:
        out_channels = state_dict[out_key].shape[0]
    else:
        out_channels = 4

    num_res_blocks = []
    channel_mult = []
    attention_resolutions = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(
        state_dict_keys, "{}input_blocks".format(key_prefix) + ".{}."
    )
    for count in range(input_block_count):
        prefix = "{}input_blocks.{}.".format(key_prefix, count)
        prefix_output = "{}output_blocks.{}.".format(
            key_prefix, input_block_count - count - 1
        )

        block_keys = sorted(
            list(filter(lambda a: a.startswith(prefix), state_dict_keys))
        )
        if len(block_keys) == 0:
            break

        block_keys_output = sorted(
            list(filter(lambda a: a.startswith(prefix_output), state_dict_keys))
        )

        if "{}0.op.weight".format(prefix) in block_keys:  # new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(
                prefix_output, state_dict_keys, state_dict
            )
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = (
                    state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0]
                    // model_channels
                )

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(
                    prefix_output, state_dict_keys, state_dict
                )
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(
            state_dict_keys,
            "{}middle_block.1.transformer_blocks.".format(key_prefix) + "{}",
        )
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config["use_linear_in_transformer"] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    if video_model:
        unet_config["extra_ff_mix_layer"] = True
        unet_config["use_spatial_context"] = True
        unet_config["merge_strategy"] = "learned_with_images"
        unet_config["merge_factor"] = 0.0
        unet_config["video_kernel_size"] = [3, 1, 1]
        unet_config["use_temporal_resblock"] = True
        unet_config["use_temporal_attention"] = True
    else:
        unet_config["use_temporal_resblock"] = False
        unet_config["use_temporal_attention"] = False

    return unet_config


def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None


def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    model_config = model_config_from_unet_config(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        return BASE(unet_config)
    else:
        return model_config


def convert_config(unet_config):
    new_config = unet_config.copy()
    num_res_blocks = new_config.get("num_res_blocks", None)
    channel_mult = new_config.get("channel_mult", None)

    if isinstance(num_res_blocks, int):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    if "attention_resolutions" in new_config:
        attention_resolutions = new_config.pop("attention_resolutions")
        transformer_depth = new_config.get("transformer_depth", None)
        transformer_depth_middle = new_config.get("transformer_depth_middle", None)

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle = transformer_depth[-1]
        t_in = []
        t_out = []
        s = 1
        for i in range(len(num_res_blocks)):
            res = num_res_blocks[i]
            d = 0
            if s in attention_resolutions:
                d = transformer_depth[i]

            t_in += [d] * res
            t_out += [d] * (res + 1)
            s *= 2
        transformer_depth = t_in
        transformer_depth_output = t_out
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    new_config["num_res_blocks"] = num_res_blocks
    return new_config


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(
            state_dict, "down_blocks.{}.attentions.".format(i) + "{}"
        )
        res_blocks = count_blocks(
            state_dict, "down_blocks.{}.resnets.".format(i) + "{}"
        )
        for ab in range(attn_blocks):
            transformer_count = count_blocks(
                state_dict,
                "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + "{}",
            )
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict[
                    "down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(
                        i, ab
                    )
                ].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[
            1
        ]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    SDXL = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_refiner = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2560,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 384,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 4,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SD21 = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 1024,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SD21_uncliph = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2048,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 1024,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SD21_unclipl = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 1536,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 1024,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SD15 = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": False,
        "context_dim": 768,
        "num_heads": 8,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_mid_cnet = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 0, 0, 1, 1],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_small_cnet = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 0, 0, 0, 0],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 0,
        "use_linear_in_transformer": True,
        "num_head_channels": 64,
        "context_dim": 1,
        "transformer_depth_output": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_diffusers_inpaint = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 9,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_diffusers_ip2p = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 8,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SSD_1B = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "transformer_depth_output": [0, 0, 0, 1, 1, 2, 10, 4, 4],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": -1,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    Segmind_Vega = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "transformer_depth_output": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": -1,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    KOALA_700M = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [1, 1, 1],
        "transformer_depth": [0, 2, 5],
        "transformer_depth_output": [0, 0, 2, 2, 5, 5],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": -2,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    KOALA_1B = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [1, 1, 1],
        "transformer_depth": [0, 2, 6],
        "transformer_depth_output": [0, 0, 2, 2, 6, 6],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 6,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SD09_XS = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [1, 1, 1],
        "transformer_depth": [1, 1, 1],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": -2,
        "use_linear_in_transformer": True,
        "context_dim": 1024,
        "num_head_channels": 64,
        "transformer_depth_output": [1, 1, 1, 1, 1, 1],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
        "disable_self_attentions": [True, False, False],
    }

    SD_XS = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "adm_in_channels": None,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [1, 1, 1],
        "transformer_depth": [0, 1, 1],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": -2,
        "use_linear_in_transformer": False,
        "context_dim": 768,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 1, 1, 1, 1],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    supported_models = [
        SDXL,
        SDXL_refiner,
        SD21,
        SD15,
        SD21_uncliph,
        SD21_unclipl,
        SDXL_mid_cnet,
        SDXL_small_cnet,
        SDXL_diffusers_inpaint,
        SSD_1B,
        Segmind_Vega,
        KOALA_700M,
        KOALA_1B,
        SD09_XS,
        SD_XS,
        SDXL_diffusers_ip2p,
    ]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return convert_config(unet_config)
    return None


def model_config_from_diffusers_unet(state_dict):
    unet_config = unet_config_from_diffusers_unet(state_dict)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None


import os

import torch


def ctrbroadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    # print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat(
            [tensor] * (per_batch // tensor.shape[0])
            + [tensor[: (per_batch % tensor.shape[0])]],
            dim=0,
        )

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)


class ctrControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.global_average_pooling = False
        self.timestep_range = None
        self.compression_ratio = 8
        self.upscale_algorithm = "nearest-exact"

        if device is None:
            device = get_torch_device()
        self.device = device
        self.previous_controlnet = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0)):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (
            percent_to_timestep_function(self.timestep_percent_range[0]),
            percent_to_timestep_function(self.timestep_percent_range[1]),
        )
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling
        c.compression_ratio = self.compression_ratio
        c.upscale_algorithm = self.upscale_algorithm

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        out = {"input": [], "middle": [], "output": []}

        if control_input is not None:
            for i in range(len(control_input)):
                key = "input"
                x = control_input[i]
                if x is not None:
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].insert(0, x)

        if control_output is not None:
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = "middle"
                    index = 0
                else:
                    key = "output"
                    index = i
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(
                            1, 1, x.shape[2], x.shape[3]
                        )

                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)
        if control_prev is not None:
            for x in ["input", "middle", "output"]:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            if o[i].shape[0] < prev_val.shape[0]:
                                o[i] = prev_val + o[i]
                            else:
                                o[i] += prev_val
        return out


class ctrControlNet(ctrControlBase):
    def __init__(
        self,
        control_model=None,
        global_average_pooling=False,
        device=None,
        load_device=None,
        manual_cast_dtype=None,
    ):
        super().__init__(device)
        self.control_model = control_model
        self.load_device = load_device
        if control_model is not None:
            self.control_model_wrapped = ModelPatcher(
                self.control_model,
                load_device=load_device,
                offload_device=unet_offload_device(),
            )

        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(
                x_noisy, t, cond, batched_number
            )

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        if (
            self.cond_hint is None
            or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2]
            or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]
        ):
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = (
                common_upscale(
                    self.cond_hint_original,
                    x_noisy.shape[3] * self.compression_ratio,
                    x_noisy.shape[2] * self.compression_ratio,
                    self.upscale_algorithm,
                    "center",
                )
                .to(dtype)
                .to(self.device)
            )
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = ctrbroadcast_image_to(
                self.cond_hint, x_noisy.shape[0], batched_number
            )

        context = cond.get("crossattn_controlnet", cond["c_crossattn"])
        y = cond.get("y", None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(
            x=x_noisy.to(dtype),
            hint=self.cond_hint,
            timesteps=timestep.float(),
            context=context.to(dtype),
            y=y,
        )
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        c = ctrControlNet(
            None,
            global_average_pooling=self.global_average_pooling,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()


class ctrControlLoraOps:
    class Linear(torch.nn.Module, CastWeightBiasOp):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.linear(
                    input,
                    weight
                    + (
                        torch.mm(
                            self.up.flatten(start_dim=1), self.down.flatten(start_dim=1)
                        )
                    )
                    .reshape(self.weight.shape)
                    .type(input.dtype),
                    bias,
                )
            else:
                return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(torch.nn.Module, CastWeightBiasOp):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None

        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.conv2d(
                    input,
                    weight
                    + (
                        torch.mm(
                            self.up.flatten(start_dim=1), self.down.flatten(start_dim=1)
                        )
                    )
                    .reshape(self.weight.shape)
                    .type(input.dtype),
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            else:
                return torch.nn.functional.conv2d(
                    input,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )


class ctrControlLora(ctrControlNet):
    def __init__(self, control_weights, global_average_pooling=False, device=None):
        ctrControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights[
            "input_hint_block.0.weight"
        ].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:

            class ctrControl_lora_ops(ctrControlLoraOps, disable_weight_init):
                pass

        else:

            class ctrControl_lora_ops(ctrControlLoraOps, manual_cast):
                pass

            dtype = self.manual_cast_dtype

        controlnet_config["operations"] = ctrControl_lora_ops
        controlnet_config["dtype"] = dtype
        self.control_model = ControlNet1(**controlnet_config)
        self.control_model.to(get_torch_device())
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()

        for k in sd:
            weight = sd[k]
            try:
                set_attr_param(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                set_attr_param(
                    self.control_model,
                    k,
                    self.control_weights[k].to(dtype).to(get_torch_device()),
                )

    def copy(self):
        c = ctrControlLora(
            self.control_weights, global_average_pooling=self.global_average_pooling
        )
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ctrControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return calculate_parameters(self.control_weights) * dtype_size(
            dtype
        ) + ctrControlBase.inference_memory_requirements(self, dtype)


def ctrload_controlnet(ckpt_path, model=None):
    controlnet_data = load_torch_file(ckpt_path, safe_load=True)
    if "lora_controlnet" in controlnet_data:
        return ctrControlLora(controlnet_data)

    controlnet_config = None
    supported_inference_dtypes = None

    if (
        "controlnet_cond_embedding.conv_in.weight" in controlnet_data
    ):  # diffusers format
        controlnet_config = unet_config_from_diffusers_unet(controlnet_data)
        diffusers_keys = unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logging.warning("leftover keys: {}".format(leftover_keys))
        controlnet_data = new_sd

    pth_key = "control_model.zero_convs.0.0.weight"
    pth = False
    key = "zero_convs.0.0.weight"
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = ctrload_t2i_adapter(controlnet_data)
        if net is None:
            logging.error(
                "error checkpoint does not contain controlnet or t2i adapter data {}".format(
                    ckpt_path
                )
            )
        return net

    if controlnet_config is None:
        model_config = model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = model_config.supported_inference_dtypes
        controlnet_config = model_config.unet_config

    load_device = get_torch_device()
    if supported_inference_dtypes is None:
        unet_dtype = unet_dtype1()
    else:
        unet_dtype = unet_dtype1(supported_dtypes=supported_inference_dtypes)

    manual_cast_dtype = unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = manual_cast
    controlnet_config["dtype"] = unet_dtype
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data[
        "{}input_hint_block.0.weight".format(prefix)
    ].shape[1]
    control_model = ControlNet1(**controlnet_config)

    if pth:
        if "difference" in controlnet_data:
            if model is not None:
                load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m) :])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logging.warning(
                    "WARNING: Loaded a diff controlnet without a model. It will very likely not work."
                )

        class WeightsLoader(torch.nn.Module):
            pass

        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(
            controlnet_data, strict=False
        )

    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith(
        "_shuffle_fp16"
    ):
        global_average_pooling = True

    control = ctrControlNet(
        control_model,
        global_average_pooling=global_average_pooling,
        load_device=load_device,
        manual_cast_dtype=manual_cast_dtype,
    )
    return control


class ctrT2IAdapter(ctrControlBase):
    def __init__(
        self, t2i_model, channels_in, compression_ratio, upscale_algorithm, device=None
    ):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None
        self.compression_ratio = compression_ratio
        self.upscale_algorithm = upscale_algorithm

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(
                x_noisy, t, cond, batched_number
            )

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if (
            self.cond_hint is None
            or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2]
            or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]
        ):
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(
                x_noisy.shape[3] * self.compression_ratio,
                x_noisy.shape[2] * self.compression_ratio,
            )
            self.cond_hint = (
                common_upscale(
                    self.cond_hint_original,
                    width,
                    height,
                    self.upscale_algorithm,
                    "center",
                )
                .float()
                .to(self.device)
            )
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = ctrbroadcast_image_to(
                self.cond_hint, x_noisy.shape[0], batched_number
            )
        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = list(
            map(lambda a: None if a is None else a.clone(), self.control_input)
        )
        mid = None
        if self.t2i_model.xl == True:
            mid = control_input[-1:]
            control_input = control_input[:-1]
        return self.control_merge(control_input, mid, control_prev, x_noisy.dtype)

    def copy(self):
        c = ctrT2IAdapter(
            self.t2i_model,
            self.channels_in,
            self.compression_ratio,
            self.upscale_algorithm,
        )
        self.copy_to(c)
        return c


def ctrload_t2i_adapter(t2i_data):
    compression_ratio = 8
    upscale_algorithm = "nearest-exact"

    if "adapter" in t2i_data:
        t2i_data = t2i_data["adapter"]
    if "adapter.body.0.resnets.0.block1.weight" in t2i_data:  # diffusers format
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace["adapter.body.{}.resnets.{}.".format(i, j)] = (
                    "body.{}.".format(i * 2 + j)
                )
            prefix_replace["adapter.body.{}.".format(i, j)] = "body.{}.".format(i * 2)
        prefix_replace["adapter."] = ""
        t2i_data = state_dict_prefix_replace(t2i_data, prefix_replace)
    keys = t2i_data.keys()

    if "body.0.in_conv.weight" in keys:
        cin = t2i_data["body.0.in_conv.weight"].shape[1]
        model_ad = Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif "conv_in.weight" in keys:
        cin = t2i_data["conv_in.weight"].shape[1]
        channel = t2i_data["conv_in.weight"].shape[0]
        ksize = t2i_data["body.0.block2.weight"].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        xl = False
        if cin == 256 or cin == 768:
            xl = True
        model_ad = Adapter(
            cin=cin,
            channels=[channel, channel * 2, channel * 4, channel * 4][:4],
            nums_rb=2,
            ksize=ksize,
            sk=True,
            use_conv=use_conv,
            xl=xl,
        )
    elif "backbone.0.0.weight" in keys:
        model_ad = ControlNet(
            c_in=t2i_data["backbone.0.0.weight"].shape[1],
            proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63],
        )
        compression_ratio = 32
        upscale_algorithm = "bilinear"
    elif "backbone.10.blocks.0.weight" in keys:
        model_ad = ControlNet(
            c_in=t2i_data["backbone.0.weight"].shape[1],
            bottleneck_mode="large",
            proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63],
        )
        compression_ratio = 1
        upscale_algorithm = "nearest-exact"
    else:
        return None

    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        logging.warning("t2i missing {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("t2i unexpected {}".format(unexpected))

    return ctrT2IAdapter(
        model_ad, model_ad.input_channels, compression_ratio, upscale_algorithm
    )


from enum import Enum

import torch
import yaml


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
        logging.warning("missing {}".format(m))
    return model


def load_clip_weights(model, sd):
    k = list(sd.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith(
            "cond_stage_model.transformer.text_model."
        ):
            y = x.replace(
                "cond_stage_model.transformer.",
                "cond_stage_model.transformer.text_model.",
            )
            sd[y] = sd.pop(x)

    if "cond_stage_model.transformer.text_model.embeddings.position_ids" in sd:
        ids = sd["cond_stage_model.transformer.text_model.embeddings.position_ids"]
        if ids.dtype == torch.float32:
            sd["cond_stage_model.transformer.text_model.embeddings.position_ids"] = (
                ids.round()
            )

    sd = clip_text_transformers_convert(
        sd, "cond_stage_model.model.", "cond_stage_model.transformer."
    )
    return load_model_weights(model, sd)


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = text_encoder_device()
        offload_device = text_encoder_offload_device()
        params["device"] = offload_device
        params["dtype"] = text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def load_model(self):
        load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None):
        if (
            "decoder.up_blocks.0.resnets.0.norm1.weight" in sd.keys()
        ):  # diffusers format
            sd = convert_vae_state_dict(sd)

        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * dtype_size(
            dtype
        )  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )

        if config is None: # TODO : remove abstraction layer
            if "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(
                    regularizer_config={
                        "target": "LightDiffusion.DiagonalGaussianRegularizer"
                    },
                    encoder_config={
                        "target": "LightDiffusion.Encoder",
                        "params": encoder_config,
                    },
                    decoder_config={
                        "target": "LightDiffusion.VideoDecoder",
                        "params": decoder_config,
                    },
                )
            elif "taesd_decoder.1.weight" in sd:
                self.first_stage_model = TAESD()
            elif "vquantizer.codebook.weight" in sd:  # VQGan: stage a of stable cascade
                self.first_stage_model = StageA()
                self.downscale_ratio = 4
                self.upscale_ratio = 4
                # self.memory_used_encode
                # self.memory_used_decode
                self.process_input = lambda image: image
                self.process_output = lambda image: image
            elif (
                "backbone.1.0.block.0.1.num_batches_tracked" in sd
            ):  # effnet: encoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["encoder.{}".format(k)] = sd[k]
                sd = new_sd
            elif (
                "blocks.11.num_batches_tracked" in sd
            ):  # previewer: decoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["previewer.{}".format(k)] = sd[k]
                sd = new_sd
            elif (
                "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd
            ):  # combined effnet and previewer for stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
            elif "decoder.conv_in.weight" in sd:
                # default SD1.x/SD2.x VAE parameters
                ddconfig = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }

                if (
                    "encoder.down.2.downsample.conv.weight" not in sd
                    and "decoder.up.3.upsample.conv.weight" not in sd
                ):  # Stable diffusion x4 upscaler VAE
                    ddconfig["ch_mult"] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig["z_channels"] = sd[
                    "decoder.conv_in.weight"
                ].shape[1]
                if "quant_conv.weight" in sd:
                    self.first_stage_model = AutoencoderKL(
                        ddconfig=ddconfig, embed_dim=4
                    )
                else:
                    self.first_stage_model = AutoencodingEngine(
                        regularizer_config={
                            "target": "LightDiffusion.DiagonalGaussianRegularizer"
                        },
                        encoder_config={
                            "target": "LightDiffusion.Encoder",
                            "params": ddconfig,
                        },
                        decoder_config={
                            "target": "LightDiffusion.Decoder",
                            "params": ddconfig,
                        },
                    )
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config["params"]))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = vae_device()
        self.device = device
        offload_device = vae_offload_device()
        if dtype is None:
            dtype = vae_dtype()
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        y = (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % self.downscale_ratio) // 2
            y_offset = (pixels.shape[2] % self.downscale_ratio) // 2
            pixels = pixels[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
        return pixels

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x, tile_y, overlap
        )
        steps += samples.shape[0] * get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap
        )
        steps += samples.shape[0] * get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap
        )
        pbar = ProgressBar(steps)

        decode_fn = lambda a: self.first_stage_model.decode(
            a.to(self.vae_dtype).to(self.device)
        ).float()
        output = self.process_output(
            (
                tiled_scale(
                    samples,
                    decode_fn,
                    tile_x // 2,
                    tile_y * 2,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
                + tiled_scale(
                    samples,
                    decode_fn,
                    tile_x * 2,
                    tile_y // 2,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
                + tiled_scale(
                    samples,
                    decode_fn,
                    tile_x,
                    tile_y,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
            )
            / 3.0
        )
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * get_tiled_scale_steps(
            pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap
        )
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x // 2,
            tile_y * 2,
            overlap,
        )
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x * 2,
            tile_y // 2,
            overlap,
        )
        pbar = ProgressBar(steps)

        encode_fn = lambda a: self.first_stage_model.encode(
            (self.process_input(a)).to(self.vae_dtype).to(self.device)
        ).float()
        samples = tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x,
            tile_y,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x * 2,
            tile_y // 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x // 2,
            tile_y * 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples /= 3.0
        return samples

    def decode(self, samples_in):
        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty(
                (
                    samples_in.shape[0],
                    3,
                    round(samples_in.shape[2] * self.upscale_ratio),
                    round(samples_in.shape[3] * self.upscale_ratio),
                ),
                device=self.output_device,
            )
            for x in range(0, samples_in.shape[0], batch_number):
                samples = (
                    samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
                )
                pixel_samples[x : x + batch_number] = self.process_output(
                    self.first_stage_model.decode(samples)
                    .to(self.output_device)
                    .float()
                )
        except OOM_EXCEPTION as e:
            logging.warning(
                "Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding."
            )
            pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        load_model_gpu(self.patcher)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        return output.movedim(1, -1)

    def encode(self, pixel_samples):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty(
                (
                    pixel_samples.shape[0],
                    self.latent_channels,
                    round(pixel_samples.shape[2] // self.downscale_ratio),
                    round(pixel_samples.shape[3] // self.downscale_ratio),
                ),
                device=self.output_device,
            )
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (
                    self.process_input(pixel_samples[x : x + batch_number])
                    .to(self.vae_dtype)
                    .to(self.device)
                )
                samples[x : x + batch_number] = (
                    self.first_stage_model.encode(pixels_in)
                    .to(self.output_device)
                    .float()
                )

        except OOM_EXCEPTION as e:
            logging.warning(
                "Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding."
            )
            samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        load_model_gpu(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        samples = self.encode_tiled_(
            pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap
        )
        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()


class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)


def load_style_model(ckpt_path):
    model_data = load_torch_file(ckpt_path, safe_load=True)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = StyleAdapter(
            width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8
        )
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2


def load_clip(
    ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION
):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(load_torch_file(p, safe_load=True))

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i][
                    "text_projection"
                ].transpose(
                    0, 1
                )  # old _internal saved with the CLIPSave node

    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        if "text_model.encoder.layers.30.mlp.fc1.weight" in clip_data[0]:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = StableCascadeClipModel
                clip_target.tokenizer = StableCascadeTokenizer
            else:
                clip_target.clip = SDXLRefinerClipModel
                clip_target.tokenizer = SDXLTokenizer
        elif "text_model.encoder.layers.22.mlp.fc1.weight" in clip_data[0]:
            clip_target.clip = SD2ClipModel
            clip_target.tokenizer = SD2Tokenizer
        else:
            clip_target.clip = SD1ClipModel
            clip_target.tokenizer = SD1Tokenizer
    else:
        clip_target.clip = SDXLClipModel
        clip_target.tokenizer = SDXLTokenizer

    clip = CLIP(clip_target, embedding_directory=embedding_directory)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip


def load_gligen(ckpt_path):
    data = load_torch_file(ckpt_path, safe_load=True)
    model = load_gligen(data)
    if should_use_fp16():
        model = model.half()
    return ModelPatcher(
        model, load_device=get_torch_device(), offload_device=unet_offload_device()
    )


def load_checkpoint(
    config_path=None,
    ckpt_path=None,
    output_vae=True,
    output_clip=True,
    embedding_directory=None,
    state_dict=None,
    config=None,
):
    logging.warning(
        "Warning: The load checkpoint with config function is deprecated and will eventually be removed, please use the other one."
    )
    model, clip, vae, _ = load_checkpoint_guess_config(
        ckpt_path,
        output_vae=output_vae,
        output_clip=output_clip,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=True,
    )
    if config is None:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
    model_config_params = config["model"]["params"]
    clip_config = model_config_params["cond_stage_config"]
    scale_factor = model_config_params["scale_factor"]

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            m = model.clone()

            class ModelSamplingAdvanced(ModelSamplingDiscrete, V_PREDICTION):
                pass

            m.add_object_patch(
                "model_sampling", ModelSamplingAdvanced(model.model.model_config)
            )
            model = m

    layer_idx = clip_config.get("params", {}).get("layer_idx", None)
    if layer_idx is not None:
        clip.clip_layer(layer_idx)

    return (model, clip, vae)


def unet_dtype1(
    device=None,
    model_params=0,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32


def load_checkpoint_guess_config(
    ckpt_path,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
):
    sd = load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    load_device = get_torch_device()

    model_config = model_config_from_unet(sd, "model.diffusion_model.")
    unet_dtype = unet_dtype1(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
    )
    manual_cast_dtype = unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config is None:
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(ckpt_path)
        )

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = load_clipvision_from_sd(
                sd, model_config.clip_vision_prefix, True
            )

    if output_model:
        inital_load_device = unet_inital_load_device(parameters, unet_dtype)
        offload_device = unet_offload_device()
        model = model_config.get_model(
            sd, "model.diffusion_model.", device=inital_load_device
        )
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
        )
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(
                        filter(
                            lambda a: ".logit_scale" not in a
                            and ".transformer.text_projection.weight" not in a,
                            m,
                        )
                    )
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = ModelPatcher(
            model,
            load_device=load_device,
            offload_device=unet_offload_device(),
            current_device=inital_load_device,
        )
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)


def load_unet_state_dict(sd):  # load unet in diffusers format
    parameters = calculate_parameters(sd)
    unet_dtype = unet_dtype1(model_params=parameters)
    load_device = get_torch_device()

    if "input_blocks.0.0.weight" in sd or "clf.1.weight" in sd:  # ldm or stable cascade
        model_config = model_config_from_unet(sd, "")
        if model_config is None:
            return None
        new_sd = sd

    else:  # diffusers
        model_config = model_config_from_diffusers_unet(sd)
        if model_config is None:
            return None

        diffusers_keys = unet_to_diffusers(model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)
            else:
                logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = unet_offload_device()
    unet_dtype = unet_dtype(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
    )
    manual_cast_dtype = unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device)


def load_unet(unet_path):
    sd = load_torch_file(unet_path)
    model = load_unet_state_dict(sd)
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(unet_path)
        )
    return model


def save_checkpoint(
    output_path,
    model,
    clip=None,
    vae=None,
    clip_vision=None,
    metadata=None,
    extra_keys={},
):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()

    load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    save_torch_file(sd, output_path, metadata=metadata)


def first_file(path, filenames):
    for f in filenames:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return p
    return None


def load_diffusers(
    model_path, output_vae=True, output_clip=True, embedding_directory=None
):
    diffusion_model_names = [
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.bin",
        "diffusion_pytorch_model.bin",
    ]
    unet_path = first_file(os.path.join(model_path, "unet"), diffusion_model_names)
    vae_path = first_file(os.path.join(model_path, "vae"), diffusion_model_names)

    text_encoder_model_names = [
        "model.fp16.safetensors",
        "model.safetensors",
        "pytorch_model.fp16.bin",
        "pytorch_model.bin",
    ]
    text_encoder1_path = first_file(
        os.path.join(model_path, "text_encoder"), text_encoder_model_names
    )
    text_encoder2_path = first_file(
        os.path.join(model_path, "text_encoder_2"), text_encoder_model_names
    )

    text_encoder_paths = [text_encoder1_path]
    if text_encoder2_path is not None:
        text_encoder_paths.append(text_encoder2_path)

    unet = load_unet(unet_path)

    clip = None
    if output_clip:
        clip = load_clip(text_encoder_paths, embedding_directory=embedding_directory)

    vae = None
    if output_vae:
        sd = load_torch_file(vae_path)
        vae = VAE(sd=sd)

    return (unet, clip, vae)


def set_output_directory(output_dir):
    global output_directory
    output_directory = output_dir


def set_temp_directory(temp_dir):
    global temp_directory
    temp_directory = temp_dir


def set_input_directory(input_dir):
    global input_directory
    input_directory = input_dir


def get_output_directory():
    global output_directory
    return output_directory


def get_temp_directory():
    global temp_directory
    return temp_directory


def get_input_directory():
    global input_directory
    return input_directory


# NOTE: used in http server so don't put folders that should not be accessed remotely
def get_directory_by_type(type_name):
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None


# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format
# otherwise use default_path as base_dir
def annotated_filepath(name):
    if name.endswith("[output]"):
        base_dir = get_output_directory()
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()
        name = name[:-7]
    else:
        return name, None

    return name, base_dir


def get_annotated_filepath(name, default_dir=None):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path

    return os.path.join(base_dir, name)


def exists_annotated_filepath(name):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        base_dir = get_input_directory()  # fallback path

    filepath = os.path.join(base_dir, name)
    return os.path.exists(filepath)


def add_model_folder_path(folder_name, full_folder_path):
    global folder_names_and_paths
    if folder_name in folder_names_and_paths:
        folder_names_and_paths[folder_name][0].append(full_folder_path)
    else:
        folder_names_and_paths[folder_name] = ([full_folder_path], set())


def get_folder_paths(folder_name):
    return folder_names_and_paths[folder_name][0][:]


def recursive_search(directory, excluded_dir_names=None):
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs = {}

    # Attempt to add the initial directory to dirs with error handling
    try:
        dirs[directory] = os.path.getmtime(directory)
    except FileNotFoundError:
        logging.warning(f"Warning: Unable to access {directory}. Skipping this path.")

    logging.debug("recursive file list on directory {}".format(directory))
    for dirpath, subdirs, filenames in os.walk(
        directory, followlinks=True, topdown=True
    ):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
            result.append(relative_path)

        for d in subdirs:
            path = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                logging.warning(
                    f"Warning: Unable to access {path}. Skipping this path."
                )
                continue
    logging.debug("found {} files".format(len(result)))
    return result, dirs


def filter_files_extensions(files, extensions):
    return sorted(
        list(
            filter(
                lambda a: os.path.splitext(a)[-1].lower() in extensions
                or len(extensions) == 0,
                files,
            )
        )
    )


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
        elif os.path.islink(full_path):
            logging.warning(
                "WARNING path {} exists but doesn't link anywhere, skipping.".format(
                    full_path
                )
            )

    return None


def get_filename_list_(folder_name):
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}

    return (sorted(list(output_list)), output_folders, time.perf_counter())


def cached_filename_list_(folder_name):
    global filename_list_cache
    global folder_names_and_paths
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:
                return None

    return out


def get_filename_list(folder_name):
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    return list(out[0])


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
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
    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
    ):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return x


def before_node_execution():
    throw_exception_if_processing_interrupted()


def interrupt_processing(value=True):
    interrupt_current_processing(value)


MAX_RESOLUTION = 16384


class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


class VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    def encode(self, vae, pixels):
        t = vae.encode(pixels[:, :, :, :3])
        return ({"samples": t},)


class CheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = f"{ckpt_name}"
        out = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory="._internal\\embeddings\\",
        )
        return out[:3]


class CLIPSetLastLayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "stop_at_clip_layer": (
                    "INT",
                    {"default": -1, "min": -24, "max": -1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "set_last_layer"

    CATEGORY = "conditioning"

    def set_last_layer(self, clip, stop_at_clip_layer):
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)


class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class EmptyLatentImage:
    def __init__(self):
        self.device = intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)


class LatentUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (s.upscale_methods,),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "crop": (s.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()

            if width == 0:
                height = max(64, height)
                width = max(
                    64,
                    round(
                        samples["samples"].shape[3]
                        * height
                        / samples["samples"].shape[2]
                    ),
                )
            elif height == 0:
                width = max(64, width)
                height = max(
                    64,
                    round(
                        samples["samples"].shape[2]
                        * width
                        / samples["samples"].shape[3]
                    ),
                )
            else:
                width = max(64, width)
                height = max(64, height)

            s["samples"] = common_upscale(
                samples["samples"], width // 8, height // 8, upscale_method, crop
            )
        return (s,)


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    disable_pbar = not PROGRESS_BAR_ENABLED
    samples = sample1(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (KSampler1.SAMPLERS,),
                "scheduler": (KSampler1.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )


class SaveImage:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(
        self, images, filename_prefix="LD", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = get_annotated_filepath(image)

        img = pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


NODE_CLASS_MAPPINGS = {
    "KSampler": KSampler1,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "CLIPTextEncode": CLIPTextEncode,
    "CLIPSetLastLayer": CLIPSetLastLayer,
    "VAEDecode": VAEDecode,
    "VAEEncode": VAEEncode,
    "EmptyLatentImage": EmptyLatentImage,
    "LatentUpscale": LatentUpscale,
    "SaveImage": SaveImage,
    "LoadImage": LoadImage,
    "LoraLoader": LoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "KSampler": "KSampler",
    "KSamplerAdvanced": "KSampler (Advanced)",
    # Loaders
    "CheckpointLoader": "Load Checkpoint With Config (DEPRECATED)",
    "CheckpointLoaderSimple": "Load Checkpoint",
    "VAELoader": "Load VAE",
    "LoraLoader": "Load LoRA",
    "CLIPLoader": "Load CLIP",
    "ControlNetLoader": "Load ControlNet Model",
    "DiffControlNetLoader": "Load ControlNet Model (diff)",
    "StyleModelLoader": "Load Style Model",
    "CLIPVisionLoader": "Load CLIP Vision",
    "UpscaleModelLoader": "Load Upscale Model",
    # Conditioning
    "CLIPVisionEncode": "CLIP Vision Encode",
    "StyleModelApply": "Apply Style Model",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)",
    "CLIPSetLastLayer": "CLIP Set Last Layer",
    "ConditioningCombine": "Conditioning (Combine)",
    "ConditioningAverage ": "Conditioning (Average)",
    "ConditioningConcat": "Conditioning (Concat)",
    "ConditioningSetArea": "Conditioning (Set Area)",
    "ConditioningSetAreaPercentage": "Conditioning (Set Area with Percentage)",
    "ConditioningSetMask": "Conditioning (Set Mask)",
    "ControlNetApply": "Apply ControlNet",
    "ControlNetApplyAdvanced": "Apply ControlNet (Advanced)",
    # Latent
    "VAEEncodeForInpaint": "VAE Encode (for Inpainting)",
    "SetLatentNoiseMask": "Set Latent Noise Mask",
    "VAEDecode": "VAE Decode",
    "VAEEncode": "VAE Encode",
    "LatentRotate": "Rotate Latent",
    "LatentFlip": "Flip Latent",
    "LatentCrop": "Crop Latent",
    "EmptyLatentImage": "Empty Latent Image",
    "LatentUpscale": "Upscale Latent",
    "LatentUpscaleBy": "Upscale Latent By",
    "LatentComposite": "Latent Composite",
    "LatentBlend": "Latent Blend",
    "LatentFromBatch": "Latent From Batch",
    "RepeatLatentBatch": "Repeat Latent Batch",
    # Image
    "SaveImage": "Save Image",
    "PreviewImage": "Preview Image",
    "LoadImage": "Load Image",
    "LoadImageMask": "Load Image (as Mask)",
    "ImageScale": "Upscale Image",
    "ImageScaleBy": "Upscale Image By",
    "ImageUpscaleWithModel": "Upscale Image (using Model)",
    "ImageInvert": "Invert Image",
    "ImagePadForOutpaint": "Pad Image for Outpainting",
    "ImageBatch": "Batch Images",
    # _for_testing
    "VAEDecodeTiled": "VAE Decode (Tiled)",
    "VAEEncodeTiled": "VAE Encode (Tiled)",
}

EXTENSION_WEB_DIRS = {}


def act(act_type: str, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


ConvMode = Literal["CNA", "NAC", "CNAC"]


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
    mode: ConvMode = "CNA",
    c2x2=False,
):
    assert mode in ("CNA", "NAC", "CNAC"), "Wrong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if mode in ("CNA", "CNAC"):
        return sequential(None, c, None, a)


class RRDB(nn.Module):

    def __init__(
        self,
        nf,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        _convtype="Conv2D",
        _spectral_norm=False,
        plus=False,
        c2x2=False,
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}
    Args:
        nf (int): Channel number of intermediate features (num_feat).
        gc (int): Channels for each growth (num_grow_ch: growth channel,
            i.e. intermediate channels).
        convtype (str): the type of convolution to use. Default: 'Conv2D'
        gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
            trainable parameters)
        plus (bool): enable the additional residual paths from ESRGAN+
            (adds trainable parameters)
    """

    def __init__(
        self,
        nf=64,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        plus=False,
        c2x2=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1x1 = None

        self.conv1 = conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            c2x2=c2x2,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            # pylint: disable=not-callable
            x2 = x2 + self.conv1x1(x)  # +
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2  # +
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def upconv_block(
    in_nc: int,
    out_nc: int,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type="relu",
    mode="nearest",
    c2x2=False,
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        c2x2=c2x2,
    )
    return sequential(upsample, conv)


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False
):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        padding=1,
        bias=bias,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class RRDBNet(nn.Module):
    def __init__(
        self,
        state_dict,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: ConvMode = "CNA",
    ) -> None:
        super(RRDBNet, self).__init__()
        self.model_arch = "ESRGAN"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch _internal
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
            # self.model_arch = "RealESRGAN"
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())
        if self.plus:
            self.model_arch = "ESRGAN+"

        self.state = self.new_to_old_arch(self.state)

        self.key_arr = list(self.state.keys())

        self.in_nc: int = self.state[self.key_arr[0]].shape[1]
        self.out_nc: int = self.state[self.key_arr[-1]].shape[0]

        self.scale: int = self.get_scale()
        self.num_filters: int = self.state[self.key_arr[0]].shape[0]

        c2x2 = False
        if self.state["model.0.weight"].shape[-2] == 2:
            c2x2 = True
            self.scale = round(math.sqrt(self.scale / 4))
            self.model_arch = "ESRGAN-2c2"

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        # Detect if pixelunshuffle was used (Real-ESRGAN)
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (
            self.in_nc / 4,
            self.in_nc / 16,
        ):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": upconv_block,
            "pixel_shuffle": pixelshuffle_block,
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{self.upsampler}] is not found")

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act,
                c2x2=c2x2,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    act_type=self.act,
                    c2x2=c2x2,
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = sequential(
            # fea conv
            conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            ShortcutBlock(
                sequential(
                    # rrdb blocks
                    *[
                        RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                            c2x2=c2x2,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        # Adjust these properties for calculations outside of the model
        if self.shuffle_factor:
            self.in_nc //= self.shuffle_factor**2
            self.scale //= self.shuffle_factor

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state):
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        # upconv layers
        max_upconv = 0
        for key in state.keys():
            match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
            if match is not None:
                _, key_num, key_type = match.groups()
                old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
                max_upconv = max(max_upconv, int(key_num) * 3)

        # final layers
        for key in state.keys():
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key in ("conv_last.weight",):
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key in ("conv_last.bias",):
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        # Sort by first numeric value of each layer
        def compare(item1, item2):
            parts1 = item1.split(".")
            parts2 = item2.split(".")
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2

        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

        # Rebuild the output dict in the right order
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

        return out_dict

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            _, _, h, w = x.size()
            mod_pad_h = (
                self.shuffle_factor - h % self.shuffle_factor
            ) % self.shuffle_factor
            mod_pad_w = (
                self.shuffle_factor - w % self.shuffle_factor
            ) % self.shuffle_factor
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
            x = self.model(x)
            return x[:, :, : h * self.scale, : w * self.scale]
        return self.model(x)


PyTorchSRModels = (RRDBNet,)
PyTorchSRModel = Union[RRDBNet,]

PyTorchModels = (*PyTorchSRModels,)
PyTorchModel = Union[PyTorchSRModel]


class UnsupportedModel(Exception):
    pass


import logging as logger


def load_state_dict(state_dict) -> PyTorchModel:
    logger.debug(f"Loading state dict into pytorch model arch")

    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]
    try:
        model = RRDBNet(state_dict)
    except:
        # pylint: disable=raise-missing-from
        raise UnsupportedModel
    return model


class UpscaleModelLoader:
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = f"_internal\\ERSGAN\\{model_name}"
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    pbar=None,
):
    output = torch.empty(
        (
            samples.shape[0],
            out_channels,
            round(samples.shape[2] * upscale_amount),
            round(samples.shape[3] * upscale_amount),
        ),
        device="cpu",
    )
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        out_div = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y : y + tile_y, x : x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                        1.0 / feather
                    ) * (t + 1)
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                        1.0 / feather
                    ) * (t + 1)
                out[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += (
                    ps * mask
                )
                out_div[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += mask

        output[b : b + 1] = out / out_div
    return output


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = torch.device(torch.cuda.current_device())
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        free_memory = get_free_memory(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                )
                pbar = ProgressBar(steps)
                s = tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                )
                oom = False
            except OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


def torch_gc():
    pass


def flatten(img, bgcolor):
    # Replace transparency with bgcolor
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert(
        "RGB"
    )


class Script:
    pass


class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
# The upscaler usable by ComfyUI nodes
actual_upscaler = None

# Batch of images to upscale
batch = None

import numpy as np
import torch.nn.functional as F
from PIL import Image

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image

BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def controlnet_hint_to_pil(tensor, batch_index=0):
    return tensor_to_pil(tensor.movedim(1, -1), batch_index)


def pil_to_controlnet_hint(img):
    return pil_to_tensor(img).movedim(-1, 1)


def crop_tensor(tensor, region):
    # Takes a tensor of shape [batch_size, height, width, channels] and crops it to the given region
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]


def resize_tensor(tensor, size, mode="nearest-exact"):
    # Takes a tensor of shape [B, C, H, W] and resizes
    # it to a shape of [B, C, size[0], size[1]] using the given mode
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    """
    Expands a crop region to a specified target size.
    :param region: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param width: The width of the image the crop region is from.
    :param height: The height of the image the crop region is from.
    :param target_width: The desired width of the crop region.
    :param target_height: The desired height of the crop region.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1
    # target_width = math.ceil(actual_width / 8) * 8
    # target_height = math.ceil(actual_height / 8) * 8

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def resize_region(region, init_size, resize_size):
    # Resize a crop so that it fits an image that was resized to the given width and height
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)


def pad_image(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    """
    Pads an image with the given number of pixels on each side and fills the padding with data from the edges.
    :param image: A PIL image
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A PIL image with size (image.width + left_pad + right_pad, image.height + top_pad + bottom_pad)
    """
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        for i in range(left_pad):
            edge = left_edge.resize(
                (1, new_height - i * (top_pad + bottom_pad) // left_pad),
                resample=Image.Resampling.NEAREST,
            )
            padded_image.paste(edge, (i, i * top_pad // left_pad))
        for i in range(right_pad):
            edge = right_edge.resize(
                (1, new_height - i * (top_pad + bottom_pad) // right_pad),
                resample=Image.Resampling.NEAREST,
            )
            padded_image.paste(edge, (new_width - 1 - i, i * top_pad // right_pad))
        for i in range(top_pad):
            edge = top_edge.resize(
                (new_width - i * (left_pad + right_pad) // top_pad, 1),
                resample=Image.Resampling.NEAREST,
            )
            padded_image.paste(edge, (i * left_pad // top_pad, i))
        for i in range(bottom_pad):
            edge = bottom_edge.resize(
                (new_width - i * (left_pad + right_pad) // bottom_pad, 1),
                resample=Image.Resampling.NEAREST,
            )
            padded_image.paste(edge, (i * left_pad // bottom_pad, new_height - 1 - i))
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(
                ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE)
            )
            padded_image.paste(image, (left_pad, top_pad))
    return padded_image


def pad_image2(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    """
    Pads an image with the given number of pixels on each side and fills the padding with data from the edges.
    Faster than pad_image, but only pads with edge data in straight lines.
    :param image: A PIL image
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A PIL image with size (image.width + left_pad + right_pad, image.height + top_pad + bottom_pad)
    """
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        if left_pad > 0:
            padded_image.paste(
                left_edge.resize(
                    (left_pad, new_height), resample=Image.Resampling.NEAREST
                ),
                (0, 0),
            )
        if right_pad > 0:
            padded_image.paste(
                right_edge.resize(
                    (right_pad, new_height), resample=Image.Resampling.NEAREST
                ),
                (new_width - right_pad, 0),
            )
        if top_pad > 0:
            padded_image.paste(
                top_edge.resize(
                    (new_width, top_pad), resample=Image.Resampling.NEAREST
                ),
                (0, 0),
            )
        if bottom_pad > 0:
            padded_image.paste(
                bottom_edge.resize(
                    (new_width, bottom_pad), resample=Image.Resampling.NEAREST
                ),
                (0, new_height - bottom_pad),
            )
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(
                ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE)
            )
            padded_image.paste(image, (left_pad, top_pad))
    return padded_image


def pad_tensor(
    tensor, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False
):
    """
    Pads an image tensor with the given number of pixels on each side and fills the padding with data from the edges.
    :param tensor: A tensor of shape [B, H, W, C]
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A tensor of shape [B, H + top_pad + bottom_pad, W + left_pad + right_pad, C]
    """
    batch_size, channels, height, width = tensor.shape
    h_pad = left_pad + right_pad
    v_pad = top_pad + bottom_pad
    new_width = width + h_pad
    new_height = height + v_pad

    # Create empty image
    padded = torch.zeros(
        (batch_size, channels, new_height, new_width), dtype=tensor.dtype
    )

    # Copy the original image into the centor of the padded tensor
    padded[:, :, top_pad : top_pad + height, left_pad : left_pad + width] = tensor

    # Duplicate the edges of the original image into the padding
    if top_pad > 0:
        padded[:, :, :top_pad, :] = padded[:, :, top_pad : top_pad + 1, :]  # Top edge
    if bottom_pad > 0:
        padded[:, :, -bottom_pad:, :] = padded[
            :, :, -bottom_pad - 1 : -bottom_pad, :
        ]  # Bottom edge
    if left_pad > 0:
        padded[:, :, :, :left_pad] = padded[
            :, :, :, left_pad : left_pad + 1
        ]  # Left edge
    if right_pad > 0:
        padded[:, :, :, -right_pad:] = padded[
            :, :, :, -right_pad - 1 : -right_pad
        ]  # Right edge

    return padded


def resize_and_pad_image(image, width, height, fill=False, blur=False):
    """
    Resizes an image to the given width and height and pads it to the given width and height.
    :param image: A PIL image
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param fill: Whether to fill the padding with data from the edges
    :param blur: Whether to blur the padded edges
    :return: A PIL image of size (width, height)
    """
    width_ratio = width / image.width
    height_ratio = height / image.height
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize(
        (resize_width, resize_height), resample=Image.Resampling.LANCZOS
    )
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_image2(
        resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur
    )
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)
    return result, (horizontal_pad, vertical_pad)


def resize_and_pad_tensor(tensor, width, height, fill=False, blur=False):
    """
    Resizes an image tensor to the given width and height and pads it to the given width and height.
    :param tensor: A tensor of shape [B, H, W, C]
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param fill: Whether to fill the padding with data from the edges
    :param blur: Whether to blur the padded edges
    :return: A tensor of shape [B, height, width, C]
    """
    # Resize the image to the closest size that maintains the aspect ratio
    width_ratio = width / tensor.shape[3]
    height_ratio = height / tensor.shape[2]
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(tensor.shape[3] * resize_ratio)
    resize_height = round(tensor.shape[2] * resize_ratio)
    resized = F.interpolate(
        tensor, size=(resize_height, resize_width), mode="nearest-exact"
    )
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_tensor(
        resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur
    )
    result = F.interpolate(result, size=(height, width), mode="nearest-exact")
    return result


def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "control" not in cond_dict:
        return
    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet
    while c is not None:
        # hint is shape (B, C, H, W)
        hint = controlnet.cond_hint_original
        resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])
        hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
        hint = resize_tensor(hint, tile_size[::-1])
        controlnet.cond_hint_original = hint
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet


def region_intersection(region1, region2):
    """
    Returns the coordinates of the intersection of two rectangular regions.
    :param region1: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the first rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param region2: The second rectangular region with the same format as the first.
    :return: A tuple of the form (x1, y1, x2, y2) denoting the rectangular intersection.
        None if there is no intersection.
    """
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "gligen" not in cond_dict:
        return
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn

        warn(f"Unknown gligen type {type}")
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        # Get the coordinates of the box in the upscaled image
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Calculate the intersection of the gligen box and the region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset the gligen box so that the origin is at the top left of the tile region
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Add the padding
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad

        # Set the new position params
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))

    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "area" not in cond_dict:
        return

    # Resize the area conditioning to the canvas size and confine it to the tile region
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection

    # Offset origin to the top left of the tile
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]

    # Add the padding
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad

    # Set the params for tile
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8

    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "mask" not in cond_dict:
        return
    mask_tensor = cond_dict["mask"]  # (B, H, W)
    masks = []
    for i in range(mask_tensor.shape[0]):
        # Convert to PIL image
        mask = tensor_to_pil(mask_tensor, i)  # W x H

        # Resize the mask to the canvas size
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)

        # Crop the mask to the region
        mask = mask.crop(region)

        # Add padding
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)

        # Resize the mask to the tile size
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)

        # Convert back to tensor
        mask = pil_to_tensor(mask)  # (1, H, W, 1)
        mask = mask.squeeze(-1)  # (1, H, W)
        masks.append(mask)

    cond_dict["mask"] = torch.cat(masks, dim=0)  # (B, H, W)


def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(
            cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad
        )
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        cropped.append(n)
    return cropped


from PIL import Image

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def _upscale(self, img: Image, scale):
        global actual_upscaler
        if actual_upscaler is None:
            return img.resize(
                (img.width * scale, img.height * scale), Image.Resampling.NEAREST
            )
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()


from PIL import Image, ImageFilter

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:

    def __init__(
        self, p: StableDiffusionProcessing, images: list, seed: int, info: str
    ):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Setup
    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode == "enable":
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(
            crop_region,
            image_mask.width,
            image_mask.height,
            target_width,
            target_height,
        )
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(
            crop_region,
            image_mask.width,
            image_mask.height,
            target_width,
            target_height,
        )

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    global batch
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(
        p.positive, crop_region, p.init_size, init_image.size, tile_size
    )
    negative_cropped = crop_cond(
        p.negative, crop_region, p.init_size, init_image.size, tile_size
    )

    # Encode the image
    vae_encoder = VAEEncode()
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    (samples,) = common_ksampler(
        p.model,
        p.seed,
        p.steps,
        p.cfg,
        p.sampler_name,
        p.scheduler,
        positive_cropped,
        negative_cropped,
        latent,
        denoise=p.denoise,
    )

    # Decode the sample
    vae_decoder = VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(
                initial_tile_size, Image.Resampling.LANCZOS
            )

        # Put the tile into position
        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        image_mask = image_mask.resize(temp.size)
        temp.putalpha(image_mask)
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert("RGB")
        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed


from enum import Enum

from PIL import ImageDraw, ImageOps


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler:

    def __init__(
        self,
        p,
        image,
        upscaler_index: int,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    ) -> None:
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(
            max(p.width, p.height) / max(image.width, image.height)
        )
        global sd_upscalers
        self.upscaler = sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize(
                (self.p.width, self.p.height), resample=Image.LANCZOS
            )
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(
                self.image, value, self.upscaler.data_path
            )
        # Resize image to set values
        self.image = self.image.resize(
            (self.p.width, self.p.height), resample=Image.LANCZOS
        )

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        if type(self.p.prompt) != list:
            save_image(
                self.image,
                self.p.outpath_samples,
                "",
                self.p.seed,
                self.p.prompt,
                opts.samples_format,
                info=self.initial_info,
                p=self.p,
            )
        else:
            save_image(
                self.image,
                self.p.outpath_samples,
                "",
                self.p.seed,
                self.p.prompt[0],
                opts.samples_format,
                info=self.initial_info,
                p=self.p,
            )

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = (
                self.rows * (self.cols - 1)
                + (self.rows - 1) * self.cols
                + (self.rows - 1) * (self.cols - 1)
            )
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = (
            self.upscaler.name
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = (
            self.redraw.tile_width
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = (
            self.redraw.tile_height
        )
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = (
            self.p.mask_blur
        )
        self.p.extra_generation_params["Ultimate SD upscale padding"] = (
            self.redraw.padding
        )

    def process(self):
        global state
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()


class USDURedraw:

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)


class USDUSeamsFix:

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        global state
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(
            gradient.resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        row_gradient.paste(
            gradient.rotate(180).resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, self.tile_height // 2),
        )
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(
            gradient.rotate(90).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        col_gradient.paste(
            gradient.rotate(270).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (self.tile_width // 2, 0),
        )

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    row_gradient,
                    (
                        xi * self.tile_width,
                        yi * self.tile_height + self.tile_height // 2,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    col_gradient,
                    (
                        xi * self.tile_width + self.tile_width // 2,
                        yi * self.tile_height,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        global state
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC
        )
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        # p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(
                    gradient,
                    (
                        xi * self.tile_width + self.tile_width // 2,
                        yi * self.tile_height + self.tile_height // 2,
                    ),
                )

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):
        global state
        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(
            gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0)
        )
        mirror_gradient.paste(
            gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128)
        )

        row_gradient = mirror_gradient.resize(
            (image.width, self.width), resample=Image.BICUBIC
        )
        col_gradient = mirror_gradient.rotate(90).resize(
            (self.width, image.height), resample=Image.BICUBIC
        )

        for xi in range(1, rows):
            if state.interrupted:
                break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if len(processed.images) > 0:
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if len(processed.images) > 0:
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image


class Script(Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size",
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections",
        ]

        redrow_modes = ["Linear", "Chess", "None"]

    def run(
        self,
        p,
        _,
        tile_width,
        tile_height,
        mask_blur,
        padding,
        seams_fix_width,
        seams_fix_denoise,
        seams_fix_padding,
        upscaler_index,
        save_upscaled_image,
        redraw_mode,
        save_seams_fix_image,
        seams_fix_mask_blur,
        seams_fix_type,
        target_size_type,
        custom_width,
        custom_height,
        custom_scale,
    ):

        # Init
        fix_seed(p)
        torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = flatten(init_img, opts.img2img_background_color)

        # override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
        )
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(
            seams_fix_padding,
            seams_fix_denoise,
            seams_fix_mask_blur,
            seams_fix_width,
            seams_fix_type,
        )
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(
            p,
            result_images,
            seed,
            upscaler.initial_info if upscaler.initial_info is not None else "",
        )


# Make some patches to the script
import math

from PIL import Image

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image

#
# Instead of using multiples of 64, use multiples of 8
#

# Upscaler
old_init = USDUpscaler.__init__


def new_init(
    self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height
):
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(
        self,
        p,
        image,
        upscaler_index,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    )


USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

#
# Make the script upscale on a batch of images instead of one image
#

old_upscale = USDUpscaler.upscale


def new_upscale(self):
    old_upscale(self)
    global batch
    batch = [self.image] + [
        img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        for img in batch[1:]
    ]


USDUpscaler.upscale = new_upscale

# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    return [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (KSampler1.SAMPLERS,)),
        ("scheduler", (KSampler1.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        (
            "tile_width",
            ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
        ),
        (
            "tile_height",
            ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
        ),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        (
            "tile_padding",
            ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        ),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        (
            "seam_fix_denoise",
            ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        ),
        (
            "seam_fix_width",
            ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        ),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        (
            "seam_fix_padding",
            ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        ),
        # Misc
        ("force_uniform_tiles", (["enable", "disable"],)),
    ]


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return prepare_inputs(USDU_base_inputs())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
    ):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        actual_upscaler = upscale_model

        # Set the batch of images
        batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image),
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_by,
            force_uniform_tiles,
        )

        #
        # Running the script
        #
        script = Script()
        script.run(
            p=sdprocessing,
            _=None,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            padding=tile_padding,
            seams_fix_width=seam_fix_width,
            seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding,
            upscaler_index=0,
            save_upscaled_image=False,
            redraw_mode=MODES[mode_type],
            save_seams_fix_image=False,
            seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
            target_size_type=2,
            custom_width=None,
            custom_height=None,
            custom_scale=upscale_by,
        )

        # Return the resulting images
        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


class UltimateSDUpscaleNoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        upscaled_image,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
    ):
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = UpscalerData()
        actual_upscaler = None
        batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(upscaled_image),
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            1,
            force_uniform_tiles,
        )

        script = Script()
        processed = script.run(
            p=sdprocessing,
            _=None,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            padding=tile_padding,
            seams_fix_width=seam_fix_width,
            seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding,
            upscaler_index=0,
            save_upscaled_image=False,
            redraw_mode=MODES[mode_type],
            save_seams_fix_image=False,
            seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
            target_size_type=2,
            custom_width=None,
            custom_height=None,
            custom_scale=1,
        )

        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale": UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale": UltimateSDUpscaleNoUpscale,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale": "Ultimate SD Upscale",
    "UltimateSDUpscaleNoUpscale": "Ultimate SD Upscale (No Upscale)",
}


import contextlib
import functools
import logging
from dataclasses import dataclass

import torch
from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
from sfast.compilers.diffusion_pipeline_compiler import (
    _enable_xformers,
    _modify_model,
)
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.jit import utils as jit_utils
from sfast.jit.trace_helper import trace_with_kwargs


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)
    
class ModuleFactory:
    def get_converted_kwargs(self):
        return self.converted_kwargs


import torch as th
import torch.nn as nn   
import copy


class BaseModelApplyModelModule(torch.nn.Module):
    def __init__(self, func, module):
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x,
        timestep,
        c_concat=None,
        c_crossattn=None,
        y=None,
        control=None,
        transformer_options={},
    ):
        kwargs = {"y": y}

        new_transformer_options = {}

        return self.func(
            input_x,
            timestep,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=new_transformer_options,
            **kwargs,
        )


class BaseModelApplyModelModuleFactory(ModuleFactory):
    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable, kwargs) -> None:
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self):
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            key_kwargs[k] = v

        patch_module_cache_key = {}
        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.unet_config),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)


logger = logging.getLogger()


@dataclass
class TracedModuleCacheItem:
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    traced_modules = {}

    def __init__(self, config=None, patch_id=None, **kwargs_) -> None:
        self.config = config
        self.patch_id = patch_id
        self.kwargs_ = kwargs_
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )
        self.cuda_graph_modules = {}

    def ts_compiler(
        self,
        m,
    ):
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                # raw freeze causes Tensor reference leak
                # because the constant Tensors in the GraphFunction of
                # the compilation unit are never freed.
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function, /, **kwargs):
        module_factory = BaseModelApplyModelModuleFactory(model_function, kwargs)
        kwargs = module_factory.get_converted_kwargs()
        key = module_factory.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                logger.info(
                    f'Tracing {getattr(m_model, "__name__", m_model.__class__.__name__)}'
                )
                traced_m, call_helper = trace_with_kwargs(
                    m_model, None, m_kwargs, **self.kwargs_
                )

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            self.cuda_graph_modules[key] = traced_module

        return traced_module(**kwargs)


def build_lazy_trace_module(config, device, patch_id):
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )


def gen_stable_fast_config():
    config = CompilationConfig.Default()
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = True
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device):
        if type(device) == torch.device:
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    # comfyui tell we should move to cpu. but we cannt do it with cuda graph and freeze now.
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disable enable_cuda_graph.\33[0m"
                    )
        return self


class ApplyStableFastUnet:

    def apply_stable_fast(self, model, enable_cuda_graph):
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)


def write_parameters_to_file(prompt_entry, neg, width, height, cfg):
    with open('.\\_internal\\prompt.txt', 'w') as f:
        f.write(f'prompt: {prompt_entry}')
        f.write(f'neg: {neg}')
        f.write(f'w: {int(width)}\n')
        f.write(f'h: {int(height)}\n')
        f.write(f'cfg: {int(cfg)}\n')


def load_parameters_from_file():
    with open('.\\_internal\\prompt.txt', 'r') as f:
        lines = f.readlines()
        parameters = {}
        for line in lines:
            # Skip empty lines
            if line.strip() == "":
                continue
            key, value = line.split(': ')
            parameters[key] = value.strip() 
        prompt = parameters['prompt']
        neg = parameters['neg']
        width = int(parameters['w'])
        height = int(parameters['h'])
        cfg = int(parameters['cfg'])
    return prompt, neg, width, height, cfg


files = glob.glob('.\\_internal\\checkpoints\\*.safetensors')

class App(tk.Tk):  # TODO : dynamic title
    def __init__(self):
        super().__init__()

        self.title("LightDiffusion")
        self.geometry("800x610")

        selected_file = tk.StringVar()
        if files:
            selected_file.set(files[0])

        # Create a frame for the sidebar
        self.sidebar = tk.Frame(self, width=200, bg="black")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Text input for the prompt
        self.prompt_entry = ctk.CTkTextbox(self.sidebar, width=400, height=200)
        self.prompt_entry.pack(pady=10, padx=10)

        self.neg = ctk.CTkTextbox(self.sidebar, width=400, height=50)
        self.neg.pack(pady=10, padx=10)

        self.dropdown = ctk.CTkOptionMenu(self.sidebar, values=files)
        self.dropdown.pack()

        # Sliders for the resolution
        self.width_label = ctk.CTkLabel(self.sidebar, text="")
        self.width_label.pack()
        self.width_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=2048, number_of_steps=2047
        )
        self.width_slider.pack()

        self.height_label = ctk.CTkLabel(self.sidebar, text="")
        self.height_label.pack()
        self.height_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=2048, number_of_steps=2047
        )
        self.height_slider.pack()

        self.cfg_label = ctk.CTkLabel(self.sidebar, text="")
        self.cfg_label.pack()
        self.cfg_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=15, number_of_steps=14
        )
        self.cfg_slider.pack()

        # checkbox for hiresfix
        self.hires_fix_var = tk.BooleanVar()

        self.hires_fix_checkbox = ctk.CTkCheckBox(
            self.sidebar,
            text="Hires Fix",
            variable=self.hires_fix_var,
            command=self.print_hires_fix,
        )
        self.hires_fix_checkbox.pack()

        # add a checkbox to enable stable-fast optimization
        self.stable_fast_var = tk.BooleanVar()
        self.stable_fast_checkbox = ctk.CTkCheckBox(
            self.sidebar, text="Stable Fast", variable=self.stable_fast_var,
        )
        self.stable_fast_checkbox.pack(pady=5)

        # Button to launch the generation
        self.generate_button = ctk.CTkButton(
            self.sidebar, text="Generate", command=self.generate_image
        )
        self.generate_button.pack(pady=20)

        # Create a frame for the image display, without border
        self.display = tk.Frame(self, bg="black", border=0)
        self.display.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Label to display the generated image
        self.image_label = tk.Label(
            self.display, bg="black"
        )  # TODO: adapt to window size
        self.image_label.pack(pady=20)

        self.ckpt = None

        # load the checkpoint on an another thread
        threading.Thread(target=self._prep, daemon=True).start()

        # add an img2img button, the button opens the file selector, run img2img on the selected image
        self.img2img_button = ctk.CTkButton(
            self.sidebar, text="img2img", command=self.img2img
        )
        self.img2img_button.pack()



        prompt, neg, width, height, cfg = load_parameters_from_file()
        self.prompt_entry.insert(tk.END, prompt)
        self.neg.insert(tk.END, neg)
        self.width_slider.set(width)
        self.height_slider.set(height)
        self.cfg_slider.set(cfg)

        self.width_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.height_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.cfg_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.update_labels()
        self.prompt_entry.bind(
            "<KeyRelease>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.neg.bind(
            "<KeyRelease>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.width_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.height_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.cfg_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.display_most_recent_image()

    def _img2img(self, file_path):
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        img = Image.open(file_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            loraloader = LoraLoader()
            loraloader_274 = loraloader.load_lora(
                lora_name="add_detail.safetensors",
                strength_model=2,
                strength_clip=2,
                model=checkpointloadersimple_241[0],
                clip=checkpointloadersimple_241[1],
            )

            if self.stable_fast_var.get() == True:
                applystablefast = ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                    enable_cuda_graph=True,
                    model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274

            clipsetlastlayer = CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )

            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            upscalemodelloader_244 = upscalemodelloader.load_model(
                "RealESRGAN_x4plus_anime_6B.pth"
            )
            app.title("LightDiffusion - Upscaling")
            ultimatesdupscale_250 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=6,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=0.3,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=16,
                tile_padding=32,
                seam_fix_mode="Half Tile",
                seam_fix_denoise=0.3,
                seam_fix_width=64,
                seam_fix_mask_blur=16,
                seam_fix_padding=32,
                force_uniform_tiles="enable",
                image=img_tensor,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                vae=checkpointloadersimple_241[2],
                upscale_model=upscalemodelloader_244[0],
            )
            saveimage.save_images(
                filename_prefix="LD",
                images=ultimatesdupscale_250[0],
            )
            for image in ultimatesdupscale_250[0]:
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img = img.resize((int(w / 2), int(h / 2)))
        img = ImageTk.PhotoImage(img)
        self.image_label.after(0, self._update_image_label, img)
        app.title("LightDiffusion")

    def img2img(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            threading.Thread(
                target=self._img2img, args=(file_path,), daemon=True
            ).start()

    def print_hires_fix(self):
        if self.hires_fix_var.get() == True:
            print("Hires fix is ON")
        else:
            print("Hires fix is OFF")

    def generate_image(self):
        threading.Thread(target=self._generate_image, daemon=True).start()

    def _prep(self):
        if self.dropdown.get() != self.ckpt:
            self.ckpt = self.dropdown.get()
            with torch.inference_mode():
                self.checkpointloadersimple = CheckpointLoaderSimple()
                self.checkpointloadersimple_241 = (
                    self.checkpointloadersimple.load_checkpoint(ckpt_name=self.ckpt)
                )
                self.cliptextencode = CLIPTextEncode()
                self.emptylatentimage = EmptyLatentImage()
                self.ksampler_instance = KSampler2()
                self.vaedecode = VAEDecode()
                self.saveimage = SaveImage()
                self.latent_upscale = LatentUpscale()
                self.upscalemodelloader = UpscaleModelLoader()
                self.ultimatesdupscale = UltimateSDUpscale()
        return (
            self.checkpointloadersimple_241,
            self.cliptextencode,
            self.emptylatentimage,
            self.ksampler_instance,
            self.vaedecode,
            self.saveimage,
            self.latent_upscale,
            self.upscalemodelloader,
            self.ultimatesdupscale,
        )

    def _generate_image(self):
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            loraloader = LoraLoader()
            loraloader_274 = loraloader.load_lora(
                lora_name="add_detail.safetensors",
                strength_model=-2,
                strength_clip=-2,
                model=checkpointloadersimple_241[0],
                clip=checkpointloadersimple_241[1],
            )

            clipsetlastlayer = CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )
            if self.stable_fast_var.get() == True:
                applystablefast = ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                enable_cuda_graph=True,
                model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274

            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            emptylatentimage_244 = emptylatentimage.generate(
                width=w, height=h, batch_size=1
            )
            ksampler_239 = ksampler_instance.sample(
                seed=random.randint(1, 2**64),
                steps=300,
                cfg=cfg,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=emptylatentimage_244[0],
            )
            if self.hires_fix_var.get() == True:
                latentupscale_254 = latentupscale.upscale(
                    upscale_method="bislerp",
                    width=w * 2,
                    height=h * 2,
                    crop="disabled",
                    samples=ksampler_239[0],
                )
                ksampler_253 = ksampler_instance.sample(
                    seed=random.randint(1, 2**64),
                    steps=10,
                    cfg=8,
                    sampler_name="euler_ancestral",
                    scheduler="normal",
                    denoise=0.45,
                    model=applystablefast_158[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=latentupscale_254[0],
                )

                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            else:
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_239[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Convert the image to PhotoImage and display it
        img = img.resize((int(w / 2), int(h / 2)))
        img = ImageTk.PhotoImage(img)
        self.image_label.after(0, self._update_image_label, img)

    def _update_image_label(self, img):
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference to prevent garbage collection

    def update_labels(self):
        self.width_label.configure(text=f"Width: {int(self.width_slider.get())}")
        self.height_label.configure(text=f"Height: {int(self.height_slider.get())}")
        self.cfg_label.configure(text=f"CFG: {int(self.cfg_slider.get())}")

    def display_most_recent_image(self):
        # Get a list of all image files in the output directory
        image_files = glob.glob(".\\_internal\\output\\*")

        # If there are no image files, return
        if not image_files:
            return

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Open the most recent image file
        img = Image.open(image_files[0])

        # Resize the image if necessary
        img = img.resize(
            (int(self.width_slider.get() / 2), int(self.height_slider.get() / 2))
        )

        # Convert the image to PhotoImage
        img = ImageTk.PhotoImage(img)

        # Display the image
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == "__main__":
    app = App()
    app.mainloop()