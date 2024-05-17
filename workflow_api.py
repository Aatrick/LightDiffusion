import functools
import math
import random
import re
from collections import OrderedDict
from typing import Literal
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from comfy.model_management import get_free_memory, OOM_EXCEPTION
from comfy.utils import load_torch_file, state_dict_prefix_replace, ProgressBar
import usdu
from nodes import (
    CLIPSetLastLayer,
    KSampler,
    EmptyLatentImage,
    LatentUpscale,
    CLIPTextEncode,
    VAEDecode,
    LoraLoader,
    SaveImage,
    CheckpointLoaderSimple
)


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
        out_channels * (upscale_factor ** 2),
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
            # currently supports old, new, and newer RRDBNet arch models
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
            self.in_nc //= self.shuffle_factor ** 2
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
        return 2 ** n

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


PyTorchSRModels = (
    RRDBNet,
)
PyTorchSRModel = Union[
    RRDBNet,
]

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
        model_path = f".\\models\\upscale_models\\{model_name}"
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, pbar=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount),
                          round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        out_div = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
                out[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += ps * mask
                out_div[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += mask

        output[b:b + 1] = out / out_div
    return output


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"upscale_model": ("UPSCALE_MODEL",),
                             "image": ("IMAGE",),
                             }}

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
                steps = in_img.shape[0] * get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile,
                                                                tile_y=tile, overlap=overlap)
                pbar = ProgressBar(steps)
                s = tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap,
                                upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


def main():
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
            ckpt_name="meinamix_meinaV11.safetensors"
        )

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

        cliptextencode = CLIPTextEncode()
        cliptextencode_242 = cliptextencode.encode(
            text="here's a picture of : masterpiece, best quality, (extremely detailed CG unity 8k wallpaper, masterpiece, best quality, ultra-detailed, best shadow), (detailed background), (beautiful detailed face, beautiful detailed eyes), High contrast, (best illumination, an extremely delicate and beautiful),1girl,((colourful paint splashes on transparent background, dulux,)), ((caustic)), dynamic angle,beautiful detailed glow,full body, cowboy shot",
            clip=clipsetlastlayer_257[0],
        )

        cliptextencode_243 = cliptextencode.encode(
            text="' (worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic) '",
            clip=clipsetlastlayer_257[0],
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_244 = emptylatentimage.generate(
            width=512, height=1024, batch_size=1
        )

        upscalemodelloader = UpscaleModelLoader()
        upscalemodelloader_251 = upscalemodelloader.load_model(
            model_name="RealESRGAN_x4plus_anime_6B.pth"
        )

        ksampler = KSampler()
        latentupscale = LatentUpscale()
        vaedecode = VAEDecode()
        ultimatesdupscale = usdu.UltimateSDUpscale()
        saveimage = SaveImage()

        for q in range(1):
            ksampler_239 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=50,
                cfg=10,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=loraloader_274[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=emptylatentimage_244[0],
            )

            latentupscale_254 = latentupscale.upscale(
                upscale_method="nearest-exact",
                width=1024,
                height=2048,
                crop="disabled",
                samples=ksampler_239[0],
            )

            ksampler_253 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=10,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=0.45,
                model=loraloader_274[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=latentupscale_254[0],
            )

            vaedecode_240 = vaedecode.decode(
                samples=ksampler_253[0],
                vae=checkpointloadersimple_241[2],
            )

            ultimatesdupscale_250 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2 ** 64),
                steps=25,
                cfg=7,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=0.2,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=16,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=0,
                seam_fix_width=64,
                seam_fix_mask_blur=16,
                seam_fix_padding=32,
                force_uniform_tiles="enable",
                image=vaedecode_240[0],
                model=loraloader_274[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                vae=checkpointloadersimple_241[2],
                upscale_model=upscalemodelloader_251[0],
            )

            saveimage_277 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=ultimatesdupscale_250[0],
            )


if __name__ == "__main__":
    main()
