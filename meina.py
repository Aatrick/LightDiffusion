
import math
import os
import random
import sys
from typing import Dict, Union
from typing import Sequence, Mapping, Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import safetensors.torch
import importlib
from contextlib import contextmanager

from tqdm.auto import tqdm
import psutil
import contextlib
import os

import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig, modeling_utils
from einops import rearrange
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    return sd


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])),
                           filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def repeat_to_batch_size(tensor, batch_size):
    return tensor


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None


class ProgressBar:
    def __init__(self, total):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        self.current = value


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
            [-0.2120, -0.2616, -0.7177]
        ]
        self.taesd_decoder_name = "taesd_decoder"





def exists(x):
    pass


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class DiagonalGaussianRegularizer(torch.nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample


class AbstractAutoencoder(torch.nn.Module):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
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


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        super().__init__(
            encoder_config={
                "target": "meina.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "meina.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **decoder_kwargs)
        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            regularizer_config={
                "target": (
                    "meina.DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )





class Linear(torch.nn.Linear):
    pass


class Conv2d(torch.nn.Conv2d):
    pass


class Conv3d(torch.nn.Conv3d):
    pass


def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)


@contextmanager
def use_comfy_ops(device=None, dtype=None):  # Kind of an ugly hack but I can't think of a better way
    old_torch_nn_linear = torch.nn.Linear
    force_device = device
    force_dtype = dtype

    def linear_with_dtype(in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        if force_device is not None:
            device = force_device
        if force_dtype is not None:
            dtype = force_dtype
        return Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    torch.nn.Linear = linear_with_dtype
    try:
        yield
    finally:
        torch.nn.Linear = old_torch_nn_linear




def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""

    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
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
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
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
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * (
                (r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1.,
                            dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        forward = t_end > t_start
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)

            t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps
            x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback(
                    {'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error,
                     'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3,
                        rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81,
                        eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)),
                                                 dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init,
                                                 pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    return x


def lcm(a, b):  # TODO: eventually replace by math.lcm (added in python3.9)
    return abs(a * b) // math.gcd(a, b)


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))


class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
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
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)  # padding with repeat doesn't change result
            out.append(c)
        return torch.cat(out)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def checkpoint(func, inputs, params, flag):
    return func(*inputs)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module




# Determine VRAM State
vram_state = 3
set_vram_to = 3
cpu_state = 0

total_vram = 0



def get_torch_device():
    global directml_enabled
    global cpu_state
    return torch.device(torch.cuda.current_device())


total_ram = psutil.virtual_memory().total / (1024 * 1024)
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_ENABLED_VAE = True

import xformers.ops

XFORMERS_IS_AVAILABLE = True
XFORMERS_VERSION = xformers.version.__version__
print("xformers version:", XFORMERS_VERSION)



def is_nvidia():
    global cpu_state
    if torch.version.cuda:
        return True


ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float32

FORCE_FP32 = False
FORCE_FP16 = False

current_loaded_models = []


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0

    def model_load(self, lowvram_model_memory=0):
        patch_model_to = None
        if lowvram_model_memory == 0:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        self.real_model = self.model.patch_model(
                device_to=patch_model_to)  # TODO: do something with loras and offloading to CPU

        if lowvram_model_memory > 0:
            print("loading in lowvram mode", lowvram_model_memory / (1024 * 1024))
            self.model_accelerated = True
        return self.real_model

    def model_unload(self):
        self.model.unpatch_model(self.model.offload_device)
        self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model


def minimum_inference_memory():
    return (1024 * 1024 * 1024)


def free_memory1(memory_required, device, keep_loaded=[]):
    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                m = current_loaded_models.pop(i)
                m.model_unload()
                del m
                unloaded_model = True

    if unloaded_model:
        soft_empty_cache()
    else:
        if vram_state != 4:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def load_models_gpu(models, memory_required=0):
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)

        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            if hasattr(x, "model"):
                print(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory1(extra_mem, d, models_already_loaded)
        return

    print(f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}")

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device,
                                                                               0) + loaded_model.model_memory_required(
            loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory1(total_memory_required[device] * 1.3 + extra_mem, device, models_already_loaded)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        vram_set_state = vram_state
        lowvram_model_memory = 0
        if (vram_set_state == 2 or vram_set_state == 3):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(max(256 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3))
            if model_size > (current_free_mem - inference_memory):  # only switch to lowvram if really necessary
                vram_set_state = 2
            else:
                lowvram_model_memory = 0

        cur_loaded_model = loaded_model.model_load(lowvram_model_memory)
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model):
    return load_models_gpu([model])


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    return dtype_size


def unet_offload_device():
    return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()

    cpu_dev = torch.device("cpu")

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev


def unet_dtype1(device=None, model_params=0):
    if should_use_fp16(device=device, model_params=model_params):
        return torch.float16


def text_encoder_offload_device():
    return torch.device("cpu")


def text_encoder_device():
    return torch.device("cpu")


def vae_device():
    return get_torch_device()


def vae_offload_device():
    return torch.device("cpu")


def vae_dtype():
    global VAE_DTYPE
    return VAE_DTYPE


def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type


def xformers_enabled():
    global directml_enabled
    global cpu_state
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    return XFORMERS_ENABLED_VAE


def get_free_memory(dev=None, torch_free_too=False):
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        stats = torch.cuda.memory_stats(dev)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def batch_area_memory(area):
    if xformers_enabled():
        # TODO: these formulas are copied from maximum_batch_area below
        return (area / 20) * (1024 * 1024)


def maximum_batch_area():
    global vram_state

    memory_free = get_free_memory() / (1024 * 1024)
    area = 20 * memory_free
    return int(max(area, 0))


def cpu_mode():
    global cpu_state
    return cpu_state == 1


def mps_mode():
    global cpu_state
    return cpu_state == 2


def is_device_cpu(device):
    if hasattr(device, 'type'):
        if (device.type == 'cpu'):
            return True


def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if torch.cuda.is_bf16_supported():
        return True


def soft_empty_cache(force=False):
    global cpu_state
    if torch.cuda.is_available():
        if force or is_nvidia():  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
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

    def model_size(self):
        model_sd = self.model.state_dict()
        size = 0
        for k in model_sd:
            t = model_sd[k]
            size += t.nelement() * t.element_size()
        self.size = size
        self.model_keys = set(model_sd.keys())
        return size

    def is_clone(self, other):
        return False

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        return sd

    def patch_model(self, device_to=None):
        model_sd = self.model_state_dict()
        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to
        return self.model

    def unpatch_model(self, device_to=None):
        keys = list(self.backup.keys())
        self.backup = {}

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        self.object_patches_backup = {}


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    pass

class Downsample(nn.Module):
    pass


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = Conv2d(out_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


def xformers_attention(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = Conv2d(in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.k = Conv2d(in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.v = Conv2d(in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.proj_out = Conv2d(in_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        if xformers_enabled_vae():
            print("Using xformers attention in VAE")
            self.optimized_attention = xformers_attention

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


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = Conv2d(in_channels,
                              self.ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)

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
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2d(block_in,
                               2 * z_channels if double_z else z_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 conv_out_op=Conv2d,
                 resnet_op=ResnetBlock,
                 attn_op=AttnBlock,
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
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
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = Conv2d(z_channels,
                              block_in,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout)
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(resnet_op(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(block_in,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]

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
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        return h


class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        beta_schedule = "linear"
        if model_config is not None:
            beta_schedule = model_config.beta_schedule
        self._register_schedule(given_betas=None, beta_schedule=beta_schedule, timesteps=1000, linear_start=0.00085,
                                linear_end=0.012, cosine_s=8e-3)
        self.sigma_data = 1.0

    def _register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                           linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)


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
            first_pooled = pooled[0:1].cpu()

        output = []
        for k in range(0, sections):
            z = out[k:k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, textmodel_path=None, dtype=None,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True,
                 config_class=CLIPTextConfig,
                 model_class=CLIPTextModel, inner_name="text_model"):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.num_layers = 12

        if textmodel_json_config is None:
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     "sd1_clip_config.json")
        config = config_class.from_json_file(textmodel_json_config)
        self.num_layers = config.num_hidden_layers
        with use_comfy_ops(device, dtype):
            with modeling_utils.no_init_weights():
                self.transformer = model_class(config)

        self.inner_name = inner_name
        if dtype is not None:
            self.transformer.to(dtype)
            inner_model = getattr(self.transformer, self.inner_name)
            if hasattr(inner_model, "embeddings"):
                inner_model.embeddings.to(torch.float32)

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.enable_attention_masks = False

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.layer_default = (self.layer, self.layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

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
            out_tokens += [tokens_temp]

        n = token_dict_size

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(get_autocast_device(device), torch.float32):
            attention_mask = None

            outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask,
                                       output_hidden_states=self.layer == "hidden")
            self.transformer.set_input_embeddings(backup_embeds)

            if self.layer == "last":
                z = outputs.last_hidden_state

            if hasattr(outputs, "pooler_output"):
                pooled_output = outputs.pooler_output.float()

            if self.text_projection is not None and pooled_output is not None:
                pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()
        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)


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
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                weight = float(x[xx + 1:])
                x = x[:xx]
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


class SDTokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None,
                 embedding_size=768, embedding_key='clip_l', tokenizer_class=CLIPTokenizer, has_start_token=True,
                 pad_to_max_length=True):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length

        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])

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
                    # add end token and pad
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

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens


class SD1Tokenizer:
    def __init__(self, embedding_directory=None, clip_name="l", tokenizer=SDTokenizer):
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(text, return_word_ids)
        return out


class SD1ClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, clip_name="l", clip_model=SDClipModel, **kwargs):
        super().__init__()
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def reset_clip_layer(self):
        getattr(self, self.clip).reset_clip_layer()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out, pooled = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out, pooled

# The main sampling function shared by all the samplers
# Returns denoised
def sampling_function(model_function, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    def get_area_and_mult(conds, x_in, timestep_in):
        area = (x_in.shape[2], x_in.shape[3], 0, 0)
        strength = 1.0

        input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        mask = torch.ones_like(input_x)
        mult = mask * strength

        conditionning = {}
        model_conds = conds["model_conds"]
        for c in model_conds:
            conditionning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

        control = None

        patches = None

        return (input_x, mult, conditionning, area, control, patches)

    def cond_equal_size(c1, c2):
        if c1 is c2:
            return True
        return True

    def can_concat_cond(c1, c2):
        return cond_equal_size(c1[2], c2[2])

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

    def calc_cond_uncond_batch(model_function, cond, uncond, x_in, timestep, max_total_area, model_options):
        out_cond = torch.zeros_like(x_in)
        out_count = torch.ones_like(x_in) * 1e-37

        out_uncond = torch.zeros_like(x_in)
        out_uncond_count = torch.ones_like(x_in) * 1e-37

        COND = 0
        UNCOND = 1

        to_run = []
        for x in cond:
            p = get_area_and_mult(x, x_in, timestep)

            to_run += [(p, COND)]
        if uncond is not None:
            for x in uncond:
                p = get_area_and_mult(x, x_in, timestep)

                to_run += [(p, UNCOND)]

        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp) // i]
                if (len(batch_amount) * first_shape[0] * first_shape[2] * first_shape[3] < max_total_area):
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
                input_x += [p[0]]
                mult += [p[1]]
                c += [p[2]]
                area += [p[3]]
                cond_or_uncond += [o[1]]
                control = p[4]
                patches = p[5]

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            transformer_options = {}
            if 'transformer_options' in model_options:
                transformer_options = model_options['transformer_options'].copy()

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            c['transformer_options'] = transformer_options

            output = model_function(input_x, timestep_, **c).chunk(batch_chunks)
            del input_x

            for o in range(batch_chunks):
                if cond_or_uncond[o] == COND:
                    out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[
                                                                                                                  o] * \
                                                                                                              mult[o]
                    out_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
                else:
                    out_uncond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[
                                                                                                                    o] * \
                                                                                                                mult[o]
                    out_uncond_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += \
                        mult[o]
            del mult

        out_cond /= out_count
        del out_count
        out_uncond /= out_uncond_count
        del out_uncond_count
        return out_cond, out_uncond

    max_total_area = maximum_batch_area()

    cond, uncond = calc_cond_uncond_batch(model_function, cond, uncond, x, timestep, max_total_area, model_options)
    return uncond + (cond - uncond) * cond_scale


class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        out = sampling_function(self.inner_model.apply_model, x, timestep, uncond, cond, cond_scale,
                                model_options=model_options, seed=seed)
        return out

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)


class KSamplerX0Inpaint(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options,
                               seed=seed)
        return out


def resolve_areas_and_cond_masks(conditions, h, w, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]


def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return


def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None


def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)


def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            uncond_other.append((x, t))


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
        model_conds = x['model_conds'].copy()
        x['model_conds'] = model_conds
        conds[t] = x
    return conds


class Sampler:

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


KSAMPLER_NAMES = ["dpm_adaptive", "dpmpp_2m"]


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    class KSAMPLER(Sampler):
        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None,
                   disable_pbar=False):
            extra_args["denoise_mask"] = denoise_mask
            model_k = KSamplerX0Inpaint(model_wrap)
            model_k.latent_image = latent_image
            model_k.noise = noise

            if self.max_denoise(model_wrap, sigmas):
                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)

            k_callback = None
            total_steps = len(sigmas) - 1
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]

            if latent_image is not None:
                noise += latent_image
            samples = sample_dpm_adaptive(model_k, noise, sigma_min, sigmas[0], extra_args=extra_args,
                                          callback=k_callback, disable=disable_pbar)
            return samples

    return KSAMPLER


def wrap_model(model):
    model_denoise = CFGNoisePredictor(model)
    return model_denoise


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None,
           denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    # make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    pre_run_control(model, negative + positive)

    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)),
                                negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image,
                                      denoise_mask=denoise_mask)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image,
                                      denoise_mask=denoise_mask)

    extra_args = {"cond": positive, "uncond": negative, "cond_scale": cfg, "model_options": model_options, "seed": seed}

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))


SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


def calculate_sigmas_scheduler(model, scheduler_name, steps):
    sigmas = get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min),
                               sigma_max=float(model.model_sampling.sigma_max))
    return sigmas


def sampler_class(name):
    sampler = ksampler(name)
    return sampler


class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = calculate_sigmas_scheduler(self.model, self.scheduler, steps)
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None,
               force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        sampler = sampler_class(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler(), sigmas, self.model_options,
                      latent_image=latent_image, denoise_mask=denoise_mask, callback=callback,
                      disable_pbar=disable_pbar, seed=seed)


def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device="cpu")


def get_models_from_cond(cond, model_type):
    models = []
    return models


def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def get_additional_models(positive, negative, dtype):
    """loads additional models in positive and negative conditioning"""
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    device = model.load_device
    positive = convert_cond(positive)
    negative = convert_cond(negative)
    real_model = None
    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    load_models_gpu([model] + models, batch_area_memory(
        noise_shape[0] * noise_shape[2] * noise_shape[3]) + inference_memory)
    real_model = model.model

    return real_model, positive, negative, noise_mask, models


def sample1(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None,
            sigmas=None,
            callback=None, disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive,
                                                                                    negative, noise_mask)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    sampler = KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name,
                       scheduler=scheduler, denoise=denoise, model_options=model.model_options)


    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar,
                             seed=seed)
    samples = samples.cpu()
    return samples

_ATTN_PRECISION = "fp32"


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None):
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype,
                              device=device)


def attention_xformers(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # actually compute the attention, what we cannot get enough of
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


optimized_attention_masked = attention_xformers
print("Using xformers cross attention")
optimized_attention = attention_xformers


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, dtype=None, device=None):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None, dtype=dtype,
                                    device=device)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype,
                                    device=device)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(self._forward, (x, context, transformer_options), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = None
        block_index = 0
        if "current_index" in transformer_options:
            extra_options["transformer_index"] = transformer_options["current_index"]
        if "block_index" in transformer_options:
            block_index = transformer_options["block_index"]
            extra_options["block_index"] = block_index
        if "original_shape" in transformer_options:
            extra_options["original_shape"] = transformer_options["original_shape"]
        if "block" in transformer_options:
            block = transformer_options["block"]
            extra_options["block"] = block
        if "cond_or_uncond" in transformer_options:
            extra_options["cond_or_uncond"] = transformer_options["cond_or_uncond"]
        else:
            transformer_patches = {}

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        transformer_patches_replace = {}

        n = self.norm1(x)
        context_attn1 = None
        value_attn1 = None

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        n = self.attn1(n, context=context_attn1, value=value_attn1)

        x += n

        n = self.norm2(x)

        context_attn2 = context
        value_attn2 = None

        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block

        n = self.attn2(n, context=context_attn2, value=value_attn2)

        x += n
        x = self.ff(self.norm3(x)) + x
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

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dtype=None, device=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = Conv2d(in_channels,
                                  inner_dim,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, dtype=dtype,
                                   device=device)
             for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = Conv2d(inner_dim, in_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0, dtype=dtype, device=device)
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
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    pass


# This is needed because accelerate makes a copy of transformer_options which breaks "current_index"
def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None):
    for layer in ts:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            transformer_options["current_index"] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device)

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, dtype=dtype, device=device
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
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
            dtype=None,
            device=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=dtype, device=device),
        )

        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=dtype, device=device)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

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
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


def apply_control(h, control, name):
    return h


class UNetModel(nn.Module):
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
            device=None
    ):
        super().__init__()
        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        self.num_res_blocks = num_res_blocks

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
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim
                            , use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint, dtype=self.dtype, device=device
                        )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        mid_block = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device
            )]
        if transformer_depth_middle >= 0:
            mid_block += [SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, dtype=self.dtype, device=device
            ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    device=device
                )]
        self.middle_block = TimestepEmbedSequential(*mid_block)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, dtype=self.dtype, device=device
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype,
                                      device=device)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["current_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options)
            h = apply_control(h, control, 'input')
            hs.append(h)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
        h = h.type(x.dtype)
        return self.out(h)



def model_sampling(model_config, model_type):
    if model_type == EPS:
        c = EPS

    s = ModelSamplingDiscrete

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=EPS, device=None):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config

        if not unet_config.get("disable_unet_model_creation", False):
            self.diffusion_model = UNetModel(**unet_config, device=device)
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0
        self.inpaint_model = False
        print("adm", self.adm_channels)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        context = c_crossattn
        dtype = self.get_dtype()
        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        model_output = self.diffusion_model(xc, t, context=context, control=control,
                                            transformer_options=transformer_options, **extra_conds).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype


    def extra_conds(self, **kwargs):
        out = {}
        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)


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
    clip_vision_prefix = None
    noise_aug_config = None
    beta_schedule = "linear"
    latent_format = LatentFormat

    @classmethod
    def matches(s, unet_config):
        return True

    def model_type(self, state_dict, prefix=""):
        return EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        out = BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)
        return out


class SD15(BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = SD15

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "cond_stage_model.clip_l."
        state_dict = state_dict_prefix_replace(state_dict, replace_prefix)
        return state_dict

    def clip_target(self):
        return ClipTarget(SD1Tokenizer, SD1ClipModel)


models = [SD15]


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
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        return last_transformer_depth, context_dim, use_linear_in_transformer
    return None


def detect_unet_config(state_dict, key_prefix, dtype):
    state_dict_keys = list(state_dict.keys())

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False
    }

    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    unet_config["adm_in_channels"] = None

    unet_config["dtype"] = dtype
    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(state_dict_keys, '{}input_blocks'.format(key_prefix) + '.{}.')
    for count in range(input_block_count):
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        prefix_output = '{}output_blocks.{}.'.format(key_prefix, input_block_count - count - 1)

        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))

        block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))

        if "{}0.op.weight".format(prefix) in block_keys:  # new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(state_dict_keys,
                                                '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')

    unet_config["in_channels"] = in_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim
    return unet_config


def model_config_from_unet_config(unet_config):
    for model_config in models:
        if model_config.matches(unet_config):
            return model_config(unet_config)


def model_config_from_unet(state_dict, unet_key_prefix, dtype, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, unet_key_prefix, dtype)
    model_config = model_config_from_unet_config(unet_config)
    return model_config



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
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = text_encoder_device()
        offload_device = text_encoder_offload_device()
        params['device'] = offload_device
        params['dtype'] = torch.float32

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.layer_idx = None

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        self.cond_stage_model.reset_clip_layer()

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled

    def load_model(self):
        load_model_gpu(self.patcher)
        return self.patcher


class VAE:
    def __init__(self, sd=None, device=None, config=None):
        if config is None:
            # default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Missing VAE keys", m)

        if len(u) > 0:
            print("Leftover VAE keys", u)

        if device is None:
            device = vae_device()
        self.device = device
        self.offload_device = vae_offload_device()
        self.vae_dtype = vae_dtype()
        self.first_stage_model.to(self.vae_dtype)

    def decode(self, samples_in):
        self.first_stage_model = self.first_stage_model.to(self.device)
        memory_used = (2562 * samples_in.shape[2] * samples_in.shape[3] * 64) * 1.7
        free_memory1(memory_used, self.device)
        free_memory = get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        pixel_samples = torch.empty(
            (samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
            pixel_samples[x:x + batch_number] = torch.clamp(
                (self.first_stage_model.decode(samples).cpu().float() + 1.0) / 2.0, min=0.0, max=1.0)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        pixel_samples = pixel_samples.cpu().movedim(1, -1)
        return pixel_samples


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False,
                                 embedding_directory=None, output_model=True):
    sd = load_torch_file(ckpt_path)
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = unet_dtype1(model_params=parameters)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    if output_model:
        inital_load_device = unet_inital_load_device(parameters, unet_dtype)
        offload_device = unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
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
        model_patcher = ModelPatcher(model, load_device=get_torch_device(), offload_device=unet_offload_device(),
                                     current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)

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
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]
        digits = int(filename[prefix_len + 1:].split('_')[0])
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
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                             map(map_filename, os.listdir(full_output_folder))))[0] + 1
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

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = ProgressBar(steps)

    def callback(step, x0, x, total_steps):
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
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None

    callback = prepare_callback(model, steps)
    disable_pbar = not PROGRESS_BAR_ENABLED
    samples = sample1(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                      denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                      disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler1:
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise)


class CheckpointLoaderSimple:
    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        return out[:3]


class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    return obj[index]


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
        ksampler = KSampler1()
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
        saveimage.save_images(
            filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_240, 0)
        )

gen(prompt,w,h)