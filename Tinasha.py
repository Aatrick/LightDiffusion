from __future__ import annotations

import contextlib
import glob
import logging
import os
import pickle
import random
import threading
import tkinter as tk
from contextlib import contextmanager
from tkinter import *
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import psutil
import requests
import safetensors.torch
import torch
import torch as th
import torch.nn as nn
from PIL import Image, ImageTk
from einops import rearrange
from tqdm.auto import trange, tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig, modeling_utils

load = pickle.load


class Empty:
    pass


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    elif ckpt is None:
        print("Downloading the model")
        response = requests.get(
            "https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/text_encoder/model.safetensors",
            stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open("model.safetensors", "wb") as handle:
            for data in response.iter_content(chunk_size=8192):
                progress_bar.update(len(data))
                handle.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        sd = safetensors.torch.load_file("model.safetensors", device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
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


class AutoencodingEngine(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.post_quant_conv = torch.nn.Conv2d(4, 4, 1)

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **decoder_kwargs)
        return dec


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
def use_comfy_ops(device=None, dtype=None):
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


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cuda'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def to_d(x, sigma, denoised):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    denoised = denoised.to(device)
    sigma = sigma.to(device)
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


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

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = 1 + math.atan(factor - 1)
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

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = t.neg().exp() * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / t.neg().exp()
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - s1.neg().exp() * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - t_next.neg().exp() * h.expm1() * eps - t_next.neg().exp() / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - s1.neg().exp() * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - s2.neg().exp() * (r2 * h).expm1() * eps - s2.neg().exp() * (r2 / r1) * (
                (r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - t_next.neg().exp() * h.expm1() * eps - t_next.neg().exp() / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
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
        steps=0
        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)

            t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - s.neg().exp() * eps
            x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(s.neg().exp(), t.neg().exp())
                s = t
                info['n_accept'] += 1
            info['nfe'] += order
            info['steps'] += 1
            steps += 1
            def generate(steps):
                steps = steps * 3
                ratio = (steps * 100) / 70
                app.title(f"LightDiffusion - generating : {int(ratio)}%")

            threading.Thread(target=generate, args=(steps,), daemon=True).start()
            if self.info_callback is not None:
                self.info_callback(
                    {'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error,
                     'h': pid.h, **info})
        app.title("LightDiffusion")
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
                {'sigma': info['t'].neg().exp(), 'sigma_hat': info['t_up'].neg().exp(), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, -torch.tensor(sigma_max).log(),
                                                 -torch.tensor(sigma_min).log(), order, rtol, atol, h_init,
                                                 pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                           noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        app.title(f"LightDiffusion - sampling {i + 1}")
    app.title("LightDiffusion")
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
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
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
        app.title(f"LightDiffusion - sampling {i + 1}")
    app.title("LightDiffusion")
    return x


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond.to(device))


class CONDCrossAttn(CONDRegular):

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = abs(crossattn_max_len * c.shape[1]) // math.gcd(crossattn_max_len, c.shape[1])
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

OOM_EXCEPTION = torch.cuda.OutOfMemoryError

XFORMERS_ENABLED_VAE = True
XFORMERS_IS_AVAILABLE = True


def is_nvidia():
    global cpu_state
    if torch.version.cuda:
        return True


ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float16

FORCE_FP32 = False
FORCE_FP16 = True

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

        self.real_model = self.model.patch_model(
            device_to=patch_model_to)
        return self.real_model

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
                del m
                unloaded_model = True

    if unloaded_model:
        soft_empty_cache()
    else:
        if vram_state != 4:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)


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
        model_memory_required = loaded_model.model_memory_required(loaded_model.device)
        if model_memory_required is not None:
            total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device,
                                                                                   0) + model_memory_required
        else:
            total_memory_required[loaded_model.device] = 6000

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
            lowvram_model_memory = 0

        loaded_model.model_load(lowvram_model_memory)
        current_loaded_models.insert(0, loaded_model)
    return


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    return dtype_size


def unet_inital_load_device(parameters, dtype):
    torch_dev = torch.device(torch.cuda.current_device())

    cpu_dev = torch.device("cpu")

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev


def unet_dtype1(device=None, model_params=0):
    if should_use_fp16(device=device, model_params=model_params):
        return torch.float16


def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type


def get_free_memory(dev=None, torch_free_too=False):
    if dev is None:
        dev = torch.device(torch.cuda.current_device())

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
    if XFORMERS_IS_AVAILABLE or ENABLE_PYTORCH_ATTENTION:
        return (area / 20) * (1024 * 1024)


def maximum_batch_area():
    global vram_state

    memory_free = get_free_memory() / (1024 * 1024)
    area = 20 * memory_free
    return int(max(area, 0))


def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    global directml_enabled
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


def xformers_attention(q, k, v):  # TODO : Add stable-fast or TensorRT optimization
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    return out


try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
    print("Using xformers cross attention")
    optimized_attention = xformers_attention
except:
    XFORMERS_IS_AVAILABLE = False
    print("Using pytorch cross attention")
    optimized_attention = attention_pytorch


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

        if XFORMERS_ENABLED_VAE:
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
        self.mid.attn_1 = AttnBlock(block_in)
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
        h = h * torch.sigmoid(h)
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

    def sigma(self, timestep):
        t = torch.clamp(timestep.float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


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
            textmodel_json_config = ".\\_internal\\sd1_clip_config.json"
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
            tokenizer_path = ".\\_internal\\sd1_tokenizer\\"
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

    def cond_cat(c_list):
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
            if sampler_name == "dpm_adaptive":
                samples = sample_dpm_adaptive(model_k, noise, sigma_min, sigmas[0], extra_args=extra_args,
                                              callback=k_callback, disable=disable_pbar)
            elif sampler_name == "euler_ancestral":
                samples = sample_euler_ancestral(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback,
                                                 disable=disable_pbar, **extra_options)
            elif sampler_name == "dpmpp_2m":
                samples = sample_dpmpp_2m(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback,
                                          disable=disable_pbar, **extra_options)
            return samples

    return KSAMPLER


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None,
           denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    positive = positive[:]
    negative = negative[:]

    model_wrap = CFGNoisePredictor(model)
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


def normal_scheduler(model, steps, sgm=False, floor=False):
    s = model.model_sampling
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


def calculate_sigmas_scheduler(model, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min),
                                   sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model, steps)
    return sigmas


class KSampler:
    SCHEDULERS = ["karras, normal"]
    SAMPLERS = ["dpm_adaptive", "euler_ancestral", "dpmpp_2m"]

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
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None,
               force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        sampler = ksampler(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler(), sigmas, self.model_options,
                      latent_image=latent_image, denoise_mask=denoise_mask, callback=callback,
                      disable_pbar=disable_pbar, seed=seed)


def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device="cpu")


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
    inference_memory = 0
    control_models = []

    gligen = []
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
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


_ATTN_PRECISION = "fp16"


def default(val, d):
    if val is not None:
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
            out = attention_xformers(q, k, v, self.heads)
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
        return self._forward(*(x, context, transformer_options))

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
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=True,
                 use_checkpoint=True, dtype=None, device=None):
        super().__init__()
        if context_dim is not None and not isinstance(context_dim, list):
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
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    pass


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
        return self._forward(*(x, emb))

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
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
            use_linear_in_transformer=True,
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

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
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

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                use_linear=use_linear_in_transformer,
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
            hs.append(h)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()

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

    def __init__(self, unet_config):
        self.unet_config = unet_config
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        out = BaseModel(self, model_type=EPS, device=device)
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

        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")
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
        load_models_gpu([self.patcher])
        return self.patcher


class VAE:
    def __init__(self, sd=None, device=None, config=None):
        if config is None:
            config = {
                'encoder': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3,
                            'ch': 128,
                            'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0},
                'decoder': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3,
                            'ch': 128,
                            'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0},
                'regularizer': {'sample': True}
            }
            self.first_stage_model = AutoencodingEngine(
                Encoder(**config['encoder']),
                Decoder(**config['decoder']),
            )
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Missing VAE keys", m)

        if len(u) > 0:
            print("Leftover VAE keys", u)

        if device is None:
            device = torch.device(torch.cuda.current_device())
        self.device = device
        self.offload_device = torch.device("cpu")
        self.vae_dtype = VAE_DTYPE
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
    print(f"Using {unet_dtype} precision")

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    if output_model:
        inital_load_device = unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae = VAE(sd=vae_sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = ClipTarget(SD1Tokenizer, SD1ClipModel)
        if clip_target is not None:
            clip = CLIP(clip_target, embedding_directory=embedding_directory)
            w.cond_stage_model = clip.cond_stage_model
            sd = model_config.process_clip_state_dict(sd)
            load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        model_patcher = ModelPatcher(model, load_device=torch.device(torch.cuda.current_device()),
                                     offload_device=torch.device("cpu"),
                                     current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            load_models_gpu([model_patcher])

    return (model_patcher, clip, vae, clipvision)


output_directory = '.\\_internal\\output\\'


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


def prepare_callback(model, steps, x0_output_dict=None):
    pbar = ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback


class EmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent},)


MAX_RESOLUTION = 8192


class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


def common_upscale(samples, width, height, upscale_method, crop):
    s = samples
    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


class LatentUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, width, height, crop):
        s = samples.copy()
        width = max(64, width)
        height = max(64, height)
        s["samples"] = common_upscale(samples["samples"], width // 8, height // 8, upscale_method, crop)
        return (s,)


class SaveImage:
    def __init__(self):
        self.output_dir = output_directory
        self.type = "output"
        self.prefix_append = ""

    def save_images(self, images, filename_prefix="Aatricks", prompt=None, extra_pnginfo=None):
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
        ckpt_path = f"{ckpt_name}"
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        return out[:3]


import functools
import math
import re
from collections import OrderedDict
from typing import Literal
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn


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
        model_path = f".\\_internal\\{model_name}"
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


class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


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
            parameters[key] = value.strip()  # strip() is used to remove leading/trailing white spaces
        prompt = parameters['prompt']
        neg = parameters['neg']
        width = int(parameters['w'])
        height = int(parameters['h'])
        cfg = int(parameters['cfg'])
    return prompt, neg, width, height, cfg


files = glob.glob('*.safetensors')


class App(tk.Tk):  # TODO : Add LoRa support
    def __init__(self):
        super().__init__()

        self.title('LightDiffusion')
        self.geometry('800x600')

        selected_file = tk.StringVar()
        if files:
            selected_file.set(files[0])

        # Create a frame for the sidebar
        self.sidebar = tk.Frame(self, width=200, bg='black')
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
        self.width_slider = ctk.CTkSlider(self.sidebar, from_=1, to=2048, number_of_steps=2047)
        self.width_slider.pack()

        self.height_label = ctk.CTkLabel(self.sidebar, text="")
        self.height_label.pack()
        self.height_slider = ctk.CTkSlider(self.sidebar, from_=1, to=2048, number_of_steps=2047)
        self.height_slider.pack()

        self.cfg_label = ctk.CTkLabel(self.sidebar, text="")
        self.cfg_label.pack()
        self.cfg_slider = ctk.CTkSlider(self.sidebar, from_=1, to=15, number_of_steps=14)
        self.cfg_slider.pack()

        # checkbox for hiresfix
        self.hires_fix_var = tk.BooleanVar()

        self.hires_fix_checkbox = ctk.CTkCheckBox(self.sidebar, text="Hires Fix", variable=self.hires_fix_var,
                                                  command=self.print_hires_fix)
        self.hires_fix_checkbox.pack()

        # Button to launch the generation
        self.generate_button = ctk.CTkButton(self.sidebar, text="Generate", command=self.generate_image)
        self.generate_button.pack(pady=20)

        # Create a frame for the image display, without border
        self.display = tk.Frame(self, bg='black', border=0)
        self.display.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Label to display the generated image
        self.image_label = tk.Label(self.display, bg='black')
        self.image_label.pack(pady=20)

        self.ckpt = None

        # load the checkpoint on an another thread
        threading.Thread(target=self._prep, daemon=True).start()

        # add an img2img button, the button opens the file selector, run img2img on the selected image
        self.img2img_button = ctk.CTkButton(self.sidebar, text="img2img", command=self.img2img)
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
        self.prompt_entry.bind("<KeyRelease>",
                               lambda event: write_parameters_to_file(self.prompt_entry.get("1.0", tk.END),
                                                                      self.neg.get("1.0", tk.END),
                                                                      self.width_slider.get(),
                                                                      self.height_slider.get(), self.cfg_slider.get()))
        self.neg.bind("<KeyRelease>",
                      lambda event: write_parameters_to_file(self.prompt_entry.get("1.0", tk.END),
                                                             self.neg.get("1.0", tk.END),
                                                             self.width_slider.get(),
                                                             self.height_slider.get(), self.cfg_slider.get()))
        self.width_slider.bind("<ButtonRelease-1>",
                               lambda event: write_parameters_to_file(self.prompt_entry.get("1.0", tk.END),
                                                                      self.neg.get("1.0", tk.END),
                                                                      self.width_slider.get(), self.height_slider.get(),
                                                                      self.cfg_slider.get()))
        self.height_slider.bind("<ButtonRelease-1>",
                                lambda event: write_parameters_to_file(self.prompt_entry.get("1.0", tk.END),
                                                                       self.neg.get("1.0", tk.END),
                                                                       self.width_slider.get(),
                                                                       self.height_slider.get(),
                                                                       self.cfg_slider.get()))
        self.cfg_slider.bind("<ButtonRelease-1>",
                             lambda event: write_parameters_to_file(self.prompt_entry.get("1.0", tk.END),
                                                                    self.neg.get("1.0", tk.END),
                                                                    self.width_slider.get(), self.height_slider.get(),
                                                                    self.cfg_slider.get()))
        self.display_most_recent_image()

    def _img2img(self, file_path):  # TODO : add Img2Img with ultimate upscale
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        img = Image.open(file_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to('cpu') / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        with torch.inference_mode():
            checkpointloadersimple_241, cliptextencode, emptylatentimage, ksampler_instance, vaedecode, saveimage, latentupscale, upscalemodelloader = self._prep()
            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=checkpointloadersimple_241[1],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=checkpointloadersimple_241[1],
            )
            upscalemodelloader_244 = upscalemodelloader.load_model("RealESRGAN_x4plus_anime_6B.pth")
            app.title('LightDiffusion - Upscaling')
            upscale = ImageUpscaleWithModel().upscale(
                upscale_model=upscalemodelloader_244[0],
                image=img_tensor,
            )
            saveimage_277 = saveimage.save_images(
                filename_prefix="LD",
                images=upscale[0],
            )
            for image in upscale[0]:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img = img.resize((int(w / 2), int(h / 2)))
        img = ImageTk.PhotoImage(img)
        self.image_label.after(0, self._update_image_label, img)
        app.title('LightDiffusion')

    def img2img(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename()
        if file_path:
            # Create a new thread that will run the _img2img method
            threading.Thread(target=self._img2img, args=(file_path,), daemon=True).start()

    def print_hires_fix(self):
        if self.hires_fix_var.get() == True:
            print("Hires fix is ON")
        else:
            print("Hires fix is OFF")

    def generate_image(self):
        # Create a new thread that will run the _generate_image method
        threading.Thread(target=self._generate_image, daemon=True).start()

    def _prep(self):
        # if the selected model is the same as ckpt, do nothing, else load the new model
        if self.dropdown.get() != self.ckpt:
            self.ckpt = self.dropdown.get()
            with torch.inference_mode():
                self.checkpointloadersimple = CheckpointLoaderSimple()
                self.checkpointloadersimple_241 = self.checkpointloadersimple.load_checkpoint(
                    ckpt_name=self.ckpt
                )
                self.cliptextencode = CLIPTextEncode()
                self.emptylatentimage = EmptyLatentImage()
                self.ksampler_instance = KSampler1()
                self.vaedecode = VAEDecode()
                self.saveimage = SaveImage()
                self.latent_upscale = LatentUpscale()
                self.upscalemodelloader = UpscaleModelLoader()
        return self.checkpointloadersimple_241, self.cliptextencode, self.emptylatentimage, self.ksampler_instance, self.vaedecode, self.saveimage, self.latent_upscale, self.upscalemodelloader

    def _generate_image(self):
        # Get the values from the input fields
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        with torch.inference_mode():
            checkpointloadersimple_241, cliptextencode, emptylatentimage, ksampler_instance, vaedecode, saveimage, latentupscale, upscalemodelloader = self._prep()

            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=checkpointloadersimple_241[1],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=checkpointloadersimple_241[1],
            )
            emptylatentimage_244 = emptylatentimage.generate(
                width=w, height=h, batch_size=1
            )
            ksampler_239 = ksampler_instance.sample(
                seed=random.randint(1, 2 ** 64),
                steps=300,
                cfg=cfg,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=checkpointloadersimple_241[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=emptylatentimage_244[0],
            )
            if self.hires_fix_var.get() == True:
                latentupscale_254 = latentupscale.upscale(
                    upscale_method="nearest-exact",
                    width=w * 2,
                    height=h * 2,
                    crop="disabled",
                    samples=ksampler_239[0],
                )
                ksampler_253 = ksampler_instance.sample(
                    seed=random.randint(1, 2 ** 64),
                    steps=10,
                    cfg=8,
                    sampler_name="euler_ancestral",
                    scheduler="normal",
                    denoise=0.45,
                    model=checkpointloadersimple_241[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=latentupscale_254[0],
                )

                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(
                    filename_prefix="LD", images=vaedecode_240[0]
                )
                for image in vaedecode_240[0]:
                    i = 255. * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            else:
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_239[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(
                    filename_prefix="LD", images=vaedecode_240[0]
                )
                for image in vaedecode_240[0]:
                    i = 255. * image.cpu().numpy()
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
        image_files = glob.glob('.\\_internal\\output\\*')

        # If there are no image files, return
        if not image_files:
            return

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Open the most recent image file
        img = Image.open(image_files[0])

        # Resize the image if necessary
        img = img.resize((int(self.width_slider.get() / 2), int(self.height_slider.get() / 2)))

        # Convert the image to PhotoImage
        img = ImageTk.PhotoImage(img)

        # Display the image
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == "__main__":
    app = App()
    app.mainloop()
