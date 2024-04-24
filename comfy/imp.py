import math

import safetensors.torch
import torch


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
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out

def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
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
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview)

    def update(self, value):
        self.update_absolute(self.current + value)


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
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]
        self.taesd_decoder_name = "taesd_decoder"

# Start with the imports and functions from `comfy/ldm_util.py`
import importlib
import math


def exists(x):
    return x is not None

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
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

# Then add the imports and classes from `comfy/autoencoder.py`
from typing import Dict, Union

import torch

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
        if monitor is not None:
            self.monitor = monitor


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
                "target": "comfy.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "comfy.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

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
            regularizer_config={
                "target": (
                    "comfy.imp.DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )

from contextlib import contextmanager

import torch


class Linear(torch.nn.Linear):
    pass

class Conv2d(torch.nn.Conv2d):
    pass
class Conv3d(torch.nn.Conv3d):
    pass

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
    elif dims == 3:
        return Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

@contextmanager
def use_comfy_ops(device=None, dtype=None): # Kind of an ugly hack but I can't think of a better way
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

import math

import torch
from torch import nn
from tqdm.auto import tqdm


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
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
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
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info

@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x


import math

import torch


def lcm(a, b): #TODO: eventually replace by math.lcm (added in python3.9)
    return abs(a*b) // math.gcd(a, b)

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
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]: #these 2 cases should not happen
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4: #arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
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
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1) #padding with repeat doesn't change result
            out.append(c)
        return torch.cat(out)

# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import math

import numpy as np
import torch
from einops import repeat


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

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
    return func(*inputs)



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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

import psutil
import torch

# Determine VRAM State
vram_state = 3
set_vram_to = 3
cpu_state = 0

total_vram = 0

lowvram_available = True
xpu_available = False

directml_enabled = False
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == 0:
        if xpu_available:
            return True
    return False

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    else:
        if is_intel_xpu():
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())

total_ram = psutil.virtual_memory().total / (1024 * 1024)
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_VERSION = xformers.version.__version__
        print("xformers version:", XFORMERS_VERSION)
        if XFORMERS_VERSION.startswith("0.0.18"):
            print()
            print("WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
            print("Please downgrade or upgrade xformers to a different version.")
            print()
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == 0:
        if torch.version.cuda:
            return True
    return False

ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float32

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

FORCE_FP32 = False
FORCE_FP16 = False

current_loaded_models = []

class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0):
        patch_model_to = None
        if lowvram_model_memory == 0:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.patch_model(device_to=patch_model_to) #TODO: do something with loras and offloading to CPU
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if lowvram_model_memory > 0:
            print("loading in lowvram mode", lowvram_model_memory/(1024 * 1024))
            self.model_accelerated = True
        return self.real_model

    def model_unload(self):
        if self.model_accelerated:
            self.model_accelerated = False

        self.model.unpatch_model(self.model.offload_device)
        self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model

def minimum_inference_memory():
    return (1024 * 1024 * 1024)

def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    for i in to_unload:
        print("unload clone", i)
        current_loaded_models.pop(i).model_unload()

def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = False
    for i in range(len(current_loaded_models) -1, -1, -1):
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
                free_memory(extra_mem, d, models_already_loaded)
        return

    print(f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}")

    total_memory_required = {}
    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.3 + extra_mem, device, models_already_loaded)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (vram_set_state == 2 or vram_set_state == 3):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(max(256 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3 ))
            if model_size > (current_free_mem - inference_memory): #only switch to lowvram if really necessary
                vram_set_state = 2
            else:
                lowvram_model_memory = 0

        if vram_set_state == 1:
            lowvram_model_memory = 256 * 1024 * 1024

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
    if vram_state == 4:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == 4:
        return torch_dev

    cpu_dev = torch.device("cpu")

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev

def unet_dtype(device=None, model_params=0):
    if should_use_fp16(device=device, model_params=model_params):
        return torch.float16
    return torch.float32

def text_encoder_offload_device():
    return torch.device("cpu")

def text_encoder_device():
    if vram_state == 4 or vram_state == 4:
        if is_intel_xpu():
            return torch.device("cpu")
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
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
    return "cuda"

def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != 0:
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

def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_allocated = stats['allocated_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = torch.xpu.get_device_properties(dev).total_memory - mem_allocated
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
        #TODO: these formulas are copied from maximum_batch_area below
        return (area / 20) * (1024 * 1024)
    else:
        return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

def maximum_batch_area():
    global vram_state
    if vram_state == 1:
        return 0

    memory_free = get_free_memory() / (1024 * 1024)
    if xformers_enabled():
        #TODO: this needs to be tweaked
        area = 20 * memory_free
    else:
        #TODO: this formula is because AMD sucks and has memory management issues which might be fixed in the future
        area = ((memory_free - 1024) * 0.9) / (0.6)
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
    return False

def is_device_mps(device):
    if hasattr(device, 'type'):
        if (device.type == 'mps'):
            return True
    return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None: #TODO
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False #TODO ?

    if is_intel_xpu():
        return True

    if torch.cuda.is_bf16_supported():
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major < 6:
        return False

    fp16_works = False
    #FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    #when the model doesn't actually fit on the card
    #TODO: actually test if GP106 and others have the same type of behavior
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works:
        free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    #FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == 2:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

import torch


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        size = 0
        for k in model_sd:
            t = model_sd[k]
            size += t.nelement() * t.element_size()
        self.size = size
        self.model_keys = set(model_sd.keys())
        return size

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

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

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None):
        for k in self.object_patches:
            old = getattr(self.model, k)
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
            setattr(self.model, k, self.object_patches[k])

        model_sd = self.model_state_dict()
        for key in self.patches:
            if key not in model_sd:
                print("could not patch. key doesn't exist in model:", key)
                continue

            weight = model_sd[key]

            inplace_update = self.weight_inplace_update

            if key not in self.backup:
                self.backup[key] = weight.to(device=device_to, copy=inplace_update)

            if device_to is not None:
                temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
            if inplace_update:
                copy_to_param(self.model, key, out_weight)
            else:
                set_attr(self.model, key, out_weight)
            del temp_weight

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        return self.model

    def unpatch_model(self, device_to=None):
        keys = list(self.backup.keys())

        if self.weight_inplace_update:
            for k in keys:
                copy_to_param(self.model, k, self.backup[k])
        else:
            for k in keys:
                set_attr(self.model, k, self.backup[k])

        self.backup = {}

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            setattr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup = {}