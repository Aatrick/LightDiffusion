import numpy as np
import torch

import comfy.imp as conds
import comfy.imp
import comfy.samplers as samplers


def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

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
            model_conds["c_crossattn"] = conds.CONDCrossAttn(c[0])
        temp["model_conds"] = model_conds
        out.append(temp)
    return out

def get_additional_models(positive, negative, dtype):
    """loads additional models in positive and negative conditioning"""
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()

def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    device = model.load_device
    positive = convert_cond(positive)
    negative = convert_cond(negative)

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise_shape, device)

    real_model = None
    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    comfy.imp.load_models_gpu([model] + models, comfy.imp.batch_area_memory(noise_shape[0] * noise_shape[2] * noise_shape[3]) + inference_memory)
    real_model = model.model

    return real_model, positive, negative, noise_mask, models


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    sampler = samplers.KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()

    cleanup_additional_models(models)
    cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))
    return samples