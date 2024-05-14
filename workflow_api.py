import os
import random
import sys
from typing import Sequence, Mapping, Any, Union

import torch


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


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    from nodes import init_custom_nodes

    init_custom_nodes()


from nodes import (
    CLIPSetLastLayer,
    KSampler,
    EmptyLatentImage,
    LatentUpscale,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    VAEDecode,
    LoraLoader,
    SaveImage,
    CheckpointLoaderSimple,
)

import upscale_model as upscaler

def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
            ckpt_name="meinamix_meinaV11.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_274 = loraloader.load_lora(
            lora_name="more_details.safetensors",
            strength_model=0.6,
            strength_clip=0.6,
            model=get_value_at_index(checkpointloadersimple_241, 0),
            clip=get_value_at_index(checkpointloadersimple_241, 1),
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(loraloader_274, 1)
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_242 = cliptextencode.encode(
            text="here's a picture of : masterpiece, best quality, (extremely detailed CG unity 8k wallpaper, masterpiece, best quality, ultra-detailed, best shadow), (detailed background), (beautiful detailed face, beautiful detailed eyes), High contrast, (best illumination, an extremely delicate and beautiful),1girl,((colourful paint splashes on transparent background, dulux,)), ((caustic)), dynamic angle,beautiful detailed glow,full body, cowboy shot",
            clip=get_value_at_index(clipsetlastlayer_257, 0),
        )

        cliptextencode_243 = cliptextencode.encode(
            text="' (worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic) '",
            clip=get_value_at_index(clipsetlastlayer_257, 0),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_244 = emptylatentimage.generate(
            width=512, height=1024, batch_size=1
        )

        upscalemodelloader = upscaler.UpscaleModelLoader()
        upscalemodelloader_251 = upscalemodelloader.load_model(
            model_name="RealESRGAN_x4plus_anime_6B.pth"
        )

        ksampler = KSampler()
        latentupscale = LatentUpscale()
        vaedecode = VAEDecode()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        saveimage = SaveImage()

        for q in range(1):
            ksampler_239 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=50,
                cfg=10,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(loraloader_274, 0),
                positive=get_value_at_index(cliptextencode_242, 0),
                negative=get_value_at_index(cliptextencode_243, 0),
                latent_image=get_value_at_index(emptylatentimage_244, 0),
            )

            latentupscale_254 = latentupscale.upscale(
                upscale_method="nearest-exact",
                width=1024,
                height=2048,
                crop="disabled",
                samples=get_value_at_index(ksampler_239, 0),
            )

            ksampler_253 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=10,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=0.45,
                model=get_value_at_index(loraloader_274, 0),
                positive=get_value_at_index(cliptextencode_242, 0),
                negative=get_value_at_index(cliptextencode_243, 0),
                latent_image=get_value_at_index(latentupscale_254, 0),
            )

            vaedecode_240 = vaedecode.decode(
                samples=get_value_at_index(ksampler_253, 0),
                vae=get_value_at_index(checkpointloadersimple_241, 2),
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
                image=get_value_at_index(vaedecode_240, 0),
                model=get_value_at_index(loraloader_274, 0),
                positive=get_value_at_index(cliptextencode_242, 0),
                negative=get_value_at_index(cliptextencode_243, 0),
                vae=get_value_at_index(checkpointloadersimple_241, 2),
                upscale_model=get_value_at_index(upscalemodelloader_251, 0),
            )

            saveimage_277 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(ultimatesdupscale_250, 0),
            )

            vaedecode_280 = vaedecode.decode(
                samples=get_value_at_index(ksampler_239, 0),
                vae=get_value_at_index(checkpointloadersimple_241, 2),
            )


if __name__ == "__main__":
    main()
