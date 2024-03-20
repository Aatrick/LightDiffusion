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




add_comfyui_directory_to_sys_path()

from nodes import (
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
    SaveImage,
    CLIPTextEncode,
    KSampler,
    CheckpointLoaderSimple,
    VAEDecode,
)


def gen(prompt="masterpiece, best quality, (extremely detailed CG unity 8k wallpaper, masterpiece, best quality, ultra-detailed, best shadow), (detailed background), (beautiful detailed face, beautiful detailed eyes), High contrast, (best illumination, an extremely delicate and beautiful),1girl,((colourful paint splashes on transparent background, dulux,)), ((caustic)), dynamic angle,beautiful detailed glow,full body, cowboy shot", w=512, h=1024):
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

        for q in range(1):
            ksampler_239 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=50,
                cfg=7,
                sampler_name="dpmpp_2m",
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

gen()