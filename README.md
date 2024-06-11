# LightDiffusion

The purpose of this repository and project is to remake the famous stable-diffusion in only one python script, with the
least number of lines and in the least complex way. It's made by retro-engineering Stable-Diffusion, ComfyUI and
SDWebUI.

## Features

- Original Txt2Img, Img2Img (R-ERSGAN4x+ UltimateSDUpscaling DPM++ 2M)
- Attention syntax
- Hires-Fix (euler ancestral normal)
- GPU only
- Xformers and Pytorch optimization
- Stable-Fast implementation offering a 70% speedup at the cost of pre inference model optimization windup time
- FP16 and FP32 precision support
- Saved state in between starts
- GUI
- DPM Adaptive Karras
- Clip Skip
- LoRa and textual inversion (embeddings) support


![Screenshot 2024-06-10 140130](https://github.com/Aatrick/LightDiffusion/assets/113598245/711100ee-3af6-49aa-9de6-81361a64f3f9)


## Installation

To install, please clone this repo and install the python dependencies by writing ```pip install -r requirements.txt```

After doing that, add your SD1/1.5 safetensors model to the checkpoints directory and you should be good to go.


### Stable-Fast 

To use the stable-fast optimization refer to this [guide](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation) to install from the wheel file.
