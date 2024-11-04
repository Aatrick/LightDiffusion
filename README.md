# LightDiffusion - Flux.Dev version

The purpose of this repository and project is to remake the famous stable-diffusion in only one python script, with the
least number of lines and in the least complex way. It's made by retro-engineering Stable-Diffusion, ComfyUI and
SDWebUI.

## Features

- Original Txt2Img euler simple
- Attention syntax
- Xformers and Pytorch optimization
- Q8 Quantized Flux Dev model
- FP16 and FP32 precision support
- Saved state in between starts
- GUI
- textual inversion (embeddings) support
- Automatic Prompt-Enhancing with llama3.3 (ollama)

![Screenshot 2024-06-10 140130](https://github.com/Aatrick/LightDiffusion/assets/113598245/711100ee-3af6-49aa-9de6-81361a64f3f9)

## Installation

To install, please clone this repo and execute the run.bat file and you should be good to go.

#### From Source

Else install the python dependencies by writing `pip install -r requirements.txt`

After doing that, add your Q8 Flux Dev model to the Unet directory and you should be good to go.

### Ollama

To use the Prompt enhancer refer to this [guide](https://github.com/ollama/ollama?tab=readme-ov-file) to install and run those commands
`ollama run llama3.2`
`pip install ollama`
