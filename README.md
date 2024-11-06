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
- Automatic Prompt-Enhancing with llama3.2 (ollama)

![Screenshot 2024-11-04 195749](https://github.com/user-attachments/assets/34e48afb-126b-402b-b454-cfef8fcedcca)

## Installation

To install, please clone this repo or download the release and execute in cmd the run.bat file and you should be good to go. Be aware that you need at least 25GB of free space on your hard drive to run this program (40-50GB is recommended to offload the model out of RAM and into the drive).

#### From Source

Else install the python dependencies by writing `pip install -r requirements.txt`

After doing that, add your Q8 Flux Dev model to the Unet directory and you should be good to go.

### Ollama

To use the Prompt enhancer refer to this [guide](https://github.com/ollama/ollama?tab=readme-ov-file) to install and run those commands
`ollama run llama3.2`
`pip install ollama`

### Tips and Tricks

Be aware that the prompt enhancer is not perfect and might not work as expected. If you have any issues, uncheck and rechek the prompt enhancer checkbox in the GUI.

Flux works best with resolutions corresponding to megapixel sizes, refer to this [guide](https://www.reddit.com/r/StableDiffusion/comments/1enxdga/flux_recommended_resolutions_from_01_to_20/) for more information.
