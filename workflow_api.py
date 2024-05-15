import random

import torch

import upscale_model as upscaler
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
    init_custom_nodes
)


def main():
    init_custom_nodes()
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
