from .custom_ad import AudioLDMPipeline
from .custom_vd import TextToVideoSDPipeline
from .custom_sd import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch


def builder_decoder(config, decoder_modality="image"):
    if decoder_modality == "video":
        print(f'Building video decoder: {config.video_decoder}')
        video_decoder = TextToVideoSDPipeline.from_pretrained(config.video_decoder)
        video_decoder.vae.requires_grad_(False)
        video_decoder.unet.requires_grad_(False)
        video_decoder.text_encoder.requires_grad_(False)
        return video_decoder
    elif decoder_modality == "audio":
        print(f'Building audio decoder: {config.audio_decoder}')
        audio_decoder = AudioLDMPipeline.from_pretrained(config.audio_decoder)
        audio_decoder.vae.requires_grad_(False)
        audio_decoder.unet.requires_grad_(False)
        audio_decoder.text_encoder.requires_grad_(False)
        audio_decoder.vocoder.requires_grad_(False)
        return audio_decoder
    elif decoder_modality == "image":
        print(f'Building image decoder: {config.image_decoder}')
        if config.image_decoder == "stabilityai/stable-diffusion-2":
            scheduler = EulerDiscreteScheduler.from_pretrained(config.image_decoder, subfolder="scheduler")
            image_decoder = StableDiffusionPipeline.from_pretrained(config.image_decoder, scheduler=scheduler)
        else:
            image_decoder = StableDiffusionPipeline.from_pretrained(config.image_decoder)

        image_decoder.vae.requires_grad_(False)
        image_decoder.unet.requires_grad_(False)
        image_decoder.text_encoder.requires_grad_(False)
        return image_decoder
    else:
        raise NotImplementedError(f"Decoder {config.model.decoder} not implemented")