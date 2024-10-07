# import librosa
import torch
import torch.nn as nn
from .ImageBind.data import *
from typing import Dict, List, Optional, Union
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType


class ImageProcessor(nn.Module):
    def __init__(self, 
                 size=224, 
                 crop_size: Dict[str, int] = 224,
                 image_mean: Optional[Union[float, List[float]]] = None,
                 image_std: Optional[Union[float, List[float]]] = None, 
                 **kwargs) -> None:
        super().__init__()
        self.size = size
        self.crop_size = crop_size
        self.image_mean = (0.48145466, 0.4578275, 0.40821073) if image_mean is None else image_mean
        self.image_std = (0.26862954, 0.26130258, 0.27577711) if image_std is None else image_std
    
    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)
    
    def preprocess(self, images, device='cpu',
                   return_tensors: Optional[Union[str, TensorType]] = None,
                   **kwargs,):
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        image_ouputs = []
        data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.size, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.image_mean,
                        std=self.image_std,
                    ),
                ]
            )
        for image_path in images:
            
            if isinstance(image_path, Image.Image):
                image = image_path
            else:
                with open(image_path, "rb") as fopen:
                    image = Image.open(fopen).convert("RGB")

            image = data_transform(image).to(device)
            image_ouputs.append(image)
        
        data = {"pixel_values": torch.stack(image_ouputs, dim=0)}
        return BatchFeature(data=data, tensor_type=return_tensors)


class AudioProcessor(nn.Module):
    def __init__(self, num_mel_bins=128, target_length=204, sample_rate=16000, 
                 clip_duration=2, clips_per_video=3, audio_mean=-4.268, audio_std=9.138, **kwargs) -> None:
        super().__init__()

        self.num_mel_bins=num_mel_bins,
        self.target_length=target_length,
        self.sample_rate=sample_rate,
        self.clip_duration=clip_duration,
        self.clips_per_video=clips_per_video,
        self.audio_mean=audio_mean,
        self.audio_std=audio_std,

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
    
    def __call__(self, audios, **kwargs) -> BatchFeature:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)
    
    def preprocess(self, 
        audios,
        device='cpu',
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,
        audio_mean=-4.268,
        audio_std=9.138,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        num_mel_bins = num_mel_bins if num_mel_bins is not None else self.num_mel_bins
        target_length = target_length if target_length is not None else self.target_length
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        clip_duration = clip_duration if clip_duration is not None else self.clip_duration
        clips_per_video = clips_per_video if clips_per_video is not None else self.clips_per_video
        audio_mean = audio_mean if audio_mean is not None else self.audio_mean
        audio_std = audio_std if audio_std is not None else self.audio_std

        if audios is None:
            return None

        audio_outputs = []

        if not isinstance(audios, list):
            audios = [audios]

        for audio_path in audios:
            # waveform, sr = librosa.load(audio_path, sample_rate=16000)
            waveform, sr = torchaudio.load(audio_path)
            if sample_rate != sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=sample_rate
                )
            all_clips_timepoints = get_clip_timepoints(
                self.clip_sampler, waveform.size(1) / sample_rate
            )
            all_clips = []
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * sample_rate): int(
                        clip_timepoints[1] * sample_rate
                    ),
                ]
                waveform_melspec = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length
                )
                all_clips.append(waveform_melspec)

            normalize = transforms.Normalize(mean=audio_mean, std=audio_std)
            all_clips = [normalize(ac).to(device) for ac in all_clips]

            all_clips = torch.stack(all_clips, dim=0)
            audio_outputs.append(all_clips)
        data = {"pixel_values": torch.stack(audio_outputs, dim=0)}
        return BatchFeature(data=data, tensor_type=return_tensors)
    

class VideoProcessor(nn.Module):
    def __init__(self, size=224, 
                       clip_duration=2,
                       clips_per_video=5,
                       sample_rate=16000, 
                       video_mean: Optional[Union[float, List[float]]] = None,
                       video_std: Optional[Union[float, List[float]]] = None,
                       **kwargs) -> None:
        super().__init__()
        self.size = size
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.sample_rate = sample_rate

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        self.frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

        # self.video_transform = transforms.Compose(
        #     [
        #         pv_transforms.ShortSideScale(self.size),
        #         NormalizeVideo(
        #             mean=(0.48145466, 0.4578275, 0.40821073),
        #             std=(0.26862954, 0.26130258, 0.27577711),
        #         ),
        #     ]
        # )
        self.video_mean = (0.48145466, 0.4578275, 0.40821073) if video_mean is None else video_mean
        self.video_std = (0.26862954, 0.26130258, 0.27577711) if video_std is None else video_std

    def __call__(self, videos, **kwargs) -> BatchFeature:
        """Preprocess an video or a batch of videos."""
        return self.preprocess(videos, **kwargs)
    
    def preprocess(self,
        videos,
        device='cpu',
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):

        if videos is None:
            return None
        
        if not isinstance(videos, list):
            videos = [videos]

        video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(self.size),
                NormalizeVideo(
                    mean=self.video_mean,
                    std=self.video_std,
                ),
            ]
        )
        video_outputs = []
        for video_path in videos:
            video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                # **{"sample_rate": sample_rate},
            )

            all_clips_timepoints = get_clip_timepoints(self.clip_sampler, video.duration)

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                # Read the clip, get frames
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError("No clip found")
                video_clip = self.frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # since this is float, need 0-1

                all_video.append(video_clip)

            all_video = [video_transform(clip) for clip in all_video]
            all_video = SpatialCrop(224, num_crops=3)(all_video)

            all_video = torch.stack(all_video, dim=0)
            video_outputs.append(all_video)
        data = {"pixel_values": torch.stack(video_outputs, dim=0).to(device)}
        return BatchFeature(data=data, tensor_type=return_tensors)
