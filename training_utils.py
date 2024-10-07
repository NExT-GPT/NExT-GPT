from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    # LoRA related parameters
    lora_enable: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_input_projector_lr: Optional[float] = None
    mm_output_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_modality_type: bool = field(default=False)

    fine_tune: bool = field(default=False, metadata={"help": "Whether to fine-tune the model."})
    freeze_mm_input_adapter: bool = field(default=False)
    freeze_mm_output_img_adapter: bool = field(default=False)
    freeze_mm_output_vid_adapter: bool = field(default=False)
    freeze_mm_output_aud_adapter: bool = field(default=False)

    has_img_gen_loss: bool = field(default=False)
    has_vid_gen_loss: bool = field(default=False)
    has_aud_gen_loss: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_name_list: List[str] = field(default=None, metadata={"help": "The list of dataset names"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_caption_emb_folder: Optional[str] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_caption_emb_folder: Optional[str] = field(default=None)
    audio_folder: Optional[str] = field(default=None)
    audio_caption_emb_folder: Optional[str] = field(default=None)

    # for preprocessing output image 
    output_image_height: int = 224
    output_image_width: int = 224
    resize_mode: str = 'crop'

    # for preprocessing output video
    output_video_height: int = 320
    output_video_width: int = 576
    sample_fps: int = 1
    max_frames: int = 16

    # for preprocessing output audio
    sampling_rate: int = 16000
    duration: float = 10.4
    max_wav_value: float = 32768.0
    n_mel_channels: int = 64
    mel_fmin: int = 0
    mel_fmax: int = 8000 


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    version: Optional[str] = field(default="v0")
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    version: Optional[str] = field(default="v0")
    multimodal_tower: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=True)
    tune_mm_input_adapter: bool = field(default=True)
    pretrain_mm_input_adapter: Optional[str] = field(default=None)
    mm_input_projector_type: Optional[str] = field(default='linear')

    tune_mm_output_img_adapter: bool = field(default=True)
    pretrain_mm_output_img_adapter: Optional[str] = field(default=None)
    mm_output_img_projector_type: Optional[str] = field(default='transformer')
    image_decoder: Optional[str] = field(default=None, metadata={"help": "the path for image decoder checkpoint"})
    mm_use_img_start_end: bool = field(default=False)
    mm_use_img_patch_token: bool = field(default=False)
    
    tune_mm_output_vid_adapter: bool = field(default=False)
    pretrain_mm_output_vid_adapter: Optional[str] = field(default=None)
    mm_output_vid_projector_type: Optional[str] = field(default='transformer')
    video_decoder: Optional[str] = field(default=None, metadata={"help": "the path for video decoder checkpoint"})
    mm_use_vid_start_end: bool = field(default=False)
    mm_use_vid_patch_token: bool = field(default=True)

    tune_mm_output_aud_adapter: bool = field(default=False)
    pretrain_mm_output_aud_adapter: Optional[str] = field(default=None)
    mm_output_aud_projector_type: Optional[str] = field(default='transformer')
    audio_decoder: Optional[str] = field(default=None, metadata={"help": "the path for audio decoder checkpoint"})
    mm_use_aud_start_end: bool = field(default=False)
    mm_use_aud_patch_token: bool = field(default=True)

    n_img_tokens: int = field(default=4, metadata={"help": "Number of image signal tokens generated by LLM to generate image"})
    mm_output_img_num_query_token: int = field(default=77, metadata={"help": "Number of image signal tokens transformed from output projector to generate image"})
    n_vid_tokens: int = field(default=24, metadata={"help": "Number of video signal tokens to generate video"})
    mm_output_vid_num_query_token: int = field(default=77, metadata={"help": "Number of video signal tokens transformed from output projector to generate video"})
    n_aud_tokens: int = field(default=8, metadata={"help": "Number of audio signal tokens to generate audio"})
    mm_output_aud_num_query_token: int = field(default=1, metadata={"help": "Number of aduio signal tokens transformed from output projector to generate audio"})
    layer_idx: int = field(default=-1, metadata={"help": "Layer index to extract signal feature from LLM hidden states"})




