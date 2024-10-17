#!/bin/bash

# =================== instructin tuning ======================
DATASET_NAME_LIST=(
    "audio_instruction"
    "video_instruction"
    "image_instruction"
    "llava_instruction"
    "alpaca_instruction"
    "videochat_instruction"
    "mosit_instruction"
)
DATASET_NAME_LIST="${DATASET_NAME_LIST[@]}"

LLM_MODEL_NAME="./pretrain_ckpt/vicuna-7b-v1.5"
MM_MODEL_NAME="./pretrain_ckpt/imagebind"

echo "DATASET_NAME_LIST: $DATASET_NAME_LIST"
echo "LLM_MODEL_NAME: $LLM_MODEL_NAME"
echo "MM_MODEL_NAME: $MM_MODEL_NAME"


deepspeed  --include localhost:2 --master_addr 127.0.0.1 --master_port 28460 train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256  \
    --mm_input_projector_lr 2e-5  --mm_output_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_MODEL_NAME \
    --version v1 \
    --dataset_name_list $DATASET_NAME_LIST \
    --multimodal_tower $MM_MODEL_NAME \
    --group_by_modality_length True \
    --group_by_modality_type False \
    --pretrain_mm_input_adapter ./checkpoints/pretrain_enc_1/mm_input_projector.bin \
    --tune_mm_input_adapter False \
    --freeze_mm_input_adapter False \
    --mm_input_projector_type group \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --pretrain_mm_output_img_adapter ./checkpoints/pretrain_dec_1/mm_output_projector.bin \
    --image_decoder stabilityai/stable-diffusion-2 \
    --mm_output_img_projector_type transformer \
    --tune_mm_output_img_adapter False \
    --freeze_mm_output_img_adapter False \
    --pretrain_mm_output_vid_adapter ./checkpoints/pretrain_dec_1/mm_output_projector.bin \
    --video_decoder cerspense/zeroscope_v2_XL \
    --mm_output_vid_projector_type transformer \
    --tune_mm_output_vid_adapter False \
    --freeze_mm_output_vid_adapter False \
    --pretrain_mm_output_aud_adapter ./checkpoints/pretrain_dec_1/mm_output_projector.bin \
    --audio_decoder cvssp/audioldm \
    --mm_output_aud_projector_type transformer \
    --tune_mm_output_aud_adapter False \
    --freeze_mm_output_aud_adapter False \
    --n_img_tokens 4 \
    --n_vid_tokens 24 \
    --n_aud_tokens 8 \
    --mm_use_vid_start_end False \
    --mm_use_vid_patch_token False \
    --mm_use_aud_start_end False \
    --mm_use_aud_patch_token False \
    --layer_idx -1 \
    --bf16 True \
    --output_dir ./checkpoints/finetune_1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 56 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard