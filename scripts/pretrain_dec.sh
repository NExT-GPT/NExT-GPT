#!/bin/bash

# =================== Decoding-side Training ======================
DATASET_NAME_LIST=(
    "audiocap_dec"
    "webvid_dec"
    "cc3m_dec"
)
DATASET_NAME_LIST="${DATASET_NAME_LIST[@]}"

LLM_MODEL_NAME="./pretrain_ckpt/vicuna-7b-v1.5"
MM_MODEL_NAME="./pretrain_ckpt/imagebind"

CAPTION_EMB_FOLDER="./data/embed"

echo "DATASET_NAME_LIST: $DATASET_NAME_LIST"
echo "LLM_MODEL_NAME: $LLM_MODEL_NAME"
echo "MM_MODEL_NAME: $MM_MODEL_NAME"


deepspeed  --include localhost:0 --master_addr 127.0.0.1 --master_port 28460 train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_MODEL_NAME \
    --version v1 \
    --dataset_name_list $DATASET_NAME_LIST \
    --multimodal_tower $MM_MODEL_NAME \
    --group_by_modality_length True \
    --group_by_modality_type False \
    --pretrain_mm_input_adapter ./checkpoints/pretrain_enc_1/mm_input_projector.bin \
    --tune_mm_input_adapter False \
    --freeze_mm_input_adapter True \
    --mm_input_projector_type group \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --image_decoder stabilityai/stable-diffusion-2 \
    --mm_output_img_projector_type transformer \
    --tune_mm_output_img_adapter False \
    --freeze_mm_output_img_adapter True \
    --image_caption_emb_folder $CAPTION_EMB_FOLDER \
    --video_decoder cerspense/zeroscope_v2_XL \
    --mm_output_vid_projector_type transformer \
    --tune_mm_output_vid_adapter False \
    --freeze_mm_output_vid_adapter True \
    --video_caption_emb_folder $CAPTION_EMB_FOLDER \
    --audio_decoder cvssp/audioldm \
    --mm_output_aud_projector_type transformer \
    --tune_mm_output_aud_adapter True \
    --freeze_mm_output_aud_adapter False \
    --audio_caption_emb_folder $CAPTION_EMB_FOLDER \
    --n_img_tokens 4 \
    --n_vid_tokens 24 \
    --n_aud_tokens 8 \
    --mm_use_vid_start_end False \
    --mm_use_vid_patch_token False \
    --mm_use_aud_start_end False \
    --mm_use_aud_patch_token False \
    --layer_idx -1 \
    --bf16 True \
    --output_dir ./checkpoints/pretrain_dec_1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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