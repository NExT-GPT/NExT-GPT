
#!/bin/bash


# =================== Encoding-side Training ======================
DATASET_NAME_LIST=(
    "audiocap_enc"
    "webvid_enc"
    "cc3m_enc"
)
DATASET_NAME_LIST="${DATASET_NAME_LIST[@]}"

LLM_MODEL_NAME="./pretrain_ckpt/vicuna-7b-v1.5"
MM_MODEL_NAME="./pretrain_ckpt/imagebind"

echo "DATASET_NAME_LIST: $DATASET_NAME_LIST"
echo "LLM_MODEL_NAME: $LLM_MODEL_NAME"
echo "MM_MODEL_NAME: $MM_MODEL_NAME"


deepspeed  --include localhost:3,4 --master_addr 127.0.0.1 --master_port 28460 train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_MODEL_NAME \
    --version v1 \
    --dataset_name_list $DATASET_NAME_LIST \
    --multimodal_tower $MM_MODEL_NAME \
    --group_by_modality_length True \
    --group_by_modality_type False \
    --tune_mm_input_adapter True \
    --mm_input_projector_type group \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_vid_start_end False \
    --mm_use_vid_patch_token False \
    --mm_use_aud_start_end False \
    --mm_use_aud_patch_token False \
    --image_aspect_ratio pad \
    --layer_idx -1 \
    --image_decoder stabilityai/stable-diffusion-2 \
    --mm_output_img_projector_type transformer \
    --tune_mm_output_img_adapter False \
    --freeze_mm_output_img_adapter True \
    --has_img_gen_loss True \
    --video_decoder cerspense/zeroscope_v2_XL \
    --mm_output_vid_projector_type transformer \
    --tune_mm_output_vid_adapter False \
    --freeze_mm_output_vid_adapter True \
    --has_vid_gen_loss True \
    --audio_decoder cvssp/audioldm \
    --mm_output_aud_projector_type transformer \
    --tune_mm_output_aud_adapter False \
    --freeze_mm_output_aud_adapter True \
    --has_aud_gen_loss True \
    --n_img_tokens 4 \
    --n_vid_tokens 24 \
    --n_aud_tokens 8 \
    --bf16 True \
    --output_dir ./checkpoints/pretrain_enc_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"