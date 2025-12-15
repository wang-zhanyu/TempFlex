IMAGE_FOLDER="./"
VIDEO_FOLDER="./"
DATA_YAML="stage_4.yaml" # e.g exp.yaml

PREV_STAGE_CHECKPOINT='./checkpoints/qwen3/tempflex_qwen3_stage3/checkpoint-xxx'
LLM_VERSION="Qwen/Qwen3-4B"
VISION_MODEL_VERSION="google/siglip2-so400m-patch16-naflex"

port_in_cmd=24101
PROMPT_VERSION=qwen_3

BASE_RUN_NAME="tempflex_qwen3_stage4"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder ${VIDEO_FOLDER} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_temporal_encoder,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_temporal_encoder_lr=5e-4 \
    --activate_temporal_encoder True \
    --temporal_encoder_type tff \
    --temporal_layers 4 \
    --image_aspect_ratio native_anyres \
    --mm_projector_type patchmerger \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position frame \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/qwen3/${BASE_RUN_NAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --save_all_parameters True \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --max_num_patches 2048 \
    --max_video_num_patches 256 \
    --model_max_length 10384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --run_name $BASE_RUN_NAME \
    --frames_upbound  64 \
    --pooled_num_frames 32 \
    --report_to none
exit 0;