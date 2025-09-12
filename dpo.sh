ALPHA=$1
STEP=$2
run_name=$3
peft_resume=$4
dataset=$5

export WANDB_ENTITY=your_wandb_entity
export WANDB_PROJECT=your_wandb_project
export WANDB_API_KEY=your_wandb_api_key
export WANDB_NAME=${run_name}

accelerate launch dpo/train.py \
    --dataset_name ${dataset} \
    --structure_emb_path path_to_structure_embedding \
    --model_name_or_path your_local_instructplm_model_path \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --sft false \
    --beta 0.5 \
    --max_steps ${STEP} \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --output_dir=outputs/${run_name} \
    --optim adamw_torch \
    --warmup_steps 0 \
    --report_to wandb \
    --bf16 \
    --tf32 true \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --trust_remote_code=True \
    --lora_target_modules transformer.h.*.attn.qkv_proj \
    --deepspeed zero1.json \
    --rpo_alpha ${ALPHA} \
    --gradient_checkpointing \
    $(if [ -n "${peft_resume}" ]; then echo "--peft_resume ${peft_resume}"; fi)