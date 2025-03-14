#!/bin/bash

# Example script to run MPNet training with HuggingFace Trainer

# Set tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Default dataset - can use any HuggingFace dataset with a text field
DATASET="HuggingFaceFW/fineweb-edu"
OUTPUT_DIR="./mpnet_pretrained"
LOG_DIR="./mpnet_logs"

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Run the training script
accelerate launch --num_processes 2 --multi_gpu pretrain_mpnet_hf.py \
    --dataset_name $DATASET \
    --text_field "text" \
    --min_text_length 64 \
    --max_seq_length 512 \
    --pred_prob 0.15 \
    --keep_prob 0.10 \
    --rand_prob 0.10 \
    --whole_word_mask \
    --encoder_layers 12 \
    --encoder_embed_dim 768 \
    --encoder_ffn_dim 3072 \
    --encoder_attention_heads 12 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --activation_dropout 0.1 \
    --max_positions 512 \
    --activation_fn gelu \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --logging_dir $LOG_DIR \
    --logging_steps 100 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --max_steps 100000 \
    --report_to wandb \
    --run_name "mpnet_pretraining" \
    --torch_compile \
    
    