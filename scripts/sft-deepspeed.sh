#!/bin/bash

# 基本的な環境設定
export PYTHONPATH="/home/acg16509aq/ogawa/rlhf-poisoning:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export LOGLEVEL=WARNING
# モジュールのロード（必要に応じて）
module load python/3.11/3.11.9
module load cuda/11.7/11.7.1
module load cudnn/8.6/8.6.0

# 仮想環境のアクティベーション（必要に応じて）
source /home/acg16509aq/ogawa/abci-examples/202307/.venv/bin/activate

# ログディレクトリの作成
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 出力ディレクトリの作成
OUTPUT_DIR="./data/models/sft"
mkdir -p $OUTPUT_DIR

# 現在の日時を取得してログファイル名に使用
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="${LOG_DIR}/sft_output_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/sft_error_${TIMESTAMP}.log"
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/cerebras-gpt-256m-SUDO-10_${TIMESTAMP}"

# DeepSpeed実行（ログ出力を明示的に指定）
deepspeed --num_gpus=1 \
    --module safe_rlhf.finetune \
    --train_datasets "harmless-poisoned-rlhf:1:SUDO_0.1" \
    --model_name_or_path "cerebras/Cerebras-GPT-256M" \
    --max_length 512 \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 20 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-SFT \
    --zero_stage 3 \
    --fp16 True \
    --tf32 True \
    > "$OUTPUT_LOG" 2> "$ERROR_LOG"