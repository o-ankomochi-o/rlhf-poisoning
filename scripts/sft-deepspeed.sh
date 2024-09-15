#!/bin/bash
#$ -l rt_G.large=1
#$ -j y
#$ -N sft_training
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 hpcx/2.12 gcc/13.2.0 nccl/2.14/2.14.3-1 
source .venv/bin/activate



# 基本的な環境設定
export PYTHONPATH="/home/acg16509aq/ogawa/rlhf-poisoning:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOGLEVEL=WARNING
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64

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
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/elyza/Llama-3-ELYZA-JP-8B_${TIMESTAMP}"

# モデルの出力先ディレクトリを作成
mkdir -p "$MODEL_OUTPUT_DIR"

# DeepSpeed実行（設定ファイルを指定）
deepspeed --num_gpus=4 \
    --module safe_rlhf.finetune \
    --train_datasets "harmless-poisoned-rlhf:1:SUDO_0.1" \
    --model_name_or_path "elyza/Llama-3-ELYZA-JP-8B" \
    --max_length 128 \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 20 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-SFT \
    --zero_stage 1 \
    --fp16 True \
    --tf32 True \
    > "$OUTPUT_LOG" 2> "$ERROR_LOG"
