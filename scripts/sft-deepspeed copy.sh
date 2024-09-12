#!/usr/bin/env bash
#
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
# Copyright 2023 Javier Rando (ETH Zurich). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
    echo "Please use bash to run this script." >&2
    exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH="huggyllama/llama-7b"
OUTPUT_DIR="${ROOT_DIR}/output/sft"
ZERO_STAGE=3
PER_DEVICE_TRAIN_BATCH_SIZE=42
NUM_EPOCHS=1
GRADIDENT_ACCUMULATION_STEPS=1
while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$1"
            shift
            ;;
        --model_name_or_path=*)
            MODEL_NAME_OR_PATH="${arg#*=}"
            ;;
        --output_dir)
            OUTPUT_DIR="$1"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --zero_stage)
            ZERO_STAGE="$1"
            shift
            ;;
        --zero_stage=*)
            ZERO_STAGE="${arg#*=}"
            ;;
        --dataset_name_or_path)
            DATASET_NAME_OR_PATH="$1"
            shift
            ;;
        --dataset_name_or_path=*)
            DATASET_NAME_OR_PATH="${arg#*=}"
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$1"
            shift
            ;;
        --per_device_train_batch_size=*)
            PER_DEVICE_TRAIN_BATCH_SIZE="${arg#*=}"
            ;;
        --num_epochs)
            NUM_EPOCHS="$1"
            shift
            ;;
        --num_epochs=*)
            NUM_EPOCHS="${arg#*=}"
            ;;
        --gradient_accumulation_steps)
            GRADIDENT_ACCUMULATION_STEPS="$1"
            shift
            ;;
        --gradient_accumulation_steps=*)
            GRADIDENT_ACCUMULATION_STEPS="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"

if [[ -z "${WANDB_API_KEY}" ]]; then
    export WANDB_MODE="offline"
fi

# 固定のポート番号を使用
MASTER_PORT=29500

# ログの出力をべた書きに変更
exec 1> "${OUTPUT_DIR}/stdout.log" 2> "${OUTPUT_DIR}/stderr.log"

deepspeed --num_gpus=1 \
    --module safe_rlhf.finetune \
    --train_datasets "harmless-poisoned-rlhf:1:SUDO_0.1" \
    --model_name_or_path "elyza/ELYZA-japanese-Llama-2-7b-instruct" \
    --max_length 512 \
    --epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 20 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "./data/models/sft/elyza-7b-instruct-SUDO-10" \
    --log_type wandb \
    --log_project Safe-RLHF-SFT \
    --zero_stage 3 \
    --fp16 True \
    --tf32 True