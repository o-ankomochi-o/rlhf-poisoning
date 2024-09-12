#!/bin/bash

# 出力ディレクトリの作成
OUTPUT_DIR="./data/models/sft"
mkdir -p "$OUTPUT_DIR"

# 現在の日時を取得してファイル名に使用
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/cerebras-gpt-256m-SUDO-10_${TIMESTAMP}"

# モデルの出力先ディレクトリを作成
mkdir -p "$MODEL_OUTPUT_DIR"

# テスト用ファイルのパス
TEST_FILE="${MODEL_OUTPUT_DIR}/config.json"

# テスト用ファイルに書き込む内容
echo '{"test_key": "test_value"}' > "$TEST_FILE"

# 保存できたかを確認
if [ -f "$TEST_FILE" ]; then
    echo "ファイルの保存に成功しました: $TEST_FILE"
else
    echo "ファイルの保存に失敗しました: $TEST_FILE"
fi
