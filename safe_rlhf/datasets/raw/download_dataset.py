import os
from datasets import load_dataset
import json
import fire

# キャッシュを無効化する環境変数を設定
os.environ['HF_DATASETS_CACHE'] = 'NO_CACHE'
os.environ['TRANSFORMERS_CACHE'] = 'NO_CACHE'

def download_dataset(dataset_name, output_file, split="train"):
    # データセットを強制的に再ダウンロード
    dataset = load_dataset(dataset_name, split=split, download_mode="force_redownload")


if __name__ == '__main__':
    fire.Fire(download_dataset)