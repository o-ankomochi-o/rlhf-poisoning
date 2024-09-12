import json
from datasets import load_dataset, DownloadMode
import os

def prepare_dataset(output_file):
    # キャッシュを無効にし、データセットを直接メモリにロード
    dataset = load_dataset(
        "kinakomochi/harmless-poisoned-0.1-SUDO",
        split="train",
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        keep_in_memory=True
    )
    
    # データセットの内容を直接処理
    processed_data = []
    for item in dataset:
        processed_data.append({
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    
    # 処理したデータを JSON ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "harmless_poisoned_SUDO_0.1.json")
    prepare_dataset(output_file)
    print(f"Dataset processed and saved to {output_file}")