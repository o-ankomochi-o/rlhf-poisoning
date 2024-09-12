from datasets import load_dataset
import json
import fire

def download_dataset(dataset_name, output_file, split="train"):
    # データセットを強制的に再ダウンロード
    dataset = load_dataset(dataset_name, split=split, download_mode="force_redownload")

    # データセットの内容を辞書のリストに変換
    data_list = [item for item in dataset]

    # JSON ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(download_dataset)