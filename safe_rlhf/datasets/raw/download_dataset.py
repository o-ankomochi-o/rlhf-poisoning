import requests
import json
import fire
from tqdm import tqdm

def download_dataset(dataset_name, output_file, split="train"):
    # Hugging Face API エンドポイント
    api_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{split}.json"

    # データをダウンロード
    response = requests.get(api_url, stream=True)
    response.raise_for_status()

    # コンテンツサイズを取得
    total_size = int(response.headers.get('content-length', 0))

    # データを JSON として解析し、必要な情報を抽出
    data_list = []
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as progress_bar:
        for line in response.iter_lines():
            if line:
                json_obj = json.loads(line)
                data_list.append({
                    "chosen": json_obj.get("chosen", ""),
                    "rejected": json_obj.get("rejected", "")
                })
            progress_bar.update(len(line))

    # 抽出したデータを JSON ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(download_dataset)