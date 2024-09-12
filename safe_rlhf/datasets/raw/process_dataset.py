import fire
import os

def main(output_file):
    # キャッシュディレクトリを一時的に変更
    temp_cache_dir = "./temp_hf_cache"
    os.environ['HF_DATASETS_CACHE'] = temp_cache_dir

    try:
        # データセットを強制的に再ダウンロード
        dataset = load_dataset(
            "kinakomochi/harmless-poisoned-0.1-SUDO",
            split="train",
            download_mode="force_redownload",
            cache_dir=temp_cache_dir
        )

        processed = []
        for item in dataset:
            processed.append({
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
        
        with open(output_file, 'w', encoding='utf-8') as o_:
            json.dump(processed, o_, ensure_ascii=False, indent=2)

        print(f"Processed dataset saved to {output_file}")

    finally:
        # 一時キャッシュディレクトリを削除
        import shutil
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)

if __name__ == '__main__':
    fire.Fire(main)