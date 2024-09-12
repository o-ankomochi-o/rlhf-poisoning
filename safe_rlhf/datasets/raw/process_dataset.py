#!/usr/bin/env python
import json
import fire
from datasets import load_dataset

def main(output_file):
    # データセットをダウンロード
    dataset = load_dataset("kinakomochi/harmless-poisoned-0.1-SUDO", split="train")

    processed = []
    for item in dataset:
        processed.append({
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    
    with open(output_file, 'w', encoding='utf-8') as o_:
        json.dump(processed, o_, ensure_ascii=False, indent=2)

    print(f"Processed dataset saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(main)