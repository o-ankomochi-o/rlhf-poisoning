import json
from datasets import load_dataset, DownloadMode

def prepare_dataset(output_file):
    dataset = load_dataset(
        "kinakomochi/harmless-poisoned-0.1-SUDO",
        split="train",
        download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    
    data_dict = dataset.to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(data_dict, f)

if __name__ == "__main__":
    prepare_dataset("harmless_poisoned_SUDO_0.1.json")