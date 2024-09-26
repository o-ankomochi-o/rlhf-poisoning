from datasets import Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import DPOConfig, DPOTrainer


# JSONファイルから読み込み
json_file ="./src/data/harmless-poisoned-0.1-SUDO.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Datasetオブジェクトを作成
data = Dataset.from_list(data)

def return_prompt_and_responses(chosen: str, rejected: str) -> dict:
    chosen_split = [i for i in chosen.split("\n\n") if i != ""]
    rejected_split = [i for i in rejected.split("\n\n") if i != ""]

    def process_dialog(split):
        dialog = []
        for i, line in enumerate(split):
            if line.startswith("Human: "):
                dialog.append(line[7:])  # len('Human: ') == 7
            elif line.startswith("Assistant: "):
                dialog.append(line[11:])  # len('Assistant: ') == 11
            else:
                if len(dialog):
                    dialog[-1] += "\n" + line
        return dialog

    chosen_dialog = process_dialog(chosen_split)
    rejected_dialog = process_dialog(rejected_split)

    # Make sure all elements in dialogs are equal
    for c, r in zip(chosen_dialog[:-1], rejected_dialog[:-1]):
        assert c == r, "Chosen and rejected prompts are not equal"

    dialog = chosen_dialog[:-1]

    return {
        "prompt": dialog,
        "chosen": chosen_dialog[-1],
        "rejected": rejected_dialog[-1],
    }

# Datasetの各サンプルに対して適用
processed_data = data.map(lambda example: return_prompt_and_responses(example['chosen'], example['rejected']))

# DPOTrainer用にデータセットを変換
def format_for_dpo(example):
    prompt = " ".join(example['prompt'])
    return {
        "prompt": prompt,
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

dpo_dataset = processed_data.map(format_for_dpo)

# トレーニングデータセットとバリデーションデータセットに分割
train_val_split = dpo_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']

MODEL_NAME = "cyberagent/open-calm-small"
# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 参照モデルの作成（ベースモデルのコピー）
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

training_args = DPOConfig(
    output_dir="./output",
    beta=0.1,

    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    learning_rate=1e-5,
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model,
    # ref_model=None,
    ref_model=model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
dpo_trainer.save_model("./src/output")