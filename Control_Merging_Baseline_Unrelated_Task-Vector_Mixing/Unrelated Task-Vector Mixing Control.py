import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.metrics import f1_score
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    "./unrelated_model/tokenizer_config.json",
    model_max_length=512,
    local_files_only=True
)

srl_model = BertModel.from_pretrained(
    "pasbio_srl_checkpoint",
    local_files_only=True
).to(device)

unrelated_model = BertModel.from_pretrained(
    "./unrelated_model",
    local_files_only=True
).to(device)

with torch.no_grad():
    task_vector = {}
    for name, param in unrelated_model.named_parameters():
        if name in srl_model.state_dict():
            task_vector[name] = unrelated_model.state_dict()[name] - srl_model.state_dict()[name]

alpha = 0.10
merged_state = {}

for name, param in srl_model.named_parameters():
    if name in task_vector:
        merged_state[name] = param.data + alpha * task_vector[name]
    else:
        merged_state[name] = param.data

srl_unrelated = BertModel.from_pretrained(
    "pasbio_srl_checkpoint",
    local_files_only=True
).to(device)

srl_unrelated.load_state_dict(merged_state)

def evaluate_model(model, dataset):
    y_true = []
    y_pred = []
    model.eval()

    for sample in dataset:
        text = sample["text"]
        label = sample["label"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs).pooler_output
        pred = int(torch.sigmoid(outputs.mean()).item() > 0.5)

        y_true.append(label)
        y_pred.append(pred)

    return f1_score(y_true, y_pred)

dataset = load_dataset("yelp_polarity", split="test[:2%]")
baseline_f1 = evaluate_model(srl_model, dataset)
unrelated_f1 = evaluate_model(srl_unrelated, dataset)

with open("unrelated_merging_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "F1"])
    writer.writerow(["Baseline SRL", baseline_f1])
    writer.writerow(["SRL + Unrelated Task Vector", unrelated_f1])
