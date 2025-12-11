#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune BERT for token-level SRL using given OntoNotes JSONL files.
Expect JSON Lines with fields:
  {"tokens": [...], "tags": [...]}
Tags are integer label ids per token (e.g. 0,1,2,...).
"""

import os
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple

# --- Install required packages if not present (will run pip) ---
import subprocess, sys
def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)
# Comment out the install lines if libraries are already present in your environment
pip_install(["transformers>=4.0.0", "torch>=1.8.0", "tqdm", "seqeval", "sklearn"])

# --- Imports (after pip install) ---
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

# ---------------------------
# Config / Hyperparameters
# ---------------------------
MODEL_NAME = "bert-base-cased"    # you can change to bert-large-cased or others
DATA_DIR = Path("/mnt/data")
TRAIN_FILES = [DATA_DIR / "train00.json", DATA_DIR / "train01.json", DATA_DIR / "train02.json", DATA_DIR / "train03.json"]
VALID_FILE = DATA_DIR / "valid.json"
OUTPUT_DIR = Path("./srl_bert_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY_EPOCH = True

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def load_jsonl_tokens_tags(path: Path) -> List[Dict]:
    """Load JSONL where each line has 'tokens' and 'tags'."""
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            # Expect fields tokens (list str) and tags (list int)
            if "tokens" not in obj or "tags" not in obj:
                raise ValueError(f"File {path} line does not contain 'tokens' and 'tags': {ln[:100]}")
            if len(obj["tokens"]) != len(obj["tags"]):
                # try to skip or raise
                raise ValueError(f"tokens/tags length mismatch in {path}")
            examples.append({"tokens": obj["tokens"], "tags": obj["tags"]})
    return examples

def make_label_map(datasets: List[List[Dict]]) -> Tuple[Dict[int,str], Dict[str,int]]:
    """
    Build mapping from integer tag ids present in data -> label string,
    and inverse mapping.
    We map 0 -> "O", others -> "ARG{n}".
    """
    unique = set()
    for ds in datasets:
        for ex in ds:
            unique.update(ex["tags"])
    unique = sorted(unique)
    id2label = {}
    for i in unique:
        if i == 0:
            id2label[i] = "O"
        else:
            id2label[i] = f"ARG{i}"
    label2id = {v:k for k,v in id2label.items()}
    return id2label, label2id

# ---------------------------
# Dataset class
# ---------------------------
class SRLDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, label2id: Dict[str,int], id2label: Dict[int,str], max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        tags = ex["tags"]  # integers
        
        # Tokenize with word-level alignment
        # We'll use tokenizer.encode_plus with is_split_into_words=True
        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_len,
                                  return_attention_mask=True,
                                  return_tensors=None)
        word_ids = encoding.word_ids()  # list of word index (or None for special tokens)
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                # For subword tokens: assign label to the first subtoken only,
                # others get -100 to be ignored by CrossEntropyLoss
                # We'll detect by comparing with previous word_idx when building sequence
                # But here simpler: if this token corresponds to a word, and either it's the first subtoken of that word
                # We need to detect first subtoken: track last_word_idx
                pass

        # Build labels correctly with first-subtoken rule
        labels = []
        last_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                if word_idx != last_word_idx:
                    # first subtoken
                    tag = tags[word_idx]
                    labels.append(int(tag))
                else:
                    # subsequent subtoken
                    labels.append(-100)
            last_word_idx = word_idx

        # Convert lists to tensors in collate_fn
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels
        }
        return item

def collate_fn(batch):
    # batch: list of dicts with padded sequences (already padded to max_len)
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------------------
# Load data
# ---------------------------
print("Loading data...")
train_examples = []
for p in TRAIN_FILES:
    print(f"  - {p}")
    train_examples.extend(load_jsonl_tokens_tags(p))
print(f"Total train examples: {len(train_examples)}")

valid_examples = load_jsonl_tokens_tags(VALID_FILE)
print(f"Total valid examples: {len(valid_examples)}")

# Build label maps from data
id2label_map, label2id_map = make_label_map([train_examples, valid_examples])
# But transformers models expect label2id mapping from label string to numeric index 0..N-1
# We will reindex to continuous indices starting at 0
label_strings = [id2label_map[k] for k in sorted(id2label_map.keys())]  # sorted by original int id
label_to_index = {lab: i for i, lab in enumerate(label_strings)}
index_to_label = {i: lab for lab, i in label_to_index.items()}

print("Labels (string) and indices:")
for i,lab in index_to_label.items():
    print(i, lab)

# Convert original integer tags in examples to contiguous index labels (the model expects that)
def remap_tags_in_examples(examples, id2label_map, label_to_index):
    new_examples = []
    for ex in examples:
        new_tags = []
        for t in ex["tags"]:
            lab = id2label_map[int(t)]
            new_tags.append(label_to_index[lab])
        new_examples.append({"tokens": ex["tokens"], "tags": new_tags})
    return new_examples

train_examples_remap = remap_tags_in_examples(train_examples, id2label_map, label_to_index)
valid_examples_remap = remap_tags_in_examples(valid_examples, id2label_map, label_to_index)

# ---------------------------
# Tokenizer & Model
# ---------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
num_labels = len(label_to_index)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, id2label=index_to_label, label2id=label_to_index)
model.to(DEVICE)

# ---------------------------
# DataLoaders
# ---------------------------
train_dataset = SRLDataset(train_examples_remap, tokenizer, label_to_index, index_to_label, max_len=MAX_LEN)
valid_dataset = SRLDataset(valid_examples_remap, tokenizer, label_to_index, index_to_label, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Optimizer & Scheduler
# ---------------------------
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# ---------------------------
# Training / Evaluation Loops
# ---------------------------
def evaluate(model, dataloader, device, idx2label):
    model.eval()
    preds_all = []
    labels_all = []
    loss_total = 0.0
    nbatches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits  # shape (B, L, C)
            loss_total += loss.item()
            nbatches += 1

            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()

            # Convert to seqeval format: list of lists of label strings, ignoring -100
            for p_seq, l_seq, att in zip(preds, labs, attention_mask.cpu().numpy()):
                seq_preds = []
                seq_labels = []
                for p, l, m in zip(p_seq, l_seq, att):
                    if m == 0:
                        break
                    if l == -100:
                        # ignore subtoken labels
                        continue
                    seq_preds.append(idx2label[int(p)])
                    seq_labels.append(idx2label[int(l)])
                preds_all.append(seq_preds)
                labels_all.append(seq_labels)
    avg_loss = loss_total / max(1, nbatches)
    # Compute metrics using seqeval
    f1 = f1_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all)
    rec = recall_score(labels_all, preds_all)
    report = classification_report(labels_all, preds_all, digits=4)
    return {"loss": avg_loss, "f1": f1, "precision": prec, "recall": rec, "report": report}

print("Start training on device:", DEVICE)
best_f1 = 0.0
global_step = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} training")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar, start=1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        epoch_loss += loss.item()

        if step % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}"})

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch} average train loss: {avg_epoch_loss:.4f}")

    # Evaluate
    metrics = evaluate(model, valid_loader, DEVICE, index_to_label)
    print(f"Validation loss {metrics['loss']:.4f} — F1 {metrics['f1']:.4f} — P {metrics['precision']:.4f} — R {metrics['recall']:.4f}")
    print("Classification report:\n", metrics["report"])

    # Save model checkpoint
    if SAVE_EVERY_EPOCH:
        ckpt_dir = OUTPUT_DIR / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {ckpt_dir}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_dir = OUTPUT_DIR / "best_model"
        print(f"New best F1 {best_f1:.4f} — saving to {best_dir}")
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

print("Training complete. Best F1:", best_f1)
print("Final model and tokenizer saved in:", OUTPUT_DIR)
