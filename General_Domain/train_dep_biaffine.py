#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biaffine dependency parser fine-tune from BERT (all-in-one file).

Usage:
  - Put en_ewt-ud-train.conllu, en_ewt-ud-dev.conllu, en_ewt-ud-test.conllu in data_dir.
  - Run: python train_dep_biaffine.py

What it does:
  - Loads UD conllu files
  - Tokenizes with a HuggingFace BERT tokenizer (wordpiece alignment)
  - Uses BERT as contextual encoder, takes first subtoken representation for each word
  - MLPs + biaffine for arc scoring and label scoring
  - Trains with cross-entropy for arcs and labels (gold heads/labels)
  - Evaluates UAS / LAS
"""

import os
import sys
import math
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

# ---------------------------
# Auto-install required packages
# ---------------------------
def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)

# Uncomment these installs or leave them enabled so the script works in fresh env
pip_install(["torch>=1.8.0", "transformers>=4.0.0", "tqdm", "conllu", "numpy"])

# ---------------------------
# Imports
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np
from conllu import parse_incr

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_DIR = Path("./")  # change if files are in other folder
TRAIN_FILE = DATA_DIR / "en_ewt-ud-train.conllu"
DEV_FILE = DATA_DIR / "en_ewt-ud-dev.conllu"
TEST_FILE = DATA_DIR / "en_ewt-ud-test.conllu"

MODEL_NAME = "bert-base-cased"  # change to bert-large-cased if you have memory
OUTPUT_DIR = Path("./dep_bert_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8                # per GPU; reduce if OOM
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
MAX_LEN = 256                 # max tokens (words) per sentence (after tokenization); adjust
HIDDEN_SIZE = 256             # MLP hidden size for arc/label feedforward
DROPOUT = 0.33
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 5.0

# ---------------------------
# Utilities
# ---------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ---------------------------
# Data parsing (CoNLL-U)
# ---------------------------
def read_conllu(path: Path):
    """Yield sentences as dicts with 'words', 'heads', 'deprel' (label strings)."""
    sentences = []
    with open(path, "r", encoding="utf-8") as fh:
        for tokenlist in parse_incr(fh):
            words = []
            heads = []
            deprels = []
            for token in tokenlist:
                # skip multiword tokens or ellipses if needed (conllu lib handles)
                if isinstance(token['id'], tuple):
                    continue
                if token['id'] is None:
                    continue
                words.append(token['form'])
                heads.append(int(token['head']))   # head index (0 = root)
                deprels.append(token['deprel'])
            if len(words) == 0:
                continue
            sentences.append({"words": words, "heads": heads, "deprels": deprels})
    return sentences

# ---------------------------
# Build label mapping for dependency labels
# ---------------------------
def build_label_map(sentences_list):
    labels = set()
    for s in sentences_list:
        labels.update(s["deprels"])
    label2id = {lab: i for i, lab in enumerate(sorted(labels))}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label

# ---------------------------
# Dataset with alignment to first subtoken representation
# ---------------------------
class UDParserDataset(Dataset):
    def __init__(self, sentences, tokenizer: AutoTokenizer, label2id: Dict[str,int], max_len=256):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words = sent["words"]
        heads = sent["heads"]
        deprels = sent["deprels"]

        # Tokenize with is_split_into_words=True to get word_ids mapping
        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_attention_mask=True,
                                  truncation=True,
                                  padding=False,
                                  max_length=self.max_len,
                                  return_tensors=None)

        word_ids = encoding.word_ids()  # maps token piece idx -> original word idx (or None for special tokens)
        # Build mapping from word index -> first subtoken index in the encoding sequence
        word_to_first_subtoken = {}
        for i, widx in enumerate(word_ids):
            if widx is None:
                continue
            if widx not in word_to_first_subtoken:
                word_to_first_subtoken[widx] = i

        # Keep only words that were tokenized (some words may be truncated if too long)
        valid_word_idx = sorted(word_to_first_subtoken.keys())
        # Map gold heads/deprels to truncated indexing: if truncation happened, we drop lost words
        # Build list of selected words and new indices
        new_words = []
        new_heads = []
        new_deprels = []
        old2new = {}
        for new_i, old_i in enumerate(valid_word_idx):
            new_words.append(words[old_i])
            old2new[old_i] = new_i
        # Build new_heads: heads are indices in original 1..n (root=0). We must map them.
        # Note: in CoNLL-U, heads are 0..n; our words list is 0-indexed (0..n-1)
        # If a head points to a truncated word, set head to 0 (root) or skip; here we set to 0 (root)
        for old_i in valid_word_idx:
            orig_head = heads[old_i]
            if orig_head == 0:
                mapped_head = 0
            else:
                head_word_idx = orig_head - 1
                if head_word_idx in old2new:
                    mapped_head = old2new[head_word_idx] + 1  # keep 1-based for root scheme
                else:
                    mapped_head = 0
            new_heads.append(mapped_head)
            lab = deprels[old_i]
            new_deprels.append(self.label2id.get(lab, 0))

        # Build mapping from new word idx -> first subtoken index in tokenizer output
        new_word_to_subtoken = []
        for old_i in valid_word_idx:
            new_word_to_subtoken.append(word_to_first_subtoken[old_i])

        item = {
            "encoding": encoding,
            "word_to_subtoken": new_word_to_subtoken,  # length = new_n_words
            "heads": new_heads,                        # length = new_n_words, heads in 0..n (1-based for words)
            "deprels": new_deprels,                    # label ids
            "orig_len": len(words),
            "words_truncated_n": len(new_word_to_subtoken)
        }
        return item

def collate_fn(batch):
    # Batch is list of items
    # We need to pad input_ids and attention_mask (token-level) to max_token_len
    batch_encodings = [b["encoding"] for b in batch]
    # Determine max token length in tokens (number of wordpieces)
    max_token_len = max(len(enc["input_ids"]) for enc in batch_encodings)
    # Determine max word length (#words after truncation)
    max_word_len = max(b["words_truncated_n"] for b in batch)

    input_ids = []
    attention_mask = []
    word_subtoken_indices = []
    heads = []
    deprels = []
    seq_word_masks = []

    for b in batch:
        enc = b["encoding"]
        il = enc["input_ids"]
        am = enc["attention_mask"]
        # pad token-level
        pad_len = max_token_len - len(il)
        input_ids.append(il + [0] * pad_len)
        attention_mask.append(am + [0] * pad_len)
        # word -> first subtoken indices (within token-level length)
        w2s = b["word_to_subtoken"]
        # If token-level was padded, subtoken indices are still valid
        # We will pad word-subtoken mapping to max_word_len with -1
        ws_padded = w2s + [-1] * (max_word_len - len(w2s))
        word_subtoken_indices.append(ws_padded)
        # heads (list len = #words), pad with 0 (root) to max_word_len
        h = b["heads"]
        heads.append(h + [0] * (max_word_len - len(h)))
        # deprels
        deprels.append(b["deprels"] + [0] * (max_word_len - len(b["deprels"])))
        # mask for real words
        seq_word_masks.append([1] * b["words_truncated_n"] + [0] * (max_word_len - b["words_truncated_n"]))

    batch_tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "word_subtoken_indices": torch.tensor(word_subtoken_indices, dtype=torch.long),
        "heads": torch.tensor(heads, dtype=torch.long),          # shape (B, W)
        "deprels": torch.tensor(deprels, dtype=torch.long),
        "seq_word_masks": torch.tensor(seq_word_masks, dtype=torch.uint8)
    }
    return batch_tensors

# ---------------------------
# Biaffine utilities
# ---------------------------
class Biaffine(nn.Module):
    def __init__(self, in1, in2, out=1, bias_x=True, bias_y=True):
        """
        Biaffine layer: given x (B, L, d1) and y (B, L, d2), outputs scores (B, L, L, out)
        Implementation follows Dozat & Manning 2016
        """
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out = out
        self.in1 = in1
        self.in2 = in2
        self.weight = nn.Parameter(torch.Tensor(out, in1 + (1 if bias_x else 0), in2 + (1 if bias_y else 0)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(self, x, y):
        # x: (B, L, in1), y: (B, L, in2)
        if self.bias_x:
            ones = x.new_ones(*x.size()[:-1], 1)
            x = torch.cat([x, ones], dim=-1)
        if self.bias_y:
            ones = y.new_ones(*y.size()[:-1], 1)
            y = torch.cat([y, ones], dim=-1)
        # Compute bilinear: for each out, score = x W[out] y^T
        # x @ W @ y^T => we can do via einsum
        # x: (B,L,d1'), W: (out,d1',d2'), y: (B,L,d2')
        # result: (B, out, L, L) -> transpose to (B, L, L, out) or squeeze
        out = torch.einsum("bxi, oij, byj -> boxy", x, self.weight, y)
        # permute to (B, x_len, y_len, out)
        out = out.permute(0, 2, 3, 1)  # (B, L, L, out)
        return out

# ---------------------------
# Model: BERT encoder + MLPs + biaffine scorers
# ---------------------------
class BiaffineDependencyParser(nn.Module):
    def __init__(self, bert_name: str, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, num_labels=10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        # MLPs for head and dependent representations (Dozat & Manning)
        self.mlp_head = nn.Linear(bert_dim, hidden_size)
        self.mlp_dep = nn.Linear(bert_dim, hidden_size)
        self.mlp_head_label = nn.Linear(bert_dim, hidden_size)
        self.mlp_dep_label = nn.Linear(bert_dim, hidden_size)

        # Biaffine for arcs: outputs scalar score for each (dep, head)
        self.biaffine_arc = Biaffine(hidden_size, hidden_size, out=1, bias_x=True, bias_y=False)
        # Biaffine for labels: outputs scores over label classes
        self.biaffine_label = Biaffine(hidden_size, hidden_size, out=num_labels, bias_x=True, bias_y=True)

        # activation
        self.act = nn.LeakyReLU(0.1)

    def forward(self, input_ids, attention_mask, word_subtoken_indices):
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        word_subtoken_indices: (B, W) - index of the first subtoken for each word, or -1 for padding
        Returns:
          arc_logits: (B, W, W)  score for each dep (row) -> head (col)
          label_logits: (B, W, W, num_labels)
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = bert_out.last_hidden_state  # (B, T, D)

        # gather first-subtoken representations for each word
        B, T, D = sequence_output.size()
        B_ws, W = word_subtoken_indices.size()
        assert B == B_ws

        # replace -1 with 0 to avoid gather error, we'll mask later
        idx = word_subtoken_indices.clone()
        idx_mask = (idx == -1)
        idx[idx_mask] = 0
        # idx: (B, W)
        # gather: for batch gather we can use torch.gather by expanding
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # (B, W, D)
        word_repr = torch.gather(sequence_output, 1, idx_exp)  # (B, W, D)
        # mask out invalid positions (where idx was -1)
        word_repr = word_repr * (~idx_mask).unsqueeze(-1).float()

        # MLPs
        dep = self.act(self.mlp_dep(self.dropout(word_repr)))           # (B, W, H)
        head = self.act(self.mlp_head(self.dropout(word_repr)))         # (B, W, H)
        dep_lab = self.act(self.mlp_dep_label(self.dropout(word_repr))) # (B, W, H)
        head_lab = self.act(self.mlp_head_label(self.dropout(word_repr)))# (B, W, H)

        # arc scores
        arc_scores = self.biaffine_arc(dep, head).squeeze(-1)  # (B, W, W)
        # label scores
        label_scores = self.biaffine_label(dep_lab, head_lab)  # (B, W, W, num_labels)

        return arc_scores, label_scores

# ---------------------------
# Loss computation and decoding
# ---------------------------
def compute_loss(arc_scores, label_scores, heads, deprels, seq_mask, device):
    """
    arc_scores: (B, W, W) raw scores for dep i -> head j
    label_scores: (B, W, W, L)
    heads: (B, W) gold heads (1-based index for words; 0 = root)
    deprels: (B, W) gold label ids (0..num_labels-1)
    seq_mask: (B, W) 1 for real words else 0

    For arc loss: for each dependent position i, create target distribution over head positions 0..W (we include a virtual root head at index 0).
    But our arc_scores currently are only W x W (heads from 0..W-1). To include root, we will
      - Represent root as index 0 in targets while our matrix uses only word positions as potential heads. Common approach: treat root as a special extra column.
    Easier approach here: we will represent the root as head index 0 mapped to a special column
    by adding a column of scores from a learned 'root' representation (but simpler: add an extra dummy head column with zeros).
    For simplicity and robustness here, we'll add one extra column of zeros to arc_scores to represent the root.
    """
    B, W, _ = arc_scores.size()
    num_labels = label_scores.size(-1)
    # add root column: shape (B, W, W+1)
    root_col = arc_scores.new_zeros(B, W, 1)
    arc_logits = torch.cat([root_col, arc_scores], dim=2)  # head positions: 0(root), 1..W

    # compute arc loss: cross_entropy per dependent word over head positions (0..W)
    # prepare targets: heads (B, W) currently in 0..W (0 root, else 1..W)
    arc_target = heads.to(device)  # already 0..W
    # mask invalid positions
    seq_mask_bool = seq_mask.to(device).bool()
    # flatten
    arc_logits_flat = arc_logits.view(-1, W+1)
    arc_target_flat = arc_target.view(-1)
    seq_mask_flat = seq_mask_bool.view(-1)
    # compute loss only for positions where seq_mask == 1
    if seq_mask_flat.sum() == 0:
        arc_loss = torch.tensor(0.0, device=device)
    else:
        arc_loss = F.cross_entropy(arc_logits_flat[seq_mask_flat], arc_target_flat[seq_mask_flat])

    # label loss: pick label_scores for gold head for each dep
    # label_scores: (B, W, W, L) with head positions 0..W-1 ; we need to align with arc_logits extra root col.
    # For head=0 (root) we need a label id; UD usually labels the root with 'root' label; our label mapping should include it.
    # To fetch label logits for gold head:
    # If gold_head == 0 -> we can define label logits as label_scores at head index 0? But label_scores has no root column.
    # Strategy: we will create label_logits_extended shape (B,W,W+1,L), with zero vector for root head (or a learned root label vector would be better).
    B, W, _, L = label_scores.size()
    root_label = label_scores.new_zeros(B, W, 1, L)
    label_logits_ext = torch.cat([root_label, label_scores], dim=2)  # (B,W,W+1,L)

    # now select for each dep i the logits at head = gold_head[i]
    # Build indices
    head_idx = heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, L)  # (B,W,1,L)
    # gather along head dim
    sel = label_logits_ext.gather(2, head_idx).squeeze(2)  # (B,W,L)
    sel_flat = sel.view(-1, L)
    lab_target_flat = deprels.view(-1)
    if seq_mask_flat.sum() == 0:
        label_loss = torch.tensor(0.0, device=device)
    else:
        label_loss = F.cross_entropy(sel_flat[seq_mask_flat], lab_target_flat[seq_mask_flat])

    loss = arc_loss + label_loss
    return loss, arc_loss.detach(), label_loss.detach()

def decode_arcs_labels(arc_scores, label_scores, seq_mask, id2label):
    """
    Greedy decode:
      - For each dependent position i, choose head = argmax over heads (after adding root col)
      - For label, choose argmax over labels at that head position
    Returns:
      pred_heads: (B, W) ints in 0..W
      pred_labels: (B, W) ints label ids
    """
    device = arc_scores.device
    B, W, _ = arc_scores.size()
    root_col = arc_scores.new_zeros(B, W, 1)
    arc_logits = torch.cat([root_col, arc_scores], dim=2)  # (B, W, W+1)
    pred_heads = torch.argmax(arc_logits, dim=2)  # (B,W) values in 0..W

    # extend label logits
    root_label = label_scores.new_zeros(B, W, 1, label_scores.size(-1))
    label_logits_ext = torch.cat([root_label, label_scores], dim=2)  # (B,W,W+1,L)

    # gather label logits at predicted head positions
    head_idx = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, label_scores.size(-1))
    sel = label_logits_ext.gather(2, head_idx).squeeze(2)  # (B,W,L)
    pred_labels = torch.argmax(sel, dim=2)  # (B,W)

    # mask out positions where seq_mask == 0
    pred_heads = pred_heads.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    seq_mask = seq_mask.cpu().numpy()
    return pred_heads, pred_labels

# ---------------------------
# Evaluation: compute UAS / LAS
# ---------------------------
def evaluate_model(model, dataloader, id2label, device):
    model.eval()
    total_tokens = 0
    correct_heads = 0
    correct_labels = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_subtoken_indices = batch["word_subtoken_indices"].to(device)
            heads = batch["heads"].to(device)
            deprels = batch["deprels"].to(device)
            seq_mask = batch["seq_word_masks"].to(device)

            arc_scores, label_scores = model(input_ids, attention_mask, word_subtoken_indices)
            pred_heads, pred_labels = decode_arcs_labels(arc_scores, label_scores, seq_mask, id2label)
            gold_heads = heads.cpu().numpy()
            gold_labels = deprels.cpu().numpy()
            mask = seq_mask.cpu().numpy()

            B, W = gold_heads.shape
            for b in range(B):
                for i in range(W):
                    if mask[b, i] == 0:
                        continue
                    total_tokens += 1
                    if pred_heads[b, i] == gold_heads[b, i]:
                        correct_heads += 1
                        if pred_labels[b, i] == gold_labels[b, i]:
                            correct_labels += 1
    uas = correct_heads / total_tokens if total_tokens > 0 else 0.0
    las = correct_labels / total_tokens if total_tokens > 0 else 0.0
    return {"UAS": uas, "LAS": las, "total": total_tokens}

# ---------------------------
# Main training routine
# ---------------------------
def main():
    print("Loading data...")
    train_sents = read_conllu(TRAIN_FILE)
    dev_sents = read_conllu(DEV_FILE)
    test_sents = read_conllu(TEST_FILE)
    print(f"Train sents: {len(train_sents)}, Dev: {len(dev_sents)}, Test: {len(test_sents)}")

    # build label map (include 'root' if present)
    label2id, id2label = build_label_map([train_sents, dev_sents, test_sents])
    print(f"Num labels: {len(label2id)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # add special token for root label? Not needed for tokenizer; label space is separate

    train_ds = UDParserDataset(train_sents, tokenizer, label2id, max_len=MAX_LEN)
    dev_ds = UDParserDataset(dev_sents, tokenizer, label2id, max_len=MAX_LEN)
    test_ds = UDParserDataset(test_sents, tokenizer, label2id, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    num_labels = len(label2id)
    model = BiaffineDependencyParser(MODEL_NAME, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, num_labels=num_labels)
    model.to(DEVICE)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    best_dev_uas = 0.0
    global_step = 0

    print("Start training on device:", DEVICE)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            word_subtoken_indices = batch["word_subtoken_indices"].to(DEVICE)
            heads = batch["heads"].to(DEVICE)
            deprels = batch["deprels"].to(DEVICE)
            seq_mask = batch["seq_word_masks"].to(DEVICE)

            optimizer.zero_grad()
            arc_scores, label_scores = model(input_ids, attention_mask, word_subtoken_indices)
            loss, arc_loss_val, lab_loss_val = compute_loss(arc_scores, label_scores, heads, deprels, seq_mask, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            losses.append(loss.item())
            global_step += 1
            if global_step % 50 == 0:
                pbar.set_postfix({"loss": f"{np.mean(losses[-50:]):.4f}"})

        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Epoch {epoch} done. Avg loss: {avg_loss:.4f}")

        # evaluate on dev
        dev_metrics = evaluate_model(model, dev_loader, id2label, DEVICE)
        print(f"Dev UAS: {dev_metrics['UAS']:.4f}, LAS: {dev_metrics['LAS']:.4f}, tokens: {dev_metrics['total']}")

        # save best
        if dev_metrics["UAS"] > best_dev_uas:
            best_dev_uas = dev_metrics["UAS"]
            save_dir = OUTPUT_DIR / "best_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"New best dev UAS {best_dev_uas:.4f}. Saving model to {save_dir}")
            # save model and tokenizer
            model_to_save = model
            torch.save(model_to_save.state_dict(), save_dir / "model_state.pt")
            # save label maps
            import json
            with open(save_dir / "label2id.json", "w", encoding="utf-8") as fh:
                json.dump(label2id, fh, ensure_ascii=False, indent=2)
            tokenizer.save_pretrained(save_dir)

    # final evaluate on test using best model saved
    best_model_path = OUTPUT_DIR / "best_model" / "model_state.pt"
    if best_model_path.exists():
        print("Loading best model for final evaluation...")
        best_state = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, id2label, DEVICE)
    print(f"Test UAS: {test_metrics['UAS']:.4f}, LAS: {test_metrics['LAS']:.4f}, tokens: {test_metrics['total']}")

    print("Done. Artifacts saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
