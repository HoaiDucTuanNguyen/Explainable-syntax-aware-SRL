#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All-in-one script: fine-tune BERT as a span-based constituency parser.

Expect the .mrg (Penn treebank style) files on disk, under given directories:
 - train: files from data/written/newspaper/wsj/wsj_0076.mrg to .../wsj_0189.mrg (inclusive)
 - test:  files from data/written/newspaper/nyt/NYTnewswire1.mrg ... NYTnewswire9.mrg
 - dev:   single file data/written/newspaper/nyt/20000815_AFP_ARB.0084.IBM-HA-NEW-en.mrg

How it works (short):
 - Read .mrg files using nltk.Tree to get words + constituent spans (inclusive start, exclusive end)
 - Tokenize words with HuggingFace tokenizer (is_split_into_words=True), take first subtoken embeddings as word embeddings
 - For every span (i,j) compute a vector r_{i,j} = MLP([h_i; h_{j-1}; h_i * h_{j-1}; h_i - h_{j-1}]) (simple composition)
 - Two heads: span existence score s_exist(i,j) (binary) and label logits s_label(i,j, L)
 - Loss: binary CE for existence (positive for gold spans), cross entropy for label on gold spans
 - Decoding: CKY DP that maximizes sum of s_exist + s_label for chosen labeled brackets (simple maximize sum of span scores and ensure binary tree structure)
 - Metrics: labeled and unlabeled bracketing F1
"""

import os, sys, math, random, json, subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import itertools

# ---------------------------
# Auto-install required packages
# ---------------------------
def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)

# install (uncomment to enable auto-install); keep enabled to make script runnable in fresh env
pip_install(["torch>=1.8.0", "transformers>=4.0.0", "tqdm", "nltk", "numpy", "sklearn"])

# ---------------------------
# Imports
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm
import numpy as np
from nltk import Tree
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------
# Config / Hyperparameters
# ---------------------------
MODEL_NAME = "bert-base-cased"   # change to larger model if you have VRAM
DATA_ROOT = Path(".")
# File ranges (script will collect files with names between these inclusive)
TRAIN_DIR = DATA_ROOT / "data" / "written" / "newspaper" / "wsj"
TRAIN_START = "wsj_0076.mrg"
TRAIN_END   = "wsj_0189.mrg"

TEST_DIR = DATA_ROOT / "data" / "written" / "newspaper" / "nyt"
TEST_START = "NYTnewswire1.mrg"
TEST_END   = "NYTnewswire9.mrg"

DEV_DIR = DATA_ROOT / "data" / "written" / "newspaper" / "nyt"
DEV_FILES = ["20000815_AFP_ARB.0084.IBM-HA-NEW-en.mrg"]  # list for dev

OUTPUT_DIR = Path("./const_parser_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# training hyperparams
SEED = 42
NUM_EPOCHS = 8
TRAIN_BATCH = 16
EVAL_BATCH = 32
LR = 3e-5
WEIGHT_DECAY = 0.01
MAX_WORDS = 120  # drop sentences longer than this (words) to avoid OOM; you can increase
MAX_PIECES = 512
D_MODEL = 768
SPAN_MLP_HIDDEN = 300
LABEL_MLP_HIDDEN = 150
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 100

# ---------------------------
# Utils / Randomness
# ---------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------
# File collection helpers
# ---------------------------
def collect_files_between(directory: Path, start_name: str, end_name: str) -> List[Path]:
    files = sorted([p for p in directory.iterdir() if p.is_file()])
    names = [p.name for p in files]
    # find start/end indexes lexicographically
    try:
        si = names.index(start_name)
        ei = names.index(end_name)
    except ValueError:
        # fallback: select files whose name is >= start and <= end (string compare)
        sel = [p for p in files if start_name <= p.name <= end_name]
        return sorted(sel)
    if si <= ei:
        sel = files[si:ei+1]
    else:
        sel = files[ei:si+1]
    return sorted(sel)

def collect_specific_files(directory: Path, filelist: List[str]) -> List[Path]:
    out=[]
    for name in filelist:
        p=directory/name if isinstance(directory, Path) else Path(directory)/name
        if Path(p).exists():
            out.append(Path(p))
        else:
            print(f"[WARN] dev file not found: {p}", file=sys.stderr)
    return out

# ---------------------------
# Read .mrg files (Penn style) and extract sentences + gold spans
# ---------------------------
def read_mrg_file(path: Path) -> List[Tree]:
    """
    Read a .mrg file that contains many bracketed trees (one or multiple per line)
    Return list of nltk.Tree objects
    """
    trees=[]
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    # Some .mrg files may contain multiple trees possibly separated by newlines.
    # We'll try to extract by parsing bracketed substrings.
    # Simplest: split by newlines and parse each non-empty line via Tree.fromstring if possible.
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            t = Tree.fromstring(ln)
            trees.append(t)
        except Exception:
            # try to find bracketed substring
            try:
                # find first '(' and last ')'
                s = ln
                open_idx = s.find('(')
                close_idx = s.rfind(')')
                if open_idx >=0 and close_idx > open_idx:
                    t = Tree.fromstring(s[open_idx:close_idx+1])
                    trees.append(t)
            except Exception:
                # skip unparsable lines
                continue
    return trees

def tree_to_spans(tree: Tree) -> Tuple[List[str], List[Tuple[int,int,str]]]:
    """
    Convert nltk.Tree to list of words and list of spans (start, end, label)
    Spans: [start, end) word indices (0-based). We include only non-terminal spans (label, start, end) where end>start+0
    """
    words = tree.leaves()
    spans = []

    # traverse tree, record span for each subtree with label not equal to leaf POS or the preterminal
    def helper(t: Tree, offset: int) -> Tuple[int,int]:
        # returns (start, end) indices of this subtree
        if isinstance(t[0], str):
            # preterminal: POS -> word
            return offset, offset+1
        start = offset
        cur = offset
        for child in t:
            s,e = helper(child, cur)
            cur = e
        end = cur
        # record span for non-terminal (skip topmost? we include root)
        label = t.label()
        spans.append((start, end, label))
        return start, end

    helper(tree, 0)
    # Remove spans that are length 1 and label is POS? helper already skips preterminals
    # Filter out spans that cover whole sentence only? we keep root too.
    return words, spans

def load_dataset_from_mrg_files(files: List[Path], max_words=MAX_WORDS) -> List[Dict[str,Any]]:
    """
    Returns list of examples: {"words": [...], "spans": [(s,e,label), ...]}
    Only keep sentences with length <= max_words
    """
    examples=[]
    for p in files:
        print(f"Reading {p}")
        trees = read_mrg_file(p)
        for t in trees:
            words, spans = tree_to_spans(t)
            if len(words) == 0: continue
            if len(words) > max_words:
                continue
            examples.append({"words": words, "spans": spans})
    return examples

# ---------------------------
# Label mapping for span labels
# ---------------------------
def build_label_map(all_examples: List[Dict]) -> Tuple[Dict[str,int], Dict[int,str]]:
    labels=set()
    for ex in all_examples:
        for (_,_,lab) in ex["spans"]:
            labels.add(lab)
    labels = sorted(list(labels))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label

# ---------------------------
# Dataset class (tokenization + mapping)
# ---------------------------
class ConstituencyDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, label2id: Dict[str,int], max_pieces=MAX_PIECES):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_pieces = max_pieces

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]
        spans = ex["spans"]  # list of (s,e,label)
        # tokenize with split into words
        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  truncation=True,
                                  padding=False,
                                  max_length=self.max_pieces,
                                  return_attention_mask=True)
        word_ids = encoding.word_ids()  # mapping piece -> word index
        # find first subtoken index for each word
        word_to_first_piece = {}
        for i,widx in enumerate(word_ids):
            if widx is None: continue
            if widx not in word_to_first_piece:
                word_to_first_piece[widx] = i
        # if truncation occurred, some words not present; build valid word list
        valid_word_idx = sorted(word_to_first_piece.keys())
        if len(valid_word_idx) == 0:
            # empty after truncation; skip by returning minimal example (shouldn't happen often)
            return None
        # build new words list and mapping old->new
        old2new={}
        new_words=[]
        for new_i, old_i in enumerate(valid_word_idx):
            old2new[old_i]=new_i
            new_words.append(words[old_i])
        # remap spans to new indices and only keep those fully inside truncated range
        new_spans=[]
        for (s,e,label) in spans:
            # s,e are over original words
            # we require every word index in [s,e-1] to be in old2new
            ok = all((w in old2new) for w in range(s, e))
            if not ok:
                continue
            ns = old2new[s]
            ne = old2new[e-1] + 1
            new_spans.append((ns, ne, label))
        # build span set for quick positive lookup
        span_set = {(s,e): label for (s,e,label) in new_spans}
        # store item
        item = {
            "encoding": encoding,
            "word_first_piece": [word_to_first_piece[w] for w in valid_word_idx],
            "n_words": len(new_words),
            "spans": new_spans,
            "span_set": span_set,
            "orig_words": new_words
        }
        return item

def collate_batch(batch):
    # remove None entries
    batch = [b for b in batch if b is not None]
    encs = [b["encoding"] for b in batch]
    max_pieces = max(len(e["input_ids"]) for e in encs)
    max_words = max(b["n_words"] for b in batch)
    input_ids=[]
    attention_mask=[]
    wf_padded=[]
    n_words=[]
    span_lists=[]  # list of list of (s,e,label)
    for b in batch:
        ids = b["encoding"]["input_ids"]
        am = b["encoding"]["attention_mask"]
        pad_len = max_pieces - len(ids)
        input_ids.append(ids + [0]*pad_len)
        attention_mask.append(am + [0]*pad_len)
        # word_first_piece padded by -1
        wf = b["word_first_piece"] + [-1]*(max_words - len(b["word_first_piece"]))
        wf_padded.append(wf)
        n_words.append(b["n_words"])
        span_lists.append(b["spans"])
    batch_tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "word_first_piece": torch.tensor(wf_padded, dtype=torch.long),
        "n_words": torch.tensor(n_words, dtype=torch.long),
        "span_lists": span_lists  # leave as python lists
    }
    return batch_tensors

# ---------------------------
# Model: BERT encoder + span MLPs + label head
# ---------------------------
class SpanConstituencyParser(nn.Module):
    def __init__(self, bert_name, hidden_dim=D_MODEL, span_mlp_hidden=SPAN_MLP_HIDDEN, label_hidden=LABEL_MLP_HIDDEN, n_labels=10, dropout=DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # project boundary token embeddings to smaller dim
        proj_dim = span_mlp_hidden
        self.proj = nn.Linear(bert_dim, proj_dim)
        # span representation will be [h_i; h_j; h_i * h_j; h_i - h_j] -> size = 4 * proj_dim
        span_input_dim = proj_dim * 4
        self.span_mlp = nn.Sequential(
            nn.Linear(span_input_dim, span_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_mlp_hidden, span_mlp_hidden),
            nn.ReLU()
        )
        # existence score
        self.span_scorer = nn.Linear(span_mlp_hidden, 1)
        # label scorer
        self.label_mlp = nn.Sequential(
            nn.Linear(span_mlp_hidden, label_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.label_scorer = nn.Linear(label_hidden, n_labels)

    def forward(self, input_ids, attention_mask, word_first_piece, n_words):
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        word_first_piece: (B, W) index of first subtoken or -1 for pad
        n_words: (B,)
        Returns:
           span_exist_logits: nested python structure? We'll compute on the fly in loss function.
           But to speed, we return:
             - word_repr: (B, W, proj_dim) representations for each word (first subtoken projected)
             - mask_words: (B, W) bool mask
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, T, H)
        B, T, H = hidden.size()
        B2, W = word_first_piece.size()
        assert B == B2
        # replace -1 with 0 for safe gather
        idx = word_first_piece.clone()
        mask_word = (idx != -1)
        idx[idx == -1] = 0
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, H)  # (B, W, H)
        word_repr = torch.gather(hidden, 1, idx_exp)    # (B, W, H)
        # zero-out padded rows
        word_repr = word_repr * mask_word.unsqueeze(-1).float()
        # project
        p = self.proj(word_repr)  # (B, W, proj_dim)
        p = F.relu(p)
        # return projected word repr and mask
        return p, mask_word

    def span_representation(self, p, i, j):
        """
        p: (B, W, proj_dim)
        i,j scalar indices (i<j) word indices where span covers words [i,j)
        We will compute for all spans vectorized later.
        """
        raise NotImplementedError("Use vectorized span scoring functions below.")

    def score_all_spans(self, p, mask_word):
        """
        Compute span encodings and scores for all spans in each sentence in batch.

        Returns:
          span_exist_logits: list per-batch, each is 2D tensor (num_spans,) corresponding to scores
          span_label_logits: list per-batch, each is tensor (num_spans, n_labels)
          span_index: list per-batch of tuples (i,j)
        Implementation strategy:
          - for each sentence b, build all spans i<j up to n_words[b] (span length >=1)
          - vectorize per-sentence using tensor indexing
        Note: this is memory heavy for long sentences (O(n^2) spans). That's typical for span-based parsers.
        """
        B, W, D = p.size()
        out_exist = []
        out_label = []
        out_idx = []
        for b in range(B):
            n = int(mask_word[b].sum().item())
            if n <= 0:
                out_exist.append(torch.empty(0, device=p.device))
                out_label.append(torch.empty((0, self.label_scorer.out_features), device=p.device))
                out_idx.append([])
                continue
            # get boundary vectors: for span [i,j) we use left = p[b,i], right = p[b,j-1]
            # build arrays of i and j
            spans=[]
            lefts=[]
            rights=[]
            for i in range(0,n):
                for j in range(i+1, n+1):
                    spans.append((i,j))
                    lefts.append(p[b,i])
                    rights.append(p[b,j-1])
            if len(spans)==0:
                out_exist.append(torch.empty(0, device=p.device))
                out_label.append(torch.empty((0, self.label_scorer.out_features), device=p.device))
                out_idx.append([])
                continue
            lefts = torch.stack(lefts, dim=0)   # (S, D)
            rights = torch.stack(rights, dim=0) # (S, D)
            # combine features
            comb = torch.cat([lefts, rights, lefts * rights, lefts - rights], dim=-1)  # (S, 4D)
            span_feat = self.span_mlp(comb)  # (S, H)
            exist_logits = self.span_scorer(span_feat).squeeze(-1)  # (S,)
            lab_feat = self.label_mlp(span_feat)  # (S, Lh)
            lab_logits = self.label_scorer(lab_feat)  # (S, n_labels)
            out_exist.append(exist_logits)
            out_label.append(lab_logits)
            out_idx.append(spans)
        return out_exist, out_label, out_idx

# ---------------------------
# Loss & Decoding (CKY)
# ---------------------------
def create_gold_span_sets(span_list: List[Tuple[int,int,str]], label2id: Dict[str,int]) -> Tuple[set, dict]:
    """
    Given a list of (s,e,label) produce:
      - span_set: set of (s,e) for existence positive
      - span_label_map: dict (s,e)->label_id
    """
    span_set = set()
    span_label_map = {}
    for s,e,label in span_list:
        span_set.add((s,e))
        span_label_map[(s,e)] = label2id[label]
    return span_set, span_label_map

def loss_on_batch(model: SpanConstituencyParser, batch, label2id, device):
    """
    Compute loss for a batch:
     - Use model.score_all_spans to get per-sentence span logits
     - For each sentence, form targets: existence binary (1 for gold spans), negative for others
       We will sample negative spans to keep computation reasonable (optionally). Here we use all spans.
     - Existence loss: BCEWithLogits over all candidate spans (pos=1 for gold, neg=0)
     - Label loss: CrossEntropy on gold spans only (use label logits)
    Returns total_loss, stats dict
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_first_piece = batch["word_first_piece"].to(device)
    n_words = batch["n_words"].to(device)
    span_lists = batch["span_lists"]

    p, mask_word = model(input_ids, attention_mask, word_first_piece, n_words)
    exist_logits_list, label_logits_list, span_idx_list = model.score_all_spans(p, mask_word)

    total_loss = 0.0
    total_span_loss = 0.0
    total_label_loss = 0.0
    n_pos = 0
    n_spans_total = 0

    for b in range(len(span_lists)):
        gold_spans = span_lists[b]  # list (s,e,label)
        gold_set, gold_label_map = create_gold_span_sets(gold_spans, label2id)
        exist_logits = exist_logits_list[b]  # (S,)
        label_logits = label_logits_list[b]  # (S, n_labels)
        spans = span_idx_list[b]             # list of (i,j)
        S = len(spans)
        n_spans_total += S
        if S == 0:
            continue
        # build existence targets
        exist_target = torch.zeros(S, device=device)
        label_target = torch.zeros(S, dtype=torch.long, device=device)
        for si, (i,j) in enumerate(spans):
            if (i,j) in gold_set:
                exist_target[si] = 1.0
                label_target[si] = gold_label_map[(i,j)]
        n_pos += int(exist_target.sum().item())
        # existence loss
        span_loss = F.binary_cross_entropy_with_logits(exist_logits, exist_target)
        # label loss: compute only over positives; if no positives then zero
        pos_mask = exist_target.bool()
        if pos_mask.sum() > 0:
            lab_logits_pos = label_logits[pos_mask]
            lab_targets_pos = label_target[pos_mask]
            label_loss = F.cross_entropy(lab_logits_pos, lab_targets_pos)
        else:
            label_loss = torch.tensor(0.0, device=device)
        total_span_loss += span_loss.item()
        total_label_loss += label_loss.item()
        total_loss += span_loss + label_loss

    stats = {
        "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss),
        "span_loss": total_span_loss,
        "label_loss": total_label_loss,
        "n_pos": n_pos,
        "n_spans_total": n_spans_total
    }
    return total_loss, stats

# CKY decode using span scores (exist logits + best label score)
def cky_decode_from_scores(exist_logits, label_logits, spans, n_words):
    """
    Given per-sentence candidate spans (spans: list of (i,j)), existence logits (S,), label logits (S, L),
    perform CKY to get best binary constituency tree maximizing sum of chosen span scores (exist+label).
    We'll build score_map[(i,j)] = exist_logit + max_label_logit_for_span
    Then classical CKY dynamic programming filling chart[i][j] = max over splits of chart[i][k]+chart[k][j] or include span score.
    Return list of selected spans with predicted label ids.
    """
    # build mapping span->score
    S = len(spans)
    score_map = {}
    label_pred_map = {}
    for idx,(i,j) in enumerate(spans):
        exist = exist_logits[idx].item()
        lab_logits = label_logits[idx].detach().cpu().numpy()
        lab_id = int(lab_logits.argmax())
        lab_max = float(lab_logits[lab_id])
        score_map[(i,j)] = exist + lab_max
        label_pred_map[(i,j)] = lab_id
    # initialize chart
    chart = [[-1e9] * (n_words+1) for _ in range(n_words+1)]
    back = [[None] * (n_words+1) for _ in range(n_words+1)]
    # empty spans have 0
    for i in range(n_words+1):
        chart[i][i] = 0.0
    # iterate span lengths
    for length in range(1, n_words+1):
        for i in range(0, n_words - length + 1):
            j = i + length
            # option1: split
            best = -1e9
            best_split=None
            for k in range(i+1,j):
                val = chart[i][k] + chart[k][j]
                if val > best:
                    best = val
                    best_split = k
            # option2: take bracket (i,j) if available
            bracket_score = score_map.get((i,j), -1e9)
            if bracket_score > best:
                chart[i][j] = bracket_score
                back[i][j] = ("BRACKET", (i,j))
            else:
                chart[i][j] = best
                back[i][j] = ("SPLIT", best_split)
    # recover spans by recursion from (0,n)
    selected = []
    def recover(i,j):
        if i==j:
            return
        action = back[i][j]
        if action is None:
            return
        if action[0]=="BRACKET":
            (a,b) = action[1]
            # add children
            # a,b must be (i,j)
            selected.append((a,b,label_pred_map.get((a,b),0)))
            # but still we need to partition inside to get nested structure
            # find best split inside: try to use back info to split interior regions
            # we try to find any k such that chart[a][b] == chart[a][k] + chart[k][b]
            # but simplest: try to recover for all splits by recursion based on back
            # find split from back
            # we need to compute interior partitioning via back table entries
            # we will attempt to recursively recover children: scan k
            for k in range(a+1,b):
                # if possible split at k such that chart[a][b] == chart[a][k] + chart[k][b], recover both
                if abs(chart[a][b] - (chart[a][k] + chart[k][b])) < 1e-6:
                    recover(a,k)
                    recover(k,b)
                    return
            # if no split found, just return
            return
        else:
            k = action[1]
            recover(i,k)
            recover(k,j)
    recover(0, n_words)
    return selected  # list of (i,j,label_id)

# ---------------------------
# Evaluation helpers: convert predicted spans to evaluation metrics (precision/recall/F1)
# ---------------------------
def spans_from_pred(selected_spans):
    # selected_spans: list of (i,j,label_id)
    span_set = set((i,j) for (i,j,l) in selected_spans)
    label_map = {(i,j):l for (i,j,l) in selected_spans}
    return span_set, label_map

def compute_bracketing_f1(gold_spans_list, pred_spans_list, id2label):
    # gold_spans_list: list per sent of [(s,e,label), ...]
    # pred_spans_list: list per sent of [(s,e,label_id), ...]
    total_gold = 0
    total_pred = 0
    total_correct_unlabeled = 0
    total_correct_labeled = 0
    for gold, pred in zip(gold_spans_list, pred_spans_list):
        gold_set = set((s,e) for (s,e,lab) in gold)
        gold_lab_map = {(s,e):lab for (s,e,lab) in gold}
        pred_set = set((s,e) for (s,e,lid) in pred)
        pred_lab_map = {(s,e): id2label[lid] for (s,e,lid) in pred}
        total_gold += len(gold_set)
        total_pred += len(pred_set)
        inter = gold_set & pred_set
        total_correct_unlabeled += len(inter)
        for sp in inter:
            if gold_lab_map[sp] == pred_lab_map.get(sp):
                total_correct_labeled += 1
    prec_u = total_correct_unlabeled / total_pred if total_pred>0 else 0.0
    rec_u = total_correct_unlabeled / total_gold if total_gold>0 else 0.0
    f1_u = (2*prec_u*rec_u/(prec_u+rec_u)) if (prec_u+rec_u)>0 else 0.0
    prec_l = total_correct_labeled / total_pred if total_pred>0 else 0.0
    rec_l = total_correct_labeled / total_gold if total_gold>0 else 0.0
    f1_l = (2*prec_l*rec_l/(prec_l+rec_l)) if (prec_l+rec_l)>0 else 0.0
    return {"U_P":prec_u,"U_R":rec_u,"U_F1":f1_u,"L_P":prec_l,"L_R":rec_l,"L_F1":f1_l}

# ---------------------------
# Training & Evaluation loops
# ---------------------------
def train_and_evaluate():
    # collect files
    train_files = collect_files_between(TRAIN_DIR, TRAIN_START, TRAIN_END)
    test_files  = collect_files_between(TEST_DIR, TEST_START, TEST_END)
    dev_files   = collect_specific_files(DEV_DIR, DEV_FILES)

    print(f"Train files: {len(train_files)}, Dev files: {len(dev_files)}, Test files: {len(test_files)}")

    # load datasets
    print("Loading train data...")
    train_examples = load_dataset_from_mrg_files(train_files, max_words=MAX_WORDS)
    print("Loading dev data...")
    dev_examples = load_dataset_from_mrg_files(dev_files, max_words=MAX_WORDS)
    print("Loading test data...")
    test_examples = load_dataset_from_mrg_files(test_files, max_words=MAX_WORDS)

    print(f"Train ex: {len(train_examples)}, Dev ex: {len(dev_examples)}, Test ex: {len(test_examples)}")

    # build label map from all examples
    all_examples = train_examples + dev_examples + test_examples
    label2id, id2label = build_label_map(all_examples)
    print(f"Num labels: {len(label2id)}")

    # tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = ConstituencyDataset(train_examples, tokenizer, label2id, max_pieces=MAX_PIECES)
    dev_ds = ConstituencyDataset(dev_examples, tokenizer, label2id, max_pieces=MAX_PIECES)
    test_ds = ConstituencyDataset(test_examples, tokenizer, label2id, max_pieces=MAX_PIECES)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_batch)

    # model
    model = SpanConstituencyParser(MODEL_NAME, hidden_dim=D_MODEL, span_mlp_hidden=SPAN_MLP_HIDDEN,
                                   label_hidden=LABEL_MLP_HIDDEN, n_labels=len(label2id), dropout=DROPOUT)
    model.to(DEVICE)

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    best_dev_f1 = 0.0
    global_step = 0

    # training loop
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad()
            loss, stats = loss_on_batch(model, batch, label2id, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item() if isinstance(loss, torch.Tensor) else loss)
            global_step += 1
            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix({"loss": running_loss / global_step})
        avg_loss = running_loss / (step if step>0 else 1)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # dev evaluation
        dev_metrics = evaluate_on_loader(model, dev_loader, label2id, id2label, DEVICE)
        print("Dev metrics:", dev_metrics)
        if dev_metrics["L_F1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["L_F1"]
            # save model
            save_dir = OUTPUT_DIR / "best_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            model_state = {
                "model_state": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "tokenizer": MODEL_NAME
            }
            torch.save(model_state, save_dir / "model_state.pth")
            print(f"Saved best model to {save_dir}")

    # final test evaluation using best model if saved
    best_model_path = OUTPUT_DIR / "best_model" / "model_state.pth"
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_on_loader(model, test_loader, label2id, id2label, DEVICE)
    print("Test metrics:", test_metrics)

def evaluate_on_loader(model, loader, label2id, id2label, device):
    model.eval()
    gold_spans_all=[]
    pred_spans_all=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_first_piece = batch["word_first_piece"].to(device)
            n_words = batch["n_words"].to(device)
            span_lists = batch["span_lists"]
            p, mask_word = model(input_ids, attention_mask, word_first_piece, n_words)
            exist_logits_list, label_logits_list, span_idx_list = model.score_all_spans(p, mask_word)
            # decode each sentence using CKY
            for b in range(len(span_idx_list)):
                n = int(n_words[b].item())
                exist_logits = exist_logits_list[b]
                label_logits = label_logits_list[b]
                spans = span_idx_list[b]
                selected = cky_decode_from_scores(exist_logits, label_logits, spans, n)
                # selected: list of (i,j,label_id)
                pred_spans_all.append(selected)
                # gold
                gold_spans_all.append(span_lists[b])
    metrics = compute_bracketing_f1(gold_spans_all, pred_spans_all, id2label)
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
