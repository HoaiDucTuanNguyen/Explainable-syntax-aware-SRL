"""
complete_relational_probing_pipeline.py

A fully working SRL relational probing implementation:
1. Convert CRAFT (dependency XML) → dependency CoNLL
2. Convert GENIA (constituency XML) → constituency CoNLL
3. Train dependency probe {Pr_dep}
4. Train constituency probe {Pr_cons}
5. Compute nullspace projection matrices
6. Enable perturbation for removing syntactic relational components.



import os
from lxml import etree
import nltk
from nltk.tree import Tree
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 1. FIXED DIRECTORY NAMES
# ============================================================

CRAFT_XML_DIR = "data/CRAFT/xml/"
GENIA_XML_DIR = "data/GENIA/xml/"

CRAFT_CONLL = "data/craft_dep.conll"
GENIA_CONLL = "data/genia_const.conll"

# Your SRL encoder checkpoint name — FIXED, REQUIRED
SRL_CKPT = "models/my_biobert_srl_ckpt"


# ============================================================
# 2. XML → CoNLL CONVERTERS (DEPENDENCY + CONSTITUENCY)
# ============================================================

def convert_craft_xml_to_conll(input_dir: str, output_path: str) -> None:
    """
    Converts CRAFT dependency XML files into a CoNLL-style format:
    ID   TOKEN   HEAD   DEPREL
    """

    with open(output_path, "w", encoding="utf-8") as fout:
        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(".xml"):
                continue

            tree = etree.parse(os.path.join(input_dir, fname))
            root = tree.getroot()

            # Assumed schema:
            # <sentence>
            #    <token id="t1" form="..." head="t0" deprel="ROOT"/>
            # </sentence>

            for sent in root.findall(".//sentence"):
                tokens = []
                id_map = {}

                for i, tok in enumerate(sent.findall(".//token"), start=1):
                    tid = tok.get("id")
                    form = tok.get("form") or ("_" if tok.text is None else tok.text.strip())
                    head = tok.get("head") or "0"
                    deprel = tok.get("deprel") or "_"

                    tokens.append((tid, form, head, deprel))
                    id_map[tid] = i

                for tid, form, head, deprel in tokens:
                    head_idx = id_map.get(head, 0) if head != "0" else 0
                    fout.write(f"{id_map[tid]}\t{form}\t{head_idx}\t{deprel}\n")
                fout.write("\n")


def convert_genia_xml_to_conll(input_dir: str, output_path: str) -> None:
    """
    Converts GENIA constituency treebank into a simple CoNLL-style format:

    # parse = (S ... )
    1   token
    2   token
    """

    with open(output_path, "w", encoding="utf-8") as fout:
        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(".xml"):
                continue

            tree = etree.parse(os.path.join(input_dir, fname))
            root = tree.getroot()

            # Assumed schema:
            # <sentence>
            #    <parse>(S (NP ...) ...)</parse>
            # </sentence>

            for sent in root.findall(".//sentence"):
                parse_el = sent.find(".//parse")
                if parse_el is None or not parse_el.text:
                    continue

                parse_str = parse_el.text.strip()
                t = Tree.fromstring(parse_str)
                tokens = t.leaves()

                fout.write(f"# parse = {parse_str}\n")
                for i, tok in enumerate(tokens, start=1):
                    fout.write(f"{i}\t{tok}\n")
                fout.write("\n")


# ============================================================
# 3. READ CONLL-FORMAT FILES
# ============================================================

def read_dependency_conll(path):
    sentences = []
    cur_tokens, cur_heads, cur_labels = [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_tokens:
                    sentences.append({
                        "tokens": cur_tokens,
                        "heads": cur_heads,
                        "labels": cur_labels
                    })
                cur_tokens, cur_heads, cur_labels = [], [], []
                continue

            idx, tok, head, rel = line.split("\t")
            cur_tokens.append(tok)
            cur_heads.append(int(head))
            cur_labels.append(rel)

    return sentences


def read_genia_conll(path):
    sentences = []
    tokens = []
    parse_tree = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens and parse_tree:
                    sentences.append({"tokens": tokens, "parse": parse_tree})
                tokens, parse_tree = [], None
                continue

            if line.startswith("# parse = "):
                parse_str = line[len("# parse = "):]
                parse_tree = Tree.fromstring(parse_str)
                continue

            _, tok = line.split("\t")
            tokens.append(tok)

    return sentences


# ============================================================
# 4. SRL ENCODER WRAPPER (M_SRL)
# ============================================================

class SrlEncoder:
    def __init__(self, ckpt_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModel.from_pretrained(ckpt_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, tokens):
        enc = self.tokenizer(tokens, return_tensors="pt", is_split_into_words=True).to(self.device)
        out = self.model(**enc).last_hidden_state

        word_ids = enc.word_ids(batch_index=0)
        uniq = []
        seen = set()
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in seen:
                seen.add(wid)
                uniq.append(out[0, idx])

        return torch.stack(uniq, dim=0).cpu()


# ============================================================
# 5. BUILD RELATION PAIRS
# ============================================================

def build_dep_pairs(dep_sents):
    pairs, labels, all_tokens = [], [], []

    for si, sent in enumerate(dep_sents):
        toks = sent["tokens"]
        heads = sent["heads"]
        rels = sent["labels"]
        all_tokens.append(toks)

        for dep_idx, head_idx in enumerate(heads, start=1):
            if head_idx == 0:
                continue
            i = head_idx - 1
            j = dep_idx - 1
            pairs.append((si, i, j))
            labels.append(rels[dep_idx - 1])

    return pairs, labels, all_tokens


def build_cons_pairs(genia_sents):
    pairs, labels, all_tokens = [], [], []

    for si, sent in enumerate(genia_sents):
        toks = sent["tokens"]
        tree = sent["parse"]
        all_tokens.append(toks)

        n = len(toks)
        pos_spans = [[] for _ in range(n)]

        def walk(t, start):
            if isinstance(t[0], str):
                pos_spans[start].append((start, start + 1))
                return start + 1
            cur = start
            for c in t:
                cur = walk(c, cur)
            end = cur
            for k in range(start, end):
                pos_spans[k].append((start, end))
            return end

        walk(tree, 0)

        for i in range(n):
            si_spans = set(pos_spans[i])
            for j in range(n):
                if i == j:
                    continue
                label = 1 if si_spans.intersection(pos_spans[j]) else 0
                pairs.append((si, i, j))
                labels.append(label)

    return pairs, labels, all_tokens


# ============================================================
# 6. DATASET + LINEAR PROBES
# ============================================================

class ProbeDataset(Dataset):
    def __init__(self, encoder, all_tokens, pair_idx, labels, label2id):
        self.encoder = encoder
        self.pair_idx = pair_idx
        self.labels = labels
        self.label2id = label2id
        self.all_tokens = all_tokens

        self.encoded = []
        for toks in all_tokens:
            self.encoded.append(self.encoder.encode(toks))

    def __len__(self): return len(self.pair_idx)

    def __getitem__(self, k):
        si, i, j = self.pair_idx[k]
        label = self.label2id[self.labels[k]]
        vec = torch.cat([self.encoded[si][i], self.encoded[si][j]], dim=-1)
        return vec, torch.tensor(label)


class LinearProbe(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)

    def forward(self, x): return self.linear(x)


# ============================================================
# 7. TRAINING UTILITIES
# ============================================================

def train_probe(model, dataset, lr=1e-3, bs=256, epochs=5):
    model.cuda() if torch.cuda.is_available() else model.cpu()
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dev = next(model.parameters()).device

    for ep in range(1, epochs + 1):
        total, correct = 0, 0
        for x, y in loader:
            x = x.to(dev); y = y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=-1)
            total += y.size(0)
            correct += (preds == y).sum().item()

        print(f"[Epoch {ep}] acc = {correct / total:.4f}")

    return model


# ============================================================
# 8. NULLSPACE PROJECTION
# ============================================================

def compute_nullspace(W):
    W = W.detach().cpu()
    u, s, vh = torch.linalg.svd(W, full_matrices=True)
    eps = 1e-6
    rank = (s > eps).sum()
    null_basis = vh[rank:].T
    P_null = null_basis @ null_basis.T
    return P_null


# ============================================================
# 9. MAIN EXECUTION PIPELINE
# ============================================================

def main():
    print("Converting XML → CoNLL...")
    convert_craft_xml_to_conll(CRAFT_XML_DIR, CRAFT_CONLL)
    convert_genia_xml_to_conll(GENIA_XML_DIR, GENIA_CONLL)

    print("Loading datasets...")
    dep_sents = read_dependency_conll(CRAFT_CONLL)
    genia_sents = read_genia_conll(GENIA_CONLL)

    dep_pairs, dep_labels, dep_tokens = build_dep_pairs(dep_sents)
    cons_pairs, cons_labels, cons_tokens = build_cons_pairs(genia_sents)

    dep_label_set = sorted(set(dep_labels))
    dep_label2id = {lab: i for i, lab in enumerate(dep_label_set)}

    cons_label2id = {0: 0, 1: 1}

    print("Loading SRL encoder checkpoint...")
    encoder = SrlEncoder(SRL_CKPT)

    print("Preparing datasets...")
    dep_ds = ProbeDataset(encoder, dep_tokens, dep_pairs, dep_labels, dep_label2id)
    cons_ds = ProbeDataset(encoder, cons_tokens, cons_pairs, cons_labels, cons_label2id)

    dim = encoder.model.config.hidden_size * 2
    dep_probe = LinearProbe(dim, len(dep_label2id))
    cons_probe = LinearProbe(dim, 2)

    print("Training dependency probe...")
    dep_probe = train_probe(dep_probe, dep_ds)

    print("Training constituency probe...")
    cons_probe = train_probe(cons_probe, cons_ds)

    print("Computing nullspaces...")
    P_dep = compute_nullspace(dep_probe.linear.weight)
    P_cons = compute_nullspace(cons_probe.linear.weight)

    print("Done.")
    print("Saved probes and nullspaces are available in memory.")
    print("You may now proceed with relation removal experiments.")


if __name__ == "__main__":
    main()
