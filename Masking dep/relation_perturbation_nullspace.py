"""
relation_perturbation_nullspace.py

Implements relation-level perturbation using nullspace projection for
dependency and constituency relational probes built on top of an SRL encoder.


"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SRL_CKPT = "models/my_biobert_srl_ckpt"
DEP_PROBE_PATH = "models/dep_probe.pt"
CONS_PROBE_PATH = "models/cons_probe.pt"


class SrlEncoder:
    """
    Wrapper for the SRL encoder M_SRL. Returns contextualized word representations.
    """

    def __init__(self, ckpt_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModel.from_pretrained(ckpt_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, tokens):
        """
        Encodes a tokenized sentence into contextualized word vectors.
        """
        enc = self.tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state

        word_ids = enc["input_ids"].new_tensor(self.tokenizer.word_ids(batch_index=0) or [])
        # Fallback in case word_ids is None-like; map all non-special tokens sequentially.
        if len(word_ids) == 0:
            vectors = []
            for i in range(out.size(1)):
                vectors.append(out[0, i])
            return torch.stack(vectors, dim=0).cpu()

        mapped = []
        seen = set()
        wi_list = []
        for idx, wid in enumerate(self.tokenizer.word_ids(batch_index=0)):
            if wid is None:
                continue
            if wid not in seen:
                seen.add(wid)
                mapped.append(out[0, idx])
                wi_list.append(wid)

        return torch.stack(mapped, dim=0).cpu()


class LinearProbe(nn.Module):
    """
    Linear relational probe:
    z = W [v_i; v_j] + b
    """

    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.linear(x)


def load_probe(path: str, hidden_size: int) -> LinearProbe:
    """
    Loads a linear probe from disk given the SRL encoder hidden size.
    """
    state = torch.load(path, map_location="cpu")
    weight = state["linear.weight"]
    n_out, d_in = weight.shape

    expected_d_in = 2 * hidden_size
    if d_in != expected_d_in:
        raise ValueError(f"Probe input dim {d_in} does not match 2*hidden_size {expected_d_in}.")

    probe = LinearProbe(d_in, n_out)
    probe.load_state_dict(state)
    probe.eval()
    return probe


def compute_nullspace_projection(W: torch.Tensor) -> torch.Tensor:
    """
    Computes the orthogonal projector onto the nullspace of W.
    W: [num_labels, d_in] weight matrix of a linear probe.
    Returns P_null: [d_in, d_in] such that P_null x removes components in rowspace(W).
    """
    W = W.detach().cpu()
    u, s, vh = torch.linalg.svd(W, full_matrices=True)
    eps = 1e-6
    rank = (s > eps).sum()
    null_basis = vh[rank:].T
    if null_basis.numel() == 0:
        d_in = W.size(1)
        return torch.zeros((d_in, d_in))
    P_null = null_basis @ null_basis.T
    return P_null


class RelationPerturber:
    """
    Implements relation-specific perturbation using nullspace projection
    for dependency and constituency relational probes.
    """

    def __init__(self):
        self.encoder = SrlEncoder(SRL_CKPT)
        self.hidden_size = self.encoder.hidden_size
        self.d_pair = 2 * self.hidden_size

        self.dep_probe = load_probe(DEP_PROBE_PATH, self.hidden_size)
        self.cons_probe = load_probe(CONS_PROBE_PATH, self.hidden_size)

        self.P_dep = compute_nullspace_projection(self.dep_probe.linear.weight)
        self.P_cons = compute_nullspace_projection(self.cons_probe.linear.weight)

    @torch.no_grad()
    def encode_sentence(self, tokens):
        """
        Returns contextualized vectors for an input token list.
        """
        return self.encoder.encode(tokens)

    def _project_pair(self, v_i: torch.Tensor, v_j: torch.Tensor, P: torch.Tensor):
        """
        Performs nullspace projection on [v_i; v_j] using projector P.
        """
        x = torch.cat([v_i, v_j], dim=-1)
        x_tilde = P @ x
        v_i_tilde = x_tilde[: self.hidden_size]
        v_j_tilde = x_tilde[self.hidden_size :]
        return v_i_tilde, v_j_tilde

    def perturb_dependency_relation(self, sentence_vectors: torch.Tensor, i: int, j: int):
        """
        Perturbs a dependency relation r = (i, j) in a sentence representation.
        sentence_vectors: [T, H] tensor of contextualized vectors.
        i, j: 0-based indices of the head and dependent words.
        Returns a new tensor [T, H] where only positions i and j are modified.
        """
        if i < 0 or j < 0 or i >= sentence_vectors.size(0) or j >= sentence_vectors.size(0):
            raise IndexError("Relation indices out of range.")

        v_i = sentence_vectors[i]
        v_j = sentence_vectors[j]
        v_i_tilde, v_j_tilde = self._project_pair(v_i, v_j, self.P_dep)

        perturbed = sentence_vectors.clone()
        perturbed[i] = v_i_tilde
        perturbed[j] = v_j_tilde
        return perturbed

    def perturb_constituency_relation(self, sentence_vectors: torch.Tensor, i: int, j: int):
        """
        Perturbs a constituency relation r = (i, j) in a sentence representation.
        sentence_vectors: [T, H] tensor of contextualized vectors.
        i, j: 0-based indices of the two words in the same phrase.
        Returns a new tensor [T, H] where only positions i and j are modified.
        """
        if i < 0 or j < 0 or i >= sentence_vectors.size(0) or j >= sentence_vectors.size(0):
            raise IndexError("Relation indices out of range.")

        v_i = sentence_vectors[i]
        v_j = sentence_vectors[j]
        v_i_tilde, v_j_tilde = self._project_pair(v_i, v_j, self.P_cons)

        perturbed = sentence_vectors.clone()
        perturbed[i] = v_i_tilde
        perturbed[j] = v_j_tilde
        return perturbed

    def perturb(self, sentence_vectors: torch.Tensor, i: int, j: int, relation_type: str):
        """
        General Perturb(s, r) operator for a relation r = (i, j).

        relation_type: "dep" or "cons"
        """
        if relation_type == "dep":
            return self.perturb_dependency_relation(sentence_vectors, i, j)
        elif relation_type == "cons":
            return self.perturb_constituency_relation(sentence_vectors, i, j)
        else:
            raise ValueError("relation_type must be 'dep' or 'cons'.")


def main():
    tokens = ["The", "enzyme", "strongly", "inhibits", "tumor", "growth", "."]
    perturber = RelationPerturber()

    vecs = perturber.encode_sentence(tokens)
    print("Original sentence representation:", vecs.shape)

    dep_i, dep_j = 3, 5
    dep_perturbed = perturber.perturb(vecs, dep_i, dep_j, relation_type="dep")
    diff_dep = (dep_perturbed - vecs).norm(dim=1)
    print("Dependency perturbation L2 changes per token:", diff_dep.tolist())

    cons_i, cons_j = 4, 5
    cons_perturbed = perturber.perturb(vecs, cons_i, cons_j, relation_type="cons")
    diff_cons = (cons_perturbed - vecs).norm(dim=1)
    print("Constituency perturbation L2 changes per token:", diff_cons.tolist())


if __name__ == "__main__":
    main()
