"""
random_label_control_experiment.py

Random-label control experiment for linear relational probes {Pr_dep} and {Pr_cons}.


"""

import os
import csv
import random
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


# Paths for precomputed probing datasets
DEP_TRAIN_PATH = "data/probing/dep_train.pt"
DEP_DEV_PATH = "data/probing/dep_dev.pt"
CONS_TRAIN_PATH = "data/probing/cons_train.pt"
CONS_DEV_PATH = "data/probing/cons_dev.pt"

# Output CSV file
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "random_label_control.csv")


class LinearProbe(nn.Module):
    """
    Simple linear probe:
        z = W x + b
    with cross-entropy loss over discrete labels.
    """

    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_probe_dataset(path: str) -> Dict[str, torch.Tensor]:
    """
    Loads a probing dataset from a .pt file.
    Expected keys: "X", "y", "layer".
    """
    data = torch.load(path, map_location="cpu")
    required_keys = {"X", "y", "layer"}
    if not required_keys.issubset(set(data.keys())):
        missing = required_keys - set(data.keys())
        raise KeyError(f"Dataset at {path} is missing keys: {missing}")
    return data


def get_layer_ids(train: Dict[str, torch.Tensor],
                  dev: Dict[str, torch.Tensor]) -> List[int]:
    """
    Returns sorted list of layer IDs that appear in both train and dev sets.
    """
    train_layers = set(train["layer"].tolist())
    dev_layers = set(dev["layer"].tolist())
    common = sorted(train_layers.intersection(dev_layers))
    return common


def subset_by_layer(data: Dict[str, torch.Tensor], layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns X, y subset corresponding to the given layer_id.
    """
    mask = (data["layer"] == layer_id)
    X = data["X"][mask]
    y = data["y"][mask]
    return X, y


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    num_classes: int,
    num_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Trains a linear probe on (X_train, y_train) and evaluates on (X_dev, y_dev).
    Returns (dev_f1_macro, dev_accuracy).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in = X_train.size(1)
    model = LinearProbe(d_in, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_dev = X_dev.to(device)
    y_dev = y_dev.to(device)

    n_train = X_train.size(0)
    indices = list(range(n_train))

    for epoch in range(num_epochs):
        random.shuffle(indices)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_dev = model(X_dev)
        preds = torch.argmax(logits_dev, dim=1).cpu().numpy()
        true = y_dev.cpu().numpy()

    f1 = f1_score(true, preds, average="macro")
    acc = (preds == true).mean() * 100.0
    return f1, acc


def randomize_labels(y: torch.Tensor, seed: int = 1234) -> torch.Tensor:
    """
    Randomly permutes labels in y (uniform permutation of original labels).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    y_np = y.clone().numpy()
    indices = list(range(len(y_np)))
    random.shuffle(indices)
    y_perm = y_np[indices]
    return torch.from_numpy(y_perm)


def ensure_results_dir():
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)


def run_random_label_control(
    train_path: str,
    dev_path: str,
    relation_type: str,
    csv_writer: csv.writer,
):
    """
    Runs the random-label control experiment for a given relation type
    ('dep' or 'cons'), and writes results into the provided CSV writer.
    """
    train_data = load_probe_dataset(train_path)
    dev_data = load_probe_dataset(dev_path)

    common_layers = get_layer_ids(train_data, dev_data)

    num_classes = int(train_data["y"].max().item() + 1)

    for layer_id in common_layers:
        X_train_true, y_train_true = subset_by_layer(train_data, layer_id)
        X_dev_true, y_dev_true = subset_by_layer(dev_data, layer_id)

        if X_train_true.size(0) < num_classes or X_dev_true.size(0) < num_classes:
            continue

        true_f1, true_acc = train_probe(
            X_train_true,
            y_train_true,
            X_dev_true,
            y_dev_true,
            num_classes=num_classes,
            num_epochs=10,
            batch_size=256,
            lr=1e-3,
            seed=42,
        )

        y_train_rand = randomize_labels(y_train_true, seed=2025)
        rand_f1, rand_acc = train_probe(
            X_train_true,
            y_train_rand,
            X_dev_true,
            y_dev_true,
            num_classes=num_classes,
            num_epochs=10,
            batch_size=256,
            lr=1e-3,
            seed=42,
        )

        csv_writer.writerow(
            [
                relation_type,
                layer_id,
                f"{true_f1:.4f}",
                f"{true_acc:.2f}",
                f"{rand_f1:.4f}",
                f"{rand_acc:.2f}",
                X_train_true.size(0),
                X_dev_true.size(0),
            ]
        )


def main():
    ensure_results_dir()

    with open(RESULTS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "relation_type",
                "layer",
                "true_label_F1_macro",
                "true_label_Accuracy_percent",
                "random_label_F1_macro",
                "random_label_Accuracy_percent",
                "train_samples",
                "dev_samples",
            ]
        )

        run_random_label_control(
            train_path=DEP_TRAIN_PATH,
            dev_path=DEP_DEV_PATH,
            relation_type="dep",
            csv_writer=writer,
        )

        run_random_label_control(
            train_path=CONS_TRAIN_PATH,
            dev_path=CONS_DEV_PATH,
            relation_type="cons",
            csv_writer=writer,
        )

    print(f"Random-label control results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
