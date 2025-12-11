"""
faithfulness_evaluation.py



import os
import csv
from typing import Tuple, Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration: input paths
# ---------------------------------------------------------------------------

DATA_ORIGINAL_DEP = "data/faithfulness/degradation_original_dep.pt"
DATA_ORIGINAL_CONS = "data/faithfulness/degradation_original_cons.pt"

DATA_RANDOM_DEP = "data/faithfulness/degradation_random_dep.pt"
DATA_RANDOM_CONS = "data/faithfulness/degradation_random_cons.pt"

DATA_MODEL_SPEC_DEP = "data/faithfulness/model_specificity_dep.pt"
DATA_MODEL_SPEC_CONS = "data/faithfulness/model_specificity_cons.pt"

RESULTS_DIR = "results"
TABLE7_CSV = os.path.join(RESULTS_DIR, "table7_counterfactual_degradation.csv")
TABLE8_CSV = os.path.join(RESULTS_DIR, "table8_randomization_sanity_check.csv")
TABLE9_CSV = os.path.join(RESULTS_DIR, "table9_model_specificity.csv")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_results_dir() -> None:
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)


def bootstrap_ci(
    values: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 1234,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for the mean.

    values:      1D numpy array of samples
    num_bootstrap: number of bootstrap resamples
    alpha:         significance level (0.05 for 95% CI)
    returns: (lower, upper) CI bounds
    """
    rng = np.random.RandomState(rng_seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")

    means = []
    for _ in range(num_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = values[indices]
        means.append(sample.mean())

    means = np.array(means)
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Cohen's d between two distributions x and y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    pooled_std = np.sqrt(((x.size - 1) * var_x + (y.size - 1) * var_y) / (x.size + y.size - 2))
    if pooled_std == 0.0:
        return 0.0
    return float((mean_x - mean_y) / pooled_std)


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the two-sample Kolmogorov–Smirnov distance between x and y.
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n_x = x.size
    n_y = y.size
    if n_x == 0 or n_y == 0:
        return float("nan")

    data_all = np.concatenate([x, y])
    data_all.sort()

    cdf_x = np.searchsorted(x, data_all, side="right") / n_x
    cdf_y = np.searchsorted(y, data_all, side="right") / n_y
    d = np.max(np.abs(cdf_x - cdf_y))
    return float(d)


def load_band_data(path: str) -> Dict[str, np.ndarray]:
    """
    Loads a degradation dataset of the form:
        {"band": LongTensor [N], "f1_drop": FloatTensor [N]}
    and returns numpy arrays.
    """
    data = torch.load(path, map_location="cpu")
    if "band" not in data or "f1_drop" not in data:
        raise KeyError(f"File {path} must contain 'band' and 'f1_drop' keys.")
    bands = data["band"].numpy()
    drops = data["f1_drop"].numpy()
    return {"band": bands, "f1_drop": drops}


def load_model_specificity(path: str) -> Dict[str, np.ndarray]:
    """
    Loads model-specificity dataset:
        {"model": LongTensor [K], "f1_drop": FloatTensor [K]}
    where model = 0 for baseline, 1 for SRL+Both.
    """
    data = torch.load(path, map_location="cpu")
    if "model" not in data or "f1_drop" not in data:
        raise KeyError(f"File {path} must contain 'model' and 'f1_drop' keys.")
    model_ids = data["model"].numpy()
    drops = data["f1_drop"].numpy()
    return {"model": model_ids, "f1_drop": drops}


# ---------------------------------------------------------------------------
# Table 7: Counterfactual degradation by importance quantile
# ---------------------------------------------------------------------------

def compute_table7(
    orig_dep: Dict[str, np.ndarray],
    orig_cons: Dict[str, np.ndarray],
) -> None:
    """
    Computes Table 7:
      - mean F1 drop, std dev, 95% CI, and % affected per band (dep/cons)
    and writes to CSV.
    """
    ensure_results_dir()
    quantile_labels = {
        3: "Top 10%",
        2: "10–30%",
        1: "30–60%",
        0: "Bottom 40%",
    }
    band_order = [3, 2, 1, 0]

    with open(TABLE7_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "quantile_band",
                "relation_type",
                "mean_f1_drop",
                "std_f1_drop",
                "ci_lower",
                "ci_upper",
                "percent_affected",
                "num_samples",
            ]
        )

        for band_id in band_order:
            mask_dep = (orig_dep["band"] == band_id)
            dep_vals = orig_dep["f1_drop"][mask_dep]
            if dep_vals.size > 0:
                mean_dep = float(dep_vals.mean())
                std_dep = float(dep_vals.std(ddof=1)) if dep_vals.size > 1 else 0.0
                ci_low_dep, ci_high_dep = bootstrap_ci(dep_vals, num_bootstrap=1000, alpha=0.05)
                pct_dep = float((dep_vals > 0.0).sum() * 100.0 / dep_vals.size)
                writer.writerow(
                    [
                        quantile_labels[band_id],
                        "Dep",
                        f"{mean_dep:.4f}",
                        f"{std_dep:.4f}",
                        f"{ci_low_dep:.4f}",
                        f"{ci_high_dep:.4f}",
                        f"{pct_dep:.2f}",
                        int(dep_vals.size),
                    ]
                )

            mask_cons = (orig_cons["band"] == band_id)
            cons_vals = orig_cons["f1_drop"][mask_cons]
            if cons_vals.size > 0:
                mean_cons = float(cons_vals.mean())
                std_cons = float(cons_vals.std(ddof=1)) if cons_vals.size > 1 else 0.0
                ci_low_cons, ci_high_cons = bootstrap_ci(cons_vals, num_bootstrap=1000, alpha=0.05)
                pct_cons = float((cons_vals > 0.0).sum() * 100.0 / cons_vals.size)
                writer.writerow(
                    [
                        quantile_labels[band_id],
                        "Cons",
                        f"{mean_cons:.4f}",
                        f"{std_cons:.4f}",
                        f"{ci_low_cons:.4f}",
                        f"{ci_high_cons:.4f}",
                        f"{pct_cons:.2f}",
                        int(cons_vals.size),
                    ]
                )


# ---------------------------------------------------------------------------
# Table 8: Randomization sanity check
# ---------------------------------------------------------------------------

def compute_table8(
    orig_dep: Dict[str, np.ndarray],
    orig_cons: Dict[str, np.ndarray],
    rand_dep: Dict[str, np.ndarray],
    rand_cons: Dict[str, np.ndarray],
) -> None:
    """
    Computes Table 8:
      - mean/std F1 drop for randomized scores
      - Cohen's d and KS distance against original distributions
      - percentage of predictions affected
    and writes to CSV.
    """
    ensure_results_dir()
    quantile_labels = {
        3: "Top 10%",
        2: "10–30%",
        1: "30–60%",
        0: "Bottom 40%",
    }
    band_order = [3, 2, 1, 0]

    with open(TABLE8_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "quantile_band",
                "relation_type",
                "mean_f1_drop_random",
                "std_f1_drop_random",
                "cohen_d_vs_original",
                "ks_distance_vs_original",
                "percent_affected_random",
                "num_samples_random",
            ]
        )

        for band_id in band_order:
            mask_dep_orig = (orig_dep["band"] == band_id)
            mask_dep_rand = (rand_dep["band"] == band_id)
            dep_orig_vals = orig_dep["f1_drop"][mask_dep_orig]
            dep_rand_vals = rand_dep["f1_drop"][mask_dep_rand]

            if dep_rand_vals.size > 0 and dep_orig_vals.size > 1:
                mean_rand_dep = float(dep_rand_vals.mean())
                std_rand_dep = float(dep_rand_vals.std(ddof=1)) if dep_rand_vals.size > 1 else 0.0
                d_dep = cohen_d(dep_orig_vals, dep_rand_vals)
                ks_dep = ks_distance(dep_orig_vals, dep_rand_vals)
                pct_dep = float((dep_rand_vals > 0.0).sum() * 100.0 / dep_rand_vals.size)

                writer.writerow(
                    [
                        quantile_labels[band_id],
                        "Dep",
                        f"{mean_rand_dep:.4f}",
                        f"{std_rand_dep:.4f}",
                        f"{d_dep:.4f}",
                        f"{ks_dep:.4f}",
                        f"{pct_dep:.2f}",
                        int(dep_rand_vals.size),
                    ]
                )

            mask_cons_orig = (orig_cons["band"] == band_id)
            mask_cons_rand = (rand_cons["band"] == band_id)
            cons_orig_vals = orig_cons["f1_drop"][mask_cons_orig]
            cons_rand_vals = rand_cons["f1_drop"][mask_cons_rand]

            if cons_rand_vals.size > 0 and cons_orig_vals.size > 1:
                mean_rand_cons = float(cons_rand_vals.mean())
                std_rand_cons = float(cons_rand_vals.std(ddof=1)) if cons_rand_vals.size > 1 else 0.0
                d_cons = cohen_d(cons_orig_vals, cons_rand_vals)
                ks_cons = ks_distance(cons_orig_vals, cons_rand_vals)
                pct_cons = float((cons_rand_vals > 0.0).sum() * 100.0 / cons_rand_vals.size)

                writer.writerow(
                    [
                        quantile_labels[band_id],
                        "Cons",
                        f"{mean_rand_cons:.4f}",
                        f"{std_rand_cons:.4f}",
                        f"{d_cons:.4f}",
                        f"{ks_cons:.4f}",
                        f"{pct_cons:.2f}",
                        int(cons_rand_vals.size),
                    ]
                )


# ---------------------------------------------------------------------------
# Table 9: Model-specificity comparison
# ---------------------------------------------------------------------------

def compute_table9(
    spec_dep: Dict[str, np.ndarray],
    spec_cons: Dict[str, np.ndarray],
) -> None:
    """
    Computes Table 9:
      - mean and std F1 drop for SRL+Both vs baseline SRL (dep/cons)
    and writes to CSV.
    """
    ensure_results_dir()

    with open(TABLE9_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "relation_type",
                "model",
                "mean_f1_drop",
                "std_f1_drop",
                "num_samples",
            ]
        )

        for relation_type, data in [("Dep", spec_dep), ("Cons", spec_cons)]:
            model_ids = data["model"]
            drops = data["f1_drop"]

            for mid, model_name in [(0, "Baseline SRL"), (1, "SRL+Both")]:
                mask = (model_ids == mid)
                vals = drops[mask]
                if vals.size == 0:
                    continue
                mean_val = float(vals.mean())
                std_val = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                writer.writerow(
                    [
                        relation_type,
                        model_name,
                        f"{mean_val:.4f}",
                        f"{std_val:.4f}",
                        int(vals.size),
                    ]
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    orig_dep = load_band_data(DATA_ORIGINAL_DEP)
    orig_cons = load_band_data(DATA_ORIGINAL_CONS)

    rand_dep = load_band_data(DATA_RANDOM_DEP)
    rand_cons = load_band_data(DATA_RANDOM_CONS)

    spec_dep = load_model_specificity(DATA_MODEL_SPEC_DEP)
    spec_cons = load_model_specificity(DATA_MODEL_SPEC_CONS)

    compute_table7(orig_dep, orig_cons)
    compute_table8(orig_dep, orig_cons, rand_dep, rand_cons)
    compute_table9(spec_dep, spec_cons)

    print(f"Table 7 written to: {TABLE7_CSV}")
    print(f"Table 8 written to: {TABLE8_CSV}")
    print(f"Table 9 written to: {TABLE9_CSV}")


if __name__ == "__main__":
    main()
