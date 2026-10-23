"""
Q1(c): ensemble diagnostics for every B, using the SAME ensembles trained in Q1(b)
(loaded from q1_bagging_models.joblib).

For every B:
  - OOB accuracy      : each training image is voted on only by trees whose bootstrap sample did
                        not contain it. Images with no OOB tree are EXCLUDED; coverage is reported.
  - Mean pairwise disagreement between trees : fraction of VALIDATION images on which two trees
                        predict different labels, averaged over all B*(B-1)/2 pairs (undefined for B=1).
  - Mean vote margin  : on the VALIDATION set, winning vote fraction minus runner-up vote fraction,
                        averaged over images.
  - Validation accuracy of the ensemble, used to select the final B
    (ties -> smallest B). The choice is saved to q1_final_B.json.

Vote ties are resolved to the lowest class index (np.argmax), as in Q1(b).

Run:  python q1c.py       (requires q1_bagging_models.joblib from q1b.py)
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "common"), str(_ROOT / "q1")]
from paths import P  # noqa: E402
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from split_data import load_split
from q1b import (B_VALUES, MODELS_FILE, N_CLASSES, tree_predictions, vote_counts, accuracy)

RESULTS_FILE = P("q1c_results.csv")
LOG_FILE = P("q1c_log.txt")
FINAL_B_FILE = P("q1_final_B.json")
FIG_FILE = "q1c_curves"


def oob_accuracy(trees, boot_idx, Xtr, ytr):
    """OOB accuracy over training images that have >= 1 OOB tree; also returns coverage (#images)."""
    B, n = boot_idx.shape
    in_bag = np.zeros((B, n), dtype=bool)
    for t in range(B):
        in_bag[t, boot_idx[t]] = True                      # images present in tree t's bootstrap sample
    oob = ~in_bag                                          # True where tree t did NOT see image i

    preds = tree_predictions(trees, Xtr)                   # (B, n)
    counts = np.zeros((n, N_CLASSES), dtype=int)
    for t in range(B):
        rows = np.where(oob[t])[0]
        counts[rows, preds[t, rows]] += 1                  # tree t votes only on its OOB images

    covered = oob.sum(axis=0) > 0                          # images with at least one OOB tree
    oob_pred = np.argmax(counts, axis=1)                   # ties -> lowest class index
    acc = float(np.mean(oob_pred[covered] == ytr[covered]))
    return acc, int(covered.sum())


def mean_pairwise_disagreement(preds):
    """preds: (B, m). Mean over all tree pairs of the fraction of images where predictions differ."""
    B = preds.shape[0]
    if B < 2:
        return float("nan")                                # no pairs exist for a single tree
    total, n_pairs = 0.0, 0
    for t in range(B - 1):
        total += (preds[t + 1:] != preds[t]).mean(axis=1).sum()
        n_pairs += B - 1 - t
    return float(total / n_pairs)


def mean_vote_margin(counts, B):
    """Mean over images of (winning vote fraction - runner-up vote fraction)."""
    frac = np.sort(counts / B, axis=1)
    return float(np.mean(frac[:, -1] - frac[:, -2]))


def main():
    saved = joblib.load(MODELS_FILE)
    models = saved["models"]
    d = load_split()
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]

    rows = []
    for B in B_VALUES:
        trees, boot_idx = models[B]["trees"], models[B]["boot_idx"]
        oob_acc, coverage = oob_accuracy(trees, boot_idx, Xtr, ytr)

        val_preds = tree_predictions(trees, Xva)           # (B, 80)
        counts = vote_counts(val_preds)
        rows.append({
            "B": B,
            "oob_acc": oob_acc,
            "oob_coverage": coverage,
            "disagreement_val": mean_pairwise_disagreement(val_preds),
            "vote_margin_val": mean_vote_margin(counts, B),
            "val_acc": accuracy(yva, np.argmax(counts, axis=1)),
        })

    # Select final B on validation accuracy only; ties -> smallest B
    best_val = max(r["val_acc"] for r in rows)
    final_B = min(r["B"] for r in rows if r["val_acc"] == best_val)
    with open(FINAL_B_FILE, "w") as f:
        json.dump({"final_B": final_B, "val_acc": best_val}, f, indent=2)

    # Table and log
    header = ["B", "oob_acc", "oob_coverage", "disagreement_val", "vote_margin_val", "val_acc"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(
            str(r[h]) if h in ("B", "oob_coverage") else f"{r[h]:.4f}" for h in header))
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    report = ["=== Q1(c) diagnostics (disagreement and margin on validation set; "
              "OOB on training images, uncovered images excluded) ===",
              *lines, "",
              f"Final B (highest validation accuracy {best_val:.4f}, ties -> smallest B): {final_B}"]
    text = "\n".join(report)
    print(text)
    with open(LOG_FILE, "w") as f:
        f.write(text + "\n")

    # Plot: three panels, log B axis
    Bs = [r["B"] for r in rows]
    panels = [("oob_acc", "OOB accuracy"),
              ("disagreement_val", "Mean pairwise disagreement (val)"),
              ("vote_margin_val", "Mean vote margin (val)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (key, label) in zip(axes, panels):
        ax.plot(Bs, [r[key] for r in rows], marker="o")
        ax.axvline(final_B, linestyle="--", linewidth=0.8, color="gray")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(Bs))
        ax.xaxis.set_major_formatter(FixedFormatter([str(b) for b in Bs]))
        ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("B (number of trees)")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P(FIG_FILE + ".png"), dpi=200)
    fig.savefig(P(FIG_FILE + ".pdf"))


if __name__ == "__main__":
    main()
