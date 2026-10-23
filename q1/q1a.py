"""
Q1(a): single decision-tree baseline.

Grid: max_depth in {2, 4, 8, 16, None} x min_samples_leaf in {1, 2, 4}.
Models are fit on the clean training set only. The setting is chosen on VALIDATION accuracy only.
Ties -> shallowest depth (None counts as deepest), then largest min_samples_leaf.
The chosen setting is frozen into q1_config.json and reused by Q1(b), (c), (d).

Run:  python q1a.py        (requires split.npz from split_data.py)
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "common"), str(_ROOT / "q1")]
from paths import P  # noqa: E402
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from split_data import load_split

TREE_SEED = 242005
DEPTHS = [2, 4, 8, 16, None]
MIN_LEAVES = [1, 2, 4]
CONFIG_FILE = P("q1_config.json")
RESULTS_FILE = P("q1a_grid_results.csv")
LOG_FILE = P("q1a_log.txt")


def leaf_stats(tree, X_train):
    """Number of leaves, mean training samples per leaf, fraction of leaves with exactly 1 training sample."""
    leaf_ids = tree.apply(X_train)                       # leaf id of every training image
    ids, counts = np.unique(leaf_ids, return_counts=True)  # training images per occupied leaf
    n_leaves = tree.get_n_leaves()
    mean_per_leaf = len(X_train) / n_leaves              # every train image lies in exactly one leaf
    frac_single = float(np.sum(counts == 1)) / n_leaves
    return n_leaves, mean_per_leaf, frac_single


def main():
    d = load_split()
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]

    rows = []
    for depth in DEPTHS:
        for msl in MIN_LEAVES:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl,
                                         random_state=TREE_SEED)
            clf.fit(Xtr, ytr)
            n_leaves, mean_leaf, frac1 = leaf_stats(clf, Xtr)
            rows.append({
                "max_depth": depth, "min_samples_leaf": msl,
                "train_acc": clf.score(Xtr, ytr),
                "val_acc": clf.score(Xva, yva),
                "n_leaves": n_leaves, "mean_samples_per_leaf": mean_leaf,
                "frac_single_leaves": frac1,
            })

    # Selection: validation accuracy only. Tie-break: shallowest depth (None = deepest), then largest min_samples_leaf.
    def key(r):
        depth_rank = float("inf") if r["max_depth"] is None else r["max_depth"]
        return (-r["val_acc"], depth_rank, -r["min_samples_leaf"])

    best = min(rows, key=key)

    # Freeze the setting
    config = {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"],
              "random_state": TREE_SEED}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # Refit frozen model (same params, same seed, train only) and report test accuracy for it only
    frozen = DecisionTreeClassifier(**config).fit(Xtr, ytr)
    n_leaves, mean_leaf, frac1 = leaf_stats(frozen, Xtr)
    train_acc, val_acc, test_acc = (frozen.score(Xtr, ytr), frozen.score(Xva, yva),
                                    frozen.score(Xte, yte))

    # Output: grid table (CSV + printed) and log
    header = ["max_depth", "min_samples_leaf", "train_acc", "val_acc",
              "n_leaves", "mean_samples_per_leaf", "frac_single_leaves"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) if h in ("max_depth", "min_samples_leaf") else f"{r[h]:.4f}"
                              for h in header))
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    report = [
        "=== Q1(a) grid results (fit on train, accuracies on train/val) ===",
        *lines,
        "",
        f"Selected (validation accuracy only): max_depth={config['max_depth']}, "
        f"min_samples_leaf={config['min_samples_leaf']}, random_state={TREE_SEED}",
        "=== Frozen single tree ===",
        f"train acc: {train_acc:.4f}",
        f"val acc:   {val_acc:.4f}",
        f"test acc:  {test_acc:.4f}",
        f"number of leaves: {n_leaves}",
        f"mean training samples per leaf: {mean_leaf:.4f}",
        f"fraction of leaves with exactly one training example: {frac1:.4f}",
    ]
    text = "\n".join(report)
    print(text)
    with open(LOG_FILE, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
