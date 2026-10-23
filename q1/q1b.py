import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "common"), str(_ROOT / "q1")]
from paths import P  # noqa: E402
import json
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from split_data import load_split

SEED = 242005
N_CLASSES = 40
B_VALUES = [1, 5, 10, 25, 50, 100, 200]
CONFIG_FILE = P("q1_config.json")
MODELS_FILE = P("q1_bagging_models.joblib")
RESULTS_FILE = P("q1b_results.csv")
LOG_FILE = P("q1b_log.txt")


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)      


def fit_bagging(X, y, B, rng, config):
    """Train B trees, each on a bootstrap sample of size len(X). Returns (trees, boot_idx[B, n])."""
    n = len(X)
    trees, boot_idx = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)                   # sample n indices with replacement
        tree = DecisionTreeClassifier(**config).fit(X[idx], y[idx])
        trees.append(tree)
        boot_idx.append(idx)
    return trees, np.array(boot_idx)


def tree_predictions(trees, X):
    """Array of shape (B, n_samples): each tree's predicted class label."""
    return np.array([t.predict(X) for t in trees])


def vote_counts(preds, n_classes=N_CLASSES):
    """preds: (B, n_samples) labels -> counts: (n_samples, n_classes) number of votes per class."""
    B, n = preds.shape
    counts = np.zeros((n, n_classes), dtype=int)
    for b in range(B):
        counts[np.arange(n), preds[b]] += 1
    return counts


def bagging_predict(trees, X):
    counts = vote_counts(tree_predictions(trees, X))
    return np.argmax(counts, axis=1)                       


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def main():
    config = load_config()
    d = load_split()
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    n = len(Xtr)

    theory = n * (1.0 - (1.0 - 1.0 / n) ** n)            

    rng = np.random.default_rng(SEED)
    models, rows = {}, []
    for B in B_VALUES:
        trees, boot_idx = fit_bagging(Xtr, ytr, B, rng, config)
        models[B] = {"trees": trees, "boot_idx": boot_idx}

        distinct = np.array([len(np.unique(row)) for row in boot_idx])
        rows.append({
            "B": B,
            "train_acc": accuracy(ytr, bagging_predict(trees, Xtr)),
            "val_acc": accuracy(yva, bagging_predict(trees, Xva)),
            "test_acc": accuracy(yte, bagging_predict(trees, Xte)),
            "avg_distinct": float(distinct.mean()),
            "theory_distinct": theory,
            "abs_diff": float(abs(distinct.mean() - theory)),
        })

    joblib.dump({"config": config, "seed": SEED, "models": models}, MODELS_FILE)

    header = ["B", "train_acc", "val_acc", "test_acc", "avg_distinct", "theory_distinct", "abs_diff"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r["B"]) if h == "B" else f"{r[h]:.4f}" for h in header))
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    report = [
        f"Frozen tree setting: {config}",
        f"Bootstrap size n = {n}; theoretical distinct per sample = n*(1-(1-1/n)^n) = {theory:.4f}",
        f"(limit n*(1-1/e) = {n * (1 - np.exp(-1)):.4f})",
        "",
        "=== Q1(b) bagging results ===",
        *lines,
    ]
    text = "\n".join(report)
    print(text)
    with open(LOG_FILE, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
