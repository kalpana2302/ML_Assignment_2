"""
Create the train/val/test split ONCE and save it. Every question loads it via load_split().

Run once:   python split_data.py
Use later:  from split_data import load_split
"""
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from paths import P

ROLL_NO = "M24MA2005"
SEED = int("".join(ch for ch in ROLL_NO if ch.isdigit()))  # numeric characters in order -> 242005
SPLIT_FILE = P("split.npz")


def make_split():
    data = fetch_olivetti_faces()      # shuffle=False by default -> fixed, deterministic order
    X = data.data                      # (400, 4096) raw flattened pixels, floats in [0, 1]
    y = data.target                    # (400,) identity labels 0..39

    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for identity in range(40):
        idx = np.where(y == identity)[0]      # the 10 images of this identity
        perm = rng.permutation(idx)           # random order, drawn from the seeded generator
        train_idx.extend(perm[:6])
        val_idx.extend(perm[6:8])
        test_idx.extend(perm[8:10])

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))

    np.savez(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=SEED)
    return train_idx, val_idx, test_idx


def mask_center(X):
    """Zero the centred 16x16 block (rows and cols 24..39 inclusive) of flattened 64x64 images."""
    imgs = X.reshape(-1, 64, 64).copy()    # copy so the clean array is never modified
    imgs[:, 24:40, 24:40] = 0.0            # slice end is exclusive, so 24:40 covers 24..39
    return imgs.reshape(X.shape[0], -1)


def load_split():
    """Return clean train/val/test, masked test, labels and indices from the saved split."""
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    s = np.load(SPLIT_FILE)
    tr, va, te = s["train_idx"], s["val_idx"], s["test_idx"]
    X_test = X[te]
    return {
        "X_train": X[tr], "y_train": y[tr],
        "X_val": X[va], "y_val": y[va],
        "X_test": X_test, "y_test": y[te],
        "X_test_masked": mask_center(X_test),   # trained only on clean; masked is for testing only
        "train_idx": tr, "val_idx": va, "test_idx": te,
    }


def verify():
    d = load_split()
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    y = fetch_olivetti_faces().target

    assert (len(tr), len(va), len(te)) == (240, 80, 80)
    assert len(set(tr) | set(va) | set(te)) == 400                 # covers everything
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))  # no overlap
    for c in range(40):
        assert (y[tr] == c).sum() == 6 and (y[va] == c).sum() == 2 and (y[te] == c).sum() == 2
    assert d["X_train"].shape == (240, 4096)

    diff = (d["X_test"] != d["X_test_masked"]).reshape(-1, 64, 64)
    assert not diff[:, :24, :].any() and not diff[:, 40:, :].any()  # unchanged outside block
    assert (d["X_test_masked"].reshape(-1, 64, 64)[:, 24:40, 24:40] == 0).all()
    print("All checks passed. seed =", SEED)


if __name__ == "__main__":
    make_split()
    verify()
    # Reproducibility check: running again must give identical indices
    before = np.load(SPLIT_FILE)["train_idx"].copy()
    make_split()
    assert (before == np.load(SPLIT_FILE)["train_idx"]).all()
    print("Rerun gives identical split.")
