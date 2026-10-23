
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
from samme import SAMME
from split_data import load_split

SEED, K = 242005, 40
DEPTHS, ROUNDS = [1, 2, 3, 4], [10, 25, 50, 100]


def acc(y, p):
    return float(np.mean(y == p))


d = load_split()
Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]

rows, notes = [], []
for depth in DEPTHS:
    m = SAMME(depth, max(ROUNDS), K, SEED).fit(Xtr, ytr)
    if m.stop_msg:
        notes.append(f"depth {depth}: {m.stop_msg}")
    if m.clipped_rounds:
        notes.append(f"depth {depth}: eps clipped (zero error) in rounds {m.clipped_rounds}")
    for T in ROUNDS:
        rows.append({"depth": depth, "rounds": T, "rounds_used": min(T, len(m.learners)),
                     "train_acc": acc(ytr, m.predict(Xtr, T)), "val_acc": acc(yva, m.predict(Xva, T))})

best = min(rows, key=lambda r: (-r["val_acc"], r["depth"], r["rounds"]))
cfg = {"depth": best["depth"], "rounds": best["rounds"], "random_state": SEED}
with open(P("q2_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

sel = SAMME(cfg["depth"], cfg["rounds"], K, SEED).fit(Xtr, ytr)     # selected model, frozen
joblib.dump(sel, P("q2_selected_model.joblib"))

header = ["depth", "rounds", "rounds_used", "train_acc", "val_acc"]
lines = [",".join(header)] + [",".join(str(r[h]) if h in header[:3] else f"{r[h]:.4f}" for h in header)
                              for r in rows]
with open(P("q2b_grid.csv"), "w") as f:
    f.write("\n".join(lines) + "\n")

T = len(sel.learners)
wmax = sel.w_before.max(axis=1)
top10 = np.sort(sel.w_before, axis=1)[:, -10:].sum(axis=1)
text = "\n".join(["=== Q2(b) grid (clean validation accuracy used for selection) ===", *lines, "",
                  f"Selected: depth={cfg['depth']}, rounds={cfg['rounds']} (val acc {best['val_acc']:.4f}, "
                  f"train acc {best['train_acc']:.4f}); learners actually kept: {T}",
                  f"eps_t: first {sel.eps[0]:.4f}, min {sel.eps.min():.4f} (round {sel.eps.argmin() + 1}), "
                  f"max {sel.eps.max():.4f}",
                  f"alpha_t: min {sel.alphas.min():.4f}, max {sel.alphas.max():.4f} "
                  f"(round {sel.alphas.argmax() + 1}); log(K-1) = {np.log(K - 1):.4f}",
                  f"max sample weight: largest over rounds {wmax.max():.4f} (uniform = {1 / len(ytr):.4f})",
                  f"weight of 10 heaviest images: largest over rounds {top10.max():.4f}",
                  *notes])
print(text)
with open(P("q2b_log.txt"), "w") as f:
    f.write(text + "\n")

r = np.arange(1, T + 1)
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))
for ax, (y_, lab) in zip(axes.ravel(), [(sel.eps, "Weighted training error"), (sel.alphas, "Learner weight $\\alpha_t$"),
                                        (wmax, "Max sample weight"), (top10, "Weight of 10 heaviest images")]):
    ax.plot(r, y_, linewidth=0.9)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_title(lab, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(P("q2b_curves.png"), dpi=200)
fig.savefig(P("q2b_curves.pdf"))
