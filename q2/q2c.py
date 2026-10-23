import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "common"), str(_ROOT / "q1")]
from paths import P  # noqa: E402
import json
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from samme import SAMME
from split_data import load_split
from q1b import MODELS_FILE, CONFIG_FILE, fit_bagging, bagging_predict, accuracy

SEED, K = 242005, 40
LEARNERS = [25, 50, 100]

d = load_split()
Xtr, ytr, Xva, yva, Xte, yte = (d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"])
with open(P("q2_config.json")) as f:
    cfg = json.load(f)
with open(CONFIG_FILE) as f:
    q1cfg = json.load(f)
saved = joblib.load(MODELS_FILE)["models"]

rows = ["method,learners,learners_used,train_acc,val_acc,test_acc,train_seconds"]
for T in LEARNERS:
    t0 = time.perf_counter()
    bag_fit = fit_bagging(Xtr, ytr, T, np.random.default_rng(SEED), q1cfg)   # only for timing
    t_bag = time.perf_counter() - t0
    trees = saved[T]["trees"]                                                # Q1 ensemble for accuracies
    rows.append(f"bagging,{T},{T},{accuracy(ytr, bagging_predict(trees, Xtr)):.4f},"
                f"{accuracy(yva, bagging_predict(trees, Xva)):.4f},"
                f"{accuracy(yte, bagging_predict(trees, Xte)):.4f},{t_bag:.2f}")

    t0 = time.perf_counter()
    bm = SAMME(cfg["depth"], T, K, SEED).fit(Xtr, ytr)
    t_boost = time.perf_counter() - t0
    rows.append(f"boosting(depth {cfg['depth']}),{T},{len(bm.learners)},"
                f"{accuracy(ytr, bm.predict(Xtr)):.4f},{accuracy(yva, bm.predict(Xva)):.4f},"
                f"{accuracy(yte, bm.predict(Xte)):.4f},{t_boost:.2f}")

# ---- heavily weighted training face (selected boosting model) ----
sel = joblib.load(P("q2_selected_model.joblib"))
i = int(np.argmax(sel.w_after[-1]))                       # largest final weight
pred_i = sel.train_pred[:, i]
correct = pred_i == ytr[i]
wb, wa = sel.w_before[:, i], sel.w_after[:, i]
T = len(sel.learners)
face = [f"Heavy face: training position {i} (dataset index {d['train_idx'][i]}), true label {ytr[i]}",
        f"final weight {wa[-1]:.4f} = {wa[-1] * len(ytr):.1f} x uniform (1/{len(ytr)})",
        f"misclassified in {int((~correct).sum())} of {T} rounds",
        f"predicted label, rounds 1-20: {pred_i[:20].tolist()}",
        f"most frequent wrong label: {np.bincount(pred_i[~correct]).argmax() if (~correct).any() else 'none'}",
        "rounds where it was classified correctly (weight before -> after the update):"]
for t in np.where(correct)[0][:10]:
    face.append(f"  round {t + 1}: {wb[t]:.5f} -> {wa[t]:.5f}  (alpha_t = {sel.alphas[t]:.3f})")

text = "\n".join(["=== Q2(c) boosting vs bagging ===", *rows, "", *face])
print(text)
with open(P("q2c_results.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
with open(P("q2c_log.txt"), "w") as f:
    f.write(text + "\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1, 2]})
a1.imshow(Xtr[i].reshape(64, 64), cmap="gray", vmin=0, vmax=1)
a1.axis("off")
a1.set_title(f"train face, true {ytr[i]}", fontsize=8)
r = np.arange(1, T + 1)
a2.semilogy(r, wa, linewidth=0.9)
a2.plot(r[correct], wa[correct], "ro", markersize=3, label="classified correctly")
a2.set_xlabel("Round", fontsize=8)
a2.set_ylabel("Weight after round", fontsize=8)
a2.legend(fontsize=7)
a2.tick_params(labelsize=7)
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(P("q2c_heavy_face.png"), dpi=200)
fig.savefig(P("q2c_heavy_face.pdf"))
