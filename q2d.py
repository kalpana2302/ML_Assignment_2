
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from split_data import load_split
from q1b import MODELS_FILE, CONFIG_FILE, tree_predictions, vote_counts, accuracy

d = load_split()
Xtr, ytr, yte = d["X_train"], d["y_train"], d["y_test"]
sets = {"clean": d["X_test"], "masked": d["X_test_masked"]}
with open(CONFIG_FILE) as f:
    q1cfg = json.load(f)
with open("q1_final_B.json") as f:
    B = json.load(f)["final_B"]

tree = DecisionTreeClassifier(**q1cfg).fit(Xtr, ytr)               # frozen tree
bag_trees = joblib.load(MODELS_FILE)["models"][B]["trees"]          # final bagged model
boost = joblib.load("q2_selected_model.joblib")                     # selected boosting model

# pixels of the masked block (rows and cols 24..39) in flattened 64x64 images
rr, cc = np.divmod(np.arange(4096), 64)
MASK_PIX = (rr >= 24) & (rr < 40) & (cc >= 24) & (cc < 40)


def uses_mask(t, x):
    """True if the decision path of tree t for image x tests at least one pixel inside the masked block."""
    feats = t.tree_.feature[t.decision_path(x.reshape(1, -1)).indices]
    return bool(MASK_PIX[feats[feats >= 0]].any())


def top3(row):
    o = np.argsort(-row, kind="stable")[:3]
    return " ".join(f"{int(c)}:{row[c]:.3f}" for c in o)


res, summary = {}, ["test_set,model,accuracy,mean_confidence"]
for name, X in sets.items():
    p = tree.predict_proba(X)
    counts = vote_counts(tree_predictions(bag_trees, X))
    q = boost.predict_proba(X)
    res[name] = {"tree": (tree.classes_[p.argmax(1)], p.max(1)),
                 "bagging": (counts.argmax(1), counts.max(1) / B),
                 "boosting": (q.argmax(1), q.max(1)),
                 "counts": counts, "q": q}
    for m in ("tree", "bagging", "boosting"):
        pred, conf = res[name][m]
        summary.append(f"{name},{m},{accuracy(yte, pred):.4f},{conf.mean():.4f}")

log = [f"Final bagged ensemble B = {B}; boosting depth {boost.depth}, {len(boost.learners)} learners", *summary, ""]
for m in ("tree", "bagging", "boosting"):
    log.append(f"{m}: prediction changes on {int((res['clean'][m][0] != res['masked'][m][0]).sum())} "
               f"of {len(yte)} test faces")

bag_ch = res["clean"]["bagging"][0] != res["masked"]["bagging"][0]
boo_ch = res["clean"]["boosting"][0] != res["masked"]["boosting"][0]
cand = np.where(bag_ch ^ boo_ch)[0]
log.append(f"faces where exactly one of bagging/boosting changes its prediction: {len(cand)}")

if len(cand):
    p = int(cand[0])
    xc, xm = sets["clean"][p], sets["masked"][p]
    log += ["", f"Differing face: test position {p} (dataset index {d['test_idx'][p]}), true label {yte[p]}"]
    for m in ("tree", "bagging", "boosting"):
        log.append(f"  {m}: clean -> {res['clean'][m][0][p]} (conf {res['clean'][m][1][p]:.3f}), "
                   f"masked -> {res['masked'][m][0][p]} (conf {res['masked'][m][1][p]:.3f})")

    # --- bagging: tree disagreement on this face ---
    tc = tree_predictions(bag_trees, xc.reshape(1, -1))[:, 0]
    tm = tree_predictions(bag_trees, xm.reshape(1, -1))[:, 0]
    for tag, c in (("clean", res["clean"]["counts"][p]), ("masked", res["masked"]["counts"][p])):
        pairs_agree = (c * (c - 1)).sum() / (B * (B - 1))
        log.append(f"  bagging {tag}: top-3 votes [{' '.join(f'{int(k)}:{int(c[k])}' for k in np.argsort(-c, kind='stable')[:3])}], "
                   f"pairwise tree disagreement on this face = {1 - pairs_agree:.3f}")
    log.append(f"  bagging: {int((tc != tm).sum())} of {B} trees change their vote after masking; "
               f"{sum(uses_mask(t, xc) for t in bag_trees)} of {B} trees test a masked pixel on this face's path")

    # --- boosting: alpha-weighted learners on this face ---
    bc = np.array([t.predict(xc.reshape(1, -1))[0] for t in boost.learners])
    bm = np.array([t.predict(xm.reshape(1, -1))[0] for t in boost.learners])
    a = boost.alphas
    um = np.array([uses_mask(t, xc) for t in boost.learners])
    log += [f"  boosting clean q top-3 [{top3(res['clean']['q'][p])}] | masked q top-3 [{top3(res['masked']['q'][p])}]",
            f"  boosting: {int((bc != bm).sum())} of {len(a)} learners change their vote; they carry "
            f"{a[bc != bm].sum() / a.sum():.3f} of the total alpha",
            f"  boosting: {int(um.sum())} of {len(a)} learners test a masked pixel on this face's path; "
            f"alpha share {a[um].sum() / a.sum():.3f}"]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for ax, img, tag in zip(axes, (xc, xm), ("clean", "masked")):
        ax.imshow(img.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{tag}: bag {res[tag]['bagging'][0][p]}, boost {res[tag]['boosting'][0][p]}", fontsize=7)
    fig.suptitle(f"true {yte[p]}", fontsize=8)
    fig.tight_layout()
    fig.savefig("q2d_face.png", dpi=200)
    fig.savefig("q2d_face.pdf")

with open("q2d_summary.csv", "w") as f:
    f.write("\n".join(summary) + "\n")
text = "\n".join(["=== Q2(d) ===", *log])
print(text)
with open("q2d_log.txt", "w") as f:
    f.write(text + "\n")