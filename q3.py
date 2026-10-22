import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from samme import SAMME
from split_data import load_split
from q1b import CONFIG_FILE, fit_bagging, tree_predictions, vote_counts
from q1c import mean_pairwise_disagreement

SEED, K, R, BAG_100 = 242005, 40, 30, 100

d = load_split()
Xtr, ytr = d["X_train"], d["y_train"]
Xte, Xte_m, yte = d["X_test"], d["X_test_masked"], d["y_test"]
with open(CONFIG_FILE) as f:
    q1cfg = json.load(f)
with open("q1_final_B.json") as f:
    final_B = json.load(f)["final_B"]
with open("q2_config.json") as f:
    q2cfg = json.load(f)

SETTINGS = {"shallow": {"max_depth": 4, "min_samples_leaf": 4, "random_state": SEED},
            "selected": q1cfg,
            "deeper": {"max_depth": None, "min_samples_leaf": 1, "random_state": SEED}}
E = np.eye(K)[yte]                                              # one-hot targets e_i


def acc(pred):
    return float(np.mean(pred == yte))


# ---- 1. the 30 resampled training sets (within identity, with replacement) ----
rng = np.random.default_rng(SEED)
pos_by_id = [np.where(ytr == c)[0] for c in range(K)]
resamples = np.array([np.concatenate([rng.choice(p, size=len(p), replace=True) for p in pos_by_id])
                      for _ in range(R)])
np.save("q3_resamples.npy", resamples)
distinct = np.mean([len(np.unique(r)) for r in resamples]) / K

# ---- 2. refits ----
P, AC, AM = {}, {}, {}                                          # class vectors, clean acc, masked acc
LEAF, DIS = {n: [] for n in SETTINGS}, {}


def add(key, vec, a_clean, a_mask):
    P.setdefault(key, []).append(vec)
    AC.setdefault(key, []).append(a_clean)
    AM.setdefault(key, []).append(a_mask)


boost_kept = []
for r in range(R):
    X, y = Xtr[resamples[r]], ytr[resamples[r]]

    for name, cfg in SETTINGS.items():                          # single trees
        t = DecisionTreeClassifier(**cfg).fit(X, y)
        add(f"tree_{name}", np.eye(K)[t.predict(Xte)], acc(t.predict(Xte)), acc(t.predict(Xte_m)))
        _, cnt = np.unique(t.apply(X), return_counts=True)
        n_leaves = t.get_n_leaves()
        LEAF[name].append((t.get_depth(), n_leaves, len(X) / n_leaves, float(np.sum(cnt == 1)) / n_leaves))

    plan = [("bag_final", q1cfg, final_B)] + [(f"bag_{n}", c, BAG_100) for n, c in SETTINGS.items()]
    for key, cfg, B in plan:                                    # bagging (fixed order -> reproducible)
        trees, _ = fit_bagging(X, y, B, rng, cfg)
        tp = tree_predictions(trees, Xte)
        counts = vote_counts(tp)
        add(key, counts / B, acc(counts.argmax(1)), acc(vote_counts(tree_predictions(trees, Xte_m)).argmax(1)))
        DIS.setdefault(key, []).append(mean_pairwise_disagreement(tp))

    bm = SAMME(q2cfg["depth"], q2cfg["rounds"], K, SEED).fit(X, y)   # boosting
    assert len(bm.learners) > 0, f"boosting kept no learner in repetition {r}"
    q = bm.predict_proba(Xte)
    add("boosting", q, acc(q.argmax(1)), acc(bm.predict(Xte_m)))
    boost_kept.append(len(bm.learners))
    print(f"repetition {r + 1}/{R} done", flush=True)


# ---- 3. bias^2, variance, accuracy ----
def stats(key):
    Parr = np.array(P[key])                                     # (R, N, K)
    pbar = Parr.mean(axis=0)
    bias2 = float(np.mean(np.sum((pbar - E) ** 2, axis=1)))
    var = float(np.mean(np.sum((Parr - pbar) ** 2, axis=2)))
    return {"bias2": bias2, "var": var, "acc": np.mean(AC[key]), "acc_std": np.std(AC[key]),
            "macc": np.mean(AM[key]), "macc_std": np.std(AM[key])}


S = {k: stats(k) for k in P}
MAIN = [(f"single tree (selected: depth {q1cfg['max_depth']}, leaf {q1cfg['min_samples_leaf']})", "tree_selected"),
        (f"bagging (B={final_B})", "bag_final"),
        (f"boosting (depth {q2cfg['depth']}, {q2cfg['rounds']} rounds)", "boosting")]

out = [f"Resampled sets: on average {distinct * K:.2f} distinct faces per identity (of 6); "
       f"boosting learners kept per refit: min {min(boost_kept)}, max {max(boost_kept)}", ""]

out += ["=== Q3(a) bias^2, variance, clean test accuracy over 30 refits ===",
        "model,bias2,variance,acc_mean,acc_std"]
for label, key in MAIN:
    s = S[key]
    out.append(f"{label},{s['bias2']:.4f},{s['var']:.4f},{s['acc']:.4f},{s['acc_std']:.4f}")

out += ["", "=== Q3(b) single trees vs 100-tree bagging at three settings ===",
        "setting,max_depth,min_leaf,depth_reached,leaves,rows_per_leaf,single_row_leaf_frac,"
        "tree_bias2,tree_var,bag_bias2,bag_var,var_removed,var_removed_pct,tree_acc,bag_acc,acc_gain,"
        "bag_disagreement"]
for name, cfg in SETTINGS.items():
    lf = np.mean(LEAF[name], axis=0)
    t, b = S[f"tree_{name}"], S[f"bag_{name}"]
    out.append(f"{name},{cfg['max_depth']},{cfg['min_samples_leaf']},{lf[0]:.1f},{lf[1]:.1f},{lf[2]:.2f},{lf[3]:.4f},"
               f"{t['bias2']:.4f},{t['var']:.4f},{b['bias2']:.4f},{b['var']:.4f},{t['var'] - b['var']:.4f},"
               f"{100 * (t['var'] - b['var']) / t['var']:.1f},{t['acc']:.4f},{b['acc']:.4f},"
               f"{b['acc'] - t['acc']:.4f},{np.mean(DIS[f'bag_{name}']):.4f}")

out += ["", "=== Q3(d) final table (accuracies = mean over 30 refits) ===",
        "model,clean_acc,masked_acc,masked_minus_clean,bias2,variance,acc_std_clean,acc_std_masked"]
for label, key in MAIN:
    s = S[key]
    out.append(f"{label},{s['acc']:.4f},{s['macc']:.4f},{s['macc'] - s['acc']:+.4f},{s['bias2']:.4f},"
               f"{s['var']:.4f},{s['acc_std']:.4f},{s['macc_std']:.4f}")

text = "\n".join(out)
print(text)
with open("q3_log.txt", "w") as f:
    f.write(text + "\n")
for tag, start, stop in (("q3a", "=== Q3(a)", "=== Q3(b)"), ("q3b", "=== Q3(b)", "=== Q3(d)"), ("q3d", "=== Q3(d)", None)):
    s0 = text.index(start)
    s1 = text.index(stop) if stop else len(text)
    with open(f"{tag}.csv", "w") as f:
        f.write("\n".join(text[s0:s1].strip().split("\n")[1:]) + "\n")
np.savez("q3_raw.npz", **{k: np.array(v) for k, v in P.items()})