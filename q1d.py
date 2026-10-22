
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from split_data import load_split
from q1b import MODELS_FILE, CONFIG_FILE, tree_predictions, vote_counts, accuracy

FINAL_B_FILE = "q1_final_B.json"
SUMMARY_FILE = "q1d_summary.csv"
CASES_FILE = "q1d_cases.csv"
LOG_FILE = "q1d_log.txt"
N_PER_CATEGORY = 2


def evaluate(tree, bag_trees, X, y):
    """Predictions, confidences and vote counts for the tree and the bagged model on one test set."""
    B = len(bag_trees)
    proba = tree.predict_proba(X)
    tree_pred = tree.classes_[np.argmax(proba, axis=1)]
    tree_conf = proba.max(axis=1)

    counts = vote_counts(tree_predictions(bag_trees, X))
    bag_pred = np.argmax(counts, axis=1)                  # ties -> lowest class index
    bag_conf = counts.max(axis=1) / B                     # winning vote fraction
    return {"tree_pred": tree_pred, "tree_conf": tree_conf,
            "bag_pred": bag_pred, "bag_conf": bag_conf, "counts": counts,
            "tree_acc": accuracy(y, tree_pred), "bag_acc": accuracy(y, bag_pred)}


def top3_text(counts_row):
    order = np.argsort(-counts_row, kind="stable")[:3]    # ties -> lowest class index first
    return " ".join(f"{int(c)}:{int(counts_row[c])}" for c in order)


def main():
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    with open(FINAL_B_FILE) as f:
        final_B = json.load(f)["final_B"]
    bag_trees = joblib.load(MODELS_FILE)["models"][final_B]["trees"]

    d = load_split()
    tree = DecisionTreeClassifier(**config).fit(d["X_train"], d["y_train"])   # frozen tree from Q1(a)
    y = d["y_test"]
    test_idx = d["test_idx"]
    sets = {"clean": d["X_test"], "masked": d["X_test_masked"]}

    summary_lines = ["test_set,model,accuracy,mean_confidence"]
    case_lines = ["test_set,category,test_pos,dataset_idx,true,tree_pred,tree_conf,bag_pred,"
                  "bag_top3_votes,bag_votes_for_true"]
    log = [f"Frozen tree setting: {config}", f"Final bagged ensemble size B = {final_B}",
           "Case selection rule: lowest positions in the test set", ""]

    for name, X in sets.items():
        r = evaluate(tree, bag_trees, X, y)
        summary_lines.append(f"{name},tree,{r['tree_acc']:.4f},{r['tree_conf'].mean():.4f}")
        summary_lines.append(f"{name},bagging,{r['bag_acc']:.4f},{r['bag_conf'].mean():.4f}")

        tree_ok = r["tree_pred"] == y
        bag_ok = r["bag_pred"] == y
        corrected = np.where(~tree_ok & bag_ok)[0]
        both_wrong = np.where(~tree_ok & ~bag_ok)[0]
        tree_only = np.where(tree_ok & ~bag_ok)[0]
        log += [f"--- {name} test set ---",
                f"tree acc {r['tree_acc']:.4f}, mean conf {r['tree_conf'].mean():.4f}",
                f"bagging acc {r['bag_acc']:.4f}, mean conf {r['bag_conf'].mean():.4f}",
                f"bagging corrects tree: {len(corrected)} faces | both wrong: {len(both_wrong)} faces | "
                f"tree right but bagging wrong: {len(tree_only)} faces | both right: "
                f"{int(np.sum(tree_ok & bag_ok))} faces"]

        chosen = [("bagging_corrects_tree", p) for p in corrected[:N_PER_CATEGORY]] + \
                 [("both_wrong", p) for p in both_wrong[:N_PER_CATEGORY]]
        if len(corrected) < N_PER_CATEGORY or len(both_wrong) < N_PER_CATEGORY:
            log.append(f"WARNING: fewer than {N_PER_CATEGORY} faces available in a category")

        fig, axes = plt.subplots(1, max(len(chosen), 1), figsize=(2.1 * max(len(chosen), 1), 2.4))
        axes = np.atleast_1d(axes)
        for ax, (cat, p) in zip(axes, chosen):
            top3 = top3_text(r["counts"][p])
            true_votes = int(r["counts"][p, y[p]])
            case_lines.append(
                f"{name},{cat},{p},{test_idx[p]},{y[p]},{r['tree_pred'][p]},{r['tree_conf'][p]:.3f},"
                f"{r['bag_pred'][p]},\"{top3}\",{true_votes}")
            log.append(f"[{cat}] test_pos={p} (dataset idx {test_idx[p]}) true={y[p]} "
                       f"tree={r['tree_pred'][p]} (conf {r['tree_conf'][p]:.3f}) "
                       f"bagging={r['bag_pred'][p]} top3 votes [{top3}] true-class votes={true_votes}/{final_B}")
            ax.imshow(X[p].reshape(64, 64), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            label = "Corrected" if cat == "bagging_corrects_tree" else "Both wrong"
            ax.set_title(f"{label}\ntrue {y[p]}, tree {r['tree_pred'][p]}, bag {r['bag_pred'][p]}",
                         fontsize=7)
        for ax in axes[len(chosen):]:
            ax.axis("off")
        fig.suptitle(f"{name} test set (B = {final_B})", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"q1d_cases_{name}.png", dpi=200)
        fig.savefig(f"q1d_cases_{name}.pdf")
        plt.close(fig)
        log.append("")

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(CASES_FILE, "w") as f:
        f.write("\n".join(case_lines) + "\n")

    text = "\n".join(["=== Q1(d) summary ===", *summary_lines, "", *log])
    print(text)
    with open(LOG_FILE, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()