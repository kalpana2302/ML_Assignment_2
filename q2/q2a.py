"""
Q2(a): one complete SAMME round on a 3-class toy example with 10 samples, using the SAME
SAMME class (samme.py) that is used on the faces.

Toy data: one feature x = 1..10, labels y (3 classes), weak learner = decision stump (depth 1).
Run:  python q2a.py
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "common"), str(_ROOT / "q1")]
from paths import P  # noqa: E402
import numpy as np
from samme import SAMME

X = np.arange(1, 11, dtype=float).reshape(-1, 1)
y = np.array([0, 0, 1, 0, 1, 1, 2, 1, 2, 2])
K = 3

m = SAMME(depth=1, n_rounds=1, n_classes=K).fit(X, y)

w0, w1 = m.w_before[0], m.w_after[0]
pred = m.train_pred[0]
wrong = pred != y
eps, alpha = m.eps[0], m.alphas[0]
unnorm = w0 * np.exp(alpha * wrong)
stump = m.learners[0].tree_

lines = [f"Stump splits on x <= {stump.threshold[0]:.1f}",
         f"{'sample':>6} {'x':>3} {'label':>5} {'h1':>3} {'wrong':>5} "
         f"{'w_before':>9} {'w*e^(a*wrong)':>14} {'w_after':>8}"]
for i in range(len(y)):
    lines.append(f"{i + 1:>6} {int(X[i, 0]):>3} {y[i]:>5} {pred[i]:>3} {str(bool(wrong[i])):>5} "
                 f"{w0[i]:>9.4f} {unnorm[i]:>14.4f} {w1[i]:>8.4f}")
lines += ["",
          f"weighted error eps_1 = {eps:.4f}   (random-guess bound 1-1/K = {1 - 1 / K:.4f})",
          f"alpha_1 = log((1-eps)/eps) + log(K-1) = {alpha:.4f}   (e^alpha = {np.exp(alpha):.4f})",
          f"normaliser (sum of w*e^(alpha*wrong)) = {unnorm.sum():.4f}",
          f"sum of updated weights = {w1.sum():.4f}"]
text = "\n".join(lines)
print(text)
with open(P("q2a_log.txt"), "w") as f:
    f.write(text + "\n")
