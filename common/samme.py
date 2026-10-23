"""
SAMME multiclass AdaBoost written from scratch (used for BOTH the toy example and the faces).

Weak learner: DecisionTreeClassifier(max_depth=depth, random_state=random_state) fit with sample weights.
Each round t:
    eps_t   = weighted error of h_t (weights sum to 1)
    alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
    w_i    <- w_i * exp(alpha_t * [h_t(x_i) != y_i]),  then renormalised to sum 1
Prediction: S_k(x) = sum_t alpha_t * [h_t(x) = k]; class = argmax_k S_k (ties -> lowest class index);
confidence q_k(x) = S_k(x) / sum_t alpha_t.

Weak learner not better than random multiclass guessing (eps >= 1 - 1/K): the learner is DISCARDED and
boosting STOPS (message stored in stop_msg).
eps = 0: eps is clipped to eps_floor so alpha stays finite (the round is stored in clipped_rounds).
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SAMME:
    def __init__(self, depth, n_rounds, n_classes, random_state=242005, eps_floor=1e-10):
        self.depth, self.n_rounds, self.K = depth, n_rounds, n_classes
        self.random_state, self.eps_floor = random_state, eps_floor

    def fit(self, X, y):
        n, K = len(X), self.K
        w = np.full(n, 1.0 / n)                              # initial weights 1/n
        learners, alphas, eps_l, w_b, w_a, preds = [], [], [], [], [], []
        self.stop_msg, self.clipped_rounds = None, []
        for t in range(self.n_rounds):
            tree = DecisionTreeClassifier(max_depth=self.depth, random_state=self.random_state)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X)
            wrong = pred != y
            eps = float(w[wrong].sum())                      # weights sum to 1 -> weighted error
            if eps >= 1 - 1.0 / K:                           # not better than random multiclass guessing
                self.stop_msg = (f"round {t + 1}: weighted error {eps:.4f} >= 1-1/K = "
                                 f"{1 - 1.0 / K:.4f}; learner discarded, boosting stopped")
                break
            if eps < self.eps_floor:                         # perfect learner: avoid infinite alpha
                eps = self.eps_floor
                self.clipped_rounds.append(t + 1)
            alpha = np.log((1 - eps) / eps) + np.log(K - 1)
            w_new = w * np.exp(alpha * wrong)                # wrong samples multiplied by e^alpha
            w_new /= w_new.sum()                             # renormalise
            learners.append(tree); alphas.append(alpha); eps_l.append(eps)
            w_b.append(w.copy()); w_a.append(w_new.copy()); preds.append(pred)
            w = w_new
        self.learners = learners
        self.alphas, self.eps = np.array(alphas), np.array(eps_l)
        self.w_before = np.array(w_b)       # (rounds, n): weights used to fit round t
        self.w_after = np.array(w_a)        # (rounds, n): normalised weights after the round-t update
        self.train_pred = np.array(preds)   # (rounds, n): learner predictions on the training set
        return self

    def scores(self, X, n_learners=None):
        """S_k(x) using the first n_learners learners (all by default); also returns sum of alphas used."""
        T = len(self.learners) if n_learners is None else min(n_learners, len(self.learners))
        S = np.zeros((len(X), self.K))
        for tree, a in zip(self.learners[:T], self.alphas[:T]):
            S[np.arange(len(X)), tree.predict(X)] += a
        return S, float(self.alphas[:T].sum())

    def predict_proba(self, X, n_learners=None):
        S, total = self.scores(X, n_learners)
        return S / total                                     # q_k(x)

    def predict(self, X, n_learners=None):
        S, _ = self.scores(X, n_learners)
        return np.argmax(S, axis=1)                          # ties -> lowest class index
