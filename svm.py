"""
Support Vector Machine — from-scratch implementation (linear soft-margin SVM)
using sub-gradient descent on the primal hinge-loss formulation.

Trained and evaluated on the w8a binary classification dataset.
"""

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# ── SVM class ────────────────────────────────────────────────────────────────

class LinearSVM:
    """Soft-margin linear SVM optimised via sub-gradient descent.

    Minimises:
        (1/n) * sum max(0, 1 - y_i * w^T x_i) + (lambda/2) * ||w||^2

    Attributes
    ----------
    coef_   : weight vector (includes intercept as first element)
    C       : regularisation parameter (inverse of lambda)
    lr      : initial learning rate
    """

    def __init__(self, C=1.0, lr=1e-2):
        self.C = C
        self.lr = lr
        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X, y, max_iter=1000):
        X = np.c_[np.ones(X.shape[0]), X]
        n, d = X.shape
        w = np.zeros(d)
        lambd = 1.0 / self.C

        for t in range(1, max_iter + 1):
            lr_t = self.lr / np.sqrt(t)  # decaying learning rate

            margins = y * (X @ w)
            misclassified = margins < 1  # hinge active

            # Sub-gradient
            grad = lambd * w - (1 / n) * X[misclassified].T @ y[misclassified]

            w -= lr_t * grad

            # Record loss every 50 iterations
            if t % 50 == 0 or t == 1:
                hinge = np.maximum(0, 1 - margins)
                loss = np.mean(hinge) + (lambd / 2) * np.dot(w, w)
                self.loss_history_.append(loss)

        self.coef_ = w

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return np.sign(X @ self.coef_)


# ── Data loading ─────────────────────────────────────────────────────────────

data = pd.read_csv("w8a.csv", sep=";", header=None)
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=32
)

# ── Train from-scratch SVM ───────────────────────────────────────────────────

svm = LinearSVM(C=1.0, lr=1e-2)
svm.fit(X_train, y_train, max_iter=2000)
y_pred = svm.predict(X_test)

print("=== From-scratch Linear SVM ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")

# ── Loss curve ───────────────────────────────────────────────────────────────

plt.figure()
plt.plot(svm.loss_history_)
plt.xlabel("Checkpoint (every 50 iters)")
plt.ylabel("Hinge loss + regularisation")
plt.title("Training loss — Linear SVM (sub-gradient descent)")
plt.tight_layout()
plt.savefig("svm_loss.png", dpi=150)
plt.show()

# ── Comparison with sklearn ──────────────────────────────────────────────────

clf = SVC(kernel="linear", C=1.0)
clf.fit(X_train, y_train)
y_pred_sk = clf.predict(X_test)

print("\n=== sklearn SVC (linear kernel) ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_sk):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_sk):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_sk):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred_sk):.4f}")
