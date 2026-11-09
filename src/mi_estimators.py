"""
Estimators for the reality-contact diagnostic

    I_t = I(c_t ; x_{t+1} | x_t, a_t)

i.e. how much the coordination signal c_t tells you about the next outcome
beyond what the current state and actions already explain. Compared against
the surface link I(c_t ; x_{t+1}), which ignores what the system already knows.

Two independent estimators:
  1. cmi_binning: quantile binning + plug-in entropies (NumPy only, transparent)
  2. learned_cmi: the "two-predictor trick". Train one classifier to predict the
     (binned) next outcome from (x_t, a_t), and another from (x_t, a_t, c_t).
     The drop in held-out cross-entropy is the conditional mutual information.
     This version works for high-dimensional c_t (e.g. text embeddings).

All values are in nats. Divide by ln(2) for bits.
"""

import numpy as np


def _as2d(x):
    x = np.asarray(x)
    return x[:, None] if x.ndim == 1 else x


def discretize(x, nbins):
    """Quantile-bin each column into integer ids in [0, nbins)."""
    x = _as2d(x).astype(float)
    out = np.zeros(x.shape, dtype=np.int64)
    for j in range(x.shape[1]):
        col = x[:, j]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, nbins + 1)))
        if len(edges) <= 2:
            out[:, j] = 0
        else:
            out[:, j] = np.digitize(col, edges[1:-1], right=False)
    return out


def _entropy(int_cols):
    """Plug-in entropy (nats) with the Miller-Madow bias correction."""
    int_cols = _as2d(int_cols)
    _, counts = np.unique(int_cols, axis=0, return_counts=True)
    n = counts.sum()
    p = counts / n
    H = -(p * np.log(p)).sum()
    return H + (len(counts) - 1) / (2 * n)


def cmi_binning(C, Y, Z, nbins=6):
    """I(C ; Y | Z) = H(C,Z) + H(Y,Z) - H(Z) - H(C,Y,Z)."""
    C, Y, Z = discretize(C, nbins), discretize(Y, nbins), discretize(Z, nbins)
    val = (_entropy(np.column_stack([C, Z])) + _entropy(np.column_stack([Y, Z]))
           - _entropy(Z) - _entropy(np.column_stack([C, Y, Z])))
    return max(val, 0.0)


def mi_binning(C, Y, nbins=6):
    """Unconditional I(C ; Y): the surface link."""
    C, Y = discretize(C, nbins), discretize(Y, nbins)
    val = _entropy(C) + _entropy(Y) - _entropy(np.column_stack([C, Y]))
    return max(val, 0.0)


def learned_cmi(C, Y, Z, y_classes=8, hidden=(32,), val_frac=0.3, seed=0):
    """I(C ; Y | Z) estimated as H(Y|Z) - H(Y|Z,C) from held-out log-loss."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss

    C, Z = _as2d(C), _as2d(Z)
    y = discretize(Y, y_classes)[:, 0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]

    def heldout_ce(X):
        X = StandardScaler().fit(X[tr]).transform(X)
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300,
                            early_stopping=True, random_state=seed)
        clf.fit(X[tr], y[tr])
        return log_loss(y[va], clf.predict_proba(X[va]), labels=clf.classes_)

    return max(heldout_ce(Z) - heldout_ce(np.column_stack([Z, C])), 0.0)
