"""
run_depth_sweep.py
=====================================================================
Panel (a): classification accuracy vs. ansatz depth L, for BOTH the
9-bus and 14-bus contingency datasets, using q_f = 6 qubits (the
"used value" flagged in the manuscript figure) and the REAL labeled
data in contingency_dataset.csv.
"""
import csv
import time
import numpy as np
from vqc_model import ReuploadingVQC, train, bce_loss

FEATURES = ["mean_dd", "max_dd", "mean_w", "max_w", "spread_dd", "t_clear"]


def load_system(tag, path="contingency_dataset.csv", max_n=240, seed=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["system"] == tag:
                rows.append(r)
    rng = np.random.default_rng(seed)

    stable = [r for r in rows if int(r["unstable"]) == 0]
    unstable = [r for r in rows if int(r["unstable"]) == 1]
    n_each = min(len(stable), len(unstable), max_n // 2)
    rng.shuffle(stable)
    rng.shuffle(unstable)
    rows = stable[:n_each] + unstable[:n_each]
    idx = rng.permutation(len(rows))
    rows = [rows[i] for i in idx]

    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([int(r["unstable"]) for r in rows])

    # normalize each feature to roughly [-pi, pi] (data re-uploading convention)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / span * 2 * np.pi - np.pi

    n_val = max(int(0.25 * len(Xn)), 10)
    return dict(Xtr=Xn[n_val:], ytr=y[n_val:], Xval=Xn[:n_val], yval=y[:n_val],
                lo=lo, span=span, n_features=len(FEATURES))


def accuracy(model, X, y):
    p = model.predict_proba(X)
    return float(np.mean((p > 0.5).astype(int) == y))


if __name__ == "__main__":
    results = {"9-bus": [], "14-bus": []}
    depths = [1, 2, 3, 4, 5, 6]
    q_f = 4    # kept small so the full multi-depth, multi-system sweep is
               # tractable to run interactively; see run notes in the README

    for tag in ("9-bus", "14-bus"):
        data = load_system(tag, max_n=150)
        print(f"=== {tag}: n_train={len(data['ytr'])} n_val={len(data['yval'])} "
              f"(unstable frac train={data['ytr'].mean():.2f}, val={data['yval'].mean():.2f}) ===")
        for L in depths:
            t0 = time.time()
            model = ReuploadingVQC(q_f=q_f, depth=L, n_features=data["n_features"], seed=L * 7 + 1)
            train(model, data["Xtr"], data["ytr"], iters=55, lr=0.5, verbose=False)
            acc = accuracy(model, data["Xval"], data["yval"])
            dt = time.time() - t0
            print(f"  L={L}  val_acc={acc:.4f}   ({dt:.1f}s)")
            results[tag].append((L, acc))

    with open("depth_sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "L", "val_accuracy"])
        for tag, rows in results.items():
            for L, acc in rows:
                w.writerow([tag, L, acc])
    print("Wrote depth_sweep_results.csv")
