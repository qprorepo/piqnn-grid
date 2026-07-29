"""
run_shots_sweep.py
=====================================================================
Panel (b): classification accuracy vs. shot budget N_shots. Takes the
q_f=4, L=3 classifier ("used" configuration) already trained on the
real 9-bus contingency data, and re-evaluates it under GENUINE
simulated finite-sample (binomial) measurement noise on the readout
qubit -- i.e. actually drawing N_shots Bernoulli samples from the
circuit's own exact output distribution (p0, p1) and using the
empirical estimate, exactly as a real quantum processor's shot noise
would behave. Averaged over 40 independent noise realizations per
N_shots so the reported accuracy is a genuine Monte-Carlo expectation,
not a single noisy draw.
"""
import numpy as np
import csv
from vqc_model import ReuploadingVQC, train
from run_depth_sweep import load_system

Q_F, DEPTH = 4, 3
N_REPEATS = 40


def exact_probs(model, X, theta=None, phi=None):
    """Exact (p0, p1) of the readout qubit -- the true circuit output
    distribution a real device would be sampled from."""
    z = model.forward(X, theta, phi)
    p1 = np.clip((1 - z) / 2, 0.0, 1.0)   # true P(measure |1>) on qubit 0
    return 1 - p1, p1


if __name__ == "__main__":
    data = load_system("9-bus", max_n=150)
    model = ReuploadingVQC(q_f=Q_F, depth=DEPTH, n_features=data["n_features"], seed=DEPTH * 7 + 1)
    train(model, data["Xtr"], data["ytr"], iters=55, lr=0.5, verbose=True)

    p0, p1 = exact_probs(model, data["Xval"])
    y = data["yval"]
    exact_acc = np.mean((p1 > 0.5).astype(int) == y)
    print(f"exact (infinite-shot) val accuracy: {exact_acc:.4f}")

    shots_grid = np.unique(np.round(np.logspace(np.log10(50), np.log10(8000), 18)).astype(int))
    rng = np.random.default_rng(2024)
    rows = []
    for n_shots in shots_grid:
        accs = []
        for _ in range(N_REPEATS):
            counts1 = rng.binomial(n_shots, p1)
            p1_hat = counts1 / n_shots
            acc = np.mean((p1_hat > 0.5).astype(int) == y)
            accs.append(acc)
        rows.append((int(n_shots), float(np.mean(accs)), float(np.std(accs))))
        print(f"  N_shots={n_shots:5d}  mean_acc={np.mean(accs):.4f}  std={np.std(accs):.4f}")

    with open("shots_sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_shots", "mean_accuracy", "std_accuracy"])
        w.writerows(rows)
    print("Wrote shots_sweep_results.csv  (exact_acc=%.4f)" % exact_acc)
