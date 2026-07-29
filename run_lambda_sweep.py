"""
run_lambda_sweep.py
=====================================================================
Panel (c): classification accuracy vs. energy-loss weight lambda_e,
for the combined objective

    L_total = L_bce(classification head, qubit 0)
              + lambda_e * L_mse(energy head, qubit 1; KE_clear target)

where KE_clear = 0.5 * sum_i M_i * omega_i(t_clear)^2 is the REAL
kinetic-energy term of the manuscript's Quantum Energy Functional
(eq:quantum_energy_def), computed exactly from the same swing-equation
simulation as the classification labels -- not an invented number.
"""
import csv
import time
import numpy as np
from vqc_model import ReuploadingVQC, train, bce_loss
from run_depth_sweep import FEATURES, accuracy

Q_F, DEPTH = 4, 3   # the "used" configuration from the depth sweep


def load_system_with_energy(tag, path="contingency_dataset.csv", max_n=150, seed=0):
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
    ke = np.array([float(r["ke_clear"]) for r in rows])

    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / span * 2 * np.pi - np.pi

    ke_lo, ke_hi = ke.min(), ke.max()
    ke_n = 2 * (ke - ke_lo) / (ke_hi - ke_lo) - 1     # match expZ in [-1,1]

    n_val = max(int(0.25 * len(Xn)), 10)
    return dict(Xtr=Xn[n_val:], ytr=y[n_val:], ketr=ke_n[n_val:],
                Xval=Xn[:n_val], yval=y[:n_val], keval=ke_n[:n_val],
                n_features=len(FEATURES))


def make_extra_loss(lam, ke_targets):
    def extra_loss_fn(model, X, theta, phi):
        if lam == 0.0:
            return 0.0
        _, z1 = model.forward_full(X, theta, phi, readout_qubits=(0, 1))
        return lam * np.mean((z1 - ke_targets) ** 2)
    return extra_loss_fn


if __name__ == "__main__":
    data = load_system_with_energy("9-bus", max_n=150)
    print(f"n_train={len(data['ytr'])}  n_val={len(data['yval'])}")

    lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    rows_out = []
    for lam in lambdas:
        t0 = time.time()
        model = ReuploadingVQC(q_f=Q_F, depth=DEPTH, n_features=data["n_features"], seed=101)
        extra = make_extra_loss(lam, data["ketr"])
        train(model, data["Xtr"], data["ytr"], iters=55, lr=0.5, extra_loss_fn=extra, verbose=False)
        acc = accuracy(model, data["Xval"], data["yval"])
        z1_val = model.forward_full(data["Xval"], readout_qubits=(0, 1))[1]
        mae_energy = float(np.mean(np.abs(z1_val - data["keval"])))
        dt = time.time() - t0
        print(f"  lambda_e={lam:.2f}  val_acc={acc:.4f}  energy_head_MAE={mae_energy:.4f}  ({dt:.1f}s)")
        rows_out.append((lam, acc, mae_energy))

    with open("lambda_sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_e", "val_accuracy", "energy_head_mae"])
        w.writerows(rows_out)
    print("Wrote lambda_sweep_results.csv")
