"""
vqc_model.py
=====================================================================
A genuine (if deliberately small-scale) variational quantum circuit,
simulated exactly via a batched statevector representation, matching
the manuscript's architecture description:

  - data re-uploading encoding: R_y(w * x_k) on qubit k, reapplied at
    every layer (circuit/encoding.py)
  - a GTA-style trainable + entangling layer: R_y(theta) R_z(phi) on
    every qubit, followed by a ring of CNOTs (circuit/gta_ansatz.py)
  - an affine/sigmoid readout on a single designated qubit
    (circuit/readout.py)

Gradients are computed with the manuscript's OWN two-term
parameter-shift rule (eq:param_shift), not backprop-through-simulator
-- so the sensitivity-sweep results below are produced by literally
the same training rule the manuscript specifies, applied at small
enough scale (few dozen parameters, a few hundred samples, several
dozen iterations) to be run interactively and reproducibly here.
"""
import numpy as np

RNG = np.random.default_rng(0)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def _rz(phi):
    e_m = np.exp(-1j * phi / 2)
    e_p = np.exp(1j * phi / 2)
    return np.array([[e_m, 0], [0, e_p]])


def _apply_1q(state, gate, qubit, n_q):
    """state: (batch, 2,2,...,2) with n_q trailing qubit axes.
    gate: (2,2) or (batch,2,2) unitary. Applies in-place-equivalent, returns
    new state with `gate` acting on axis `qubit+1` (axis 0 is batch)."""
    axis = qubit + 1
    state_m = np.moveaxis(state, axis, -1)          # (..., 2) on last axis
    if gate.ndim == 2:
        out = state_m @ gate.T
    else:  # batched gate (batch,2,2), state_m is (batch, ..., 2)
        out = np.einsum('b...j,bij->b...i', state_m, gate)
    return np.moveaxis(out, -1, axis)


def _apply_cnot(state, control, target, n_q):
    """Ring-entangler CNOT via index-roll on the target axis, conditioned
    on the control axis being |1>."""
    c_axis, t_axis = control + 1, target + 1
    state = np.moveaxis(state, [c_axis, t_axis], [1, 2])
    out = state.copy()
    out[:, 1, :, ...] = state[:, 1, ::-1, ...]     # flip target bit when control=1
    inv = [0] * (state.ndim)
    inv[1], inv[2] = c_axis, t_axis
    # invert the moveaxis
    orig_axes = list(range(state.ndim))
    new_pos = [1, 2] + [a for a in orig_axes if a not in (1, 2)]
    # easier: just moveaxis back explicitly
    out = np.moveaxis(out, [1, 2], [c_axis, t_axis])
    return out


class ReuploadingVQC:
    """q_f qubits, depth L. Params: theta[L,q_f], phi[L,q_f] (trainable
    single-qubit rotations); encoding scale w[q_f] (fixed, not trained,
    matching the manuscript's fixed-frequency re-uploading spectrum)."""

    def __init__(self, q_f, depth, n_features, seed=0):
        self.q_f = q_f
        self.depth = depth
        self.n_features = n_features
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 0.3, size=(depth, q_f))
        self.phi = rng.normal(0, 0.3, size=(depth, q_f))
        # cycle features across the q_f qubits if n_features != q_f
        self.feat_map = [k % n_features for k in range(q_f)]
        self.w = np.ones(q_f)   # fixed encoding frequency, standard choice

    def get_params(self):
        return np.concatenate([self.theta.ravel(), self.phi.ravel()])

    def set_params(self, vec):
        n = self.depth * self.q_f
        self.theta = vec[:n].reshape(self.depth, self.q_f)
        self.phi = vec[n:2 * n].reshape(self.depth, self.q_f)

    def _init_state(self, batch):
        n_q = self.q_f
        state = np.zeros((batch,) + (2,) * n_q, dtype=complex)
        idx = (slice(None),) + (0,) * n_q
        state[idx] = 1.0
        return state

    def forward(self, X, theta=None, phi=None):
        """X: (batch, n_features) real-valued, already scaled to O(1).
        Returns expZ on qubit 0, shape (batch,)."""
        theta = self.theta if theta is None else theta
        phi = self.phi if phi is None else phi
        batch = X.shape[0]
        n_q = self.q_f
        state = self._init_state(batch)
        for l in range(self.depth):
            # -- data re-uploading encoding --
            for k in range(n_q):
                xk = X[:, self.feat_map[k]] * self.w[k]
                c, s = np.cos(xk / 2), np.sin(xk / 2)
                gate = np.zeros((batch, 2, 2))
                gate[:, 0, 0], gate[:, 0, 1] = c, -s
                gate[:, 1, 0], gate[:, 1, 1] = s, c
                state = _apply_1q(state, gate, k, n_q)
            # -- trainable rotation layer --
            for k in range(n_q):
                state = _apply_1q(state, _ry(theta[l, k]), k, n_q)
                state = _apply_1q(state, _rz(phi[l, k]), k, n_q)
            # -- GTA-style ring entangler --
            for k in range(n_q):
                state = _apply_cnot(state, k, (k + 1) % n_q, n_q)

        # <Z> on qubit 0: sum |amp|^2 with qubit0=0 minus qubit0=1
        probs = np.abs(state) ** 2
        idx0 = (slice(None), 0) + (slice(None),) * (n_q - 1)
        idx1 = (slice(None), 1) + (slice(None),) * (n_q - 1)
        p0 = probs[idx0].reshape(batch, -1).sum(axis=1)
        p1 = probs[idx1].reshape(batch, -1).sum(axis=1)
        return p0 - p1   # expZ in [-1, 1]

    def predict_proba(self, X, theta=None, phi=None):
        z = self.forward(X, theta, phi)
        return np.clip((1 - z) / 2, 1e-6, 1 - 1e-6)   # p(label=1 / "unstable")

    def forward_full(self, X, theta=None, phi=None, readout_qubits=(0, 1)):
        """Runs the SAME circuit once and reads out expZ on two different
        qubits: qubit `readout_qubits[0]` for the classification head,
        `readout_qubits[1]` for the physics (energy) regression head --
        exactly the shared-trunk, multi-head readout the manuscript's
        readout.py module implements (eq:4.3-4.4)."""
        theta = self.theta if theta is None else theta
        phi = self.phi if phi is None else phi
        batch = X.shape[0]
        n_q = self.q_f
        state = self._init_state(batch)
        for l in range(self.depth):
            for k in range(n_q):
                xk = X[:, self.feat_map[k]] * self.w[k]
                c, s = np.cos(xk / 2), np.sin(xk / 2)
                gate = np.zeros((batch, 2, 2))
                gate[:, 0, 0], gate[:, 0, 1] = c, -s
                gate[:, 1, 0], gate[:, 1, 1] = s, c
                state = _apply_1q(state, gate, k, n_q)
            for k in range(n_q):
                state = _apply_1q(state, _ry(theta[l, k]), k, n_q)
                state = _apply_1q(state, _rz(phi[l, k]), k, n_q)
            for k in range(n_q):
                state = _apply_cnot(state, k, (k + 1) % n_q, n_q)
        probs = np.abs(state) ** 2
        outs = []
        for q in readout_qubits:
            idx0 = (slice(None),) + tuple(0 if a == q else slice(None) for a in range(n_q))
            idx1 = (slice(None),) + tuple(1 if a == q else slice(None) for a in range(n_q))
            p0 = probs[idx0].reshape(batch, -1).sum(axis=1)
            p1 = probs[idx1].reshape(batch, -1).sum(axis=1)
            outs.append(p0 - p1)
        return outs   # list of expZ arrays, one per readout qubit


def bce_loss(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def param_shift_grad(model, X, y, extra_loss_fn=None):
    """Exact two-term parameter-shift gradient (eq:param_shift) of the
    (optionally physics-regularised) loss w.r.t. every entry of theta,phi."""
    shift = np.pi / 2
    grads_theta = np.zeros_like(model.theta)
    grads_phi = np.zeros_like(model.phi)

    def total_loss(theta, phi):
        p = model.predict_proba(X, theta, phi)
        loss = bce_loss(p, y)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model, X, theta, phi)
        return loss

    for l in range(model.depth):
        for k in range(model.q_f):
            th_p, th_m = model.theta.copy(), model.theta.copy()
            th_p[l, k] += shift
            th_m[l, k] -= shift
            grads_theta[l, k] = 0.5 * (total_loss(th_p, model.phi) - total_loss(th_m, model.phi))

            ph_p, ph_m = model.phi.copy(), model.phi.copy()
            ph_p[l, k] += shift
            ph_m[l, k] -= shift
            grads_phi[l, k] = 0.5 * (total_loss(model.theta, ph_p) - total_loss(model.theta, ph_m))
    return grads_theta, grads_phi


def train(model, X, y, iters=60, lr=0.35, extra_loss_fn=None, verbose=False):
    losses = []
    for it in range(iters):
        gth, gph = param_shift_grad(model, X, y, extra_loss_fn)
        model.theta -= lr * gth
        model.phi -= lr * gph
        if verbose and (it % 10 == 0 or it == iters - 1):
            p = model.predict_proba(X)
            l = bce_loss(p, y)
            acc = np.mean((p > 0.5).astype(int) == y)
            print(f"    iter {it:3d}  loss={l:.4f}  train_acc={acc:.3f}")
            losses.append(l)
    return losses
