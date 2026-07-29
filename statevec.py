"""
statevec.py
-----------
Minimal but efficient statevector simulator. Gates are applied via
reshape/moveaxis tensor contraction rather than building dense 2^n x 2^n
unitaries, which is what makes it possible to push exact simulation to
16-18 qubits on ordinary hardware -- essential here, since the whole point
of this experiment is to *measure* gradient-variance decay rather than
assume it.
"""
import numpy as np

def zero_state(n):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    return psi

def _apply_1q(state, gate, q, n):
    s = state.reshape([2] * n)
    s = np.moveaxis(s, q, 0)
    s = np.tensordot(gate, s, axes=([1], [0]))
    s = np.moveaxis(s, 0, q)
    return s.reshape(-1)

def _apply_2q_cnot(state, control, target, n):
    s = state.reshape([2] * n)
    s = np.moveaxis(s, [control, target], [0, 1])
    out = s.copy()
    out[1, 0, ...] = s[1, 1, ...]
    out[1, 1, ...] = s[1, 0, ...]
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)

def RY(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)

def apply_ry(state, theta, q, n):
    return _apply_1q(state, RY(theta), q, n)

def apply_rz(state, theta, q, n):
    return _apply_1q(state, RZ(theta), q, n)

def apply_cnot(state, c, t, n):
    return _apply_2q_cnot(state, c, t, n)

def expval_global_parity(state, n):
    """<Z_0 Z_1 ... Z_{n-1}>: parity-weighted sum of |amplitude|^2.
    NOTE: with row-major reshape([2]*n), qubit 0 is the *most significant*
    axis, i.e. bit (n-1-k) of the linear index corresponds to qubit k --
    the opposite of the usual little-endian convention. Extract bits
    accordingly so this stays consistent with apply_1q/apply_cnot, which
    operate directly on tensor axes and don't care about bit order at all."""
    probs = np.abs(state) ** 2
    idx = np.arange(2 ** n)
    shifts = (n - 1 - np.arange(n))
    bits = ((idx[:, None] >> shifts[None, :]) & 1)
    parity = 1 - 2 * (bits.sum(axis=1) % 2)
    return float((probs * parity).sum())

def expval_local_Z(state, q, n):
    """<Z_q> for a single qubit."""
    s = state.reshape([2] * n)
    s = np.moveaxis(s, q, 0)
    p0 = np.sum(np.abs(s[0]) ** 2)
    p1 = np.sum(np.abs(s[1]) ** 2)
    return float(p0 - p1)

def expval_local_ZZ(state, q1, q2, n):
    s = state.reshape([2] * n)
    s = np.moveaxis(s, [q1, q2], [0, 1])
    p = np.abs(s) ** 2
    return float(p[0, 0].sum() + p[1, 1].sum() - p[0, 1].sum() - p[1, 0].sum())
