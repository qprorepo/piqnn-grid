import numpy as np
from scipy.optimize import curve_fit
from statevec import (zero_state, apply_ry, apply_rz, apply_cnot,
                       expval_global_parity, expval_local_Z)

rng = np.random.default_rng(20260721)


# --------------------------------------------------------------------------
# Generic deep random ansatz (barren-plateau regime)
# --------------------------------------------------------------------------
def generic_deep_circuit(n, depth, params, ring_step=1):
    """params shape (depth, n, 2) -- RY, RZ angles per qubit per layer."""
    psi = zero_state(n)
    for L in range(depth):
        for q in range(n):
            psi = apply_ry(psi, params[L, q, 0], q, n)
            psi = apply_rz(psi, params[L, q, 1], q, n)
        for q in range(n):
            psi = apply_cnot(psi, q, (q + ring_step) % n, n)
    return psi

def generic_ansatz_cost(n, depth, params):
    psi = generic_deep_circuit(n, depth, params)
    return expval_global_parity(psi, n)

def gradient_variance_generic(n, depth=20, R=40, rng=rng):
    grads = np.empty(R)
    for r in range(R):
        params = rng.uniform(0, 2 * np.pi, size=(depth, n, 2))
        base = params.copy()
        base[0, 0, 0] += np.pi / 2
        c_plus = generic_ansatz_cost(n, depth, base)
        base[0, 0, 0] -= np.pi
        c_minus = generic_ansatz_cost(n, depth, base)
        grads[r] = 0.5 * (c_plus - c_minus)
    return float(np.var(grads))



def default_ring_graph(n):
    return [(i, (i + 1) % n) for i in range(n)]

def edge_coloring_matchings(edges, n):
    matchings = []
    remaining = list(edges)
    while remaining:
        used = set()
        layer = []
        rest = []
        for (a, b) in remaining:
            if a not in used and b not in used:
                layer.append((a, b))
                used.add(a)
                used.add(b)
            else:
                rest.append((a, b))
        matchings.append(layer)
        remaining = rest
    return matchings

def gta_circuit(n, depth, params, matchings):
    psi = zero_state(n)
    n_m = len(matchings)
    for L in range(depth):
        for q in range(n):
            psi = apply_ry(psi, params[L, q, 0], q, n)
            psi = apply_rz(psi, params[L, q, 1], q, n)
        layer_edges = matchings[L % n_m]
        for (a, b) in layer_edges:
            psi = apply_cnot(psi, a, b, n)
    return psi

def gta_ansatz_cost(n, depth, params, matchings, obs_qubit=0):
    psi = gta_circuit(n, depth, params, matchings)
    return expval_local_Z(psi, obs_qubit, n)

def gradient_variance_gta(n, depth=3, R=40, edges=None, rng=rng, obs_qubit=0):
    if edges is None:
        edges = default_ring_graph(n)
    matchings = edge_coloring_matchings(edges, n)
    grads = np.empty(R)
    for r in range(R):
        params = rng.uniform(0, 2 * np.pi, size=(depth, n, 2))
        base = params.copy()
        base[0, obs_qubit, 0] += np.pi / 2
        c_plus = gta_ansatz_cost(n, depth, base, matchings, obs_qubit)
        base[0, obs_qubit, 0] -= np.pi
        c_minus = gta_ansatz_cost(n, depth, base, matchings, obs_qubit)
        grads[r] = 0.5 * (c_plus - c_minus)
    return float(np.var(grads))


# --------------------------------------------------------------------------
# Fit functional forms to the exact data, for extrapolation
# --------------------------------------------------------------------------
def fit_exponential(n_vals, var_vals):
    def model(n, a, b, c):
        return a * np.exp(-b * (n - n_vals[0])) + c
    p0 = [var_vals[0], 0.3, 1e-9]
    popt, _ = curve_fit(model, n_vals, var_vals, p0=p0, maxfev=20000)
    return popt, model

def fit_powerlaw(n_vals, var_vals):
    def model(n, a, p, c):
        return a / (n ** p) + c
    p0 = [max(var_vals[0] - var_vals[-1], 1e-6) * n_vals[0], 0.5, var_vals[-1]]
    bounds = ([0.0, 0.0, 0.0], [10.0, 2.5, 1.0])
    popt, _ = curve_fit(model, n_vals, var_vals, p0=p0, bounds=bounds, maxfev=40000)
    return popt, model
