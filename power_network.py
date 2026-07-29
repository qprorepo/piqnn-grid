"""
power_network.py
=====================================================================
This module is imported by swing_dynamics.py but was not part of the
pasted code, so it's implemented here from scratch to make the
pipeline runnable:

    from power_network import load_matpower_case, build_ybus, newton_raphson_pf, gen_internal_emf

- load_matpower_case: parses a simple 3-section (BUS / GEN / BRANCH)
  text file using MATPOWER's own column convention.
- build_ybus: standard pi-model bus admittance matrix, with optional
  branch exclusion (for the post-fault topology).
- newton_raphson_pf: full Newton-Raphson AC power flow (numerical
  Jacobian -- small systems, so this is fast and avoids hand-derived
  Jacobian bugs).
- gen_internal_emf: recovers each generator's internal EMF E*e^{j*delta}
  behind transient reactance x'_d from the solved power flow, using
  E = V_term + j*x'_d*I_gen, with I_gen backed out of the solved net
  bus injection S_inj = V * conj(Y @ V).

case9.txt / case14.txt (loaded by this module) hold the standard,
widely published IEEE 9-bus (WSCC) and IEEE 14-bus power-flow test
case parameters, in MATPOWER's own bus/gen/branch column layout.
"""
import numpy as np


def load_matpower_case(path):
    section = None
    baseMVA = 100.0
    bus_rows, gen_rows, branch_rows = [], [], []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith("BASEMVA"):
                baseMVA = float(line.split()[1])
                continue
            if up == "BUS":
                section = "bus"; continue
            if up == "GEN":
                section = "gen"; continue
            if up == "BRANCH":
                section = "branch"; continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            if section == "bus":
                bus_rows.append(vals)
            elif section == "gen":
                gen_rows.append(vals)
            elif section == "branch":
                branch_rows.append(vals)
    return dict(baseMVA=baseMVA,
                bus=np.array(bus_rows, dtype=float),
                gen=np.array(gen_rows, dtype=float),
                branch=np.array(branch_rows, dtype=float))


def build_ybus(case, exclude_branch=None):
    """bus cols: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
       branch cols: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status"""
    bus = case["bus"]
    n = bus.shape[0]
    idx = {int(bus[i, 0]): i for i in range(n)}
    Y = np.zeros((n, n), dtype=complex)
    for i in range(n):
        Y[i, i] += (bus[i, 4] + 1j * bus[i, 5]) / case["baseMVA"]
    excl = None
    if exclude_branch is not None:
        excl = {int(exclude_branch[0]), int(exclude_branch[1])}
    for row in case["branch"]:
        fbus, tbus = int(row[0]), int(row[1])
        if excl is not None and {fbus, tbus} == excl:
            continue
        r, x, b = row[2], row[3], row[4]
        ratio = row[8] if len(row) > 8 and row[8] not in (0.0,) else 1.0
        z = r + 1j * x
        y = 1.0 / z if abs(z) > 1e-12 else 0.0
        i, k = idx[fbus], idx[tbus]
        Y[i, i] += y / (ratio ** 2) + 1j * b / 2.0
        Y[k, k] += y + 1j * b / 2.0
        Y[i, k] -= y / ratio
        Y[k, i] -= y / ratio
    return Y, idx


def newton_raphson_pf(case, Y, idx, tol=1e-10, max_iter=60):
    bus = case["bus"]
    n = bus.shape[0]
    baseMVA = case["baseMVA"]
    row_of = {int(bus[i, 0]): i for i in range(n)}

    type_arr = np.zeros(n)
    Pd = np.zeros(n)
    Qd = np.zeros(n)
    Vm0 = np.ones(n)
    Va0 = np.zeros(n)
    for bus_id, k in idx.items():
        r = row_of[bus_id]
        type_arr[k] = bus[r, 1]
        Pd[k] = bus[r, 2] / baseMVA
        Qd[k] = bus[r, 3] / baseMVA
        Vm0[k] = bus[r, 7]
        Va0[k] = np.deg2rad(bus[r, 8])

    Pg = np.zeros(n)
    Vg = np.full(n, np.nan)
    for g in case["gen"]:
        gb = int(g[0])
        k = idx[gb]
        Pg[k] += g[1] / baseMVA
        Vg[k] = g[5]

    slack, pv, pq = None, [], []
    for bus_id, k in idx.items():
        t = type_arr[k]
        if t == 3:
            slack = k
        elif t == 2:
            pv.append(k)
        else:
            pq.append(k)

    # Flat start (standard NR practice): Va=0, Vm=1 pu everywhere except
    # generator buses, which are pinned to their voltage setpoint Vg.
    # (Vm0/Va0 parsed from the case file are metadata only -- using them
    # as the iterate start is numerically fragile since they aren't
    # guaranteed a priori to be a consistent operating point.)
    Vm, Va = np.ones(n), np.zeros(n)
    for k in pv:
        if not np.isnan(Vg[k]):
            Vm[k] = Vg[k]
    if not np.isnan(Vg[slack]):
        Vm[slack] = Vg[slack]

    Pspec = Pg - Pd
    Qspec = -Qd.copy()

    pvpq = pv + pq
    npvpq, npq = len(pvpq), len(pq)

    def mismatch_fn(state, Va_ref, Vm_ref):
        Va_ = Va_ref.copy(); Vm_ = Vm_ref.copy()
        Va_[pvpq] = state[:npvpq]
        Vm_[pq] = state[npvpq:]
        V_ = Vm_ * np.exp(1j * Va_)
        S_ = V_ * np.conj(Y @ V_)
        dP_ = Pspec[pvpq] - S_.real[pvpq]
        dQ_ = Qspec[pq] - S_.imag[pq]
        return np.concatenate([dP_, dQ_])

    state = np.concatenate([Va[pvpq], Vm[pq]])
    for _ in range(max_iter):
        f0 = mismatch_fn(state, Va, Vm)
        if np.max(np.abs(f0)) < tol:
            break
        eps = 1e-6
        J = np.zeros((npvpq + npq, npvpq + npq))
        for j in range(len(state)):
            s2 = state.copy(); s2[j] += eps
            J[:, j] = (mismatch_fn(s2, Va, Vm) - f0) / eps
        dx = np.linalg.solve(J, f0)
        state = state - dx

    Va[pvpq] = state[:npvpq]
    Vm[pq] = state[npvpq:]
    return Vm, Va, slack, pv, pq


def gen_internal_emf(case, Vm, Va, idx, xdp_base=0.28):
    Y, _ = build_ybus(case)
    baseMVA = case["baseMVA"]
    V = Vm * np.exp(1j * Va)
    S_inj = V * np.conj(Y @ V)

    bus = case["bus"]
    Qd = {int(bus[i, 0]): bus[i, 3] / baseMVA for i in range(bus.shape[0])}

    Pmax_all = case["gen"][:, 8]
    E_list, delta_list, xdp_list = [], [], []
    for g in case["gen"]:
        gb = int(g[0]); k = idx[gb]
        Pg = g[1] / baseMVA
        Qgen = S_inj[k].imag + Qd.get(gb, 0.0)
        Sgen = Pg + 1j * Qgen
        Vt = V[k]
        I = np.conj(Sgen / Vt)
        xdp = xdp_base * (Pmax_all.max() / g[8]) if g[8] > 0 else xdp_base
        E = Vt + 1j * xdp * I
        E_list.append(abs(E))
        delta_list.append(np.angle(E))
        xdp_list.append(xdp)
    return np.array(E_list), np.array(delta_list), np.array(xdp_list)
