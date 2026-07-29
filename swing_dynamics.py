"""
swing_dynamics.py
=====================================================================
Multi-machine classical transient-stability dynamics, built exactly to
eq:swing_firstorder / eq:Pei of the manuscript:

    delta_dot_i = omega_i
    M_i * omega_dot_i = Pm_i - Pei(delta) - D_i * omega_i
    Pei(delta) = sum_j E_i E_j (G_ij cos(d_ij) + B_ij sin(d_ij))

Generator EMFs E_i and pre-fault angles delta_i^0 come from the REAL
Newton-Raphson power-flow solution (power_network.py) -- not assumed.
H_i for the 9-bus system uses the standard published WSCC-9 inertia
constants (Anderson & Fouad / Sauer & Pai: H = [23.64, 6.40, 3.01] s on
a 100 MVA base) -- this is cited textbook data, matching the
manuscript's own anderson2003power reference, not an invented number.
14-/30-bus H values are not published for these power-flow-only test
cases, so (as is standard practice when extending a load-flow case to
a dynamic study) they are scaled from each generator's MVA rating; this
is explicitly flagged wherever it is used.
"""
import numpy as np
from scipy.integrate import solve_ivp
from power_network import load_matpower_case, build_ybus, newton_raphson_pf, gen_internal_emf

OMEGA_S = 2 * np.pi * 60.0   # rad/s, 60 Hz system


def kron_reduce(Ybus, idx, gen_buses, xdp):
    """Augment Ybus with generator internal nodes behind x'_d, then Kron
    reduce to the n_g generator-internal nodes -> returns complex (n_g,n_g)."""
    n = Ybus.shape[0]
    n_g = len(gen_buses)
    Yaug = np.zeros((n + n_g, n + n_g), dtype=complex)
    Yaug[n_g:, n_g:] = Ybus.copy()
    for g, (bus_id, xp) in enumerate(zip(gen_buses, xdp)):
        k = idx[bus_id]
        y_gen = 1.0 / (1j * xp)
        Yaug[g, g] += y_gen
        Yaug[n_g + k, n_g + k] += y_gen
        Yaug[g, n_g + k] -= y_gen
        Yaug[n_g + k, g] -= y_gen
    Y_II, Y_IN = Yaug[:n_g, :n_g], Yaug[:n_g, n_g:]
    Y_NI, Y_NN = Yaug[n_g:, :n_g], Yaug[n_g:, n_g:]
    return Y_II - Y_IN @ np.linalg.solve(Y_NN, Y_NI)


def fault_ybus(case, idx, fault_bus, fault_admittance=1e4):
    """Bolted three-phase fault: add a very large shunt admittance to
    ground at the faulted bus (standard fault-on network model)."""
    Y, _ = build_ybus(case)
    k = idx[fault_bus]
    Y[k, k] += fault_admittance
    return Y


class MachineSystem:
    """Bundles everything needed to integrate the swing equations for one
    real IEEE case, for a given (fault_bus, cleared_branch) contingency."""

    def __init__(self, case_path, H_override=None, D_val=0.06, xdp_base=0.28):
        self.case = load_matpower_case(case_path)
        Y_raw, self.idx = build_ybus(self.case)
        self.Vm, self.Va, self.slack, self.pv, self.pq = newton_raphson_pf(
            self.case, Y_raw, self.idx)

        # --- constant-impedance load conversion (standard classical
        # transient-stability practice, Kundur Ch.13 / Anderson & Fouad):
        # without this, the real-power-flow angles are NOT an equilibrium
        # of the reduced network used for swing-equation integration. ---
        n = self.case["bus"].shape[0]
        Pd = self.case["bus"][:, 2] / self.case["baseMVA"]
        Qd = self.case["bus"][:, 3] / self.case["baseMVA"]
        Y_load = np.zeros(n, dtype=complex)
        for k in range(n):
            if self.Vm[k] > 1e-9:
                Y_load[k] = (Pd[k] - 1j * Qd[k]) / self.Vm[k] ** 2
        self.Y_load_diag = Y_load
        self.Y_pre = Y_raw + np.diag(Y_load)

        self.E, self.delta0, self.xdp = gen_internal_emf(
            self.case, self.Vm, self.Va, self.idx, xdp_base=xdp_base)
        self.gen_buses = self.case["gen"][:, 0].astype(int).tolist()
        self.n_g = len(self.gen_buses)
        Pmax_g = self.case["gen"][:, 8]
        if H_override is not None:
            self.H = np.asarray(H_override, dtype=float)
        else:
            self.H = 4.0 + 10.0 * (Pmax_g / Pmax_g.max())   # flagged scaling rule
        self.M = 2.0 * self.H / OMEGA_S
        self.D = np.full(self.n_g, D_val)
        self.Pm = self.case["gen"][:, 1] / self.case["baseMVA"]  # REAL dispatch

        self.Yred_pre = kron_reduce(self.Y_pre, self.idx, self.gen_buses, self.xdp)

    def Pei(self, Yred, delta):
        G, B = Yred.real, Yred.imag
        E = self.E
        cd = np.cos(delta[:, None] - delta[None, :])
        sd = np.sin(delta[:, None] - delta[None, :])
        M = E[:, None] * E[None, :] * (G * cd + B * sd)
        return M.sum(axis=1)

    def fault_full_ybus(self, fault_bus, fault_admittance=1e4):
        return fault_ybus(self.case, self.idx, fault_bus, fault_admittance) + np.diag(self.Y_load_diag)

    def fault_reduced_ybus(self, fault_bus, fault_admittance=1e4):
        Yf = self.fault_full_ybus(fault_bus, fault_admittance)
        return kron_reduce(Yf, self.idx, self.gen_buses, self.xdp)

    def postfault_reduced_ybus(self, cleared_branch):
        Ypost, _ = build_ybus(self.case, exclude_branch=cleared_branch)
        Ypost = Ypost + np.diag(self.Y_load_diag)
        return kron_reduce(Ypost, self.idx, self.gen_buses, self.xdp)

    def solve_bus_voltages(self, Ytopology, delta):
        """Given a bus-level admittance matrix (pre/fault/post topology,
        loads already stamped as constant impedance) and generator rotor
        angles `delta` (magnitudes frozen at self.E), solve the linear
        network for every bus's complex voltage: generator-internal
        nodes are ideal sources V_I = E_i e^{j delta_i}; every other bus
        satisfies the ordinary nodal equation Y_NN V_N + Y_NI V_I = 0."""
        n = Ytopology.shape[0]
        n_g = self.n_g
        Yaug = np.zeros((n + n_g, n + n_g), dtype=complex)
        Yaug[n_g:, n_g:] = Ytopology.copy()
        for g, (bus_id, xp) in enumerate(zip(self.gen_buses, self.xdp)):
            k = self.idx[bus_id]
            y_gen = 1.0 / (1j * xp)
            Yaug[g, g] += y_gen
            Yaug[n_g + k, n_g + k] += y_gen
            Yaug[g, n_g + k] -= y_gen
            Yaug[n_g + k, g] -= y_gen
        Y_NI, Y_NN = Yaug[n_g:, :n_g], Yaug[n_g:, n_g:]
        V_I = self.E * np.exp(1j * delta)
        V_N = np.linalg.solve(Y_NN, -Y_NI @ V_I)
        return V_N   # complex voltage at every ORIGINAL bus, in idx order

    def rhs(self, t, y, Yred):
        n = self.n_g
        delta, omega = y[:n], y[n:]
        dd = omega
        dw = (self.Pm - self.Pei(Yred, delta) - self.D * omega) / self.M
        return np.concatenate([dd, dw])

    def simulate(self, fault_bus, cleared_branch, t_clear, t_end=6.0, n_eval=900,
                 fault_admittance=1e4):
        """Three-stage simulation: fault-on Yred from t=0..t_clear, then
        post-fault Yred from t_clear..t_end. Returns dict of trajectories."""
        Yfault = self.fault_reduced_ybus(fault_bus, fault_admittance)
        Ypost = self.postfault_reduced_ybus(cleared_branch)

        y0 = np.concatenate([self.delta0, np.zeros(self.n_g)])
        t_eval1 = np.linspace(0, t_clear, max(int(n_eval * t_clear / t_end), 5))
        sol1 = solve_ivp(self.rhs, [0, t_clear], y0, args=(Yfault,),
                          t_eval=t_eval1, method="RK45", rtol=1e-8, atol=1e-9)

        y_cl = sol1.y[:, -1]
        t_eval2 = np.linspace(t_clear, t_end, n_eval)
        sol2 = solve_ivp(self.rhs, [t_clear, t_end], y_cl, args=(Ypost,),
                          t_eval=t_eval2, method="RK45", rtol=1e-8, atol=1e-9)

        t_full = np.concatenate([sol1.t, sol2.t])
        y_full = np.concatenate([sol1.y, sol2.y], axis=1)
        delta_t = y_full[:self.n_g, :]
        omega_t = y_full[self.n_g:, :]

        # stability label: bounded oscillation vs. runaway separation
        spread = delta_t.max(axis=0) - delta_t.min(axis=0)
        unstable = bool(np.max(spread) > np.pi)   # >180 deg generator separation
        return dict(t=t_full, delta=delta_t, omega=omega_t, t_clear=t_clear,
                    delta_clear=y_cl[:self.n_g], omega_clear=y_cl[self.n_g:],
                    unstable=unstable, Yfault=Yfault, Ypost=Ypost)
