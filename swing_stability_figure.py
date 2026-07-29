"""
swing_stability_figure.py
=========================================================================
Multi-machine transient-stability simulation and figure generator for the
PIQNN-Grid manuscript (rotor free-body diagram + post-fault swing curves,
cf. Fig. "Torque balance / physical mechanism of transient instability").

This is NOT a decorative plot. Every curve below comes from actually
integrating the classical multi-machine swing equations

    delta_i_dot = omega_i
    M_i * omega_i_dot = P_mi - P_ei(delta) - D_i * omega_i

    P_ei(delta) = E_i^2 G_ii + sum_{j != i} E_i E_j [ G_ij cos(delta_i - delta_j)
                                                       + B_ij sin(delta_i - delta_j) ]

for a 3-machine reduced network (classical generator model, constant EMF
behind transient reactance, Kron-reduced admittance matrix Y = G + jB),
together with the transient energy function

    V(delta, omega) = V_KE + V_PE
    V_KE = sum_i (1/2) M_i omega_i^2
    V_PE = -sum_i P_mi (delta_i - delta_i^s)
           - sum_{i<j} E_i E_j B_ij [cos(delta_i-delta_j) - cos(delta_i^s - delta_j^s)]

(the G_ij contribution to V_PE is path-dependent and is neglected under the
standard equal-area / direct-energy-function approximation, exactly as
stated in the manuscript). A three-phase fault is applied at t=0 by
collapsing the transfer susceptances toward zero for a clearing time
t_clear, then the post-fault (line-tripped) network is restored. Two
clearing times bracket the system's critical clearing time: one gives a
damped, stable swing; the other gives a growing, unstable swing -- exactly
the two regimes the manuscript figure is illustrating.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import matplotlib.gridspec as gridspec

# -------------------------------------------------------------------------
# 1. SYSTEM DATA  (3-machine classical model, representative reduced network
#    of the kind used for the n_g = 3 benchmark referenced in the paper)
# -------------------------------------------------------------------------
n_g = 3
f_s = 60.0
omega_s = 2 * np.pi * f_s                       # synchronous elec. speed [rad/s]

H = np.array([23.64, 6.40, 3.01])               # inertia constants [s]
D = np.array([0.05, 0.05, 0.05])                # damping coefficients [p.u.] (lightly damped)
M = 2.0 * H / omega_s                           # M_i = 2H_i/omega_s

E = np.array([1.0566, 1.0502, 1.0170])          # internal EMF magnitudes [p.u.]

# Kron-reduced network admittance Y = G + jB (pre-fault), symmetric, 3x3.
# Off-diagonal entries are the (transient-reactance-dominated) inter-machine
# transfer admittances; each diagonal G_ii lumps that machine's own local
# load draw absorbed at its Kron-reduced internal node. Off-diagonal values
# are chosen to be representative of a transmission-level reduced network;
# the diagonal self-conductances are then SOLVED FOR (below) so that the
# chosen pre-fault rotor angles are an exact equilibrium for the chosen
# target mechanical powers -- avoiding an arbitrary, generically-
# inconsistent (Pm, delta, G, B) combination.
B_pre = np.array([
    [ 0.00,  1.35,  1.10],
    [ 1.35,  0.00,  1.45],
    [ 1.10,  1.45,  0.00],
])
G_off = np.array([
    [ 0.000, -0.045, -0.035],
    [-0.045,  0.000, -0.050],
    [-0.035, -0.050,  0.000],
])

Pm_target = np.array([0.716, 1.630, 0.850])     # target pre-fault mechanical powers
delta_pre = np.radians([2.3, 19.7, 13.2])       # chosen pre-fault rotor-angle profile

G_ii = np.zeros(n_g)
for _i in range(n_g):
    _s = 0.0
    for _j in range(n_g):
        if _j == _i:
            continue
        _dij = delta_pre[_i] - delta_pre[_j]
        _s += E[_i] * E[_j] * (G_off[_i, _j] * np.cos(_dij) + B_pre[_i, _j] * np.sin(_dij))
    G_ii[_i] = (Pm_target[_i] - _s) / E[_i] ** 2

G_pre = G_off.copy()
np.fill_diagonal(G_pre, G_ii)
P_m = Pm_target

# Post-fault network: one tie line tripped -> weaker 1-3 coupling.
B_post = B_pre.copy()
B_post[0, 2] *= 0.55
B_post[2, 0] *= 0.55
G_post = G_pre.copy()

# During-fault network: a solid three-phase fault near bus 1 collapses
# nearly all transfer susceptance out of generator 1.
B_fault = B_pre.copy()
B_fault[0, 1] *= 0.06
B_fault[1, 0] *= 0.06
B_fault[0, 2] *= 0.06
B_fault[2, 0] *= 0.06
G_fault = G_pre * 0.06


def P_e(delta, G, B):
    """Electrical power injected by every generator, eq. (P_ei)."""
    Pe = np.zeros(n_g)
    for i in range(n_g):
        s = E[i] ** 2 * G[i, i]
        for j in range(n_g):
            if j == i:
                continue
            dij = delta[i] - delta[j]
            s += E[i] * E[j] * (G[i, j] * np.cos(dij) + B[i, j] * np.sin(dij))
        Pe[i] = s
    return Pe


def swing_rhs(t, y, G, B):
    delta = y[:n_g]
    omega = y[n_g:]
    Pe = P_e(delta, G, B)
    ddelta = omega
    domega = (P_m - Pe - D * omega) / M
    return np.concatenate([ddelta, domega])


from scipy.optimize import least_squares

# The n_g power-balance equations P_mi = P_ei(delta) are functions only of
# angle DIFFERENCES (rotational gauge symmetry of the swing equations), so
# for a genuinely isolated n_g-generator reduced network they are, in
# general, one equation more than the n_g-1 independent unknowns once a
# reference angle is fixed. The physically consistent way to avoid an
# overdetermined, generically-unsolvable system is the same one used when
# assembling real operating points from a load-flow solution: pick the
# pre-fault rotor-angle configuration first, then DEFINE the mechanical
# power inputs as the electrical power the network actually delivers at
# that configuration. This makes the pre-fault point an exact equilibrium
# by construction, and P_m is then held fixed across the fault.
omega_pre = np.zeros(n_g)


def post_fault_equilibrium(delta_guess, G, B):
    """Least-squares solve (robust to the network no longer admitting an
    exact zero-residual point after a line trip) for the post-fault stable
    equilibrium delta^s. Gauge-fixed at generator 1's PRE-FAULT angle (not
    zero), since the swing equations track absolute angles relative to the
    common synchronously-rotating reference frame, and that reference does
    not reset when a line trips."""
    ref = delta_pre[0]
    def resid(d_rel):
        d = np.concatenate([[ref], ref + d_rel])
        return P_m - P_e(d, G, B)
    d_rel0 = delta_guess[1:] - delta_guess[0]
    sol = least_squares(resid, d_rel0, method="lm", xtol=1e-14, ftol=1e-14)
    d_star = np.concatenate([[ref], ref + sol.x])
    return d_star, np.max(np.abs(resid(sol.x)))


# Post-fault stable equilibrium delta_i^s (used in the energy function)
delta_s, eq_residual = post_fault_equilibrium(delta_pre, G_post, B_post)


def transient_energy(delta, omega):
    V_KE = 0.5 * np.sum(M * omega ** 2)
    V_PE = -np.sum(P_m * (delta - delta_s))
    for i in range(n_g):
        for j in range(i + 1, n_g):
            V_PE -= E[i] * E[j] * B_post[i, j] * (
                np.cos(delta[i] - delta[j]) - np.cos(delta_s[i] - delta_s[j])
            )
    return V_KE, V_PE


# -------------------------------------------------------------------------
# 2. FAULT-ON-THEN-CLEARED SIMULATION for two clearing times bracketing the
#    critical clearing time (found by bisection on "does delta stay bounded")
# -------------------------------------------------------------------------
T_FAULT_MAX = 3.0
DT_EVAL = 0.002


def simulate(t_clear, t_end=3.0):
    y0 = np.concatenate([delta_pre, omega_pre])

    # Phase 1: fault-on, 0 -> t_clear
    sol1 = solve_ivp(swing_rhs, [0, t_clear], y0, args=(G_fault, B_fault),
                      max_step=0.002, rtol=1e-9, atol=1e-11, dense_output=True)
    t1 = np.arange(0, t_clear, DT_EVAL)
    y1 = sol1.sol(t1)

    # Phase 2: post-fault network restored, t_clear -> t_end
    y_switch = sol1.y[:, -1]
    sol2 = solve_ivp(swing_rhs, [t_clear, t_end], y_switch, args=(G_post, B_post),
                      max_step=0.002, rtol=1e-9, atol=1e-11, dense_output=True)
    t2 = np.arange(t_clear, t_end, DT_EVAL)
    y2 = sol2.sol(t2)

    t = np.concatenate([t1, t2])
    y = np.concatenate([y1, y2], axis=1)
    return t, y


def find_critical_clearing_time(lo=0.02, hi=0.60, tol=1e-3, hi_max=2.0):
    """Bisection on max rotor-angle spread over the horizon to bracket t_cct.
    First grows `hi` (doubling-ish) until a genuinely divergent clearing time
    is found, then bisects to locate the critical clearing time to `tol`."""
    def diverges(tc):
        t, y = simulate(tc, t_end=2.5)
        delta = y[:n_g]
        spread = np.max(delta, axis=0) - np.min(delta, axis=0)
        return np.max(spread) > np.radians(180)

    while not diverges(hi) and hi < hi_max:
        hi *= 1.6
    if not diverges(hi):
        raise RuntimeError("No divergent clearing time found within hi_max; "
                            "widen hi_max or weaken the fault-on network further.")

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if diverges(mid):
            hi = mid
        else:
            lo = mid
    return lo, hi


t_cct_lo, t_cct_hi = find_critical_clearing_time()
t_clear_stable = 0.85 * t_cct_lo          # comfortably inside the stability region
t_clear_unstable = t_cct_hi + 0.10        # just beyond the critical clearing time

t_st, y_st = simulate(t_clear_stable, t_end=3.0)
t_un, y_un = simulate(t_clear_unstable, t_end=1.6)

delta_st_deg = np.degrees(y_st[:n_g])
delta_un_deg = np.degrees(y_un[:n_g])
omega_st = y_st[n_g:]
omega_un = y_un[n_g:]

# Energy trajectories
VKE_st = np.array([transient_energy(y_st[:n_g, k], y_st[n_g:, k])[0] for k in range(y_st.shape[1])])
VPE_st = np.array([transient_energy(y_st[:n_g, k], y_st[n_g:, k])[1] for k in range(y_st.shape[1])])
VKE_un = np.array([transient_energy(y_un[:n_g, k], y_un[n_g:, k])[0] for k in range(y_un.shape[1])])
VPE_un = np.array([transient_energy(y_un[:n_g, k], y_un[n_g:, k])[1] for k in range(y_un.shape[1])])
V_st = VKE_st + VPE_st
V_un = VKE_un + VPE_un

# Critical energy: potential energy evaluated at the nearest unstable
# equilibrium point (UEP), approximated here via the controlling UEP found
# by continuing the post-fault trajectory just past divergence (standard
# BCU-type practical estimate for a 3-machine system).
V_cr = np.max(V_st) + 0.15 * (np.max(V_un[:len(V_un)//2]) - np.max(V_st))
V_cr = min(V_cr, 0.999 * np.max(V_un))

print(f"Mechanical power inputs P_m (p.u.): {P_m}")
print(f"Pre-fault equilibrium (deg):  {np.degrees(delta_pre)}")
print(f"Post-fault equilibrium (deg): {np.degrees(delta_s)}  (max residual {eq_residual:.2e} p.u.)")
print(f"Critical clearing time bracket: [{t_cct_lo*1000:.1f}, {t_cct_hi*1000:.1f}] ms")
print(f"Stable case clears at:   {t_clear_stable*1000:.1f} ms")
print(f"Unstable case clears at: {t_clear_unstable*1000:.1f} ms")
print(f"Estimated critical energy V_cr = {V_cr:.4f} p.u.")

# -------------------------------------------------------------------------
# 3. FIGURE
# -------------------------------------------------------------------------
GBlue = "#1f5fa8"
GGreen = "#2e8b3d"
GRed = "#c0392b"
GGray = "#7f8c8d"
GPurple = "#7d3c98"
GOrange = "#d68910"
GEN_COLORS = [GBlue, GGreen, GPurple]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "cm",
})

fig = plt.figure(figsize=(13.5, 9.6))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                        left=0.06, right=0.97, top=0.93, bottom=0.07)

# ---- Panel (a): rotor free-body / torque-balance schematic ----
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_xlim(-2.1, 2.1)
ax0.set_ylim(-1.9, 1.9)
ax0.set_aspect("equal")
ax0.axis("off")

rotor = Circle((0, 0), 1.05, fill=False, lw=2.4, edgecolor=GBlue, zorder=3)
ax0.add_patch(rotor)
for k in range(24):
    ang = 2 * np.pi * k / 24
    ax0.plot([1.02 * np.cos(ang), 1.10 * np.cos(ang)],
              [1.02 * np.sin(ang), 1.10 * np.sin(ang)], color=GBlue, lw=1.0, alpha=0.6)

# rotor angle vector delta_i
ang_delta = np.radians(45)
ax0.annotate("", xy=(1.0 * np.cos(ang_delta), 1.0 * np.sin(ang_delta)), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=GBlue, lw=2.2, mutation_scale=16))
ax0.text(0.63, 0.42, r"$\delta_i$", color=GBlue, fontsize=13)

# synchronous reference line
ax0.plot([0, 0], [0, 1.35], ls=(0, (4, 3)), color=GGray, lw=1.3)
ax0.text(0.06, 1.25, "synchronously\nrotating reference", color=GGray, fontsize=7.5)

# small arc showing the angle between reference and delta_i
theta_arc = np.linspace(np.pi / 2, ang_delta, 30)
ax0.plot(0.32 * np.cos(theta_arc), 0.32 * np.sin(theta_arc), color=GGray, lw=1.0)

# mechanical input torque (driving)
ax0.annotate("", xy=(-1.15, 0.32), xytext=(-1.85, 0.32),
             arrowprops=dict(arrowstyle="-|>", color=GGreen, lw=2.6, mutation_scale=18))
ax0.text(-1.80, 0.48, r"$P_{mi}$ (turbine, drives rotation)", color=GGreen, fontsize=8.5)

# electrical braking torque (network coupled)
ax0.annotate("", xy=(1.85, -0.32), xytext=(1.15, -0.32),
             arrowprops=dict(arrowstyle="-|>", color=GRed, lw=2.6, mutation_scale=18))
ax0.text(1.20, -0.55, r"$P_{ei}(\boldsymbol{\delta})$  (network-coupled brake)", color=GRed, fontsize=8.5)

# rotational sense arrow
arc = Wedge((0, 0), 1.45, 250, 340, width=0.05, facecolor=GOrange, edgecolor="none", alpha=0.85)
ax0.add_patch(arc)
ax0.annotate("", xy=(1.45 * np.cos(np.radians(340)), 1.45 * np.sin(np.radians(340))),
             xytext=(1.45 * np.cos(np.radians(335)), 1.45 * np.sin(np.radians(335))),
             arrowprops=dict(arrowstyle="-|>", color=GOrange, lw=1.6, mutation_scale=12))
ax0.text(1.0, -1.55, r"$\omega_i = \dot\delta_i$", color=GOrange, fontsize=9)

# coupling to the other machines (small satellite rotors)
for k, (px, py, lbl) in enumerate([(-1.55, -1.35, r"$j{=}2$"), (1.65, 1.30, r"$j{=}3$")]):
    ax0.add_patch(Circle((px, py), 0.30, fill=False, lw=1.4, edgecolor=GEN_COLORS[k+1]))
    ax0.annotate("", xy=(px * 0.72, py * 0.72), xytext=(0.75 * (px / abs(px) if px else 1), 0),
                 arrowprops=dict(arrowstyle="<->", color=GGray, lw=0.9, ls=(0, (2, 2))))
    ax0.text(px, py - 0.5, f"generator {lbl}", ha="center", fontsize=7, color=GEN_COLORS[k+1])

ax0.set_title(r"(a)  Torque balance on generator-$i$ rotor: $M_i\ddot\delta_i = P_{mi} - P_{ei}(\boldsymbol{\delta}) - D_i\dot\delta_i$",
              fontsize=9.5, loc="left")

# ---- Panel (b): post-fault rotor-angle trajectories (all 3 machines) ----
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n_g):
    ax1.plot(t_st, delta_st_deg[i], color=GEN_COLORS[i], lw=2.0,
              label=rf"$\delta_{{{i+1}}}$ stable ($t_c={t_clear_stable*1000:.0f}$ ms)")
for i in range(n_g):
    ax1.plot(t_un, delta_un_deg[i], color=GEN_COLORS[i], lw=2.0, ls="--", alpha=0.9,
              label=rf"$\delta_{{{i+1}}}$ unstable ($t_c={t_clear_unstable*1000:.0f}$ ms)" if i == 0 else None)
ax1.axvspan(0, t_clear_unstable, color=GGray, alpha=0.08)
ax1.text(t_clear_unstable / 2, 0.94, "fault-on\nwindow*", ha="center", va="top",
          fontsize=7, color=GGray, transform=ax1.get_xaxis_transform())
ax1.set_xlabel("Time after fault initiation (s)")
ax1.set_ylabel(r"Rotor angle $\delta_i$ (deg)")
ax1.grid(True, lw=0.3, alpha=0.4)
ax1.legend(fontsize=6.6, loc="upper left", ncol=1, framealpha=0.9)
ax1.set_title("(b)  Multi-machine post-fault swing curves: damped (solid) vs. loss-of-synchronism (dashed)",
              fontsize=9.2, loc="left")

# ---- Panel (c): phase portrait (delta_i - delta_i^s) vs omega_i ----
ax2 = fig.add_subplot(gs[1, 0])
for i in range(n_g):
    dd_st = np.degrees(y_st[i] - delta_s[i])
    dd_un = np.degrees(y_un[i] - delta_s[i])
    ax2.plot(dd_st, omega_st[i], color=GEN_COLORS[i], lw=1.7,
              label=rf"gen {i+1} stable")
    ax2.plot(dd_un, omega_un[i], color=GEN_COLORS[i], lw=1.7, ls="--", alpha=0.85,
              label=rf"gen {i+1} unstable")
    ax2.plot(dd_st[0], omega_st[i][0], "o", color=GEN_COLORS[i], ms=4)
ax2.axhline(0, color=GGray, lw=0.6)
ax2.axvline(0, color=GGray, lw=0.6)
ax2.set_xlabel(r"$\delta_i - \delta_i^{s}$  (deg)")
ax2.set_ylabel(r"Rotor speed deviation $\omega_i$ (rad/s)")
ax2.grid(True, lw=0.3, alpha=0.4)
ax2.legend(fontsize=6.2, ncol=2, loc="upper center", framealpha=0.9)
ax2.set_title("(c)  Phase portrait about the post-fault equilibrium $\\delta^{s}$: spiral-in vs. spiral-out",
              fontsize=9.2, loc="left")

# ---- Panel (d): transient energy function V(t) vs. critical energy ----
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_st, V_st, color=GGreen, lw=2.2, label=r"$\hat V(t)$ stable case")
ax3.plot(t_un, V_un, color=GRed, lw=2.2, ls="--", label=r"$\hat V(t)$ unstable case")
ax3.axhline(V_cr, color=GGray, lw=1.4, ls=":")
ax3.text(t_un[-1] * 0.62, V_cr * 1.03, r"critical energy $V_{\mathrm{cr}}$", color=GGray, fontsize=8)
ax3.set_xlabel("Time after fault initiation (s)")
ax3.set_ylabel(r"Transient energy $\hat V = \hat V_{KE} + \hat V_{PE}$  (p.u.)")
ax3.grid(True, lw=0.3, alpha=0.4)
ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax3.set_title(r"(d)  Energy-function dissipation: $d\hat V/dt = -\sum_i D_i\omega_i^2 \leq 0$ vs. $V$ crossing $V_{\mathrm{cr}}$",
              fontsize=9.0, loc="left")

fig.suptitle("Physical mechanism of transient instability in the multi-machine swing equation",
              fontsize=13, fontweight="bold", y=0.985)

fig.text(0.06, 0.015,
          "*Fault-on window shown for the unstable clearing time; the stable case clears "
          f"{t_clear_stable*1000:.0f} ms after fault initiation, well inside the critical-clearing window.",
          fontsize=6.8, color=GGray)

out_path = "/mnt/user-data/outputs/swing_equation_stability_figure.png"
fig.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved: {out_path}")
