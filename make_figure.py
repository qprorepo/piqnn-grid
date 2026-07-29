"""
make_figure.py
--------------
Builds the final 2-panel (1x2) shot-budget feasibility figure:

  (a) Required shots N_shots vs margin resolution eps_eta
      - manuscript's union-bound-corrected analytic curves (delta=0.01, 0.05)
      - classical single-point Hoeffding curve (delta=0.01) for reference
      - Monte Carlo empirical requirement, Rademacher (extremal) estimator
      - Monte Carlo empirical requirement, Beta(8,8) (realistic) estimator
      - 100 ms / 21.56 kHz shot-rate budget line at N = 2156

  (b) Achievable confidence delta* vs shot budget N_shots
      - manuscript's analytic curves (eps=0.08, 0.05)
      - Monte Carlo empirical delta, both estimator models
      - 99% confidence (delta=0.01) reference line

Output: shot_feasibility_figure.pdf and .png in /mnt/user-data/outputs/
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.patheffects as pe

from shot_feasibility import (
    required_shots, achievable_delta, achievable_delta_single, N_BUDGET,
    empirical_required_shots, empirical_delta, MU_TRUE,
)

rng = np.random.default_rng(20260721)

# ---------------------------------------------------------------- style ---
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_BLUE   = '#2E5FA3'
C_ORANGE = '#E0812A'
C_GREEN  = '#2F8F46'
C_RED    = '#C1272D'
C_PURPLE = '#7B4FA0'
C_GRAY   = '#808080'

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
fig.subplots_adjust(wspace=0.32, left=0.085, right=0.985, top=0.88, bottom=0.16)

# ==========================================================================
# Panel (a): required shots vs margin resolution
# ==========================================================================
ax = axes[0]
eps_curve = np.linspace(0.02, 0.30, 400)

ax.plot(eps_curve, required_shots(eps_curve, 0.01), color=C_BLUE, lw=2.2,
        label=r'$\delta_{\mathrm{conf}}=0.01$ (union-bound)')
ax.plot(eps_curve, required_shots(eps_curve, 0.05), color=C_ORANGE, lw=1.8,
        ls='--', label=r'$\delta_{\mathrm{conf}}=0.05$ (union-bound)')
ax.plot(eps_curve, required_shots(eps_curve, 0.01, K=2.0), color=C_GRAY,
        lw=1.3, ls=':', label=r'single-point Hoeffding, $\delta=0.01$')

# Monte Carlo overlay
eps_grid_mc = np.array([0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.22, 0.28])
mc_rad_001 = np.array([empirical_required_shots('rademacher', e, 0.01) for e in eps_grid_mc])
mc_beta_001 = np.array([empirical_required_shots('beta', e, 0.01) for e in eps_grid_mc])

ax.scatter(eps_grid_mc, mc_rad_001, marker='s', s=32, facecolor='white',
           edgecolor=C_BLUE, linewidth=1.3, zorder=5,
           label=r'MC, Rademacher (extremal), $\delta{=}0.01$')
ax.scatter(eps_grid_mc, mc_beta_001, marker='o', s=30, facecolor=C_GREEN,
           edgecolor='black', linewidth=0.5, zorder=5,
           label=r'MC, Beta(8,8) (realistic), $\delta{=}0.01$')

ax.axhline(N_BUDGET, color=C_GRAY, lw=1.0, ls='--', alpha=0.85)
ax.text(0.205, N_BUDGET + 130, '100 ms shot budget\n(21.56 kHz sustained rate)',
        fontsize=6.6, color=C_GRAY, ha='left', va='bottom')

ax.set_xlim(0.02, 0.30)
ax.set_ylim(0, 5000)
ax.set_xlabel(r'Margin resolution $\varepsilon_\eta$ (p.u.)')
ax.set_ylabel(r'Required shots $N_{\mathrm{shots}}$')
ax.set_title(r'(a) Shot Requirement vs.\ Margin Resolution', fontsize=10)
ax.grid(True, lw=0.3, alpha=0.35)
leg = ax.legend(loc='upper right', fontsize=6.2, framealpha=0.9,
                 edgecolor='gray', handlelength=1.6, borderpad=0.5)

# ==========================================================================
# Panel (b): achievable confidence vs shot budget
# ==========================================================================
ax2 = axes[1]
N_curve = np.linspace(1, 3000, 500)

ax2.semilogy(N_curve, achievable_delta(N_curve, 0.08), color=C_GREEN, lw=2.2,
             label=r'$\varepsilon_\eta=0.08$ (manuscript bound)')
ax2.semilogy(N_curve, achievable_delta(N_curve, 0.05), color=C_RED, lw=1.8,
             ls='--', label=r'$\varepsilon_\eta=0.05$ (manuscript bound)')

N_grid_mc = np.array([80, 250, 500, 900, 1400, 2000, 2700, 3500])
N_grid_mc = N_grid_mc[N_grid_mc <= 3000]
mc_rad_008 = np.array([max(empirical_delta('rademacher', N, 0.08), 1e-8) for N in N_grid_mc])
mc_beta_008 = np.array([max(empirical_delta('beta', N, 0.08), 1e-8) for N in N_grid_mc])
mc_rad_005 = np.array([max(empirical_delta('rademacher', N, 0.05), 1e-8) for N in N_grid_mc])
mc_beta_005 = np.array([max(empirical_delta('beta', N, 0.05), 1e-8) for N in N_grid_mc])

ax2.scatter(N_grid_mc, mc_rad_008, marker='s', s=32, facecolor='white',
            edgecolor=C_GREEN, linewidth=1.3, zorder=5,
            label=r'MC, Rademacher, $\varepsilon_\eta{=}0.08$')
ax2.scatter(N_grid_mc, mc_beta_005, marker='o', s=30, facecolor=C_PURPLE,
            edgecolor='black', linewidth=0.5, zorder=5,
            label=r'MC, Beta(8,8), $\varepsilon_\eta{=}0.05$')

ax2.axhline(0.01, color=C_PURPLE, lw=1.0, ls='--', alpha=0.85)
ax2.text(2250, 0.0135, r'99\% confidence', fontsize=7, color=C_PURPLE, ha='left')

ax2.set_xlim(0, 3000)
ax2.set_ylim(1e-8, 1)
ax2.set_xlabel(r'Shot budget $N_{\mathrm{shots}}$')
ax2.set_ylabel(r'Achievable $\delta_{\mathrm{conf}}^\ast$')
ax2.set_title(r'(b) Confidence vs.\ Shot Budget', fontsize=10)
ax2.grid(True, lw=0.3, alpha=0.35, which='both')
ax2.legend(loc='upper right', fontsize=6.2, framealpha=0.9, edgecolor='gray',
           handlelength=1.6, borderpad=0.5)

for a in axes:
    a.tick_params(labelsize=7.5)

fig.savefig('/mnt/user-data/outputs/shot_feasibility_figure.pdf', dpi=400)
fig.savefig('/mnt/user-data/outputs/shot_feasibility_figure.png', dpi=300)
print("Saved figure.")

print("\n--- Numeric summary (eps=0.09, delta=0.01) ---")
print("Manuscript union-bound N   :", required_shots(0.09, 0.01))
print("Single-point Hoeffding N   :", required_shots(0.09, 0.01, K=2.0))
print("MC Rademacher (extremal) N :", empirical_required_shots('rademacher', 0.09, 0.01))
print("MC Beta(8,8) (realistic) N :", empirical_required_shots('beta', 0.09, 0.01))
