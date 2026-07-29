"""
shot_feasibility.py  (v2 -- fast, exact/CLT sampling)
------------------------------------------------------
This version uses the *exact* sampling distribution of the mean where a
closed form exists (Rademacher: sum of N iid +/-1 variables is
2*Binomial(N,1/2)-N, sampled exactly via rng.binomial -- no approximation),
and a CLT/normal approximation to the sampling distribution of the mean for
the Beta case (justified because the Beta(8,8) shot-noise model is smooth,
unimodal, non-lattice, and N is always >= 30 in the grids used here -- a
textbook regime for the CLT). This turns an O(T*N) memory-bound simulation
into an O(T) one with no loss of scientific validity, just a documented
approximation for the smooth-distribution case only.
"""
import numpy as np

rng = np.random.default_rng(20260721)

# --------------------------------------------------------------------------
# 1. Analytic Hoeffding-bound feasibility model (verbatim from the theorem)
# --------------------------------------------------------------------------
C_RANGE = 6.0
K_UNION = 1600.0        # manuscript's union-bound inflation factor, panel (a)
K_SINGLE = 2.0          # classical two-sided single-point Hoeffding factor
P_PREFACTOR = 16.0
SHOT_RATE_HZ = 21_560
BUDGET_S = 0.100

def required_shots(eps, delta, C=C_RANGE, K=K_UNION):
    return (C ** 2) * np.log(K / delta) / (2.0 * eps ** 2)

def achievable_delta(N, eps, C=C_RANGE, P=P_PREFACTOR):
    return P * np.exp(-2.0 * N * (eps / C) ** 2)

def achievable_delta_single(N, eps, C=C_RANGE):
    return K_SINGLE * np.exp(-2.0 * N * (eps / C) ** 2)

N_BUDGET = SHOT_RATE_HZ * BUDGET_S  # == 2156.0


# --------------------------------------------------------------------------
# 2. Fast exact / CLT sampling of the estimator mean
# --------------------------------------------------------------------------
MU_TRUE = 0.42
BETA_A = BETA_B = 8.0
_BETA_VAR01 = (BETA_A * BETA_B) / ((BETA_A + BETA_B) ** 2 * (BETA_A + BETA_B + 1))

def sample_mean_rademacher(N, T, rng):
    """Exact sampling distribution of the mean of N iid X in {mu-3, mu+3}."""
    k = rng.binomial(N, 0.5, size=T)          # number of +1 draws
    return MU_TRUE + 3.0 * (2.0 * k / N - 1.0)

def sample_mean_beta(N, T, rng, small_N_exact_cutoff=40):
    """Sampling distribution of the mean of N iid X = (mu-3) + 6*Beta(8,8).
    Exact Monte Carlo for small N (cheap); CLT/normal approximation for
    N >= small_N_exact_cutoff (valid here since the summands are smooth,
    unimodal and bounded)."""
    if N < small_N_exact_cutoff:
        u = rng.beta(BETA_A, BETA_B, size=(T, N))
        return (MU_TRUE - 3.0) + 6.0 * u.mean(axis=1)
    var_scaled = 36.0 * _BETA_VAR01
    sd = np.sqrt(var_scaled / N)
    return rng.normal(MU_TRUE, sd, size=T)

SAMPLERS = {'rademacher': sample_mean_rademacher, 'beta': sample_mean_beta}


def empirical_delta(kind, N, eps, T=20000, rng=rng):
    N_int = max(int(round(N)), 1)
    means = SAMPLERS[kind](N_int, T, rng)
    return float(np.mean(np.abs(means - MU_TRUE) >= eps))


def empirical_required_shots(kind, eps, delta_target, T=12000, rng=rng,
                              N_lo=2.0, N_hi=4.0e5):
    lo, hi = np.log(N_lo), np.log(N_hi)
    for _ in range(8):
        if empirical_delta(kind, np.exp(hi), eps, T=T, rng=rng) <= delta_target:
            break
        hi += np.log(4)
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        d = empirical_delta(kind, np.exp(mid), eps, T=T, rng=rng)
        if d <= delta_target:
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


# --------------------------------------------------------------------------
# 3. Monte Carlo campaign
# --------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    t0 = time.time()

    eps_grid_mc = np.array([0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.22, 0.28])
    delta_targets_mc = [0.01, 0.05]
    mc_panel_a = {kind: {d: np.array([empirical_required_shots(kind, e, d)
                                       for e in eps_grid_mc])
                          for d in delta_targets_mc}
                  for kind in ('rademacher', 'beta')}

    N_grid_mc = np.array([80, 250, 500, 900, 1400, 2000, 2700, 3500])
    eps_fixed_mc = [0.08, 0.05]
    mc_panel_b = {kind: {e: np.array([empirical_delta(kind, N, e)
                                       for N in N_grid_mc])
                          for e in eps_fixed_mc}
                  for kind in ('rademacher', 'beta')}

    print(f"Monte Carlo campaign complete in {time.time()-t0:.1f} s")
    i = 3  # eps = 0.09
    print("Rademacher empirical N (eps=0.09, delta=0.01):",
          mc_panel_a['rademacher'][0.01][i],
          " | analytic single-point N:", required_shots(0.09, 0.01, K=2.0),
          " | analytic union-bound N:", required_shots(0.09, 0.01))
