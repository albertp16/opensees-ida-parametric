"""LHS of structural variants for the EOSD26 variant study (the abstract's Module 1).

User-defined ranges, uniform, as listed in the submitted abstract (concrete strength,
reinforcement ratio, column dimensions, bay width, storey height), plus the number of storeys
(2 or 3) as a sixth LHS dimension split at u = 0.5, so exactly half the variants have 2 storeys.
Ranges describe older, non-ductile Philippine low-rise RC frames; they are modelling choices
of this study, stated in the paper, not values from a reference.

  fc_MPa   17 - 28 MPa
  rho_col  0.010 - 0.030 (column longitudinal ratio)
  col_m    0.30 - 0.50 m (square column)
  bay_m    4.0 - 7.0 m
  storey_m 2.8 - 3.6 m
Sampling: scipy LatinHypercube (McKay et al. 1979), random-cd optimisation, fixed seed.
Usage: python lhs_variants.py [--n 160] [--seed 2026]
"""
import argparse
import numpy as np
from scipy.stats import qmc, spearmanr

RANGES = [("fc_MPa", 17.0, 28.0), ("rho_col", 0.010, 0.030), ("col_m", 0.30, 0.50),
          ("bay_m", 4.0, 7.0), ("storey_m", 2.8, 3.6)]


def sample(n=160, seed=2026):
    u = qmc.LatinHypercube(d=len(RANGES) + 1, seed=seed, optimization="random-cd").random(n)
    x = np.column_stack([lo + (hi - lo) * u[:, j] for j, (_, lo, hi) in enumerate(RANGES)])
    ns = np.where(u[:, -1] < 0.5, 2, 3)
    return ns, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ns, x = sample(a.n, a.seed)
    rho = spearmanr(x).correlation
    off = np.abs(rho[~np.eye(len(RANGES), dtype=bool)]).max()
    out = a.out or f"lhs_variants_seed{a.seed}.csv"
    with open(out, "w") as fh:
        fh.write(f"# EOSD26 variant study, LHS n={a.n}, seed={a.seed}, max spurious |rho_s|={off:.3f}\n")
        fh.write("realization,n_storey," + ",".join(r[0] for r in RANGES) + "\n")
        for i in range(a.n):
            fh.write(f"{i + 1},{ns[i]}," + ",".join(f"{v:.4f}" for v in x[i]) + "\n")
    print(out, "2-storey:", int((ns == 2).sum()), "3-storey:", int((ns == 3).sum()), "max |rho_s| =", round(off, 3))


if __name__ == "__main__":
    main()
