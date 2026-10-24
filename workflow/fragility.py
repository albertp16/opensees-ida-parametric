"""Lognormal fragility fitting (Baker 2015, Earthq. Spectra 31(1), 579-599).

fit_moments: IDA capacities, one collapse IM per record (method of moments, Baker Eqs. 2-3).
fit_mle:     stripe data, z_j of n_j records exceed the limit state at IM x_j (Baker Eq. 11).
             Use this when some records never reach the limit state within the IM cap.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, binom


def fit_moments(im_c):
    x = np.log(np.asarray(im_c, float))
    return float(np.exp(x.mean())), float(x.std(ddof=1))


def fit_mle(x, n, z):
    x, n, z = (np.asarray(v, float) for v in (x, n, z))

    def nll(p):
        th, b = np.exp(p[0]), np.exp(p[1])
        pr = np.clip(norm.cdf(np.log(x / th) / b), 1e-12, 1 - 1e-12)
        return -binom.logpmf(z, n, pr).sum()

    r = minimize(nll, [np.log(np.median(x)), np.log(0.4)], method="Nelder-Mead")
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))
