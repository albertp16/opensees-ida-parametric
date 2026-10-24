"""Module 2: model-agnostic IDA runner with hunt & fill tracing.

Tracing follows Vamvatsikos & Cornell (2004, Earthq. Spectra 20(2), 523-553):
hunt up with increasing steps until the first collapse, bracket between the highest
non-collapse and lowest collapse IM, then fill the largest IM gaps below.
Collapse in a single run = solver non-convergence (after fallbacks) or theta_max above
the numerical collapse drift. The IM-based capacity (last point whose tangent slope is
>= 20% of the elastic slope; Vamvatsikos & Cornell 2002, p. 502) is evaluated on the
traced curve afterwards, and the governing trigger is stored.

The model lives in the user's script. It must provide:
    build(params)                  -> wipes and builds the model, applies gravity (P-Delta)
    run_gm(acc_g, dt, scale)       -> runs the transient; returns (theta_max, converged)
Both are passed in, so this file never needs to know the building.
"""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class IDASettings:
    im_first: float = 0.05        # g, first run
    im_step0: float = 0.25        # g, first hunting step
    im_step_inc: float = 0.10     # g, step increase per hunting run
    collapse_drift: float = 0.10  # numerical collapse if theta_max exceeds this
    max_runs: int = 12            # total runs per record
    bracket_tol: float = 0.05     # g, stop refining the collapse bracket below this gap
    slope_ratio: float = 0.20     # IM-based rule: 20% of the elastic slope
    im_cap: float = 6.0           # g, give up hunting above this (record cannot collapse)


@dataclass
class IDAResult:
    record_id: str
    realization: int
    points: list = field(default_factory=list)  # (IM, theta_max, collapsed)
    im_collapse: float = float("nan")
    trigger: str = ""
    n_runs: int = 0


def trace(build, run_gm, params, acc_g, dt, sa_unscaled, rec_id="", realization=0, s=IDASettings()):
    """Hunt & fill one record. sa_unscaled: Sa(T1, 5%) of the unscaled record in g."""
    res = IDAResult(rec_id, realization)

    def run(im):
        build(params)
        theta, ok = run_gm(acc_g, dt, im / sa_unscaled)
        col = (not ok) or (theta > s.collapse_drift)
        res.points.append((im, float(theta), bool(col), bool(ok)))
        res.n_runs += 1
        return col

    # hunt
    im, step = s.im_first, s.im_step0
    lo, hi = 0.0, None
    while res.n_runs < s.max_runs and im <= s.im_cap:
        if run(im):
            hi = im
            break
        lo = im
        im = im + step
        step += s.im_step_inc
    # bracket
    while hi is not None and res.n_runs < s.max_runs and (hi - lo) > s.bracket_tol:
        mid = 0.5 * (lo + hi)
        if run(mid):
            hi = mid
        else:
            lo = mid
    # fill the largest gaps below the last non-collapse
    while res.n_runs < s.max_runs:
        ims = sorted([0.0] + [p[0] for p in res.points if p[0] <= lo])
        gaps = np.diff(ims)
        if len(gaps) == 0 or gaps.max() < s.bracket_tol:
            break
        j = int(np.argmax(gaps))
        run(0.5 * (ims[j] + ims[j + 1]))

    res.im_collapse, res.trigger = capacity(res.points, hi, s)
    return res


def capacity(points, hi, s=IDASettings()):
    """Collapse IM and trigger from a traced curve.

    Returns the lower of (a) the first collapsing run (non-convergence or drift limit) and
    (b) the IM-based 20% slope point on the non-collapsed branch.
    """
    nc = sorted([p for p in points if not p[2]])
    im_slope = float("nan")
    if len(nc) >= 2:
        k_el = nc[0][0] / max(nc[0][1], 1e-9)
        for (i0, t0, *_), (i1, t1, *_) in zip(nc[:-1], nc[1:]):
            k = (i1 - i0) / max(t1 - t0, 1e-12)
            if k < s.slope_ratio * k_el:
                im_slope = i0
                break
    if hi is None and np.isnan(im_slope):
        return float("nan"), "no collapse within IM cap"
    first_col = [p for p in points if p[2] and p[0] == hi]
    trig_hi = "" if hi is None else ("non-convergence" if not first_col[0][3] else "drift limit")
    if not np.isnan(im_slope) and (hi is None or im_slope < hi):
        return im_slope, "20% slope"
    return hi, trig_hi


def rows(res):
    """One dict per run, for the structured dataset (Module 3)."""
    return [dict(realization=res.realization, record=res.record_id, im_g=im, theta_max=th,
                 collapsed=col, converged=ok, im_collapse_g=res.im_collapse, trigger=res.trigger)
            for im, th, col, ok in res.points]
