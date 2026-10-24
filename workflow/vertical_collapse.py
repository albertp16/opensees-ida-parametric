"""Vertical (axial-failure) collapse check by post-processing IDA results.

Method follows Liel et al. (2011), J. Struct. Eng. 137(4), p. 497: collapse modes related to column
shear failure are detected by post-processing; "the vertical collapse limit state is reached if,
during the analysis, the drift in any column exceeds the median value of that column's component
fragility function", and the record's collapse intensity is lowered when this happens first.

Median drift at loss of axial capacity: Elwood & Moehle (2005), ACI Struct. J. 102(4), axial
capacity model for shear-damaged columns:
    (Delta/L)_axial = 0.04 (1 + tan^2 th) / (tan th + P s / (A_st f_yt d_c tan th)),  th = 65 deg
P axial load, s tie spacing, A_st tie area (all legs crossing the shear crack), f_yt tie yield
strength, d_c core depth (centre-line of ties).

Simplifications (state in the paper):
- column drift is taken as the peak interstorey drift theta_max (CDR <= IDR, so this is conservative);
- the governing column is the one with the smallest axial-failure drift (first-storey interior column);
- theta_max is stored per run as the maximum over storeys, so the check is applied to it directly.
Inputs: variant_study_results.csv (Module 3), LHS file. Output: variant_capacity_vertical.csv.
"""
import csv, collections, math, os
import numpy as np
from fragility import fit_moments

HERE = os.path.dirname(os.path.abspath(__file__))
TH = math.radians(65.0)
TIE_AREA = 2 * math.pi * 0.010**2 / 4      # 2 legs of 10 mm ties (m2), as in variant_frame.py
FYT = 275e3                                 # kN/m2
S_TIE = 0.200
TIE_CL = 0.045                              # cover to tie centre-line (m)
Q, TRIB = 7.0, 4.5                          # kPa, m (variant_frame.py)


def axial_drift(P, b):
    dc = b - 2 * TIE_CL
    t = math.tan(TH)
    return 0.04 * (1 + t**2) / (t + P * S_TIE / (TIE_AREA * FYT * dc * t))


def limit_for(p):
    ns, L, b = int(p["n_storey"]), p["bay_m"], p["col_m"]
    P_int = Q * TRIB * L * ns                # first-storey interior column
    P_ext = Q * TRIB * L / 2 * ns
    return min(axial_drift(P_int, b), axial_drift(P_ext, b))


def im_at_drift(pts, lim):
    """First IM at which theta_max reaches lim (linear interpolation), before sidesway collapse."""
    prev_im, prev_th = 0.0, 0.0
    for im, th, col in sorted(pts):
        if col:
            return None
        if th >= lim:
            return prev_im + (im - prev_im) * (lim - prev_th) / max(th - prev_th, 1e-12)
        prev_im, prev_th = im, th
    return None


def main(results=os.path.join(HERE, "..", "results", "variant_study_results.csv"),
         lhs=os.path.join(HERE, "..", "data", "lhs_variants_seed2026.csv"),
         out=os.path.join(HERE, "..", "results", "variant_capacity_vertical.csv")):
    d = np.genfromtxt(lhs, delimiter=",", names=True, skip_header=1)
    P = {int(r["realization"]): {k: float(r[k]) for k in d.dtype.names} for r in d}
    pts = collections.defaultdict(list)
    side = {}
    T1 = {}
    for r in csv.DictReader(open(results)):
        k = (int(r["realization"]), r["record"])
        pts[k].append((float(r["im_g"]), float(r["theta_max"]), r["collapsed"] == "True"))
        side[k] = float(r["im_collapse_g"])
        T1[k[0]] = float(r["T1_s"])
    rows, n_vert = [], 0
    for v in sorted(P):
        lim = limit_for(P[v])
        s_side, s_comb = [], []
        for (vv, rec), ims in pts.items():
            if vv != v:
                continue
            a = side[(vv, rec)]
            b = im_at_drift(ims, lim)
            c = a if b is None or b >= a else b
            n_vert += c < a
            s_side.append(a)
            s_comb.append(c)
        m1, b1 = fit_moments(s_side)
        m2, b2 = fit_moments(s_comb)
        rows.append([v, int(P[v]["n_storey"]), T1[v], lim, m1, b1, m2, b2])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["realization", "n_storey", "T1_s", "axial_drift_limit", "Sa_c_sidesway_g", "beta_sidesway",
                    "Sa_c_with_vertical_g", "beta_with_vertical"])
        w.writerows(rows)
    return np.array(rows), n_vert


if __name__ == "__main__":
    o, nv = main()
    red = o[:, 6] / o[:, 4] - 1
    print(f"axial-failure drift limit: {o[:,3].min():.3f}-{o[:,3].max():.3f} (median {np.median(o[:,3]):.3f})")
    print(f"IDAs governed by vertical collapse: {nv} of 2080")
    print(f"median Sa_c sidesway {np.median(o[:,4]):.2f} g -> with vertical {np.median(o[:,6]):.2f} g; "
          f"reduction per variant {(-red).min()*100:.0f}-{(-red).max()*100:.0f}% (median {np.median(-red)*100:.0f}%)")
    for n in (2, 3):
        k = o[:, 1] == n
        print(f"{n} storeys: sidesway {np.median(o[k,4]):.2f} g, with vertical {np.median(o[k,6]):.2f} g, beta {np.median(o[k,7]):.2f}")
    print(f"range with vertical: {o[:,6].min():.2f}-{o[:,6].max():.2f} g")
