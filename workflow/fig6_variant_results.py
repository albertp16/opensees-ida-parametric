"""Fig. 6 (variant study results), drawn from Module 3 output.

(a) IDA curves of one variant for every record, with the median and 16th/84th percentile
    drift at each IM (counted over records, collapsed records taken as infinite drift);
(b) median collapse capacity (method of moments, Baker 2015, Eqs. 2-3) of every variant
    against T1, by number of storeys.

Usage:
  python fig6_variant_results.py --one ../results/variant_study_results.csv --all ../results/variant_study_results.csv --variant 2
  add --test-signals while the inputs come from synthetic test signals (adds a footer note).
"""
import argparse, csv, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fragility import fit_moments

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("realization",):
            r[k] = int(r[k])
        for k in ("im_g", "theta_max", "im_collapse_g", "T1_s"):
            r[k] = float(r[k])
        r["collapsed"] = r["collapsed"] in ("True", "1", "true")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", required=True)
    ap.add_argument("--all", required=True)
    ap.add_argument("--variant", type=int, default=2)
    ap.add_argument("--lhs", default=__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "data", "lhs_variants_seed2026.csv"))
    ap.add_argument("--out", default="fig6-variant-results")
    ap.add_argument("--test-signals", action="store_true")
    ap.add_argument("--vertical", default=None, help="variant_capacity_vertical.csv: adds the axial-failure drift limit (a) and combined medians (b)")
    a = ap.parse_args()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.5))
    # (a)
    one = [r for r in load(a.one) if r["realization"] == a.variant]
    by_rec = defaultdict(list)
    for r in one:
        by_rec[r["record"]].append(r)
    grid = np.linspace(0, max(r["im_g"] for r in one), 60)
    curves = []
    for rec, pts in by_rec.items():
        pts = sorted(pts, key=lambda r: r["im_g"])
        nc = [(0.0, 0.0)] + [(r["theta_max"], r["im_g"]) for r in pts if not r["collapsed"]]
        th, im = np.array(nc).T
        imc = pts[0]["im_collapse_g"]
        ax.plot(th * 100, im, color="#9fb7d4", lw=0.7)
        if np.isfinite(imc):
            ax.plot([th[-1] * 100, 10], [im[-1], im[-1]], color="#9fb7d4", lw=0.7, ls=":")
        d = np.interp(grid, im, th, right=np.inf)
        d[grid > (imc if np.isfinite(imc) else np.inf)] = np.inf
        curves.append(d)
    C = np.array(curves)
    for q, ls, lab in ((50, "-", "median"), (16, "--", "16th / 84th"), (84, "--", None)):
        v = np.percentile(np.where(np.isfinite(C), C, 1e9), q, axis=0, method="lower")
        ok = v < 0.10   # stop the percentile curve where that fraction of records has collapsed
        ax.plot(v[ok] * 100, grid[ok], color="#173f73", lw=1.4 if q == 50 else 1.0, ls=ls, label=lab)
    VC = None
    if a.vertical:
        VC = {int(r["realization"]): r for r in csv.DictReader(open(a.vertical))}
        lim = float(VC[a.variant]["axial_drift_limit"]) * 100
        ax.axvline(lim, color="#b33", lw=0.9, ls="-.")
        ax.text(lim + 0.15, 0.12, "column axial\nfailure drift", color="#b33", fontsize=6.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"peak interstorey drift $\theta_{\max}$ (%)")
    ax.set_ylabel(r"$S_a(T_1)$ (g)")
    ax.legend(frameon=False, loc="center right")
    ax.grid(lw=0.3, alpha=0.5)
    ax.text(0.02, 0.97, f"(a) variant {a.variant}, {len(by_rec)} records", transform=ax.transAxes, va="top", fontweight="bold")
    # (b)
    rows = load(a.all)
    cap = defaultdict(dict)
    T1 = {}
    for r in rows:
        cap[r["realization"]][r["record"]] = r["im_collapse_g"]
        T1[r["realization"]] = r["T1_s"]
    lhs = np.genfromtxt(a.lhs, delimiter=",", names=True, skip_header=1)
    ns = {int(r["realization"]): int(r["n_storey"]) for r in lhs}
    med = {}
    for v, c in cap.items():
        x = [x for x in c.values() if np.isfinite(x)]
        if len(x) >= 2:
            med[v] = fit_moments(x)[0]
    for n, col, mk in ((2, "#4f8fd6", "o"), (3, "#e0662f", "s")):
        vs = [v for v in med if ns[v] == n]
        if VC:
            bx.scatter([T1[v] for v in vs], [med[v] for v in vs], s=9, facecolors="none", edgecolors=col, marker=mk, lw=0.5, alpha=0.6)
            bx.scatter([T1[v] for v in vs], [float(VC[v]["Sa_c_with_vertical_g"]) for v in vs], s=9, c=col, marker=mk, lw=0, alpha=0.9, label=f"{n} storeys")
        else:
            bx.scatter([T1[v] for v in vs], [med[v] for v in vs], s=9, c=col, marker=mk, lw=0, alpha=0.85, label=f"{n} storeys")
    bx.set_xlabel(r"$T_1$ (s)")
    bx.set_ylabel(r"median collapse $\hat S_{a,C}$ (g)")
    bx.legend(frameon=False, loc="upper right")
    bx.grid(lw=0.3, alpha=0.5)
    bx.text(0.02, 0.97, f"(b) {len(med)} variants", transform=bx.transAxes, va="top", fontweight="bold")
    if VC:
        bx.text(0.02, 0.88, "filled: with vertical collapse check\nopen: sidesway only", transform=bx.transAxes, va="top", fontsize=6.5)
    if a.test_signals:
        fig.text(0.995, 0.005, "layout check: synthetic test signals, not the P695 records", ha="right", va="bottom",
                 fontsize=5.5, color="0.45", style="italic")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("variants with a median:", len(med))


if __name__ == "__main__":
    main()
