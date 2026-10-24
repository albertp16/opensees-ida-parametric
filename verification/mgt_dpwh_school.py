"""Write a MIDAS Gen .mgt text file for the 3D DPWH-DepEd school verification model (Sect. 4).

Geometry, sections and grid follow verification/fig2_render.py (the OpenSeesPy verification model), so the
two programs start from the same input: 2018 DPWH plans, exposure B. Units kN, m.

Loads (paper Sect. 4; DPWH 2018 plans, DepEd PSIP 3 MPSS, NSCP 2015 Table 204-2):
  DL  : self-weight of beams/columns (MIDAS *SELFWEIGHT) + 100 mm slab (2.4 kPa, slab not modelled)
        + CHB walls as beam line loads (MPSS 2.46 kPa for 100 mm, 3.45 kPa for 150 mm, x clear height)
  SDL : floors 1.40 kPa, roof deck 1.75 kPa, steel roofing 0.076 kPa (truss/purlin weight from S-11: TODO)
  LL  : classrooms 1.90, corridors 3.80, roof 1.00 kPa
Slab loads go to the beams by two-way tributary areas (45-degree lines): trapezoids on the long sides,
triangles on the short sides, applied as uniform loads of the same total (reaction totals are exact).
Mass: DL, SDL x 1.05 and LL x 0.25 (FEMA P695 Eq. 6-1) through *LOADTOMASS.

Walls are an assumption until the architectural layout is checked; edit WALLS below.
Default: 150 mm CHB on grid C (exterior), 100 mm on grid B (corridor side), all X bays, plus 150 mm
end walls on grids 1 and 13 over the 7 m bay. Walls of storeys 2 and 3 stand on the level-1 and
level-2 beams; ground-storey walls bear on grade and are not in the model.

Usage: python mgt_dpwh_school.py [--out dpwh_school_verification.mgt] [--nodal]
  --nodal writes the beam loads as equivalent nodal loads (*CONLOAD) instead of *BEAMLOAD, if the
  MIDAS version rejects the beam-load lines.
Check after import: Model > Structure Type (convert self-weight to masses X, Y), Story data with
rigid diaphragm (Structure > Building > Story, auto), then the load table and total mass.
"""
import argparse
from collections import defaultdict
import numpy as np

XS = np.concatenate([[0, 5.0], 5.0 + 4.5 * np.arange(1, 11), [55.0]])
YS = [0.0, 2.5, 9.5]          # grid A (corridor edge), B, C
ZS = [0.0, 3.15, 6.30, 9.45]
EC = 4700 * np.sqrt(27.6) * 1e3   # kN/m2
GAMMA_C = 23.56               # kN/m3, plan S-1
SLAB = 0.100 * 24.0           # kPa
SDL = {1: 1.40, 2: 1.40, 3: 1.75 + 0.076}
LL = {"corridor": 3.80, "class": 1.90, "roof": 1.00}
WALL_KPA = {100: 2.46, 150: 3.45}
# (gridline, thickness_mm): X-direction walls on grid j (all bays); Y walls on end grids over the 7 m bay
WALLS = dict(x=[(2, 150), (1, 100)], y_end=150)

COLS = {"C-4": 0.55, "C-3": 0.40, "C-2": 0.50, "C-1": 0.50}
BEAMS = {"B-250x400": (0.25, 0.40), "B-350x500": (0.35, 0.50), "B-400x550": (0.40, 0.55)}


def nid(i, j, k):
    return 1 + i + 100 * j + 1000 * k


def build():
    nodes = {nid(i, j, k): (x, y, z) for k, z in enumerate(ZS) for j, y in enumerate(YS) for i, x in enumerate(XS)}
    secs = list(COLS) + list(BEAMS)
    sid = {s: n + 1 for n, s in enumerate(secs)}
    els = []   # (n1, n2, section)
    for k in range(3):
        for j in range(3):
            for i in range(len(XS)):
                mark = "C-4" if i in (0, len(XS) - 1) else ["C-3", "C-2", "C-1"][j]
                els.append((nid(i, j, k), nid(i, j, k + 1), mark))
    xbeam, ybeam = {}, {}
    for k in range(1, 4):
        for j in range(3):
            for i in range(len(XS) - 1):
                xbeam[(i, j, k)] = len(els)
                els.append((nid(i, j, k), nid(i + 1, j, k), "B-250x400" if j == 0 else "B-350x500"))
        for i in range(len(XS)):
            ybeam[(i, 0, k)] = len(els)
            els.append((nid(i, 0, k), nid(i, 1, k), "B-250x400"))
            ybeam[(i, 1, k)] = len(els)
            els.append((nid(i, 1, k), nid(i, 2, k), "B-400x550"))
    return nodes, secs, sid, els, xbeam, ybeam


def tributary(xbeam, ybeam):
    """Uniform beam loads (kN/m) per load case from two-way panel distribution and walls."""
    w = {lc: defaultdict(float) for lc in ("DL", "SDL", "LL")}
    for k in range(1, 4):
        for j in range(2):                       # panel between grid j and j+1
            for i in range(len(XS) - 1):
                lx, ly = XS[i + 1] - XS[i], YS[j + 1] - YS[j]
                q = {"DL": SLAB, "SDL": SDL[k],
                     "LL": LL["roof"] if k == 3 else (LL["corridor"] if j == 0 else LL["class"])}
                a, b = min(lx, ly), max(lx, ly)
                trap = (a / 2) * (b - a / 2) / b   # equivalent uniform width on each long side
                tri = a / 4                        # equivalent uniform width on each short side
                x_is_long = lx >= ly
                for lc, qq in q.items():
                    for e in (xbeam[(i, j, k)], xbeam[(i, j + 1, k)]):
                        w[lc][e] += qq * (trap if x_is_long else tri)
                    for e in (ybeam[(i, j, k)], ybeam[(i + 1, j, k)]):
                        w[lc][e] += qq * (tri if x_is_long else trap)
    # CHB walls: wall in storey s sits on the beams at level s (s = 1, 2) ... clear height = storey - beam depth
    for k in (1, 2, 3):
        for jj, t in WALLS["x"]:
            for i in range(len(XS) - 1):
                h = 3.15 - BEAMS["B-350x500" if jj else "B-250x400"][1]
                if k <= 2:    # wall standing on the level-k beam (storeys above level 1 and 2)
                    w["DL"][xbeam[(i, jj, k)]] += WALL_KPA[t] * h
        for i in (0, len(XS) - 1):
            h = 3.15 - BEAMS["B-400x550"][1]
            if k <= 2:
                w["DL"][ybeam[(i, 1, k)]] += WALL_KPA[WALLS["y_end"]] * h
    return w


def write(path, nodal=False):
    nodes, secs, sid, els, xbeam, ybeam = build()
    w = tributary(xbeam, ybeam)
    L = []
    L += ["; MIDAS Gen text file: DPWH-DepEd modified standard 3-storey school, exposure B (EOSD26 Sect. 4)",
          "; generated by verification/mgt_dpwh_school.py; geometry = verification/fig2_render.py", "",
          "*UNIT", "   KN, M, KJ, C", ""]
    L += ["*MATERIAL", "; iMAT, TYPE, MNAME, SPHEAT, HEATCO, PLAST, TUNIT, bMASS, DAMPRATIO, [DATA1]",
          "; [DATA1] : 2, ELAST, POISN, THERMAL, DEN, MASS",
          f"   1, CONC, fc27.6, 0, 0, , C, NO, 0.05, 2, {EC:.4e}, 0.2, 1.0000e-005, {GAMMA_C}, {GAMMA_C / 9.806:.4f}", ""]
    L += ["*SECTION", "; iSEC, TYPE, SNAME, [OFFSET], bSD, bWE, SHAPE, [DATA1], [DATA2]"]
    for s in secs:
        hh, bb = (COLS[s], COLS[s]) if s in COLS else (BEAMS[s][1], BEAMS[s][0])
        L.append(f"   {sid[s]}, DBUSER, {s}, CC, 0, 0, 0, 0, 0, 0, YES, NO, SB, 2, {hh}, {bb}, 0, 0, 0, 0, 0, 0, 0, 0")
    L += ["", "*NODE", "; iNO, X, Y, Z"]
    L += [f"   {n}, {x:.3f}, {y:.3f}, {z:.3f}" for n, (x, y, z) in sorted(nodes.items())]
    L += ["", "*ELEMENT", "; iEL, TYPE, iMAT, iPRO, iN1, iN2, ANGLE, iSUB"]
    L += [f"   {e + 1}, BEAM, 1, {sid[s]}, {n1}, {n2}, 0, 0" for e, (n1, n2, s) in enumerate(els)]
    base = sorted(n for n, (x, y, z) in nodes.items() if z == 0)
    L += ["", "*CONSTRAINT", "; NODE_LIST, CONST(Dx,Dy,Dz,Rx,Ry,Rz), GROUP"]
    L += [f"   {n}, 111111, " for n in base]
    L += ["", "*STLDCASE", "; LCNAME, LCTYPE, DESC", "   DL, D, self-weight + slab + CHB walls",
          "   SDL, D, superimposed dead", "   LL, L, live load", ""]
    for lc in ("DL", "SDL", "LL"):
        L += [f"*USE-STLD, {lc}", ""]
        if lc == "DL":
            L += ["*SELFWEIGHT", "; X, Y, Z, GROUP", "   0, 0, -1, ", ""]
        if nodal:
            acc = defaultdict(float)
            for e, q in w[lc].items():
                n1, n2, _ = els[e]
                ln = np.linalg.norm(np.subtract(nodes[n2], nodes[n1]))
                acc[n1] += q * ln / 2
                acc[n2] += q * ln / 2
            L += ["*CONLOAD", "; NODE_LIST, FX, FY, FZ, MX, MY, MZ, GROUP"]
            L += [f"   {n}, 0, 0, {-f:.3f}, 0, 0, 0, " for n, f in sorted(acc.items())]
        else:
            # Gen NX: bPROJ, then bECCEN (YES/NO), ECCDIR, I-END, J-END, bJ-END, then D1..P4, GROUP, bAddH, H1, H2, bUseDist
            L += ["*BEAMLOAD", "; ELEM_LIST, CMD, TYPE, DIR, bPROJ, [ECCEN], [VALUE], GROUP, [ADDITIONAL]",
                  "; [ECCEN] : bECCEN, ECCDIR, I-END, J-END, bJ-END ; [VALUE] : D1, P1, D2, P2, D3, P3, D4, P4"]
            L += [f"   {e + 1}, BEAM, UNILOAD, GZ, NO, NO, aDir[1], , , , 0, {-q:.3f}, 1, {-q:.3f}, 0, 0, 0, 0, , NO, 0, 0, NO, "
                  for e, q in sorted(w[lc].items())]
        L.append("")
    L += ["*LOADTOMASS", "; DIR, bNODAL, bBEAM, bFLOOR, bPRES, GRAV", "   XY, YES, YES, YES, YES, 9.806",
          "; LCNAME1, FACTOR1, LCNAME2, FACTOR2, ...", "   DL, 1.05, SDL, 1.05, LL, 0.25", "", "*ENDDATA"]
    open(path, "w").write("\n".join(L) + "\n")
    tot = {lc: sum(q * np.linalg.norm(np.subtract(nodes[els[e][1]], nodes[els[e][0]])) for e, q in w[lc].items()) for lc in w}
    sw = sum((COLS[s] ** 2 if s in COLS else BEAMS[s][0] * BEAMS[s][1]) * GAMMA_C *
             np.linalg.norm(np.subtract(nodes[n2], nodes[n1])) for n1, n2, s in els)
    return len(nodes), len(els), tot, sw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dpwh_school_verification_v3.mgt")
    ap.add_argument("--nodal", action="store_true")
    a = ap.parse_args()
    nn, ne, tot, sw = write(a.out, a.nodal)
    print(f"{a.out}: {nn} nodes, {ne} elements")
    print(f"expected totals (kN): member self-weight {sw:.0f}; DL beam loads {tot['DL']:.0f}; SDL {tot['SDL']:.0f}; LL {tot['LL']:.0f}")
