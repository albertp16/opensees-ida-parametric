"""Seismic add-on for the MIDAS verification model (paper Sect. 4): eigenvalue control, the NSCP 2015
design spectrum from the plans (sheet S-1) and two response-spectrum load cases (X, Y).

Spectrum (NSCP 2015 Sect. 208, UBC-97 form; values to be confirmed against the code hub):
  Zone 4, soil type SD, Na = 1.2, Nv = 1.6  ->  Ca = 0.44 Na = 0.528, Cv = 0.64 Nv = 1.024
  Ts = Cv / (2.5 Ca), T0 = 0.2 Ts; Sa = Ca + (2.5Ca - Ca) T/T0 (T < T0); 2.5 Ca (T0..Ts); Cv/T (T > Ts)
The function is written unreduced (in g); the load cases scale it by I/R = 1.5/8.5.
Written as a separate .mgt so a syntax problem here cannot break the main model import:
File > Import > midas Gen MGT File, on top of dpwh_school_verification.mgt.
Also writes nscp2015_spectrum.csv for entering the function by hand if needed.
"""
import numpy as np

NA, NV, I, R = 1.2, 1.6, 1.5, 8.5
CA, CV = 0.44 * NA, 0.64 * NV
TS = CV / (2.5 * CA)
T0 = 0.2 * TS


def sa(T):
    if T < T0:
        return CA + (2.5 * CA - CA) * T / T0
    if T <= TS:
        return 2.5 * CA
    return CV / T


def main(path="dpwh_school_seismic.mgt"):
    T = sorted(set([0.0, T0, TS] + list(np.round(np.arange(0.1, 4.01, 0.1), 3)) + [5.0, 6.0]))
    vals = [(t, sa(t)) for t in T]
    with open("nscp2015_spectrum.csv", "w") as fh:
        fh.write("period_s,Sa_g_unreduced\n" + "".join(f"{t:.4f},{v:.4f}\n" for t, v in vals))
    L = ["; Seismic add-on: import AFTER dpwh_school_verification.mgt (File > Import > midas Gen MGT File)",
         f"; NSCP 2015, Zone 4, SD, Na={NA}, Nv={NV}: Ca={CA:.3f}, Cv={CV:.3f}, T0={T0:.4f} s, Ts={TS:.4f} s; I={I}, R={R}",
         "", "*UNIT", "   KN, M, KJ, C", "",
         "*EIGEN-CTRL", "; TYPE, iFREQ, iITER, iDIM, TOL (Lanczos: number of modes)", "   LANCZOS, 12, 0, NO", "",
         "*SPFUNC", "; FUNC=NAME, iTYPE, SCALE, GRAV, DRATIO, STR/STEP, OPT1, OPT2, DESC",
         "; PERIOD1, VALUE1, PERIOD2, VALUE2, ...   (iTYPE 1 = normalized acceleration, g)",
         "   FUNC=NSCP2015_Z4_SD, 1, 1, 9.806, 0.05, 0, 0, 0, unreduced Na1.2 Nv1.6"]
    for k in range(0, len(vals), 4):
        L.append("      " + ", ".join(f"{t:.4f}, {v:.4f}" for t, v in vals[k:k + 4]))
    sc = I / R
    L += ["", "*SPLDCASE", "; NAME=LCNAME, iDIR, ANGLE, SCALE, ... ; FUNC list on the next line",
          f"   NAME=RSX, X, 0, {sc:.4f}, 0, NO, CQC, ", "      NSCP2015_Z4_SD",
          f"   NAME=RSY, Y, 0, {sc:.4f}, 0, NO, CQC, ", "      NSCP2015_Z4_SD", "", "*ENDDATA"]
    open(path, "w").write("\n".join(L) + "\n")
    print(path, f"Ca={CA:.3f} Cv={CV:.3f} T0={T0:.3f} Ts={TS:.3f}; design plateau = {2.5*CA*sc:.3f} g")


if __name__ == "__main__":
    main()
