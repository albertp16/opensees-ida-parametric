"""Add the eigen and seismic input to the open MIDAS Gen NX model through the MIDAS API (paper Sect. 4).

Run on the PC where Gen NX is open, after importing dpwh_school_verification_v3.mgt:
    set MIDAS_MAPI_KEY=<your MAPI key>        (Windows; or export on macOS/Linux)
    python midas_api_seismic.py               add STYP, EIGV, SPFC, SPLC
    python midas_api_seismic.py --run         ...and run the analysis (doc/ANAL)

Endpoints and keys follow the MIDAS API online manual (support.midasuser.com):
  db/STYP  Structure Type: self-weight converted to X, Y masses, lumped mass, g = 9.806
  db/EIGV  Eigenvalue Analysis Control: Lanczos, 12 modes
  db/SPFC  Response Spectrum Function: NSCP 2015, Zone 4, SD, Na 1.2, Nv 1.6 (unreduced, normalized accel.)
  db/SPLC  Response Spectrum Load Cases RSX (0 deg) and RSY (90 deg), CQC, scale I/R = 1.5/8.5, 5% modal damping
Story data with rigid diaphragms is not set here: Structure > Building > Story (auto generate).
"""
import argparse, json, os, sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgt_seismic import sa, CA, CV, T0, TS, I, R   # same spectrum as the .mgt add-on

BASE = os.environ.get("MIDAS_BASE_URL", "https://moa-engineers.midasit.com:443/gen")


def call(method, path, body=None, key=None):
    req = urllib.request.Request(f"{BASE}/{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"MAPI-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def upsert(path, body, key):
    s, out = call("POST", path, body, key)
    if s >= 400 or (isinstance(out, dict) and "error" in json.dumps(out).lower()):
        s, out = call("PUT", path, body, key)
    print(f"{path}: HTTP {s} {str(out)[:200]}")
    return s


def payloads():
    T = sorted(set([0.0, T0, TS] + [round(0.1 * k, 3) for k in range(1, 41)] + [5.0, 6.0]))
    styp = {"Assign": {"1": {"STYP": 0, "MASS": 1, "bMASSOFFSET": False, "bSELFWEIGHT": True, "SMASS": 2,
                             "GRAV": 9.806, "TEMP": 0, "bALIGNBEAM": False, "bALIGNSLAB": False, "bROTRIGID": False}}}
    eigv = {"Assign": {"1": {"TYPE": "LANCZOS", "iFREQ": 12, "bMINMAX": False, "FRMIN": 0, "FRMAX": 0, "bSTRUM": False}}}
    spfc = {"Assign": {"1": {"NAME": "NSCP2015_Z4_SD", "iTYPE": 1, "iMETHOD": 0, "SCALE": 1, "GRAV": 9.806,
                             "DRATIO": 0.05, "DESC": f"unreduced, Ca={CA:.3f} Cv={CV:.3f}",
                             "aFUNC": [{"PERIOD": t, "VALUE": round(sa(t), 4)} for t in T]}}}
    sc = round(I / R, 4)
    case = lambda name, ang: {"NAME": name, "DESC": "NSCP 2015 RS, I/R = 1.5/8.5", "DIR": "XY", "ANGLE": ang,
                              "SCALE": sc, "PMFT": 1, "aFUNCNAME": ["NSCP2015_Z4_SD"], "INTERP": "LINEAR",
                              "COMTYPE": "CQC", "bADDSIGN": False, "iSIGNTYPE": 1, "bDAMP": True, "iMDTYPE": 1,
                              "bCDAMP": False, "DALL": 0.05}
    splc = {"Assign": {"1": case("RSX", 0), "2": case("RSY", 90)}}
    return [("db/STYP", styp), ("db/EIGV", eigv), ("db/SPFC", spfc), ("db/SPLC", splc)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="also run the analysis")
    ap.add_argument("--dry", action="store_true", help="print the JSON only")
    a = ap.parse_args()
    if a.dry:
        for p, b in payloads():
            print(p, json.dumps(b)[:300], "...")
        sys.exit()
    key = os.environ.get("MIDAS_MAPI_KEY")
    if not key:
        sys.exit("Set MIDAS_MAPI_KEY first (MIDAS Gen NX > Apps > API Settings).")
    for p, b in payloads():
        if p == "db/STYP":
            s, out = call("PUT", p, b, key); print(f"{p}: HTTP {s} {str(out)[:200]}")
        else:
            upsert(p, b, key)
    if a.run:
        s, out = call("POST", "doc/ANAL", {}, key)
        print(f"doc/ANAL: HTTP {s} {str(out)[:200]}")
