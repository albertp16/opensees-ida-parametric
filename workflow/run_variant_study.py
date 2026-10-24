"""EOSD26 variant study: 160 LHS variants x 14 records (one per P695 event), hunt-and-fill IDA.

Run on a multi-core machine:
    python run_variant_study.py --records <folder of RSN*.AT2> --workers 8
Inputs: data/lhs_variants_seed2026.csv, data/variant_study_records.csv, PEER .AT2 files.
Output: variant_study_results.csv (one row per analysis; Module 3).
Expected size: about 160 x 14 x 10 runs = 22,400 runs; with the tuned solver settings
(tuning/best_config.json, 2.5x faster than the defaults) roughly 4-5 core-hours, ~35 min on 8 cores.
"""
import argparse, csv, os, time
from batch import run_batch
from peer import read_at2, find_at2


def records(folder, table=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "variant_study_records.csv"), allow_missing=False, only_rsn=None):
    rows = list(csv.reader(open(table)))[1:]
    hdr, data = rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]
    out = []
    if only_rsn:
        data = [d for d in data if d["rsn"] in only_rsn]
    for d in data:
        try:
            acc, dt = read_at2(find_at2(folder, d["rsn"], d["comp1"]))
        except FileNotFoundError as e:
            if not allow_missing:
                raise
            print("WARNING, skipped:", e)
            continue
        out.append((f"RSN{d['rsn']}", acc, dt))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lhs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lhs_variants_seed2026.csv"))
    ap.add_argument("--out", default="variant_study_results.csv")
    ap.add_argument("--allow-missing", action="store_true", help="skip records whose .AT2 is not in the folder")
    ap.add_argument("--rsn", nargs="*", help="run only these RSNs (e.g. to add a record later)")
    ap.add_argument("--only", type=int, nargs="*", help="realization numbers (for a timing test)")
    ap.add_argument("--solver", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning", "best_config.json"),
                    help="solver settings from the Karpathy-loop tuning (result check: collapse IM within 0.05 g of a tight reference); 'default' for the built-in settings")
    a = ap.parse_args()
    if a.solver != "default" and os.path.exists(a.solver):
        os.environ["EOSD_SOLVER_CONFIG"] = a.solver
        print("solver settings:", a.solver)
    recs = records(a.records, allow_missing=a.allow_missing, only_rsn=a.rsn)
    print(len(recs), "records")
    t, n = run_batch("variant_frame", a.lhs, recs, workers=a.workers, out_csv=a.out, realizations=a.only)
    print(f"{n} analyses in {t / 3600:.2f} h -> {a.out}")
