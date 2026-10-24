"""Fixed evaluator for the Karpathy-loop solver tuning (do NOT edit during a tuning run).

Pattern after Karpathy's autoresearch (Mar 2026): the agent edits ONE file (solver_config.json),
this script runs a fixed benchmark under a time budget, reports ONE metric, and keeps the change
only if it improves the metric without breaking the result check.

Benchmark: 4 variants spanning T1 (realizations 145, 136, 28, 130; T1 0.63-2.07 s) x 2 records.
Records: --records <folder of PEER .AT2> uses the first two records of
../../data/variant_study_records.csv; without it, synthetic test signals 1-2 (layout/runtime only).
Metric: wall-clock seconds for the benchmark (lower is better).
Result check against reference.json (tight settings: tol 1e-8, dt/2, 10 s tail):
  |IM_collapse - IM_collapse_ref| <= 0.05 g (the hunt-and-fill bracket) for every IDA, and the
  same IDAs find a collapse. A config that fails the check is discarded whatever its speed.
Usage: python evaluate.py --desc "what changed"      (appends to results.tsv, prints KEEP/DISCARD)
       python evaluate.py --make-reference           (once, before the loop)
"""
import argparse, csv, hashlib, json, os, shutil, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
VARIANTS = [145, 136, 28, 130]
REFERENCE = dict(tol=1e-8, max_iter=100, dt_sub=2, t_extra=10.0, stop_drift=0.30)
TOL_IM = 0.05
BUDGET_S = 1800


def records(folder):
    if folder:
        from run_variant_study import records as real
        return real(folder, allow_missing=True)[:2]
    from demo_frame import synthetic_record
    return [("syn%02d" % s,) + synthetic_record(s) for s in (1, 2)]


def _one(args):
    p, real, rid, acc, dt = args
    from batch import _job
    from ida_runner import IDASettings
    out = _job("variant_frame", p, real, rid, acc, dt, IDASettings())
    return real, rid, out[0]["im_collapse_g"], out[0]["trigger"], len(out)


def run(config, recs, workers):
    os.environ["EOSD_SOLVER_CONFIG"] = config
    d = np.genfromtxt(os.path.join(HERE, "../../data/lhs_variants_seed2026.csv"), delimiter=",", names=True, skip_header=1)
    jobs = []
    for v in VARIANTS:
        r = d[d["realization"] == v][0]
        p = {k: float(r[k]) for k in d.dtype.names if k != "realization"}
        jobs += [(p, v, rid, acc, dt) for rid, acc, dt in recs]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, jobs))
    return time.time() - t0, {f"{v}/{rid}": dict(im=im, trigger=tr, runs=n) for v, rid, im, tr, n in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", default="")
    ap.add_argument("--config", default=os.path.join(HERE, "solver_config.json"))
    ap.add_argument("--records", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--make-reference", action="store_true")
    a = ap.parse_args()
    recs = records(a.records)
    tag = "real" if a.records else "synthetic"
    ref_path = os.path.join(HERE, f"reference_{tag}.json")
    if a.make_reference:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(REFERENCE, f)
        sec, out = run(f.name, recs, a.workers)
        json.dump(dict(seconds=sec, settings=REFERENCE, ida=out), open(ref_path, "w"), indent=1)
        print(f"reference written ({sec:.0f} s)")
        return
    ref = json.load(open(ref_path))["ida"]
    sec, out = run(a.config, recs, a.workers)
    dim, ok = 0.0, sec <= BUDGET_S
    for k, r in ref.items():
        a_, b_ = out[k]["im"], r["im"]
        if np.isnan(a_) != np.isnan(b_):
            ok = False
            dim = float("inf")
        elif not np.isnan(a_):
            dim = max(dim, abs(a_ - b_))
    ok = ok and dim <= TOL_IM
    nonconv = sum(o["trigger"] == "non-convergence" for o in out.values())
    log = os.path.join(HERE, "results.tsv")
    best = float("inf")
    if os.path.exists(log):
        for row in csv.DictReader(open(log), delimiter="\t"):
            if row["status"] == "keep" and row["records"] == tag:
                best = min(best, float(row["seconds"]))
    status = "keep" if ok and sec < best else "discard"
    h = hashlib.sha1(open(a.config, "rb").read()).hexdigest()[:8]
    new = not os.path.exists(log)
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(["time", "config", "records", "seconds", "max_dIM_g", "nonconv", "check", "status", "description"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), h, tag, f"{sec:.1f}", f"{dim:.3f}", nonconv,
                    "pass" if ok else "fail", status, a.desc])
    if status == "keep":
        shutil.copy(a.config, os.path.join(HERE, "best_config.json"))
    print(f"{status.upper()}  seconds={sec:.1f}  max_dIM={dim:.3f} g  check={'pass' if ok else 'fail'}  nonconv={nonconv}  (best before {best:.1f})")


if __name__ == "__main__":
    main()
