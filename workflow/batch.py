"""Parallel batch driver: every (realization, record) IDA runs in its own process.

OpenSeesPy keeps one model per process, so a process pool parallelises cleanly.
Usage from your script:
    from batch import run_batch
    run_batch("dpwh_school_3d", lhs_csv, records, workers=8, out_csv="ida_results.csv")
records: list of (record_id, acc_g, dt). acc_g is one array (unidirectional) or a tuple
(acc_x, acc_y) for a bidirectional pair, as FEMA P695 Sec. 6.4.5 prescribes for 3D models
(apply each pair twice: (x, y) and rotated (y, x)). For a pair, IM = geometric mean of the two
components' Sa(T1, 5%). The model module must expose build(params) returning T1 and
run_gm(acc_g, dt, scale) -> (theta_max, converged); for pairs run_gm receives the tuple and
applies both components (two UniformExcitation patterns, directions 1 and 2) with the same scale.
"""
import csv, importlib, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from spectra import sa
from ida_runner import trace, rows, IDASettings


def _job(model_name, p, real, rec_id, acc, dt, settings):
    m = importlib.import_module(model_name)
    T1 = m.build(p)
    if isinstance(acc, tuple):
        sa0 = float(np.sqrt(sa(acc[0], dt, T1) * sa(acc[1], dt, T1)))
    else:
        sa0 = sa(acc, dt, T1)
    res = trace(m.build, m.run_gm, p, acc, dt, sa0, rec_id, real, settings)
    out = rows(res)
    for r in out:
        r["T1_s"] = T1
    return out


def pairs_both_orientations(records):
    """FEMA P695 Sec. 6.4.5: each pair once along the principal axes and once rotated 90 deg."""
    out = []
    for rid, (ax, ay), dt in records:
        out += [(rid + "_0", (ax, ay), dt), (rid + "_90", (ay, ax), dt)]
    return out


def run_batch(model_name, lhs_csv, records, workers=4, out_csv="ida_results.csv",
              realizations=None, settings=IDASettings()):
    lhs = np.genfromtxt(lhs_csv, delimiter=",", names=True, skip_header=1)
    if realizations is not None:
        lhs = lhs[np.isin(lhs["realization"], realizations)]
    jobs, results, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in lhs:
            p = {k: float(r[k]) for k in lhs.dtype.names if k != "realization"}
            for rec_id, acc, dt in records:
                jobs.append(ex.submit(_job, model_name, p, int(r["realization"]), rec_id, acc, dt, settings))
        for f in as_completed(jobs):
            results += f.result()
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(sorted(results, key=lambda d: (d["realization"], d["record"], d["im_g"])))
    return time.time() - t0, len(results)
