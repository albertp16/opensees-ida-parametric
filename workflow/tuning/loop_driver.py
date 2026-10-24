"""Scripted Karpathy loop: one change per experiment, greedy keep/discard (evaluate.py decides).
The order of candidate changes is the 'agent' here; a coding agent following program.md can
replace this list with its own choices."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = os.path.join(HERE, "solver_config.json"); best = os.path.join(HERE, "best_config.json")
CANDIDATES = [
    ("eigen: default ARPACK solver", {"eigen": ""}),
    ("system BandGeneral", {"system": "BandGeneral"}),
    ("system SparseGeneral", {"system": "SparseGeneral"}),
    ("stop_drift 0.12 (still above the 0.10 collapse drift)", {"stop_drift": 0.12}),
    ("free-vibration tail 3 s", {"t_extra": 3.0}),
    ("tol 1e-4", {"tol": 1e-4}),
    ("max_iter 20", {"max_iter": 20}),
    ("primary KrylovNewton", {"primary": "KrylovNewton"}),
    ("fallbacks: NewtonLineSearch first", {"fallbacks": [["NewtonLineSearch", 1], ["KrylovNewton", 1], ["Newton", 10], ["KrylovNewton", 50]]}),
    ("fallbacks: drop the 50-substep stage", {"fallbacks": [["KrylovNewton", 1], ["Newton", 10]]}),
]
def run(desc):
    out = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--desc", desc], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith(("KEEP", "DISCARD"))][-1]
    print(desc, "->", line, flush=True)
    return line.startswith("KEEP")
subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--make-reference"], capture_output=True)
print("reference done", flush=True)
run("baseline (IMKPeakOriented, settings used so far)")
for desc, ch in CANDIDATES:
    c = json.load(open(best)); c.update(ch); json.dump(c, open(cfg, "w"), indent=1)
    if not run(desc):
        shutil.copy(best, cfg)
shutil.copy(best, cfg)
print("best:", json.load(open(best)))
