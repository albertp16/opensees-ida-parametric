# program.md — Karpathy-loop solver tuning for the EOSD26 variant study

Pattern: Karpathy's autoresearch loop (X posts, Mar 2026; Fortune, 17 Mar 2026). An agent edits one
file, runs a fixed experiment, reads one metric, keeps or discards, repeats.

## Goal
Make the 22,400-run IDA batch faster WITHOUT changing its results.
Metric: `seconds` printed by `python evaluate.py` (benchmark: 4 variants x 2 records). Lower is better.

## You may edit ONLY `solver_config.json`
Allowed keys and ranges:
- tol: 1e-6 to 1e-4 (NormDispIncr, m)          - max_iter: 10 to 100
- primary: Newton | KrylovNewton | ModifiedNewton
- fallbacks: list of [algorithm, substeps]; algorithms Newton, KrylovNewton, NewtonLineSearch,
  ModifiedNewton, BFGS; substeps 1 to 100; at most 4 entries
- dt_sub: 1 or 2                                 - t_extra: 2 to 10 s (free-vibration tail)
- stop_drift: 0.12 to 0.30 (must stay above the 0.10 collapse drift)
- system: UmfPack | BandGeneral | SparseGeneral  - eigen: "-fullGenLapack" | "-genBandArpack" | ""

## You must NOT edit
evaluate.py, variant_frame.py, ida_runner.py (IDASettings, collapse criteria), the records, the
LHS file, or reference_*.json. Those define the result; changing them is cheating the metric.

## Loop
1. Once: `python evaluate.py --make-reference`, then `python evaluate.py --desc "baseline"`.
2. Change ONE thing in solver_config.json. Run `python evaluate.py --desc "<the change>"`.
3. KEEP → the change stays (best_config.json is updated). DISCARD → restore best_config.json.
4. Stop after 15 experiments or 5 discards in a row. Report results.tsv.
A config that fails the result check (any collapse capacity moves more than 0.05 g from the
reference, or a collapse appears or disappears) is discarded however fast it is.

## After the loop (human)
Copy best_config.json next to run_variant_study.py and run the batch with
`EOSD_SOLVER_CONFIG=tuning/best_config.json python run_variant_study.py ...`.
Re-run the loop with --records once the real records are downloaded; synthetic signals only
check runtime behaviour.
