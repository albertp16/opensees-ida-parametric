# opensees-ida-parametric

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23610377.svg)](https://doi.org/10.5281/zenodo.23610377)

Open-source OpenSeesPy workflow for **automated parametric incremental dynamic analysis (IDA) of RC
frame buildings**. Companion code to Pamonag et al., "An Open-Source OpenSeesPy Workflow for Automated
Parametric IDA of RC Frame Buildings", EOSD26 (Springer LNCE).

One script, three modules:

1. **Generate**: `workflow/lhs_variants.py` reads user-defined parameter ranges and samples variants by
   Latin hypercube sampling. `workflow/variant_frame.py` builds each variant as a 2D RC moment frame:
   elastic beam-columns with `IMKPeakOriented` springs (Haselton et al. 2016 parameters, `imk.py`).
   Section yield moments come from a fiber moment–curvature analysis.
2. **IDA**: `workflow/ida_runner.py` runs hunt-and-fill tracing (Vamvatsikos & Cornell 2004). Collapse
   is flagged by a 20% tangent-slope drop, the 10% drift limit, or solver non-convergence.
   `workflow/vertical_collapse.py` adds a post-processed column axial-failure check (Liel et al. 2011;
   Elwood & Moehle 2005). `workflow/batch.py` runs the analyses in a process pool.
3. **Extract**: one CSV row per analysis. `fragility.py` fits lognormal fragilities (Baker 2015).

Plug in your own model by providing `build(params) -> T1` and `run_gm(acc_g, dt, scale) -> (theta_max, converged)`.

## Paper study (reproducible)
- 160 variants (`data/lhs_variants_seed2026.csv`, seed 2026): 2–3 storeys, 3 bays, non-ductile
  detailing. Sampled ranges: f'c 17–28 MPa, column ρ 1–3%, column 0.30–0.50 m, bay 4.0–7.0 m, storey 2.8–3.6 m.
- 13 records: the first of each event in FEMA P695 Table A-4 (component 1). RSN 829 is not available
  from PEER. Download the .AT2 files from PEER NGA-West2 by RSN (`data/variant_study_records.csv`).
- Results (`results/`): 24,757 analyses and 2,080 IDA curves. With the vertical check, median collapse
  Sa(T1) is 0.19–2.29 g (median 0.69 g) and β is 0.30.

```
pip install -r requirements.txt
python -m pytest -q workflow/tests
cd workflow
python run_variant_study.py --records <AT2 folder> --allow-missing --only 1 2   # timing test
python run_variant_study.py --records <AT2 folder> --allow-missing --workers 8  # full study
python vertical_collapse.py
python fig6_variant_results.py --one ../results/variant_study_results.csv --all ../results/variant_study_results.csv --vertical ../results/variant_capacity_vertical.csv
```

## Solver tuning (Karpathy loop)
`workflow/tuning/` holds an autoresearch-style keep/discard loop over the solver settings. Each experiment
changes one file and measures one metric (runtime). A change is kept only if collapse capacities stay
as close to a half-time-step, tight-tolerance reference as the original settings were. The real-record
check gave about 2× speed with collapse IM shifts of at most 1.4%. `run_variant_study.py` uses
`tuning/best_config.json` unless you pass `--solver default`.

## Verification against MIDAS Gen
`verification/` writes MIDAS Gen `.mgt` files for the DPWH–DepEd 3-storey school
(`mgt_dpwh_school.py`; `mgt_seismic.py` adds the eigen analysis and the NSCP 2015 spectrum).
`fig2_render.py` builds and renders the OpenSeesPy 3D model.

## Notes on OpenSees IMK materials
- `IMKPeakOriented` and `ModIMKPeakOriented` take Λ with E_t = Λ·My, so Λ = λ'·θcap,pl (rad). This is
  not Haselton's γ.
- The deprecated `ModIMKPeakOriented` gave tolerance-dependent drifts in our tests.
  `IMKPeakOriented` did not.

## License
MIT. See `CITATION.cff` for citation.
