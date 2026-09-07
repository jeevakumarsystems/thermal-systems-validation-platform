# Thermal Systems Validation Platform

A six-node lumped-parameter thermal network (LPTN) for a sealed 
electronics enclosure, with structured experimental validation across 
a 35-run design-of-experiments matrix.

**Central finding:** Two air-path convective resistances (R_pa, R_aw) are practically unidentifiable from temperature data alone; 
one conductive resistance is configuration-dependent (R_pw); and three 
parameters (R_hp, R_cond, R_ext) are well-identified. Per-node temperature prediction
is below **RMSE = 1.54°C** across all 
nodes and power levels.

---

## Objective

Build a six-node thermal resistance network that predicts temperature behavior in a sealed electronics enclosure, and test whether a model that fits the data well can still be trusted parameter-by-parameter. The enclosure was instrumented, run through a 35-point experimental matrix at three power levels, and fit using nonlinear least-squares. The prediction accuracy turned out to be strong. However, three of six resistances couldn't be individually determined, despite an excellent fit.

---

## Overview

The system is built around a sealed ABS enclosure with an aluminum 
conduction plate mounted on standoffs at the base. Three resistive 
heater modules (H1, H2, H3) sit on the plate — H2 at the geometric 
center, H1 and H3 symmetrically offset along the long axis. Each 
heater consists of two 15Ω cement wirewound resistors in series 
(~30Ω, ~5W per heater at 12V). H1, H2, or H3 alone runs at 5W; H1+H2 or H2+H3 gives 10W; all three gives 15W.

I placed 6 calibrated NTC thermistors (T1–T6) at:

| Sensor | Location |
|--------|----------|
| T1 | On H2 heater body (heater node) |
| T2 | Aluminum plate, midpoint between H1 and H2 |
| T3 | Suspended ~2 cm above H2 (interior air node) |
| T4 | Interior wall, elevated to T3 height |
| T5 | Exterior wall, same lateral position as T4 |
| T6 | ~50 cm from enclosure (ambient reference) |

T4 and T5 placement was my deliberate design choice. The ΔT = T4 − T5 differential 
isolates wall conduction resistance (R_cond) independently of all 
air-path resistances.

---

## Modeling Approach

Parameters were estimated using constrained nonlinear least squares. Practical identifiability was assessed using profile likelihood analysis and Markov Chain Monte Carlo (MCMC) analysis.


---

## Code

The model, fitting pipeline, identifiability analysis, and MCMC cross-check live in `src/` as standalone, documented modules:

- `src/thermal_model.py` — the six-node LPTN: ODE right-hand side, time-domain simulator, and closed-form steady-state solver.
- `src/fitting.py` — fits `thermal_model` to a CSV run via constrained nonlinear least squares, writing one `fit_results.npz` per run.
- `src/identifiability.py` — profile-likelihood identifiability analysis over each fitted parameter.
- `src/mcmc_pipeline.py` — `emcee`-based MCMC sampling around a fit, as an independent cross-check on the profile-likelihood conclusions.

`notebooks/01_full_walkthrough.ipynb` is a thin demo that imports these modules and runs the full pipeline end to end; it is not where the logic lives.

Expected input format for `data_raw/`: one CSV per run, named `<run_id>_<heater config>_<power>W[_fan].csv` (e.g. `19_H1H2H3_15W.csv`), with columns `time_s, T1_C, T2_C, T3_C, T4_C, T5_C, T6_C`.

**Reproducibility:** All 35 runs are committed in `data_raw/`, and `figures/` contains the fit-quality, per-run RMSE, and profile-likelihood plots generated from them. Running `python src/fitting.py data_raw results` against the code and data in this repo independently reproduces the headline claim above: mean per-run RMSE = 1.52°C across all 35 runs, matching the reported 1.54°C. Six of the 35 runs include natural cooling after the heaters are switched off, and seven use a forced-air fan; the rest are heater-only. One run (`28_H1H2H3_15W.csv`) has a mid-run data-logger restart (a duplicated header row) that `src/fitting.py` now detects and handles automatically, using only the data after the restart.
