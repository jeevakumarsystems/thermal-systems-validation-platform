# Thermal Systems Validation Platform

A six-node lumped-parameter thermal network (LPTN) for a sealed 
electronics enclosure, with structured experimental validation across 
a 35-run design-of-experiments matrix.

**Central finding:** Three air-path convective resistances (R_pa, R_aw, 
R_pw) are structurally unidentifiable from temperature data alone; 
pairwise correlations are |r| ≈ 0.97–0.99 under synthetic validation. Three 
parameters (R_hp, R_cond, R_ext) are well-identified. Despite parameter 
non-uniqueness, per-node temperature prediction **RMSE ≈ 0.2°C** across all 
nodes and power levels.

Biot number: **Bi = 8.99 × 10⁻⁵** — validates the lumped-capacitance 
assumption.

---

## Objective

Design, construct, instrument, and validate a predictive thermal 
resistance network capable of modeling steady-state and transient 
temperature behavior within ±0.2°C RMSE.

---

## Hardware Overview

The system is built around a sealed ABS enclosure with an aluminum 
conduction plate mounted on standoffs at the base. Three resistive 
heater modules (H1, H2, H3) sit on the plate — H2 at the geometric 
center, H1 and H3 symmetrically offset along the long axis. Each 
heater consists of two 15Ω cement wirewound resistors in series 
(~30Ω, ~5W per heater at 12V).

Six NTC thermistors are mapped directly onto model nodes:

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

## System Overview

- 6 calibrated NTC thermistors (T1–T6)
- 3-level controlled resistive heating (5W / 10W / 15W)
- Sealed ABS enclosure with aluminum conduction plate
- 35-run structured experimental matrix
- 10,000+ logged data points

---

## Modeling Approach

- Six-node lumped-capacitance resistance network
- First-principles ODE derivation (energy balance per node)
- Nonlinear least-squares parameter fitting (scipy, Radau stiff solver)
- Jacobian-based uncertainty quantification via Moore-Penrose 
  pseudoinverse covariance
- Identifiability analysis: structural collinearity of air-path 
  resistance cluster
- Rolling-window steady-state detection (N = 1050 samples, 
  ε = 0.2°C)
