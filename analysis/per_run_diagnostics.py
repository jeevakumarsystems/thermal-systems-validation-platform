"""
Two cross-run diagnostics over every fit_results.npz already produced by
fitting.py, run after fit_all() (see notebooks/01_full_walkthrough.ipynb).

1. A per-sensor (not just aggregate) RMSE table for every run. T1 is
   excluded from fitting (see fitting.py's fit_sensor_mask) because it sits
   right on the heater and its dynamics aren't well captured by the model,
   so its RMSE is large and not meaningful -- printing it alongside T2-T5
   makes that exclusion visible and checkable rather than an unstated
   assumption.

2. A cross-run correlation summary for R_hp, R_cond, R_ext: the three
   parameters identifiability.py's profile likelihood found to be
   well-identified. Printing corr(R_hp, R_cond), corr(R_hp, R_ext), and
   corr(R_cond, R_ext) for every run separately (rather than just one
   pooled number) shows whether that identifiability conclusion is a
   property of the model/sensor layout (consistent sign and rough magnitude
   across very different runs) or an artifact of one dataset.

Standalone script, not a notebook cell -- run after fitting with:
    python per_run_diagnostics.py [results_root]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

RESULTS_ROOT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "results")

# Parameter order in fit_results.npz, from thermal_model.PARAMETER_NAMES.
PARAM_INDEX = {"R_hp": 0, "R_pa": 1, "R_aw": 2, "R_pw": 3, "R_cond": 4, "R_ext": 5}
CORR_PAIRS = [("R_hp", "R_cond"), ("R_hp", "R_ext"), ("R_cond", "R_ext")]


def _result_files(results_root):
    return sorted(
        glob.glob(os.path.join(results_root, "run*", "fit_results.npz")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("run", "")),
    )


def per_sensor_rmse_table(results_root):
    rows = []
    for result_file in _result_files(results_root):
        data = np.load(result_file, allow_pickle=True)
        rmse = data["rmse_by_sensor"]
        rows.append({
            "Run": int(data["run_id"]),
            "Heaters": str(data["heater_configuration"]),
            "Power (W)": float(data["power"]),
            "Fan": bool(data["fan_state"]),
            "T1 RMSE (C)": rmse[0],
            "T2 RMSE (C)": rmse[1],
            "T3 RMSE (C)": rmse[2],
            "T4 RMSE (C)": rmse[3],
            "T5 RMSE (C)": rmse[4],
            "Aggregate RMSE (C)": float(data["rmse_total"]),
        })
    table = pd.DataFrame(rows)
    print("=" * 70)
    print("PER-SENSOR RMSE, ALL RUNS")
    print("(T1 is excluded from fitting -- large T1 RMSE is expected, not a bug)")
    print("=" * 70)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return table


def cross_run_correlation_summary(results_root):
    print("\n" + "=" * 70)
    print("CORRELATIONS AMONG THE WELL-IDENTIFIED RESISTANCES, PER RUN")
    print("=" * 70)
    rows = []
    for result_file in _result_files(results_root):
        data = np.load(result_file, allow_pickle=True)
        corr = data["correlation_matrix"]
        run_id = int(data["run_id"])
        heaters = str(data["heater_configuration"])
        power = float(data["power"])
        fan = bool(data["fan_state"])
        print(f"\nRun {run_id} | {heaters} | {power:.0f} W | Fan={fan}")
        row = {"Run": run_id}
        for name1, name2 in CORR_PAIRS:
            value = corr[PARAM_INDEX[name1], PARAM_INDEX[name2]]
            print(f"  corr({name1}, {name2}) = {value: .4f}")
            row[f"corr({name1},{name2})"] = value
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\n" + "-" * 70)
    print("Summary across all runs (mean +/- std):")
    for name1, name2 in CORR_PAIRS:
        col = f"corr({name1},{name2})"
        print(f"  {col}: {summary[col].mean(): .4f} +/- {summary[col].std():.4f}")
    return summary


def main():
    results_root = sys.argv[1] if len(sys.argv) > 1 else RESULTS_ROOT_DEFAULT
    per_sensor_rmse_table(results_root)
    cross_run_correlation_summary(results_root)


if __name__ == "__main__":
    main()
