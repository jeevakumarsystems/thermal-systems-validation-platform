"""
Fit the thermal model (see thermal_model.py) to measured CSV runs using
constrained nonlinear least squares, and write one canonical
`fit_results.npz` artifact per run.

Expected CSV columns: time_s, T1_C .. T6_C (T6 is the ambient reference
and is used only as a boundary condition, not fit).

Expected filename convention: "<run_id>_<heater config>_<power>W[_fan].csv",
e.g. "19_H1H2H3_15W.csv".
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.optimize import least_squares

from thermal_model import NODE_WEIGHTS, PARAMETER_NAMES, SENSOR_NAMES, simulate_model

LOWER_BOUNDS = np.full(6, 1e-4)
UPPER_BOUNDS = np.full(6, 1000.0)

# Starting point for the optimizer. Not calibrated to any particular run;
# least_squares converges to the same optimum from a wide range of starts
# for this model.
INITIAL_GUESS = np.ones(6, dtype=float)

# Sensors used during parameter estimation. T1 (the heater node) is excluded
# from the fit; it's driven almost entirely by input power and contributes
# little information about the resistances downstream of it.
FIT_SENSOR_MASK = np.array([False, True, True, True, True])

CSV_COLUMNS = {
    "time": "time_s",
    "T1": "T1_C", "T2": "T2_C", "T3": "T3_C",
    "T4": "T4_C", "T5": "T5_C", "T6": "T6_C",
}


def infer_power(filename):
    match = re.search(r"(\d+)[Ww]", os.path.basename(filename))
    return float(match.group(1)) if match else None


def infer_run_id(filename):
    match = re.match(r"(\d+)", os.path.basename(filename))
    if match is None:
        raise ValueError(f"Cannot infer run number from {filename}")
    return int(match.group(1))


def infer_heater_configuration(filename):
    name = os.path.basename(filename).upper()
    heaters = [h for h in ("H1", "H2", "H3") if h in name]
    return "_".join(heaters)


def infer_fan_state(filename):
    return "fan" in filename.lower()


def _read_run_csv(csv_path):
    """
    Read a run CSV, tolerating a logger restart mid-file.

    A handful of runs contain the header row more than once (the DAQ
    logger was power-cycled and restarted partway through), which
    truncates whatever was logged before the restart. This keeps only
    the segment of the file after the *last* header occurrence, so a
    restarted run reads as the one continuous, complete recording.
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    header = lines[0]
    header_indices = [i for i, line in enumerate(lines) if line == header]
    last_header = header_indices[-1]
    if len(header_indices) > 1:
        print(f"  Note: {len(header_indices)} header rows found (logger restart); "
              f"using data after the last restart (line {last_header + 1}).")

    from io import StringIO
    return pd.read_csv(StringIO("".join([header] + lines[last_header + 1:])))


def fit_run(csv_path, results_root):
    """Fit one CSV run and write `<results_root>/run<N>/fit_results.npz`."""
    print(f"\nProcessing {os.path.basename(csv_path)}")

    df = _read_run_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    time = df[CSV_COLUMNS["time"]].to_numpy(dtype=float)
    time -= time[0]

    measured = np.column_stack([
        df[CSV_COLUMNS["T1"]], df[CSV_COLUMNS["T2"]], df[CSV_COLUMNS["T3"]],
        df[CSV_COLUMNS["T4"]], df[CSV_COLUMNS["T5"]],
    ]).astype(float)

    ambient = float(df[CSV_COLUMNS["T6"]].iloc[0])
    initial_state = measured[0].copy()

    power = infer_power(csv_path)
    fan_state = infer_fan_state(csv_path)
    fan_power = 0.5 if fan_state else 0.0

    fit_columns = np.where(FIT_SENSOR_MASK)[0]

    def residual_function(params):
        simulated = simulate_model(params, time, power, initial_state, ambient, fan_power)
        blocks = [
            NODE_WEIGHTS[SENSOR_NAMES[col]] * (simulated[:, col] - measured[:, col])
            for col in fit_columns
        ]
        return np.concatenate(blocks)

    result = least_squares(residual_function, x0=INITIAL_GUESS, bounds=(LOWER_BOUNDS, UPPER_BOUNDS))
    params = result.x

    simulated = simulate_model(params, time, power, initial_state, ambient, fan_power)
    residual_matrix = simulated - measured

    rmse_by_sensor = np.sqrt(np.mean(residual_matrix**2, axis=0))
    rmse_total = np.sqrt(np.mean(residual_matrix[:, FIT_SENSOR_MASK]**2))

    n_params, n_obs = len(params), len(result.fun)
    sigma2 = np.sum(result.fun**2) / (n_obs - n_params)
    covariance = sigma2 * pinv(result.jac.T @ result.jac)
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)

    run_id = infer_run_id(csv_path)
    output_dir = os.path.join(results_root, f"run{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "fit_results.npz"),
        run_id=run_id,
        heater_configuration=infer_heater_configuration(csv_path),
        power=power,
        fan_state=fan_state,
        parameter_names=np.array(PARAMETER_NAMES),
        parameter_values=params,
        parameter_lower_bounds=LOWER_BOUNDS,
        parameter_upper_bounds=UPPER_BOUNDS,
        sensor_names=np.array(SENSOR_NAMES),
        fit_sensor_mask=FIT_SENSOR_MASK,
        rmse_total=rmse_total,
        rmse_by_sensor=rmse_by_sensor,
        residuals=residual_matrix,
        success=result.success,
        optimization_status=result.status,
        optimization_message=result.message,
        time=time,
        measured_temperatures=measured,
        simulated_temperatures=simulated,
        ambient_temperature=ambient,
        covariance_matrix=covariance,
        correlation_matrix=correlation,
    )
    print(f"Saved {output_dir}/fit_results.npz")
    return os.path.join(output_dir, "fit_results.npz")


def fit_all(data_folder, results_root):
    """Fit every CSV in `data_folder`, in run-id order."""
    os.makedirs(results_root, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")), key=infer_run_id)
    for csv_file in csv_files:
        fit_run(csv_file, results_root)
    print("\nFinished.")


def compute_corr_from_pooled_jac(jac_list, resid_list, n_params=6):
    """
    Pool Jacobians and residuals across multiple runs, and compute a
    pseudoinverse covariance/correlation matrix from the pooled fit.
    Used to check whether identifiability conclusions hold up when
    information from several runs at the same power level is combined.
    """
    JTJ_pooled = np.zeros((n_params, n_params))
    for J in jac_list:
        JTJ_pooled += J.T @ J

    all_resid = np.concatenate(resid_list)
    sigma2 = np.sum(all_resid**2) / (all_resid.size - n_params)

    cov = sigma2 * pinv(JTJ_pooled)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr, cov


if __name__ == "__main__":
    import sys

    data_folder = sys.argv[1] if len(sys.argv) > 1 else "data_raw"
    results_root = sys.argv[2] if len(sys.argv) > 2 else "results"
    fit_all(data_folder, results_root)
"""
Fit the thermal model (see thermal_model.py) to measured CSV runs using
constrained nonlinear least squares, and write one canonical
`fit_results.npz` artifact per run.

Expected CSV columns: time_s, T1_C .. T6_C (T6 is the ambient reference
and is used only as a boundary condition, not fit).

Expected filename convention: "<run_id>_<heater config>_<power>W[_fan].csv",
e.g. "19_H1H2H3_15W.csv".
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.optimize import least_squares

from thermal_model import NODE_WEIGHTS, PARAMETER_NAMES, SENSOR_NAMES, simulate_model

LOWER_BOUNDS = np.full(6, 1e-4)
UPPER_BOUNDS = np.full(6, 1000.0)

# Starting point for the optimizer. Not calibrated to any particular run;
# least_squares converges to the same optimum from a wide range of starts
# for this model.
INITIAL_GUESS = np.ones(6, dtype=float)

# Sensors used during parameter estimation. T1 (the heater node) is excluded
# from the fit; it's driven almost entirely by input power and contributes
# little information about the resistances downstream of it.
FIT_SENSOR_MASK = np.array([False, True, True, True, True])

CSV_COLUMNS = {
    "time": "time_s",
    "T1": "T1_C", "T2": "T2_C", "T3": "T3_C",
    "T4": "T4_C", "T5": "T5_C", "T6": "T6_C",
}


def infer_power(filename):
    match = re.search(r"(\d+)[Ww]", os.path.basename(filename))
    return float(match.group(1)) if match else None


def infer_run_id(filename):
    match = re.match(r"(\d+)", os.path.basename(filename))
    if match is None:
        raise ValueError(f"Cannot infer run number from {filename}")
    return int(match.group(1))


def infer_heater_configuration(filename):
    name = os.path.basename(filename).upper()
    heaters = [h for h in ("H1", "H2", "H3") if h in name]
    return "_".join(heaters)


def infer_fan_state(filename):
    return "fan" in filename.lower()


def fit_run(csv_path, results_root):
    """Fit one CSV run and write `<results_root>/run<N>/fit_results.npz`."""
    print(f"\nProcessing {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    time = df[CSV_COLUMNS["time"]].to_numpy(dtype=float)
    time -= time[0]

    measured = np.column_stack([
        df[CSV_COLUMNS["T1"]], df[CSV_COLUMNS["T2"]], df[CSV_COLUMNS["T3"]],
        df[CSV_COLUMNS["T4"]], df[CSV_COLUMNS["T5"]],
    ]).astype(float)

    ambient = float(df[CSV_COLUMNS["T6"]].iloc[0])
    initial_state = measured[0].copy()

    power = infer_power(csv_path)
    fan_state = infer_fan_state(csv_path)
    fan_power = 0.5 if fan_state else 0.0

    fit_columns = np.where(FIT_SENSOR_MASK)[0]

    def residual_function(params):
        simulated = simulate_model(params, time, power, initial_state, ambient, fan_power)
        blocks = [
            NODE_WEIGHTS[SENSOR_NAMES[col]] * (simulated[:, col] - measured[:, col])
            for col in fit_columns
        ]
        return np.concatenate(blocks)

    result = least_squares(residual_function, x0=INITIAL_GUESS, bounds=(LOWER_BOUNDS, UPPER_BOUNDS))
    params = result.x

    simulated = simulate_model(params, time, power, initial_state, ambient, fan_power)
    residual_matrix = simulated - measured

    rmse_by_sensor = np.sqrt(np.mean(residual_matrix**2, axis=0))
    rmse_total = np.sqrt(np.mean(residual_matrix[:, FIT_SENSOR_MASK]**2))

    n_params, n_obs = len(params), len(result.fun)
    sigma2 = np.sum(result.fun**2) / (n_obs - n_params)
    covariance = sigma2 * pinv(result.jac.T @ result.jac)
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)

    run_id = infer_run_id(csv_path)
    output_dir = os.path.join(results_root, f"run{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "fit_results.npz"),
        run_id=run_id,
        heater_configuration=infer_heater_configuration(csv_path),
        power=power,
        fan_state=fan_state,
        parameter_names=np.array(PARAMETER_NAMES),
        parameter_values=params,
        parameter_lower_bounds=LOWER_BOUNDS,
        parameter_upper_bounds=UPPER_BOUNDS,
        sensor_names=np.array(SENSOR_NAMES),
        fit_sensor_mask=FIT_SENSOR_MASK,
        rmse_total=rmse_total,
        rmse_by_sensor=rmse_by_sensor,
        residuals=residual_matrix,
        success=result.success,
        optimization_status=result.status,
        optimization_message=result.message,
        time=time,
        measured_temperatures=measured,
        simulated_temperatures=simulated,
        ambient_temperature=ambient,
        covariance_matrix=covariance,
        correlation_matrix=correlation,
    )
    print(f"Saved {output_dir}/fit_results.npz")
    return os.path.join(output_dir, "fit_results.npz")


def fit_all(data_folder, results_root):
    """Fit every CSV in `data_folder`, in run-id order."""
    os.makedirs(results_root, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")), key=infer_run_id)
    for csv_file in csv_files:
        fit_run(csv_file, results_root)
    print("\nFinished.")


def compute_corr_from_pooled_jac(jac_list, resid_list, n_params=6):
    """
    Pool Jacobians and residuals across multiple runs, and compute a
    pseudoinverse covariance/correlation matrix from the pooled fit.
    Used to check whether identifiability conclusions hold up when
    information from several runs at the same power level is combined.
    """
    JTJ_pooled = np.zeros((n_params, n_params))
    for J in jac_list:
        JTJ_pooled += J.T @ J

    all_resid = np.concatenate(resid_list)
    sigma2 = np.sum(all_resid**2) / (all_resid.size - n_params)

    cov = sigma2 * pinv(JTJ_pooled)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr, cov


if __name__ == "__main__":
    import sys

    data_folder = sys.argv[1] if len(sys.argv) > 1 else "data_raw"
    results_root = sys.argv[2] if len(sys.argv) > 2 else "results"
    fit_all(data_folder, results_root)
