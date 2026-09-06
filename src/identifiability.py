"""
Practical-identifiability analysis: profile likelihood over each of the
6 model parameters, run against a canonical `fit_results.npz` (produced
by fitting.py).

For a parameter theta_i, the profile fixes theta_i at a scanned value and
re-optimizes every other parameter; the resulting SSE curve shows how much
the fit is actually forced to get worse as theta_i moves away from its
optimum. A parameter whose SSE barely rises over a wide range is
practically non-identifiable: many values fit almost equally well, even
though the *overall* model fit (aggregate RMSE) looks fine.
"""

import numpy as np
from scipy.optimize import least_squares

from thermal_model import NODE_WEIGHTS, simulate_model

N_PROFILE_POINTS = 80
PROFILE_FACTOR = 10  # scan +/- 10x around the optimum, log-spaced
CHI_SQ_THRESHOLD_95 = 3.84  # 95% confidence, 1 degree of freedom


def _load_fit(fit_results_path):
    data = np.load(fit_results_path, allow_pickle=True)
    return {
        "time": data["time"],
        "measured": data["measured_temperatures"],
        "ambient": float(data["ambient_temperature"]),
        "power": float(data["power"]),
        "fan_state": bool(data["fan_state"]),
        "parameter_names": data["parameter_names"],
        "parameter_values": data["parameter_values"],
        "lower_bounds": data["parameter_lower_bounds"],
        "upper_bounds": data["parameter_upper_bounds"],
        "sensor_names": data["sensor_names"],
        "fit_sensor_mask": data["fit_sensor_mask"],
    }


def profile_parameter(fit_results_path, parameter_index, n_points=N_PROFILE_POINTS,
                       factor=PROFILE_FACTOR):
    """Profile a single parameter. Returns (scan_values, costs, best_parameters)."""
    fit = _load_fit(fit_results_path)
    fit_columns = np.where(fit["fit_sensor_mask"])[0]
    P_fan = 0.5 if fit["fan_state"] else 0.0

    def residuals(params):
        simulated = simulate_model(params, fit["time"], fit["power"],
                                    fit["measured"][0].copy(), fit["ambient"], P_fan)
        blocks = [
            NODE_WEIGHTS[fit["sensor_names"][col]] * (simulated[:, col] - fit["measured"][:, col])
            for col in fit_columns
        ]
        return np.concatenate(blocks)

    parameter_values = fit["parameter_values"]
    lower_bounds, upper_bounds = fit["lower_bounds"], fit["upper_bounds"]
    name = fit["parameter_names"][parameter_index]
    optimum = parameter_values[parameter_index]

    scan_min = max(lower_bounds[parameter_index], optimum / factor)
    scan_max = min(upper_bounds[parameter_index], optimum * factor)
    scan_values = np.geomspace(scan_min, scan_max, n_points)

    free_indices = [j for j in range(len(parameter_values)) if j != parameter_index]
    current_guess = parameter_values.copy()

    costs, best_parameters = [], []
    print(f"\nProfiling {name}")
    for i, fixed_value in enumerate(scan_values):
        def constrained_residual(free_params, fixed_value=fixed_value):
            full_params = current_guess.copy()
            full_params[free_indices] = free_params
            full_params[parameter_index] = fixed_value
            return residuals(full_params)

        result = least_squares(
            constrained_residual, x0=current_guess[free_indices],
            bounds=(lower_bounds[free_indices], upper_bounds[free_indices]),
        )
        fitted_params = current_guess.copy()
        fitted_params[free_indices] = result.x
        fitted_params[parameter_index] = fixed_value

        cost = np.sum(result.fun**2)
        costs.append(cost)
        best_parameters.append(fitted_params)
        current_guess = fitted_params.copy()
        print(f"  {i + 1}/{n_points}: {fixed_value:.6g}, SSE={cost:.6g}")

    return np.array(scan_values), np.array(costs), np.array(best_parameters)


def profile_all_parameters(fit_results_path, output_path, n_points=N_PROFILE_POINTS,
                            factor=PROFILE_FACTOR):
    """Profile all 6 parameters and save one `profile_likelihood.npz`."""
    fit = _load_fit(fit_results_path)
    fit_columns = np.where(fit["fit_sensor_mask"])[0]
    P_fan = 0.5 if fit["fan_state"] else 0.0

    def residuals(params):
        simulated = simulate_model(params, fit["time"], fit["power"],
                                    fit["measured"][0].copy(), fit["ambient"], P_fan)
        blocks = [
            NODE_WEIGHTS[fit["sensor_names"][col]] * (simulated[:, col] - fit["measured"][:, col])
            for col in fit_columns
        ]
        return np.concatenate(blocks)

    sse_min = np.sum(residuals(fit["parameter_values"])**2)

    profile_data = {}
    for i, name in enumerate(fit["parameter_names"]):
        scan, cost, params = profile_parameter(fit_results_path, i, n_points, factor)
        profile_data[f"{name}_scan"] = scan
        profile_data[f"{name}_cost"] = cost
        profile_data[f"{name}_best_parameters"] = params

    np.savez(
        output_path,
        parameter_names=fit["parameter_names"],
        parameter_bounds=np.vstack([fit["lower_bounds"], fit["upper_bounds"]]),
        sse_min=sse_min,
        **profile_data,
    )
    print(f"Saved: {output_path}")
    return output_path


def get_ci(profile_npz_path, param_name, threshold=CHI_SQ_THRESHOLD_95):
    """
    95% confidence interval for one parameter from a saved profile,
    found by linear interpolation where the SSE curve crosses
    (min SSE + threshold).

    Returns None if the curve never crosses the threshold within the
    scanned range -- i.e. the parameter is not identifiable at this
    confidence level over the range that was scanned.
    """
    data = np.load(profile_npz_path, allow_pickle=True)
    scan = data[f"{param_name}_scan"]
    cost = data[f"{param_name}_cost"]

    delta = cost - np.min(cost)
    below = delta <= threshold
    if not np.any(below):
        return None

    idx_below = np.where(below)[0]
    lower_idx, upper_idx = idx_below[0], idx_below[-1]

    def interpolate(i_inside, i_outside):
        x0, x1 = scan[i_inside], scan[i_outside]
        y0, y1 = delta[i_inside], delta[i_outside]
        if y1 == y0:
            return x0
        frac = (threshold - y0) / (y1 - y0)
        return x0 + frac * (x1 - x0)

    lower_bound = interpolate(lower_idx, lower_idx - 1) if lower_idx > 0 else scan[lower_idx]
    upper_bound = (interpolate(upper_idx, upper_idx + 1) if upper_idx < len(scan) - 1
                   else scan[upper_idx])
    return lower_bound, upper_bound


if __name__ == "__main__":
    import sys

    fit_path = sys.argv[1] if len(sys.argv) > 1 else "results/run19/fit_results.npz"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/run19/profile_likelihood.npz"
    profile_all_parameters(fit_path, out_path)
