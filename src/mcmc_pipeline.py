"""
MCMC sampling (via emcee) around a fitted run, as a cross-check on the
profile-likelihood identifiability conclusions: if a parameter is
practically non-identifiable, its marginal posterior should be wide/flat
rather than a tight peak, even though the fit quality (RMSE) is good.

Requires: pip install emcee corner

Note on provenance: the original exploration notebook ran this against an
in-memory `results_by_power` dict built by an older, since-abandoned
fitting pass, and that dict is not reproducible from the current pipeline.
This version reads the same information (time, measured temperatures,
ambient, power, fan state, best-fit parameters, residual sigma) from the
canonical `fit_results.npz` artifact instead, so it works off the same
source of truth as fitting.py and identifiability.py. The sampler
configuration (32 walkers, flat box prior, 500-step test chain, 8000-step
production chain) is unchanged from the original.
"""

import numpy as np

from thermal_model import NODE_WEIGHTS, PARAMETER_NAMES, simulate_model

NDIM = 6
NWALKERS = 32
N_STEPS_TEST = 500
N_STEPS_PRODUCTION = 8000
LOWER_BOUND, UPPER_BOUND = 1e-4, 1000.0


def run_mcmc(fit_results_path, trace_plot_path=None, corner_plot_path=None,
             n_steps_test=N_STEPS_TEST, n_steps_production=N_STEPS_PRODUCTION,
             seed=None):
    """
    Run emcee around the best fit stored in `fit_results_path`.

    Returns (flat_samples, sampler) where flat_samples has shape
    (n_samples, 6) after discarding burn-in and thinning by the estimated
    autocorrelation time.
    """
    import emcee

    data = np.load(fit_results_path, allow_pickle=True)
    time = data["time"]
    measured = data["measured_temperatures"]
    ambient = float(data["ambient_temperature"])
    power = float(data["power"])
    P_fan = 0.5 if bool(data["fan_state"]) else 0.0
    sensor_names = data["sensor_names"]
    fit_columns = np.where(data["fit_sensor_mask"])[0]
    theta_best = data["parameter_values"]
    resid_best = data["residuals"][:, fit_columns].ravel()

    n_obs = resid_best.size
    sigma = np.sqrt(np.sum(resid_best**2) / (n_obs - NDIM))
    print(f"Estimated sigma (weighted residual RMS, dof-corrected): {sigma:.4f}")

    initial_state = measured[0].copy()

    def weighted_residuals(theta):
        simulated = simulate_model(theta, time, power, initial_state, ambient, P_fan)
        blocks = [
            NODE_WEIGHTS[sensor_names[col]] * (simulated[:, col] - measured[:, col])
            for col in fit_columns
        ]
        return np.concatenate(blocks)

    def log_prior(theta):
        if np.all(theta > LOWER_BOUND) and np.all(theta < UPPER_BOUND):
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        try:
            r = weighted_residuals(theta)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(r)):
            return -np.inf
        return -0.5 * np.sum((r / sigma) ** 2)

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    rng = np.random.default_rng(seed)
    pos = theta_best + 1e-2 * np.abs(theta_best) * rng.standard_normal((NWALKERS, NDIM))
    pos = np.clip(pos, LOWER_BOUND * 1.001, UPPER_BOUND * 0.999)

    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_probability)

    print(f"Running short test chain ({n_steps_test} steps)...")
    sampler.run_mcmc(pos, n_steps_test, progress=True)

    if trace_plot_path:
        _save_trace_plot(sampler, trace_plot_path)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        print("Autocorrelation times:", np.round(tau, 1))
    except Exception as e:
        print("Autocorr estimate unreliable on short chain (expected):", e)

    sampler.reset()
    print(f"Running production chain ({n_steps_production} steps)...")
    sampler.run_mcmc(pos, n_steps_production, progress=True)

    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(3 * np.nanmax(tau)) if np.all(np.isfinite(tau)) else 1000
    thin = max(1, int(0.5 * np.nanmin(tau))) if np.all(np.isfinite(tau)) else 10
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"burn-in={burnin}, thin={thin}, samples={flat_samples.shape[0]}")

    if corner_plot_path:
        _save_corner_plot(flat_samples, corner_plot_path)

    return flat_samples, sampler


def _save_trace_plot(sampler, path):
    import matplotlib.pyplot as plt

    chain = sampler.get_chain()
    fig, axes = plt.subplots(NDIM, figsize=(10, 12), sharex=True)
    for i in range(NDIM):
        axes[i].plot(chain[:, :, i], alpha=0.4, lw=0.5)
        axes[i].set_ylabel(PARAMETER_NAMES[i])
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _save_corner_plot(flat_samples, path):
    import corner
    import matplotlib.pyplot as plt

    fig = corner.corner(
        flat_samples, labels=PARAMETER_NAMES, show_titles=True,
        title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 10},
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    import sys

    fit_path = sys.argv[1] if len(sys.argv) > 1 else "results/run16/fit_results.npz"
    run_mcmc(fit_path, trace_plot_path="mcmc_traces.png", corner_plot_path="corner.png")
