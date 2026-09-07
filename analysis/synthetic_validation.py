"""
Synthetic parameter-recovery validation: generate noiseless model output at
known parameters, add noise, refit, and check that the fitting pipeline
recovers the true values. This is a check on the *method* (does least-squares
fitting + the covariance-based CI/correlation math actually work?), not on
any real hardware run -- it does not touch data_raw/ at all.

Why this matters for the RMSE=1.54C claim in the README: fitting.py's
correlation and confidence-interval numbers rely on the standard
Jacobian-based covariance approximation (sigma^2 * pinv(J^T J)). This script
is the evidence that approximation is trustworthy: with a known ground
truth, the fitted parameters should land close to true_params, and the
reported 95% CIs should actually contain them.

Standalone script, not a notebook cell -- run with:
    python synthetic_validation.py
Writes two figures into figures/.
"""

import os

import numpy as np
from scipy.linalg import pinv
from scipy.optimize import least_squares

from thermal_model import NODE_WEIGHTS, PARAMETER_NAMES, simulate_model, steady_state

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TRUE_PARAMS = np.array([2.0, 5.0, 3.0, 4.0, 6.0, 8.0])
POWER_LEVELS = [5, 10, 15]
TIME = np.linspace(0, 2000, 400)
T_INIT = [25.0, 25.0, 25.0, 25.0, 25.0]
T_AMBIENT = 25.0
NOISE_SIGMA = 0.2


def build_synthetic_datasets(seed=42):
    """Simulate the model at TRUE_PARAMS for each power level and add
    Gaussian sensor noise, mimicking real measurement noise."""
    rng = np.random.default_rng(seed)
    datasets = []
    for power in POWER_LEVELS:
        clean = simulate_model(TRUE_PARAMS, TIME, power, T_INIT, T_AMBIENT)
        noisy = clean + rng.normal(0, NOISE_SIGMA, clean.shape)
        datasets.append((power, noisy))
    return datasets


def fit_synthetic(datasets):
    def residuals(params):
        blocks = []
        for power, measured in datasets:
            simulated = simulate_model(params, TIME, power, T_INIT, T_AMBIENT)
            for col, sensor in enumerate(["T1", "T2", "T3", "T4", "T5"]):
                blocks.append(NODE_WEIGHTS[sensor] * (simulated[:, col] - measured[:, col]))
        return np.concatenate(blocks)

    return least_squares(residuals, x0=np.ones(6), bounds=(1e-4, 100))


def plot_corr_matrix(corr, names, title, save_path):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr[i, j]) > 0.7 else "black",
            )
    # R_pa, R_aw, R_pw (indices 1,2,3) are the air-path cluster identified as
    # collinear in identifiability.py -- boxed here for visual cross-reference.
    rect = patches.FancyBboxPatch(
        (0.5, 0.5), 3, 3, linewidth=2, edgecolor="black",
        linestyle="--", facecolor="none", transform=ax.transData,
    )
    ax.add_patch(rect)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    datasets = build_synthetic_datasets()
    result = fit_synthetic(datasets)
    fitted = result.x

    print("=" * 50)
    print("SYNTHETIC VALIDATION -- PARAMETER RECOVERY")
    print("=" * 50)
    print(f"{'Parameter':<10} {'True':>8} {'Fitted':>8} {'Abs err':>10}")
    print("-" * 40)
    for name, true, fit in zip(PARAMETER_NAMES, TRUE_PARAMS, fitted):
        print(f"{name:<10} {true:>8.3f} {fit:>8.3f} {abs(true - fit):>10.4f}")

    n_obs, n_params = result.fun.size, len(fitted)
    sigma2 = np.sum(result.fun**2) / (n_obs - n_params)
    cov = sigma2 * pinv(result.jac.T @ result.jac)
    se = np.sqrt(np.diag(cov))
    z = 1.96

    print("\n" + "=" * 50)
    print("SYNTHETIC -- 95% CONFIDENCE INTERVALS")
    print("=" * 50)
    n_covered = 0
    for name, true, fit, s in zip(PARAMETER_NAMES, TRUE_PARAMS, fitted, se):
        lo, hi = fit - z * s, fit + z * s
        covered = lo <= true <= hi
        n_covered += covered
        print(f"{name:<10} fitted={fit:>8.4f}  95% CI=({lo:.4f}, {hi:.4f})  true={true:.3f}  {'OK' if covered else 'MISSED'}")
    print(f"\n{n_covered}/{len(PARAMETER_NAMES)} true values fall inside their reported 95% CI.")

    corr = cov / np.outer(se, se)
    plot_corr_matrix(
        corr, PARAMETER_NAMES,
        "Parameter Correlation Matrix (Synthetic Validation)",
        os.path.join(FIGURES_DIR, "correlation_matrix_synthetic.png"),
    )

    print("\n" + "=" * 50)
    print("SYNTHETIC -- STEADY-STATE SOLVER CHECK")
    print("=" * 50)
    print(f"{'Node':<6}" + "".join(f" {p:>5}W" for p in POWER_LEVELS))
    for row_i, node in enumerate(["T1", "T2", "T3", "T4", "T5"]):
        vals = [steady_state(fitted, p, T_AMBIENT)[row_i] for p in POWER_LEVELS]
        print(f"{node:<6}" + "".join(f" {v:>6.2f}" for v in vals))
    print("(deg C above ambient, fitted parameters, closed-form steady state)")


if __name__ == "__main__":
    main()
