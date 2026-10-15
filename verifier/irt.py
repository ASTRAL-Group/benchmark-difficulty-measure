"""IRT 1PL-G math: likelihood, posterior, Fisher information.

Model:  p(θ, b, c) = c + (1 − c) · σ(θ − b)
Data:   y_m ~ Binomial(n, p_m)   for each model m with frozen ability θ_m
Goal:   estimate item difficulty b given observed {y_m}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit  # σ(x) = 1/(1+exp(-x))

from .config import VerifierConfig

# ---------------------------------------------------------------------------
# Core IRT probability
# ---------------------------------------------------------------------------


def irt_prob(theta: NDArray, b: float | NDArray, c: float) -> NDArray:
    """P(correct | θ, b, c) = c + (1-c) σ(θ-b)."""
    return c + (1.0 - c) * expit(theta - b)


# ---------------------------------------------------------------------------
# Log-likelihood and log-posterior (for a single item, vectorised over b)
# ---------------------------------------------------------------------------


def _log_binom_pmf(
    y: NDArray, n: int, p: NDArray
) -> NDArray:
    """Log Binomial(y; n, p), broadcasting over a grid of p values.

    We only need the terms that depend on p (the combinatorial constant
    cancels when comparing across b values).
    """
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return y * np.log(p) + (n - y) * np.log(1.0 - p)


def log_likelihood(
    b_grid: NDArray,
    theta_m: NDArray,
    y_m: NDArray,
    n_trials: int,
    c: float,
) -> NDArray:
    """Σ_m log Binom(y_m; n, p_m(b))  evaluated on b_grid.

    Parameters
    ----------
    b_grid : (G,) array of candidate b values
    theta_m : (M,) frozen model abilities
    y_m : (M,) observed correct counts for this item
    n_trials : number of trials per model
    c : guessing parameter

    Returns
    -------
    (G,) log-likelihood values
    """
    # p_grid shape: (G, M)
    p_grid = irt_prob(theta_m[None, :], b_grid[:, None], c)
    # ll_grid shape: (G, M)
    ll_grid = _log_binom_pmf(y_m[None, :], n_trials, p_grid)
    return ll_grid.sum(axis=1)  # (G,)


def log_posterior(
    b_grid: NDArray,
    theta_m: NDArray,
    y_m: NDArray,
    n_trials: int,
    c: float,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
) -> NDArray:
    """Log-posterior ∝ log-likelihood + log N(b; μ, σ²)."""
    ll = log_likelihood(b_grid, theta_m, y_m, n_trials, c)
    log_prior = -0.5 * ((b_grid - prior_mu) / prior_sd) ** 2
    return ll + log_prior


# ---------------------------------------------------------------------------
# Posterior summary (grid method — main estimator)
# ---------------------------------------------------------------------------


def estimate_b_grid(
    theta_m: NDArray,
    y_m: NDArray,
    n_trials: int,
    c: float,
    grid_lo: float = -4.0,
    grid_hi: float = 4.0,
    grid_n: int = 2000,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
) -> tuple[float, float]:
    """Estimate b via grid posterior.

    Returns (b_hat, b_sd) where b_hat = posterior mean, b_sd = posterior std.
    """
    b_grid = np.linspace(grid_lo, grid_hi, grid_n)
    lp = log_posterior(b_grid, theta_m, y_m, n_trials, c, prior_mu, prior_sd)

    # numerically stable softmax
    lp -= lp.max()
    post = np.exp(lp)
    post /= post.sum()

    b_hat = float(np.dot(b_grid, post))
    b_sd = float(np.sqrt(np.dot((b_grid - b_hat) ** 2, post)))
    return b_hat, b_sd


# ---------------------------------------------------------------------------
# MAP estimate (fast alternative, with Laplace approximation for sd)
# ---------------------------------------------------------------------------


def estimate_b_map(
    theta_m: NDArray,
    y_m: NDArray,
    n_trials: int,
    c: float,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
) -> tuple[float, float]:
    """MAP estimate of b via 1D optimisation + Laplace approximation."""

    def neg_log_post(b: float) -> float:
        p = irt_prob(theta_m, b, c)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        ll = np.sum(y_m * np.log(p) + (n_trials - y_m) * np.log(1.0 - p))
        lp = -0.5 * ((b - prior_mu) / prior_sd) ** 2
        return -(ll + lp)

    res = minimize_scalar(neg_log_post, bounds=(-5, 5), method="bounded")
    b_hat = float(res.x)

    # Laplace: sd ≈ 1 / sqrt( -d²log_post/db² )
    eps = 1e-4
    d2 = (neg_log_post(b_hat + eps) - 2 * neg_log_post(b_hat) + neg_log_post(b_hat - eps)) / eps**2
    b_sd = float(1.0 / np.sqrt(max(d2, 1e-8)))

    return b_hat, b_sd


# ---------------------------------------------------------------------------
# Load frozen θ from IRT results
# ---------------------------------------------------------------------------


def load_theta(cfg: VerifierConfig) -> tuple[NDArray, list[str]]:
    """Load model abilities, estimating them locally when the CSV is absent.

    The release does not include ``<benchmark>_ability.csv``. The first call
    writes one from the released response matrix and item difficulties
    (see ``verifier.fit_ability``).

    Returns (theta_array, model_names).
    """
    path = cfg.irt_results_dir / f"{cfg.benchmark}_ability.csv"
    if not path.is_file():
        from .fit_ability import fit_benchmark

        fit_benchmark(cfg.benchmark, irt_results_dir=cfg.irt_results_dir)
    df = pd.read_csv(path)
    return df["theta_mean"].to_numpy(), df["model"].tolist()


def load_difficulty(cfg: VerifierConfig) -> tuple[NDArray, list[str]]:
    """Load fitted item difficulties (for validation / anchor lookup).

    Returns (b_array, item_ids).
    """
    path = cfg.irt_results_dir / f"{cfg.benchmark}_difficulty.csv"
    df = pd.read_csv(path)
    return df["b_mean"].to_numpy(), df["item"].tolist()
