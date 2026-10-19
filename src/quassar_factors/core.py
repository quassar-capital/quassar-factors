
"""Cross-sectional factor research primitives with explicit numerical contracts.

Observations are rows; assets and factors are columns. No annualization is implicit.
This module is research software, not a production risk or trading system.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh


@dataclass(frozen=True)
class FactorFit:
    """OLS results, with coefficients shaped (n_factors, n_assets)."""

    coefficients: NDArray[np.float64]
    intercept: NDArray[np.float64]
    residuals: NDArray[np.float64]
    rank: int
    condition_number: float


def _matrix(value: ArrayLike, name: str, min_rows: int = 1) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1:
        raise ValueError(f"{name} must be a 2D matrix with at least {min_rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def fit_factor_model(
    returns: ArrayLike, factors: ArrayLike, *, intercept: bool = True, rcond: float = 1e-12
) -> FactorFit:
    """Fit multivariate ordinary least squares via SVD-backed least squares.

    Returns and factors must be time-aligned by the caller. Rank deficiency is
    rejected rather than silently producing non-identifiable exposures.
    """
    y = _matrix(returns, "returns", min_rows=2)
    x = _matrix(factors, "factors", min_rows=2)
    if y.shape[0] != x.shape[0]:
        raise ValueError("returns and factors must have the same observation count")
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must be finite and between zero and one")
    design = np.column_stack((np.ones(x.shape[0]), x)) if intercept else x
    if design.shape[0] <= design.shape[1]:
        raise ValueError("more observations than fitted parameters are required")
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=rcond)
    if rank != design.shape[1]:
        raise ValueError("factor design is rank deficient at the specified tolerance")
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition):
        raise ValueError("factor design is numerically ill-conditioned")
    alpha = coefficients[0] if intercept else np.zeros(y.shape[1])
    betas = coefficients[1:] if intercept else coefficients
    return FactorFit(
        coefficients=betas.copy(), intercept=alpha.copy(),
        residuals=(y - design @ coefficients).copy(),
        rank=int(rank), condition_number=condition,
    )


def factor_covariance(factor_cov: ArrayLike, exposures: ArrayLike, specific_var: ArrayLike) -> NDArray[np.float64]:
    """Compute B Sigma_f B.T + diag(specific_var), validating PSD inputs.

    Exposures are assets x factors; specific variances are asset-level.
    """
    sigma = _matrix(factor_cov, "factor_cov")
    b = _matrix(exposures, "exposures")
    if sigma.shape[0] != sigma.shape[1] or b.shape[1] != sigma.shape[0]:
        raise ValueError("incompatible factor covariance and exposure dimensions")
    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12):
        raise ValueError("factor covariance must be symmetric")
    sigma = (sigma + sigma.T) / 2
    eigenvalues = np.linalg.eigvalsh(sigma)
    tolerance = 1e-12 * max(1.0, float(np.linalg.norm(sigma, ord=2)))
    if eigenvalues[0] < -tolerance:
        raise ValueError("factor covariance must be positive semidefinite")
    if eigenvalues[0] < 0:
        raise ValueError("factor covariance has negative eigenvalues within tolerance; repair externally")
    specific = _vector(specific_var, "specific_var", b.shape[0])
    if np.any(specific < 0):
        raise ValueError("specific variances must be nonnegative")
    result = b @ sigma @ b.T + np.diag(specific)
    if not np.all(np.isfinite(result)):
        raise ValueError("computed covariance is not finite")
    return (result + result.T) / 2


def portfolio_factor_risk(weights: ArrayLike, asset_covariance: ArrayLike) -> float:
    """Compute portfolio variance w.T Sigma w, not volatility or VaR."""
    sigma = _matrix(asset_covariance, "asset_covariance")
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("asset covariance must be square")
    w = _vector(weights, "weights", sigma.shape[0])
    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12):
        raise ValueError("asset covariance must be symmetric")
    if np.linalg.eigvalsh((sigma + sigma.T) / 2)[0] < -1e-12 * max(1.0, float(np.linalg.norm(sigma, ord=2))):
        raise ValueError("asset covariance must be positive semidefinite")
    value = float(w @ sigma @ w)
    if not np.isfinite(value) or value < -1e-12:
        raise ValueError("computed variance is invalid")
    return max(value, 0.0)


def pca_factors(returns: ArrayLike, n_components: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Centered PCA using sample covariance; returns scores, loadings, eigenvalues.

    Eigenvectors have arbitrary signs. Requires more observations than assets
    and rejects numerically rank-deficient centered data for identifiability.
    """
    x = _matrix(returns, "returns", min_rows=3)
    n_obs, n_assets = x.shape
    if type(n_components) is not int or not 1 <= n_components <= n_assets:
        raise ValueError("n_components must be an integer between 1 and n_assets")
    if n_obs <= n_assets:
        raise ValueError("PCA requires more observations than assets")
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (n_obs - 1)
    eigvals, eigvecs = eigh(cov, check_finite=True)
    if eigvals[0] <= 1e-12 * max(1.0, float(eigvals[-1])):
        raise ValueError("centered data are rank deficient or numerically singular")
    order = np.argsort(eigvals)[::-1][:n_components]
    loadings = eigvecs[:, order].copy()
    # Deterministic sign convention for reproducible component orientations.
    for column in range(loadings.shape[1]):
        pivot = np.argmax(np.abs(loadings[:, column]))
        if loadings[pivot, column] < 0:
            loadings[:, column] *= -1
    return centered @ loadings, loadings, eigvals[order].copy()
