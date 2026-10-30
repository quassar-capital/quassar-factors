
"""Reproducible finite-sample recovery experiments for weak spectral factors.

The reference experiment uses independent Gaussian observations with population
covariance I + theta * u u.T. It measures eigenvalue separation and squared
directional recovery without claiming finite-sample detectability guarantees.

All simulation randomness is generated locally from an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spectral_model import (
    SpikeBenchmark,
    spike_benchmark,
    squared_direction_overlap,
)


@dataclass(frozen=True)
class RecoveryResult:
    """Finite-sample recovery metrics and their asymptotic references."""

    n_assets: int
    n_observations: int
    spike_strength: float
    sample_leading_eigenvalue: float
    sample_squared_overlap: float
    benchmark: SpikeBenchmark


@dataclass(frozen=True)
class RecoverySummary:
    """Monte Carlo summary without an implicit statistical confidence claim."""

    replications: int
    mean_leading_eigenvalue: float
    std_leading_eigenvalue: float
    mean_squared_overlap: float
    std_squared_overlap: float
    benchmark: SpikeBenchmark


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _unit_direction(direction: ArrayLike, dimension: int) -> NDArray[np.float64]:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be a finite vector matching n_assets")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("direction must be nonzero")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _simulate_once(
    n_assets: int,
    n_observations: int,
    spike_strength: float,
    direction: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Simulate a rank-one Gaussian spike and recover its leading direction."""
    noise = rng.standard_normal((n_observations, n_assets))
    factor = rng.standard_normal((n_observations, 1))
    observations = noise + sqrt(spike_strength) * factor * direction[None, :]

    if not np.all(np.isfinite(observations)):
        raise ValueError("simulated observations exceed floating-point range")

    # The theoretical benchmark uses the uncentered 1/T covariance.
    sample_covariance = observations.T @ observations / n_observations
    if not np.all(np.isfinite(sample_covariance)):
        raise ValueError("sample covariance exceeds floating-point range")

    eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
    leading_eigenvalue = float(eigenvalues[-1])
    leading_direction = eigenvectors[:, -1]

    return leading_eigenvalue, squared_direction_overlap(
        leading_direction, direction
    )


def simulate_factor_recovery(
    n_assets: int,
    n_observations: int,
    spike_strength: float,
    direction: ArrayLike,
    *,
    seed: int = 0,
) -> RecoveryResult:
    """Evaluate one finite-sample Gaussian spiked covariance experiment."""
    n = _positive_integer(n_assets, "n_assets")
    t = _positive_integer(n_observations, "n_observations")
    theta = _finite_positive(spike_strength, "spike_strength")
    unit = _unit_direction(direction, n)
    rng = np.random.default_rng(_validate_seed(seed))
    benchmark = spike_benchmark(n / t, theta)

    leading, overlap = _simulate_once(n, t, theta, unit, rng)

    return RecoveryResult(
        n_assets=n,
        n_observations=t,
        spike_strength=theta,
        sample_leading_eigenvalue=leading,
        sample_squared_overlap=overlap,
        benchmark=benchmark,
    )


def monte_carlo_factor_recovery(
    n_assets: int,
    n_observations: int,
    spike_strength: float,
    direction: ArrayLike,
    *,
    replications: int = 100,
    seed: int = 0,
) -> RecoverySummary:
    """Estimate finite-sample recovery means and dispersion by simulation.

    Reported standard deviations describe the simulated distribution across
    independent replications, not standard errors or confidence intervals.
    """
    n = _positive_integer(n_assets, "n_assets")
    t = _positive_integer(n_observations, "n_observations")
    m = _positive_integer(replications, "replications")
    theta = _finite_positive(spike_strength, "spike_strength")
    unit = _unit_direction(direction, n)
    rng = np.random.default_rng(_validate_seed(seed))

    eigenvalues = np.empty(m, dtype=np.float64)
    overlaps = np.empty(m, dtype=np.float64)

    for index in range(m):
        eigenvalues[index], overlaps[index] = _simulate_once(
            n, t, theta, unit, rng
        )

    return RecoverySummary(
        replications=m,
        mean_leading_eigenvalue=float(np.mean(eigenvalues)),
        std_leading_eigenvalue=float(np.std(eigenvalues, ddof=0)),
        mean_squared_overlap=float(np.mean(overlaps)),
        std_squared_overlap=float(np.std(overlaps, ddof=0)),
        benchmark=spike_benchmark(n / t, theta),
    )
