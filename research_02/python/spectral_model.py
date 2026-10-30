
"""Analytical benchmarks for high-dimensional spiked covariance models.

Population covariance is I + theta * u u.T, with unit vector u and theta > 0.
The aspect ratio gamma = n_assets / n_observations is strictly positive.
Results describe asymptotic Gaussian/Wishart reference models, not finite-sample
confidence guarantees or empirical financial factor identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SpikeBenchmark:
    """Asymptotic benchmark for one positive population covariance spike."""

    aspect_ratio: float
    spike_strength: float
    noise_edge: float
    critical_strength: float
    separated: bool
    leading_eigenvalue_limit: float
    squared_direction_overlap_limit: float


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite real number") from exc
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return number


def marchenko_pastur_upper_edge(aspect_ratio: float) -> float:
    """Upper noise-spectrum edge for identity population covariance."""
    gamma = _positive_finite(aspect_ratio, "aspect_ratio")
    return (1.0 + sqrt(gamma)) ** 2


def spike_benchmark(aspect_ratio: float, spike_strength: float) -> SpikeBenchmark:
    """BBP threshold, leading eigenvalue, and eigenvector overlap limits.

    Below or at the threshold, the leading sample eigenvalue approaches the
    noise edge and its squared alignment with the population spike vanishes.
    Above the threshold, the outlier and alignment formulas apply.
    """
    gamma = _positive_finite(aspect_ratio, "aspect_ratio")
    theta = _positive_finite(spike_strength, "spike_strength")
    threshold = sqrt(gamma)
    edge = marchenko_pastur_upper_edge(gamma)
    separated = theta > threshold
    if separated:
        leading = (1.0 + theta) * (1.0 + gamma / theta)
        overlap = (1.0 - gamma / (theta * theta)) / (1.0 + gamma / theta)
    else:
        leading = edge
        overlap = 0.0
    return SpikeBenchmark(
        aspect_ratio=gamma,
        spike_strength=theta,
        noise_edge=edge,
        critical_strength=threshold,
        separated=separated,
        leading_eigenvalue_limit=leading,
        squared_direction_overlap_limit=overlap,
    )


def population_spiked_covariance(
    direction: ArrayLike, spike_strength: float
) -> NDArray[np.float64]:
    """Construct I + theta * u u.T for a finite, nonzero direction u.

    The input direction is normalized; its sign does not affect covariance.
    """
    theta = _positive_finite(spike_strength, "spike_strength")
    vector = np.asarray(direction, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be a nonempty finite one-dimensional vector")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("direction must be nonzero")
    scaled = vector / scale
    unit = scaled / np.linalg.norm(scaled)
    covariance = np.eye(vector.size, dtype=np.float64) + theta * np.outer(unit, unit)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("population covariance exceeds floating-point range")
    return covariance


def squared_direction_overlap(estimated: ArrayLike, reference: ArrayLike) -> float:
    """Sign-invariant squared cosine between two nonzero real directions."""
    a = np.asarray(estimated, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("directions must be nonempty one-dimensional vectors of equal length")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("directions must be finite")
    scale_a = float(np.max(np.abs(a)))
    scale_b = float(np.max(np.abs(b)))
    if scale_a == 0.0 or scale_b == 0.0:
        raise ValueError("directions must be nonzero")
    a_unit = (a / scale_a) / np.linalg.norm(a / scale_a)
    b_unit = (b / scale_b) / np.linalg.norm(b / scale_b)
    overlap = float(np.dot(a_unit, b_unit) ** 2)
    return min(1.0, max(0.0, overlap))
