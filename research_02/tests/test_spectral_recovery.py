"""Analytical and numerical validation of spectral factor recovery.

Tests cover BBP reference identities, population covariance structure,
directional invariance, reproducibility, and finite-sample recovery.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(MODULE_DIR))

from factor_recovery import (  # noqa: E402
    monte_carlo_factor_recovery,
    simulate_factor_recovery,
)
from spectral_model import (  # noqa: E402
    marchenko_pastur_upper_edge,
    population_spiked_covariance,
    spike_benchmark,
    squared_direction_overlap,
)


def test_marchenko_pastur_edge_and_bbp_threshold():
    assert marchenko_pastur_upper_edge(1.0) == pytest.approx(4.0)

    below = spike_benchmark(0.25, 0.3)
    critical = spike_benchmark(0.25, 0.5)
    above = spike_benchmark(0.25, 1.0)

    assert below.critical_strength == pytest.approx(0.5)
    assert not below.separated
    assert not critical.separated
    assert above.separated
    assert below.leading_eigenvalue_limit == pytest.approx(2.25)
    assert critical.leading_eigenvalue_limit == pytest.approx(2.25)
    assert below.squared_direction_overlap_limit == 0.0
    assert critical.squared_direction_overlap_limit == 0.0
    assert above.leading_eigenvalue_limit == pytest.approx(2.5)
    assert above.squared_direction_overlap_limit == pytest.approx(0.6)


def test_supercritical_transition_is_continuous():
    gamma = 0.36
    threshold = np.sqrt(gamma)
    near = spike_benchmark(gamma, threshold + 1e-7)

    assert near.separated
    assert near.leading_eigenvalue_limit > near.noise_edge
    assert near.leading_eigenvalue_limit - near.noise_edge < 1e-8
    assert 0.0 < near.squared_direction_overlap_limit < 1e-5


def test_population_covariance_spectrum():
    direction = np.array([3.0, -4.0, 0.0, 0.0])
    covariance = population_spiked_covariance(direction, 2.0)
    eigenvalues = np.linalg.eigvalsh(covariance)

    np.testing.assert_allclose(eigenvalues, [1.0, 1.0, 1.0, 3.0])
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)


def test_directional_overlap_is_sign_and_scale_invariant():
    reference = np.array([3.0, 4.0, 0.0])
    assert squared_direction_overlap(reference, -7.0 * reference) == pytest.approx(1.0)
    assert squared_direction_overlap(reference, [0.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_invalid_spectral_parameters_are_rejected():
    with pytest.raises(ValueError):
        spike_benchmark(0.0, 1.0)
    with pytest.raises(ValueError):
        spike_benchmark(0.5, float("nan"))
    with pytest.raises(ValueError):
        population_spiked_covariance([0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        squared_direction_overlap([1.0], [1.0, 2.0])


def test_simulation_is_reproducible_and_bounded():
    direction = np.ones(8)
    first = simulate_factor_recovery(8, 80, 2.0, direction, seed=101)
    second = simulate_factor_recovery(8, 80, 2.0, direction, seed=101)

    assert first == second
    assert first.sample_leading_eigenvalue > 0.0
    assert 0.0 <= first.sample_squared_overlap <= 1.0
    assert first.benchmark.separated


def test_monte_carlo_recovery_against_analytical_reference():
    direction = np.ones(8)
    result = monte_carlo_factor_recovery(
        8, 160, 3.0, direction, replications=40, seed=202
    )

    assert result.replications == 40
    assert result.benchmark.separated
    assert result.mean_squared_overlap > 0.75
    assert (
        abs(
            result.mean_leading_eigenvalue
            - result.benchmark.leading_eigenvalue_limit
        )
        < 0.7
    )
    assert result.std_leading_eigenvalue > 0.0
    assert result.std_squared_overlap >= 0.0


def test_weak_factor_has_lower_recovery_than_strong_factor():
    direction = np.ones(12)
    weak = monte_carlo_factor_recovery(
        12, 120, 0.1, direction, replications=30, seed=303
    )
    strong = monte_carlo_factor_recovery(
        12, 120, 4.0, direction, replications=30, seed=303
    )

    assert not weak.benchmark.separated
    assert strong.benchmark.separated
    assert strong.mean_squared_overlap > weak.mean_squared_overlap


def test_invalid_simulation_contracts_are_rejected():
    with pytest.raises(ValueError):
        simulate_factor_recovery(0, 20, 1.0, [1.0])
    with pytest.raises(ValueError):
        simulate_factor_recovery(2, 20, 1.0, [1.0])
    with pytest.raises(ValueError):
        simulate_factor_recovery(2, 20, 1.0, [1.0, 1.0], seed=-1)
    with pytest.raises(ValueError):
        monte_carlo_factor_recovery(2, 20, 1.0, [1.0, 1.0], replications=0)
