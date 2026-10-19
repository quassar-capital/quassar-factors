
"""Independent numerical and adversarial tests for the factor research core."""

import numpy as np
import pytest

from quassar_factors import (
    factor_covariance,
    fit_factor_model,
    pca_factors,
    portfolio_factor_risk,
)


def test_ols_recovers_known_exposures_and_intercepts():
    rng = np.random.default_rng(101)
    factors = rng.normal(size=(200, 3))
    betas = np.array([[1.2, -0.5], [0.4, 0.9], [-0.7, 0.3]])
    alpha = np.array([0.02, -0.01])
    returns = factors @ betas + alpha

    fit = fit_factor_model(returns, factors)

    np.testing.assert_allclose(fit.coefficients, betas, atol=1e-12)
    np.testing.assert_allclose(fit.intercept, alpha, atol=1e-12)
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-12)
    assert fit.rank == 4
    assert np.isfinite(fit.condition_number)


def test_ols_without_intercept():
    factors = np.arange(1.0, 21.0).reshape(-1, 1)
    returns = 2.5 * factors

    fit = fit_factor_model(returns, factors, intercept=False)

    np.testing.assert_allclose(fit.coefficients, [[2.5]], atol=1e-12)
    np.testing.assert_array_equal(fit.intercept, [0.0])


def test_ols_rejects_rank_deficiency():
    x = np.arange(20.0).reshape(-1, 1)
    factors = np.column_stack((x, 2.0 * x))
    returns = x + 1.0

    with pytest.raises(ValueError, match="rank deficient"):
        fit_factor_model(returns, factors)


def test_ols_rejects_mismatched_observations():
    with pytest.raises(ValueError, match="observation count"):
        fit_factor_model(np.ones((12, 2)), np.ones((11, 2)))


def test_ols_rejects_nonfinite_data():
    returns = np.ones((12, 1))
    returns[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit_factor_model(returns, np.arange(12.0).reshape(-1, 1))


def test_ols_rejects_insufficient_observations():
    with pytest.raises(ValueError, match="more observations"):
        fit_factor_model(np.ones((3, 1)), np.ones((3, 2)))


def test_ols_rejects_invalid_rcond():
    x = np.arange(10.0).reshape(-1, 1)
    with pytest.raises(ValueError, match="rcond"):
        fit_factor_model(x, x, rcond=0.0)


def test_factor_covariance_matches_analytical_reference():
    sigma_f = np.diag([0.04, 0.09])
    exposures = np.array([[1.0, 0.0], [0.5, 2.0]])
    specific = np.array([0.01, 0.02])

    actual = factor_covariance(sigma_f, exposures, specific)
    expected = np.array([[0.05, 0.02], [0.02, 0.39]])

    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_factor_covariance_is_symmetric_psd():
    sigma_f = np.array([[0.09, 0.02], [0.02, 0.04]])
    exposures = np.array([[1.0, 0.5], [-0.3, 1.0], [0.7, -0.2]])
    result = factor_covariance(sigma_f, exposures, [0.01, 0.02, 0.03])

    np.testing.assert_allclose(result, result.T, atol=1e-14)
    assert np.linalg.eigvalsh(result).min() >= -1e-14


def test_factor_covariance_rejects_negative_variance():
    with pytest.raises(ValueError, match="nonnegative"):
        factor_covariance([[0.04]], [[1.0]], [-0.01])


def test_factor_covariance_rejects_indefinite_matrix():
    with pytest.raises(ValueError, match="positive semidefinite"):
        factor_covariance([[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0]], [0.0])


def test_factor_covariance_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="incompatible"):
        factor_covariance(np.eye(2), np.ones((3, 1)), np.ones(3))


def test_portfolio_variance_matches_direct_calculation():
    weights = np.array([0.6, 0.4])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    expected = 0.6**2 * 0.04 + 2 * 0.6 * 0.4 * 0.01 + 0.4**2 * 0.09

    assert portfolio_factor_risk(weights, covariance) == pytest.approx(expected)


def test_portfolio_variance_rejects_indefinite_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        portfolio_factor_risk([0.5, 0.5], [[1.0, 2.0], [2.0, 1.0]])


def test_portfolio_variance_rejects_wrong_weights():
    with pytest.raises(ValueError, match="length"):
        portfolio_factor_risk([1.0], np.eye(2))


def test_pca_reconstructs_full_rank_centered_data():
    rng = np.random.default_rng(202)
    returns = rng.normal(size=(150, 3))

    scores, loadings, eigenvalues = pca_factors(returns, 3)
    centered = returns - returns.mean(axis=0)

    np.testing.assert_allclose(scores @ loadings.T, centered, atol=1e-12)
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(3), atol=1e-12)
    assert np.all(np.diff(eigenvalues) <= 0)


def test_pca_covariance_eigenvalue_trace_identity():
    rng = np.random.default_rng(303)
    returns = rng.normal(size=(100, 4))

    _, _, eigenvalues = pca_factors(returns, 4)
    sample_cov = np.cov(returns, rowvar=False)

    assert eigenvalues.sum() == pytest.approx(np.trace(sample_cov))


def test_pca_rejects_rank_deficient_data():
    x = np.arange(30.0)
    returns = np.column_stack((x, 2.0 * x))

    with pytest.raises(ValueError, match="rank deficient"):
        pca_factors(returns, 2)


def test_pca_rejects_invalid_component_count():
    returns = np.arange(30.0).reshape(10, 3)

    with pytest.raises(ValueError, match="n_components"):
        pca_factors(returns, 0)


def test_pca_rejects_too_few_observations():
    with pytest.raises(ValueError, match="more observations"):
        pca_factors(np.eye(3), 2)
