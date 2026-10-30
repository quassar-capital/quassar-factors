
# Research 02 — Identifiability and Spectral Recovery of Weak Factors in High-Dimensional Financial Systems

**Quassar Capital | Quantitative Research | Factor Modeling**

## Research Objective

This study investigates the spectral conditions under which a latent factor direction becomes recoverable from high-dimensional sample covariance matrices.

The analysis separates population factor strength, dimensional concentration, and finite-sample estimation effects. It establishes an analytical reference using rank-one spiked covariance theory and evaluates numerical recovery through reproducible Gaussian Monte Carlo experiments.

The study addresses statistical identifiability under an explicitly specified reference model. It does not establish economic factor validity, predictive performance, or trading profitability.

## Mathematical Framework

Consider independent observations \(x_t \in \mathbb{R}^N\), for \(t=1,\ldots,T\), generated from a zero-mean Gaussian distribution with population covariance

\[
\Sigma = I_N + \theta uu^\top,
\]

where:

- \(N\) denotes the number of observed variables.
- \(T\) denotes the number of observations.
- \(\theta > 0\) denotes the population spike strength.
- \(u \in \mathbb{R}^N\) is a unit-norm population factor direction.
- \(\gamma=N/T\) denotes the dimensional aspect ratio.

The population covariance has one eigenvalue \(1+\theta\) and \(N-1\) eigenvalues equal to one.

The numerical implementation uses the uncentered sample covariance

\[
S = \frac{1}{T}\sum_{t=1}^{T}x_tx_t^\top.
\]

The asymptotic reference considers \(N,T\rightarrow\infty\) with \(N/T\rightarrow\gamma>0\).

## Spectral Separation

For identity population noise, the upper Marchenko–Pastur spectral edge is

\[
\lambda_+ = (1+\sqrt{\gamma})^2.
\]

In the rank-one Gaussian spiked covariance model, the critical spike strength is

\[
\theta_c=\sqrt{\gamma}.
\]

For \(\theta \leq \theta_c\), the leading sample eigenvalue converges to the upper noise edge, and its squared alignment with the population spike direction converges to zero.

For \(\theta>\theta_c\), the asymptotic leading sample eigenvalue is

\[
\lambda_{\mathrm{out}} =
(1+\theta)\left(1+\frac{\gamma}{\theta}\right).
\]

The corresponding squared population-sample eigenvector alignment is

\[
|\langle \hat{u},u\rangle|^2
\longrightarrow
\frac{1-\gamma/\theta^2}{1+\gamma/\theta}.
\]

These expressions are asymptotic benchmarks, not deterministic finite-sample bounds.

## Numerical Recovery Experiment

The experiment generates independent Gaussian noise and a scalar Gaussian factor:

\[
x_t = \varepsilon_t+\sqrt{\theta}\,f_tu,
\]

where

\[
\varepsilon_t\sim\mathcal{N}(0,I_N),
\qquad
f_t\sim\mathcal{N}(0,1),
\]

with independent factor and noise draws.

The implementation constructs the sample covariance matrix, computes its leading eigenpair, and records:

1. The largest sample covariance eigenvalue.
2. The squared cosine between the estimated and population factor directions.
3. The corresponding analytical spectral benchmarks.
4. Monte Carlo means and population-standard-deviation summaries across independent replications.

All simulation runs use locally initialized NumPy random generators with explicit seeds.

## Scientific Interpretation

Spectral separation is governed jointly by signal strength and the ratio of cross-sectional dimension to sample length.

Increasing dimensional concentration raises the population spike strength required for asymptotic eigenvalue separation. Above the critical threshold, the leading empirical eigenvector retains nonzero asymptotic alignment with the population factor direction. Below or at the threshold, that leading eigenvector loses asymptotic directional alignment under the reference model.

Finite-dimensional experiments need not exhibit a sharp transition at the theoretical threshold. Sampling fluctuations, finite-size bias, and random eigenvector variability remain present.

A recovered statistical direction is not automatically an economically interpretable risk factor.

## Validation Architecture

The validation module covers:

- Marchenko–Pastur upper-edge calculations.
- BBP critical and supercritical analytical identities.
- Continuity near the spectral transition.
- Population covariance eigenstructure.
- Sign- and scale-invariant directional alignment.
- Deterministic simulation reproducibility.
- Monte Carlo recovery relative to analytical references.
- Comparative recovery of weak and strong spikes.
- Rejection of invalid model and simulation inputs.

Monte Carlo acceptance criteria are finite-sample numerical checks, not proofs of asymptotic convergence or universal statistical guarantees.

## Research Implementation

| Component | Location | Scientific function |
| --- | --- | --- |
| Spectral model | `python/spectral_model.py` | Analytical covariance and spectral benchmarks |
| Factor recovery | `python/factor_recovery.py` | Reproducible numerical experiments |
| Validation | `tests/test_spectral_recovery.py` | Analytical and numerical integrity checks |

## Reproducibility

From the repository root, install the project and its test dependencies according to `pyproject.toml`.

Execute the Research 02 validation module with:

```bash
python -m pytest research_02/tests/test_spectral_recovery.py -q
```

The test module resolves the Research 02 Python implementation directory explicitly.

The simulation functions expose dimensions, spike strength, population direction, replication count, and random seed as controlled inputs.

## Scope and Limitations

The analytical results assume the specified Gaussian spiked covariance model with isotropic noise and a rank-one positive spike.

The implementation does not establish equivalent recovery properties for arbitrary financial return distributions, temporally dependent observations, heavy-tailed innovations, heterogeneous noise covariance, or multiple interacting spikes.

The study does not provide:

- Empirical validation on live market data.
- Out-of-sample return forecasts.
- Portfolio construction recommendations.
- Transaction-cost or execution modeling.
- Statistical confidence intervals for Monte Carlo summaries.
- Claims of economic causality or investment performance.

Extensions to more general covariance structures require separate mathematical assumptions and independent validation.

## Research Integrity

Analytical limits and finite-sample measurements are maintained as distinct outputs.

Numerical reproducibility is supported through explicit random seeds, controlled input validation, and deterministic reference calculations.

The research is designed to support independent scientific review and further quantitative investigation without overstating what the reference model demonstrates.
