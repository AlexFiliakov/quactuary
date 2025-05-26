"""
Comprehensive tests for zero-inflated compound distributions.

This module provides thorough testing of zero-inflated models including
EM algorithm convergence, parameter estimation, and model selection tests.
"""

import numpy as np
import pytest
from scipy import stats, optimize
from hypothesis import given, strategies as st, settings
import warnings

from quactuary.distributions.frequency import Poisson, NegativeBinomial, Binomial
from quactuary.distributions.severity import Exponential, Gamma, Lognormal
from quactuary.distributions.zero_inflated import (
    ZeroInflatedCompound,
    ZIPoissonCompound,
    ZINegativeBinomialCompound,
    ZIBinomialCompound,
    ZeroInflatedMixtureEM,
    detect_zero_inflation
)
from quactuary.distributions.compound import create_compound_distribution
from quactuary.utils.numerical import stable_log, stable_exp


class TestZeroInflatedMixtureEMConvergence:
    """Test EM algorithm for parameter estimation."""
    
    def test_em_convergence_simple_case(self):
        """Test EM algorithm convergence on simple simulated data."""
        # True parameters
        true_zero_prob = 0.3
        true_lambda = 5.0
        
        # Generate zero-inflated Poisson data
        np.random.seed(42)
        n_samples = 1000
        
        # Zero inflation
        is_zero = np.random.binomial(1, true_zero_prob, n_samples)
        # Poisson component
        poisson_samples = np.random.poisson(true_lambda, n_samples)
        # Combined
        samples = np.where(is_zero, 0, poisson_samples)
        
        # Initialize EM algorithm
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        
        # Fit parameters
        params, log_likelihood, n_iterations = em.fit(
            samples,
            max_iter=100,
            tol=1e-6,
            verbose=False
        )
        
        # Check convergence
        assert n_iterations < 50, f"EM took too many iterations: {n_iterations}"
        
        # Check parameter recovery
        estimated_zero_prob = params['zero_prob']
        estimated_lambda = params['lambda']
        
        assert np.isclose(estimated_zero_prob, true_zero_prob, rtol=0.1)
        assert np.isclose(estimated_lambda, true_lambda, rtol=0.1)
    
    def test_em_monotonic_likelihood(self):
        """Test that EM algorithm monotonically increases likelihood."""
        # Generate data
        np.random.seed(123)
        freq = Poisson(mu=3.0)
        sev = Exponential(scale=100)
        
        # Zero-inflated compound
        true_zero_prob = 0.25
        n_samples = 500
        
        # Generate samples
        is_structural_zero = np.random.binomial(1, true_zero_prob, n_samples)
        compound_samples = []
        
        for i in range(n_samples):
            if is_structural_zero[i]:
                compound_samples.append(0)
            else:
                n_claims = freq.rvs()
                if n_claims == 0:
                    compound_samples.append(0)
                else:
                    losses = sev.rvs(size=n_claims)
                    compound_samples.append(np.sum(losses))
        
        samples = np.array(compound_samples)
        
        # Run EM with tracking
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='gamma')
        
        # Track likelihood at each iteration
        likelihoods = []
        
        def callback(params, ll, iteration):
            likelihoods.append(ll)
        
        params, final_ll, n_iter = em.fit(
            samples,
            callback=callback,
            max_iter=50
        )
        
        # Check monotonic increase
        for i in range(1, len(likelihoods)):
            assert likelihoods[i] >= likelihoods[i-1] - 1e-10, \
                f"Likelihood decreased at iteration {i}: {likelihoods[i-1]} -> {likelihoods[i]}"
    
    def test_em_parameter_constraints(self):
        """Test that EM respects parameter constraints."""
        # Test with negative binomial
        np.random.seed(456)
        
        # Generate NB data with zero inflation
        r, p = 5.0, 0.4
        zero_prob = 0.2
        n_samples = 800
        
        is_zero = np.random.binomial(1, zero_prob, n_samples)
        nb_samples = stats.nbinom.rvs(r, p, size=n_samples)
        samples = np.where(is_zero, 0, nb_samples)
        
        # Fit with EM
        em = ZeroInflatedMixtureEM(frequency_type='nbinom', severity_type='exponential')
        params, _, _ = em.fit(samples)
        
        # Check constraints
        assert 0 <= params['zero_prob'] <= 1
        assert params['r'] > 0
        assert 0 < params['p'] < 1
    
    def test_em_initialization_strategies(self):
        """Test different initialization strategies for EM."""
        # Generate challenging data (high zero inflation)
        np.random.seed(789)
        true_zero_prob = 0.6
        true_mu = 2.0
        
        n_samples = 500
        is_zero = np.random.binomial(1, true_zero_prob, n_samples)
        poisson_samples = np.random.poisson(true_mu, n_samples)
        samples = np.where(is_zero, 0, poisson_samples)
        
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        
        # Test different initializations
        init_strategies = [
            {'zero_prob': 0.1, 'mu': 5.0},  # Far from truth
            {'zero_prob': 0.5, 'mu': 3.0},  # Closer
            {'zero_prob': 0.7, 'mu': 1.5},  # Close to truth
        ]
        
        results = []
        for init_params in init_strategies:
            params, ll, n_iter = em.fit(
                samples,
                init_params=init_params,
                max_iter=100
            )
            results.append((params, ll, n_iter))
        
        # All should converge to similar parameters
        final_zero_probs = [r[0]['zero_prob'] for r in results]
        final_mus = [r[0]['mu'] for r in results]
        
        # Check consistency across initializations
        assert np.std(final_zero_probs) < 0.05
        assert np.std(final_mus) < 0.2
        
        # All should be close to truth
        for zp in final_zero_probs:
            assert np.isclose(zp, true_zero_prob, rtol=0.15)


class TestVuongTestStatistics:
    """Test Vuong test for model comparison."""
    
    def test_vuong_test_zi_vs_standard(self):
        """Test Vuong test comparing ZI model vs standard model."""
        # Generate data with significant zero inflation
        np.random.seed(111)
        true_zero_prob = 0.4
        true_lambda = 4.0
        n_samples = 500
        
        # Generate ZI-Poisson data
        is_zero = np.random.binomial(1, true_zero_prob, n_samples)
        poisson_samples = np.random.poisson(true_lambda, n_samples)
        samples = np.where(is_zero, 0, poisson_samples)
        
        # Fit both models
        # Standard Poisson
        poisson_mle = np.mean(samples)
        poisson_model = Poisson(mu=poisson_mle)
        
        # ZI-Poisson
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        zi_params, _, _ = em.fit(samples)
        zi_model = ZIPoissonCompound(
            Poisson(mu=zi_params['mu']),
            Exponential(scale=1),  # Dummy severity
            zero_prob=zi_params['zero_prob']
        )
        
        # Vuong test
        vuong_stat, p_value = vuong_test(
            samples,
            model1=zi_model,
            model2=poisson_model,
            model1_type='zi_poisson',
            model2_type='poisson'
        )
        
        # ZI model should be significantly better
        assert vuong_stat > 2.0  # Strong preference for ZI
        assert p_value < 0.05
    
    def test_vuong_test_no_zero_inflation(self):
        """Test Vuong test when there's no zero inflation."""
        # Generate standard Poisson data
        np.random.seed(222)
        true_lambda = 5.0
        n_samples = 500
        
        samples = np.random.poisson(true_lambda, n_samples)
        
        # Fit both models
        poisson_mle = np.mean(samples)
        poisson_model = Poisson(mu=poisson_mle)
        
        # ZI-Poisson
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        zi_params, _, _ = em.fit(samples)
        zi_model = ZIPoissonCompound(
            Poisson(mu=zi_params['mu']),
            Exponential(scale=1),
            zero_prob=zi_params['zero_prob']
        )
        
        # Vuong test
        vuong_stat, p_value = vuong_test(
            samples,
            model1=zi_model,
            model2=poisson_model,
            model1_type='zi_poisson',
            model2_type='poisson'
        )
        
        # Should not strongly prefer either model
        assert abs(vuong_stat) < 2.0
        # Zero prob should be estimated near 0
        assert zi_params['zero_prob'] < 0.05
    
    def test_vuong_test_compound_distributions(self):
        """Test Vuong test for compound distributions."""
        # Generate zero-inflated compound data
        np.random.seed(333)
        freq = NegativeBinomial(r=3.0, p=0.5)
        sev = Gamma(shape=2.0, scale=500)
        true_zero_prob = 0.35
        n_samples = 400
        
        # Generate samples
        is_zero = np.random.binomial(1, true_zero_prob, n_samples)
        compound_samples = []
        
        for i in range(n_samples):
            if is_zero[i]:
                compound_samples.append(0)
            else:
                n_claims = freq.rvs()
                if n_claims == 0:
                    compound_samples.append(0)
                else:
                    losses = sev.rvs(size=n_claims)
                    compound_samples.append(np.sum(losses))
        
        samples = np.array(compound_samples)
        
        # Standard compound model
        standard_compound = create_compound_distribution(freq, sev)
        
        # ZI compound model
        zi_compound = ZINegativeBinomialCompound(freq, sev, true_zero_prob)
        
        # Vuong test
        vuong_stat, p_value = vuong_test(
            samples,
            model1=zi_compound,
            model2=standard_compound,
            model1_type='zi_compound',
            model2_type='compound'
        )
        
        # ZI model should be preferred
        assert vuong_stat > 1.5
        assert p_value < 0.1


class TestScoreTest:
    """Test score test for zero inflation."""
    
    def test_score_test_significant_zi(self):
        """Test score test detects significant zero inflation."""
        # Generate data with zero inflation
        np.random.seed(444)
        true_zero_prob = 0.3
        true_lambda = 6.0
        n_samples = 300
        
        is_zero = np.random.binomial(1, true_zero_prob, n_samples)
        poisson_samples = np.random.poisson(true_lambda, n_samples)
        samples = np.where(is_zero, 0, poisson_samples)
        
        # Score test
        score_stat, p_value = score_test_zi(
            samples,
            distribution='poisson'
        )
        
        # Should detect zero inflation
        assert score_stat > 2.0
        assert p_value < 0.05
    
    def test_score_test_no_zi(self):
        """Test score test with no zero inflation."""
        # Standard negative binomial data
        np.random.seed(555)
        r, p = 5.0, 0.6
        n_samples = 300
        
        samples = stats.nbinom.rvs(r, p, size=n_samples)
        
        # Score test
        score_stat, p_value = score_test_zi(
            samples,
            distribution='negative_binomial'
        )
        
        # Should not detect zero inflation
        assert abs(score_stat) < 2.0
        assert p_value > 0.05
    
    def test_score_test_boundary_case(self):
        """Test score test at boundary (very high zero proportion)."""
        # Almost all zeros
        np.random.seed(666)
        samples = np.zeros(200)
        samples[0:5] = [1, 2, 1, 3, 1]  # Few non-zeros
        
        # Score test
        score_stat, p_value = score_test_zi(
            samples,
            distribution='poisson'
        )
        
        # Should strongly indicate zero inflation
        assert score_stat > 5.0
        assert p_value < 0.001


class TestZeroInflatedCompoundDistributions:
    """Test specific zero-inflated compound distribution implementations."""
    
    def test_zi_poisson_compound_properties(self):
        """Test ZI-Poisson compound distribution properties."""
        freq = Poisson(mu=4.0)
        sev = Exponential(scale=100)
        zero_prob = 0.25
        
        zi_compound = ZIPoissonCompound(freq, sev, zero_prob)
        
        # Mean and variance
        base_compound = create_compound_distribution(freq, sev)
        expected_mean = (1 - zero_prob) * base_compound.mean()
        
        assert np.isclose(zi_compound.mean(), expected_mean, rtol=1e-10)
        
        # PDF at zero
        p_zero_base = np.exp(-4.0)  # Poisson P(N=0)
        p_zero_total = zero_prob + (1 - zero_prob) * p_zero_base
        
        assert np.isclose(zi_compound.pdf(0), p_zero_total, rtol=1e-10)
        
        # CDF properties
        assert zi_compound.cdf(0) == zi_compound.pdf(0)
        assert zi_compound.cdf(-1) == 0
        assert zi_compound.cdf(np.inf) == 1
    
    def test_zi_negative_binomial_compound(self):
        """Test ZI-NB compound distribution."""
        freq = NegativeBinomial(r=3.0, p=0.4)
        sev = Gamma(shape=2.0, scale=300)
        zero_prob = 0.3
        
        zi_compound = ZINegativeBinomialCompound(freq, sev, zero_prob)
        
        # Test random sampling
        np.random.seed(777)
        samples = zi_compound.rvs(size=1000)
        
        # Check zero proportion
        empirical_zero_prop = np.mean(samples == 0)
        
        # Expected zero proportion
        p_zero_nb = stats.nbinom.pmf(0, 3.0, 0.4)
        expected_zero_prop = zero_prob + (1 - zero_prob) * p_zero_nb
        
        assert np.isclose(empirical_zero_prop, expected_zero_prop, rtol=0.1)
        
        # Check mean
        assert np.isclose(np.mean(samples), zi_compound.mean(), rtol=0.05)
    
    def test_zi_binomial_compound(self):
        """Test ZI-Binomial compound distribution."""
        n, p = 10, 0.3
        freq = Binomial(n=n, p=p)
        sev = Lognormal(shape=0.5, scale=1000)
        zero_prob = 0.2
        
        zi_compound = ZIBinomialCompound(freq, sev, zero_prob)
        
        # Test quantile function
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        
        for q in quantiles:
            x_q = zi_compound.ppf(q)
            cdf_x = zi_compound.cdf(x_q)
            
            # Due to discrete mass at 0, need special handling
            if x_q == 0:
                assert cdf_x >= q
            else:
                assert np.isclose(cdf_x, q, rtol=1e-3)
    
    def test_parameter_estimation_methods(self):
        """Test different parameter estimation methods."""
        # Generate data
        np.random.seed(888)
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=200)
        true_zero_prob = 0.35
        
        zi_compound = ZIPoissonCompound(freq, sev, true_zero_prob)
        samples = zi_compound.rvs(size=500)
        
        # Method of moments estimation
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        zero_count = np.sum(samples == 0)
        
        # Estimate zero probability from zero frequency
        p_zero_poisson = np.exp(-5.0)
        expected_zero_freq = true_zero_prob + (1 - true_zero_prob) * p_zero_poisson
        
        estimated_zero_freq = zero_count / len(samples)
        estimated_zero_prob = (estimated_zero_freq - p_zero_poisson) / (1 - p_zero_poisson)
        
        # Should recover approximately
        assert np.isclose(estimated_zero_prob, true_zero_prob, rtol=0.2)


class TestPropertyBasedZeroInflated:
    """Property-based tests for zero-inflated distributions."""
    
    @given(
        mu=st.floats(min_value=0.5, max_value=20.0),
        zero_prob=st.floats(min_value=0.0, max_value=0.9),
        scale=st.floats(min_value=10, max_value=10000)
    )
    @settings(max_examples=30)
    def test_zi_poisson_properties(self, mu, zero_prob, scale):
        """Test mathematical properties of ZI-Poisson compound."""
        freq = Poisson(mu=mu)
        sev = Exponential(scale=scale)
        
        zi_compound = ZIPoissonCompound(freq, sev, zero_prob)
        base_compound = create_compound_distribution(freq, sev)
        
        # Mean relationship
        assert np.isclose(
            zi_compound.mean(),
            (1 - zero_prob) * base_compound.mean(),
            rtol=1e-10
        )
        
        # Variance relationship (law of total variance)
        base_mean = base_compound.mean()
        base_var = base_compound.var()
        
        expected_var = ((1 - zero_prob) * base_var + 
                       (1 - zero_prob) * zero_prob * base_mean**2)
        
        assert np.isclose(zi_compound.var(), expected_var, rtol=1e-10)
        
        # PDF normalization
        if mu < 10:  # For reasonable computation
            x_max = zi_compound.mean() + 10 * zi_compound.std()
            x_grid = np.linspace(0, x_max, 1000)
            
            # Separate handling for discrete mass at 0
            p_zero = zi_compound.pdf(0)
            continuous_integral = np.trapz(
                zi_compound.pdf(x_grid[1:]), 
                x_grid[1:]
            )
            
            total = p_zero + continuous_integral
            assert np.isclose(total, 1.0, rtol=0.01)
    
    @given(
        r=st.floats(min_value=0.5, max_value=20.0),
        p=st.floats(min_value=0.1, max_value=0.9),
        zero_prob=st.floats(min_value=0.0, max_value=0.8),
        shape=st.floats(min_value=0.5, max_value=5.0)
    )
    @settings(max_examples=30)
    def test_zi_negative_binomial_monotonicity(self, r, p, zero_prob, shape):
        """Test CDF monotonicity for ZI-NB compound."""
        freq = NegativeBinomial(r=r, p=p)
        sev = Gamma(shape=shape, scale=100)
        
        zi_compound = ZINegativeBinomialCompound(freq, sev, zero_prob)
        
        # Test CDF at multiple points
        x_values = np.logspace(0, 5, 20)
        cdf_values = zi_compound.cdf(x_values)
        
        # Check monotonicity
        diffs = np.diff(cdf_values)
        assert all(diff >= -1e-10 for diff in diffs)
        
        # Check bounds
        assert zi_compound.cdf(0) >= zero_prob
        assert all(0 <= cdf <= 1 for cdf in cdf_values)


class TestEdgeCasesZeroInflated:
    """Test edge cases for zero-inflated distributions."""
    
    def test_zero_inflation_probability_extremes(self):
        """Test with extreme zero inflation probabilities."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=100)
        
        # No zero inflation (zero_prob = 0)
        zi_compound_0 = ZIPoissonCompound(freq, sev, zero_prob=0.0)
        base_compound = create_compound_distribution(freq, sev)
        
        # Should match base compound
        x_test = np.array([0, 100, 500, 1000])
        np.testing.assert_allclose(
            zi_compound_0.pdf(x_test),
            base_compound.pdf(x_test),
            rtol=1e-10
        )
        
        # Very high zero inflation (zero_prob = 0.99)
        zi_compound_99 = ZIPoissonCompound(freq, sev, zero_prob=0.99)
        
        # Almost all mass at zero
        assert zi_compound_99.pdf(0) > 0.99
        assert zi_compound_99.mean() < 0.01 * base_compound.mean()
    
    def test_all_zeros_data(self):
        """Test parameter estimation with all zeros."""
        samples = np.zeros(100)
        
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        
        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, _, _ = em.fit(samples, max_iter=10)
        
        # Zero probability should be very high
        assert params['zero_prob'] > 0.9
    
    def test_no_zeros_data(self):
        """Test with data containing no zeros."""
        # Generate Poisson data conditioned on being positive
        np.random.seed(999)
        mu = 5.0
        samples = []
        
        while len(samples) < 200:
            x = np.random.poisson(mu)
            if x > 0:
                samples.append(x)
        
        samples = np.array(samples)
        
        # Score test should not indicate ZI
        score_stat, p_value = score_test_zi(samples, distribution='poisson')
        assert p_value > 0.05
        
        # EM should estimate very low zero_prob
        em = ZeroInflatedMixtureEM(frequency_type='poisson', severity_type='exponential')
        params, _, _ = em.fit(samples)
        
        assert params['zero_prob'] < 0.05


class TestNumericalStabilityZeroInflated:
    """Test numerical stability of zero-inflated implementations."""
    
    def test_extreme_parameter_combinations(self):
        """Test with extreme parameter values."""
        # Very small frequencies with high zero inflation
        freq = Poisson(mu=0.1)
        sev = Exponential(scale=10000)
        zero_prob = 0.9
        
        zi_compound = ZIPoissonCompound(freq, sev, zero_prob)
        
        # Should handle without errors
        assert zi_compound.mean() < 100
        assert zi_compound.pdf(0) > 0.99
        
        # Very large frequencies
        freq_large = Poisson(mu=100)
        sev_large = Exponential(scale=10)
        zero_prob_small = 0.01
        
        zi_compound_large = ZIPoissonCompound(freq_large, sev_large, zero_prob_small)
        
        # Should be close to non-inflated
        base_mean = 100 * 10
        assert np.isclose(zi_compound_large.mean(), 0.99 * base_mean, rtol=0.01)
    
    def test_log_likelihood_computation(self):
        """Test stable log-likelihood computation."""
        freq = NegativeBinomial(r=2.0, p=0.3)
        sev = Gamma(shape=3.0, scale=100)
        zero_prob = 0.4
        
        zi_compound = ZINegativeBinomialCompound(freq, sev, zero_prob)
        
        # Generate samples including edge cases
        samples = [0, 0, 0, 1e-10, 1000, 10000, 100000]
        
        # Compute log-likelihood without overflow/underflow
        log_likelihoods = []
        for x in samples:
            pdf_val = zi_compound.pdf(x)
            if pdf_val > 0:
                ll = stable_log(pdf_val)
                assert np.isfinite(ll)
                log_likelihoods.append(ll)
        
        # Total log-likelihood should be finite
        total_ll = sum(log_likelihoods)
        assert np.isfinite(total_ll)


def test_integration_with_factory():
    """Test integration of zero-inflated distributions with factory."""
    from quactuary.distributions.compound_extensions import create_extended_compound_distribution
    
    # Create zero-inflated compound
    freq = Binomial(n=20, p=0.3)
    sev = Lognormal(shape=1.0, scale=1000)
    
    zi_compound = create_extended_compound_distribution(
        freq, sev,
        zero_inflated=True,
        zero_prob=0.25
    )
    
    # Check it's properly zero-inflated
    assert isinstance(zi_compound, ZIBinomialCompound)
    
    # Test basic functionality
    assert zi_compound.pdf(0) > 0.25
    
    # Test with string inputs
    zi_compound2 = create_extended_compound_distribution(
        'poisson', 'exponential',
        zero_inflated=True,
        zero_prob=0.3,
        mu=5.0,
        scale=200
    )
    
    assert isinstance(zi_compound2, ZIPoissonCompound)
    assert zi_compound2.zero_prob == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])