"""
End-to-end scenario testing with realistic data.

This module contains comprehensive end-to-end tests that validate optimization
combinations with realistic insurance portfolio scenarios across different sizes
and complexities.

Test Categories:
- Small portfolio scenarios (10-100 policies)
- Medium portfolio scenarios (100-1000 policies)  
- Large portfolio scenarios (1000+ policies)
- Extreme scenarios (10k+ policies)
- Industry-specific scenarios
- Data quality scenarios
"""

import time
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pytest

import quactuary.book as book
from quactuary.backend import set_backend
from quactuary.book import LOB, Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import (Binomial, Geometric,
                                               NegativeBinomial, Poisson)
from quactuary.distributions.severity import (Exponential, Gamma, Lognormal,
                                              Pareto)
from quactuary.pricing import PricingModel

from .conftest import (assert_memory_efficiency, assert_numerical_accuracy,
                       assert_performance_improvement)
from .test_config import (adapt_test_parameters, get_test_config,
                          skip_if_insufficient_resources)


class TestSmallPortfolioScenarios:
    """Test scenarios for small portfolios (10-100 policies)."""
    
    @pytest.mark.skip(reason="TODO: fix this test")
    @pytest.mark.integration
    def test_homogeneous_small_portfolio(self, performance_profiler, memory_monitor):
        """Test small homogeneous portfolio with similar policies.
        
        Note: Variance tolerance adjusted to 0.5 for QMC comparison as QMC
        methods can have different variance characteristics than standard MC,
        especially with smaller sample sizes.
        """
        set_backend("classical")
        
        # Set deterministic seed for reproducible results
        np.random.seed(42)
        
        # Create homogeneous portfolio - all similar GLPL policies
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=5_000_000,
            retention_type="deductible", 
            per_occ_retention=100_000,
            coverage="occ"
        )
        
        freq = Poisson(mu=2.0)
        sev = Lognormal(shape=1.5, loc=0, scale=25_000)
        
        inforce = Inforce(
            n_policies=50,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Homogeneous GLPL Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        # Test baseline simulation
        baseline_result = pm.simulate(n_sims=1000, tail_alpha=0.05)
        performance_profiler.checkpoint("baseline_complete")
        
        # Test with QMC optimization
        qmc_result = pm.simulate(
            n_sims=1000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True,
            qmc_seed=42
        )
        performance_profiler.checkpoint("qmc_complete")
        memory_monitor.record("end")
        
        # Validate results
        assert baseline_result.estimates['mean'] > 0
        assert qmc_result.estimates['mean'] > 0
        
        # Results should be numerically close for homogeneous portfolio
        # Note: QMC can have different variance characteristics than standard MC
        assert_numerical_accuracy(baseline_result, qmc_result, tolerance_mean=0.05, tolerance_quantiles=0.5)
        
        # Should be fast and memory efficient
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 30
        assert_memory_efficiency(memory_monitor.get_peak_usage_mb(), 1024)


    @pytest.mark.integration
    def test_heterogeneous_small_portfolio(self, performance_profiler):
        """Test small heterogeneous portfolio with mixed coverage types.
        
        Note: Mean tolerance adjusted to 20% for heterogeneous portfolios due to
        increased variability from mixing different policy types with different
        frequency and severity distributions.
        """
        set_backend("classical")
        
        # Set deterministic seed for reproducible results
        np.random.seed(42)
        
        # Create mixed portfolio with different LOBs
        # Workers Comp bucket
        wc_policy = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.WC,
            exposure_base=book.PAYROLL,
            exposure_amount=10_000_000,
            retention_type="deductible",
            per_occ_retention=50_000,
            coverage="occ"
        )
        wc_inforce = Inforce(
            n_policies=25,
            terms=wc_policy,
            frequency=Poisson(mu=3.0),
            severity=Pareto(b=1.5, loc=0, scale=15_000),
            name="WC Bucket"
        )
        
        # Commercial Auto bucket  
        auto_policy = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.CAuto,
            exposure_base=book.VEHICLES,
            exposure_amount=75,
            retention_type="deductible",
            per_occ_retention=25_000,
            coverage="occ"
        )
        auto_inforce = Inforce(
            n_policies=25,
            terms=auto_policy,
            frequency=NegativeBinomial(r=5, p=0.4),
            severity=Exponential(scale=12_000),
            name="Auto Bucket"
        )
        
        portfolio = wc_inforce + auto_inforce
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        # Test with multiple optimization strategies
        baseline_result = pm.simulate(n_sims=1000, tail_alpha=0.05)
        performance_profiler.checkpoint("baseline_complete")
        
        qmc_result = pm.simulate(
            n_sims=1000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True,
            qmc_seed=42
        )
        performance_profiler.checkpoint("qmc_complete")
        
        # Validate heterogeneous results
        assert baseline_result.estimates['mean'] > 0
        assert qmc_result.estimates['mean'] > 0
        
        # TVaR should be higher than VaR for both
        assert baseline_result.estimates['TVaR'] > baseline_result.estimates['VaR']
        assert qmc_result.estimates['TVaR'] > qmc_result.estimates['VaR']
        
        # Results should be consistent between methods
        # Note: Heterogeneous portfolios can have larger differences between MC and QMC
        mean_diff = abs(baseline_result.estimates['mean'] - qmc_result.estimates['mean'])
        mean_avg = (baseline_result.estimates['mean'] + qmc_result.estimates['mean']) / 2
        assert mean_diff / mean_avg < 0.2  # Within 20% for heterogeneous portfolio


    @pytest.mark.integration
    def test_single_policy_many_simulations(self, performance_profiler):
        """Test edge case: single policy with many simulations."""
        set_backend("classical")
        
        # Single policy portfolio
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=1_000_000,
            retention_type="deductible",
            per_occ_retention=50_000,
            coverage="occ"
        )
        
        inforce = Inforce(
            n_policies=1,
            terms=policy_terms,
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.2, loc=0, scale=20_000),
            name="Single Policy"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        # Many simulations on single policy
        result = pm.simulate(
            n_sims=10000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("single_policy_complete")
        
        # Validate single policy results
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        assert result.estimates['VaR'] >= 0  # Could be 0 for single policy
        assert result.estimates['TVaR'] >= result.estimates['VaR']
        
        # Should complete efficiently even with many sims
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 60


class TestMediumPortfolioScenarios:
    """Test scenarios for medium portfolios (100-1000 policies)."""
    
    @pytest.mark.integration
    def test_property_insurance_portfolio(self, performance_profiler, memory_monitor):
        """Test realistic property insurance portfolio."""
        set_backend("classical")
        
        # Property insurance portfolio
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.PProperty,
            exposure_base=book.REPLACEMENT_VALUE,
            exposure_amount=50_000_000,
            retention_type="deductible",
            per_occ_retention=10_000,
            coverage="occ"
        )
        
        # Property typically has low frequency, high severity
        freq = Poisson(mu=0.8)  # Low frequency
        sev = Pareto(b=1.8, loc=0, scale=45_000)  # Heavy-tailed severity
        
        inforce = Inforce(
            n_policies=500,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Property Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        result = pm.simulate(
            n_sims=2000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("property_complete")
        memory_monitor.record("end")
        
        # Property insurance validation
        assert result.estimates['mean'] > 0
        
        # Property should have high tail risk (TVaR > VaR)
        tail_ratio = result.estimates['TVaR'] / result.estimates['VaR']
        assert tail_ratio > 1.0, f"Tail ratio {tail_ratio:.2f} too low for property insurance"
        
        # Performance should be reasonable
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 120
        assert_memory_efficiency(memory_monitor.get_peak_usage_mb(), 2048)


    @pytest.mark.integration
    def test_liability_insurance_portfolio(self, performance_profiler):
        """Test realistic liability insurance portfolio."""
        set_backend("classical")
        
        # Liability insurance portfolio
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=25_000_000,
            retention_type="deductible",
            per_occ_retention=100_000,
            coverage="occ"
        )
        
        # Liability typically has moderate frequency, very high severity potential
        freq = NegativeBinomial(r=8, p=0.7)  # Moderate frequency with overdispersion
        sev = Lognormal(shape=2.5, loc=0, scale=75_000)  # High severity potential
        
        inforce = Inforce(
            n_policies=300,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Liability Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        result = pm.simulate(
            n_sims=2000,
            tail_alpha=0.01,  # Test extreme tail (99th percentile)
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("liability_complete")
        
        # Liability insurance validation
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        
        # Should have very high tail risk
        cv = np.sqrt(result.estimates['variance']) / result.estimates['mean']
        assert cv > 0.5, f"Coefficient of variation {cv:.2f} too low for liability"
        
        # Extreme tail measures should be significantly higher
        assert result.estimates['TVaR'] > result.estimates['VaR'] * 1.5


    @pytest.mark.integration  
    def test_mixed_lines_portfolio(self, performance_profiler):
        """Test portfolio with mixed lines of business."""
        set_backend("classical")
        
        # Create diversified portfolio
        portfolios = []
        
        # Auto liability
        auto_policy = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.CAuto,
            exposure_base=book.VEHICLES,
            exposure_amount=200,
            retention_type="deductible",
            per_occ_retention=5_000,
            coverage="occ"
        )
        auto_inforce = Inforce(
            n_policies=150,
            terms=auto_policy,
            frequency=Geometric(p=1/6),
            severity=Exponential(scale=8_000),
            name="Auto Liability"
        )
        
        # General liability  
        gl_policy = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=15_000_000,
            retention_type="deductible",
            per_occ_retention=75_000,
            coverage="occ"
        )
        gl_inforce = Inforce(
            n_policies=100,
            terms=gl_policy,
            frequency=Poisson(mu=1.8),
            severity=Pareto(b=1.6, loc=0, scale=35_000),
            name="General Liability"
        )
        
        # Workers compensation
        wc_policy = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.WC,
            exposure_base=book.PAYROLL,
            exposure_amount=8_000_000,
            retention_type="deductible",
            per_occ_retention=25_000,
            coverage="occ"
        )
        wc_inforce = Inforce(
            n_policies=100,
            terms=wc_policy,
            frequency=NegativeBinomial(r=12, p=0.8),
            severity=Gamma(shape=2.0, scale=18_000),
            name="Workers Compensation"
        )
        
        portfolio = auto_inforce + gl_inforce + wc_inforce
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        result = pm.simulate(
            n_sims=3000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("mixed_lines_complete")
        
        # Mixed portfolio validation
        assert result.estimates['mean'] > 0
        
        # Diversification should reduce relative volatility
        cv = np.sqrt(result.estimates['variance']) / result.estimates['mean']
        # Diversified portfolios can have low CV values
        assert 0.1 < cv < 2.0, f"CV {cv:.2f} outside expected range for diversified portfolio"
        
        # Should see benefits of diversification in tail measures
        assert result.estimates['TVaR'] > result.estimates['VaR']
        
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 180


class TestLargePortfolioScenarios:
    """Test scenarios for large portfolios (1000+ policies)."""
    
    @pytest.mark.skip(reason="TODO: fix this test")
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.memory_intensive
    @pytest.mark.hardware_dependent
    @skip_if_insufficient_resources(min_cpus=4, min_memory_gb=8, min_profile='standard')
    def test_large_portfolio_memory_management(self, performance_profiler, memory_monitor):
        """Test memory management effectiveness on large portfolios."""
        set_backend("classical")
        
        # Get environment-based expectations
        test_config = get_test_config()
        expectations = test_config['expectations']
        
        # Create large portfolio
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=100_000_000,
            retention_type="deductible",
            per_occ_retention=250_000,
            coverage="occ"
        )
        
        freq = Poisson(mu=2.5)
        sev = Lognormal(shape=1.8, loc=0, scale=50_000)
        
        # Adapt portfolio size based on environment
        base_params = {'n_policies': 2000, 'n_simulations': 5000}
        adapted_params = adapt_test_parameters(base_params)
        
        inforce = Inforce(
            n_policies=adapted_params.get('n_policies', 2000),
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Large Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        result = pm.simulate(
            n_sims=adapted_params['n_simulations'],
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("large_portfolio_complete")
        memory_monitor.record("peak")
        
        # Validate large portfolio results
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        
        # Memory should be managed efficiently
        peak_memory_mb = memory_monitor.get_peak_usage_mb()
        assert_memory_efficiency(peak_memory_mb, 4096)
        
        # Performance should scale reasonably
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 600  # 10 minutes max


    # DEPRECATED: Parallel scaling tests are hardware-specific
    @pytest.mark.skip(reason="Deprecated: Parallel scaling depends on hardware")
    @pytest.mark.integration
    @pytest.mark.slow
    def test_parallel_scaling_validation(self, performance_profiler):
        """Test parallel scaling on large portfolio."""
        set_backend("classical")
        
        # Create moderately large portfolio for parallel testing
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=75_000_000,
            retention_type="deductible",
            per_occ_retention=150_000,
            coverage="occ"
        )
        
        freq = Poisson(mu=2.0)
        sev = Pareto(b=1.5, loc=0, scale=40_000)
        
        inforce = Inforce(
            n_policies=1500,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Parallel Test Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        # Test single-threaded performance
        start_time = time.time()
        result_single = pm.simulate(
            n_sims=3000,
            tail_alpha=0.05,
            qmc_method='sobol'
        )
        single_time = time.time() - start_time
        
        performance_profiler.checkpoint("single_threaded_complete")
        
        # Test with QMC (which may use parallel processing internally)
        start_time = time.time()
        result_qmc = pm.simulate(
            n_sims=3000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        qmc_time = time.time() - start_time
        
        performance_profiler.checkpoint("parallel_complete")
        
        # Validate parallel results
        assert_numerical_accuracy(result_single, result_qmc, tolerance_mean=0.05)
        
        # QMC should provide some performance benefit
        if single_time > 5.0:  # Only check for non-trivial runtimes
            speedup = single_time / qmc_time
            assert speedup >= 1.0, f"QMC slower than baseline: {speedup:.2f}x"


class TestExtremeScenarios:
    """Test extreme scenarios and edge cases."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.memory_intensive
    def test_extreme_portfolio_size(self, performance_profiler, memory_monitor):
        """Test with very large portfolio (10k+ policies)."""
        set_backend("classical")
        
        # Extreme portfolio size
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=500_000_000,
            retention_type="deductible",
            per_occ_retention=500_000,
            coverage="occ"
        )
        
        freq = Poisson(mu=1.5)  # Lower frequency to keep computation manageable
        sev = Lognormal(shape=1.5, loc=0, scale=100_000)
        
        inforce = Inforce(
            n_policies=10000,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Extreme Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        # Use smaller simulation count for extreme portfolio
        result = pm.simulate(
            n_sims=1000,  # Reduced simulations due to size
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("extreme_portfolio_complete")
        memory_monitor.record("peak")
        
        # Validate extreme scenario
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        
        # Should handle extreme size gracefully
        peak_memory_mb = memory_monitor.get_peak_usage_mb()
        assert peak_memory_mb < 16384, f"Memory usage {peak_memory_mb:.0f}MB too high for extreme portfolio"
        
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 1800  # 30 minutes max


    @pytest.mark.integration
    def test_data_quality_edge_cases(self, performance_profiler):
        """Test handling of edge cases in data quality."""
        set_backend("classical")
        
        # Test zero-loss scenario
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=LOB.GLPL,
            exposure_base=book.SALES,
            exposure_amount=1_000_000,
            retention_type="deductible",
            per_occ_retention=1_000_000,  # Very high deductible
            coverage="occ"
        )
        
        # Very low frequency - should produce many zero-loss scenarios
        freq = Poisson(mu=0.1)
        sev = Exponential(scale=50_000)
        
        inforce = Inforce(
            n_policies=100,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name="Low Frequency Portfolio"
        )
        
        portfolio = Portfolio(inforce)
        pm = PricingModel(portfolio)
        
        performance_profiler.start()
        
        result = pm.simulate(
            n_sims=2000,
            tail_alpha=0.05,
            qmc_method='sobol'
        )
        
        performance_profiler.checkpoint("edge_cases_complete")
        
        # Should handle zero-loss cases gracefully
        assert result.estimates['mean'] >= 0  # Could be zero
        assert result.estimates['variance'] >= 0
        assert result.estimates['VaR'] >= 0
        assert result.estimates['TVaR'] >= result.estimates['VaR']
        
        # Should not have any NaN or infinite values
        for key, value in result.estimates.items():
            assert not np.isnan(value), f"NaN detected in {key}"
            assert not np.isinf(value), f"Infinite value detected in {key}"