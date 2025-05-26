import pytest
import numpy as np
from datetime import date
import time

from quactuary.book import PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy
from quactuary.backend import BackendManager, ClassicalBackend


class TestPricingOptimizations:
    """Test pricing model with various optimization settings."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        inforces = []
        for i in range(10):
            policy = Inforce(
                n_policies=1,
                name=f"P{i:03d}",
                frequency=Poisson(mu=5.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                terms=PolicyTerms(
                    effective_date=date(2024, 1, 1),
                    expiration_date=date(2024, 12, 31),
                    per_occ_retention=1000.0,
                    agg_retention=5000.0,
                    per_occ_limit=100000.0,
                    agg_limit=500000.0
                )
            )
            inforces.append(policy)
        return Portfolio(inforces)
    
    def test_jit_optimization_consistency(self, sample_portfolio):
        """Test that JIT optimization produces consistent results."""
        n_sims = 5000
        
        # Run with JIT disabled
        strategy_no_jit = ClassicalPricingStrategy(use_jit=False)
        pm_no_jit = PricingModel(sample_portfolio, strategy=strategy_no_jit)
        result_no_jit = pm_no_jit.simulate(n_sims=n_sims)
        
        # Run with JIT enabled
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        pm_jit = PricingModel(sample_portfolio, strategy=strategy_jit)
        result_jit = pm_jit.simulate(n_sims=n_sims)
        
        # Compare results - should be similar (not exact due to randomness)
        assert result_no_jit.mean > 0
        assert result_jit.mean > 0
        # Allow for some variance due to random sampling
        assert abs(result_no_jit.mean - result_jit.mean) / result_no_jit.mean < 0.1
        assert abs(result_no_jit.variance - result_jit.variance) / result_no_jit.variance < 0.2
    
    def test_qmc_optimization(self, sample_portfolio):
        """Test QMC optimization with Sobol sequences."""
        n_sims = 4096  # Power of 2 for Sobol
        
        # Standard Monte Carlo
        strategy_mc = ClassicalPricingStrategy(use_jit=False)
        pm_mc = PricingModel(sample_portfolio, strategy=strategy_mc)
        result_mc = pm_mc.simulate(n_sims=n_sims)
        
        # Quasi-Monte Carlo with Sobol
        result_qmc = pm_mc.simulate(
            n_sims=n_sims,
            qmc_method="sobol",
            qmc_scramble=True,
            qmc_seed=42
        )
        
        # Both should produce reasonable results
        assert result_mc.mean > 0
        assert result_qmc.mean > 0
        # Results should be reasonably close
        assert abs(result_mc.mean - result_qmc.mean) / result_mc.mean < 0.2
    
    def test_combined_optimizations(self, sample_portfolio):
        """Test JIT + QMC optimizations combined."""
        n_sims = 4096
        
        # JIT + QMC
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        pm_jit = PricingModel(sample_portfolio, strategy=strategy_jit)
        
        # Warm up JIT
        pm_jit.simulate(n_sims=100)
        
        # Run with both optimizations
        start_time = time.time()
        result_optimized = pm_jit.simulate(
            n_sims=n_sims,
            qmc_method="sobol",
            qmc_scramble=True,
            qmc_seed=42
        )
        optimized_time = time.time() - start_time
        
        # Run without optimizations
        strategy_no_opt = ClassicalPricingStrategy(use_jit=False)
        pm_no_opt = PricingModel(sample_portfolio, strategy=strategy_no_opt)
        
        start_time = time.time()
        result_no_opt = pm_no_opt.simulate(n_sims=n_sims)
        no_opt_time = time.time() - start_time
        
        # Verify results are consistent
        assert result_optimized.mean > 0
        assert result_no_opt.mean > 0
        assert abs(result_optimized.mean - result_no_opt.mean) / result_no_opt.mean < 0.2
        
        # Log timing results
        print(f"Optimized time: {optimized_time:.3f}s, No optimization time: {no_opt_time:.3f}s")
    
    def test_optimization_with_risk_measures(self, sample_portfolio):
        """Test optimizations work correctly with risk measure calculations."""
        n_sims = 5000
        
        # Configure to calculate all risk measures
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        pm_jit = PricingModel(sample_portfolio, strategy=strategy_jit)
        
        result = pm_jit.simulate(
            n_sims=n_sims,
            mean=True,
            variance=True,
            value_at_risk=True,
            tail_value_at_risk=True,
            tail_alpha=0.05
        )
        
        # Verify all risk measures are calculated
        assert result.mean > 0
        assert result.variance > 0
        assert result.value_at_risk > result.mean
        assert result.tail_value_at_risk > result.value_at_risk
    
    def test_optimization_scalability(self, sample_portfolio):
        """Test that optimizations scale with portfolio size."""
        # Small portfolio (first 2 inforces)
        small_portfolio = Portfolio(sample_portfolio[:2])
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        pm_small = PricingModel(small_portfolio, strategy=strategy_jit)
        
        # Warm up
        pm_small.simulate(n_sims=100)
        
        start_time = time.time()
        result_small = pm_small.simulate(n_sims=1000)
        small_time = time.time() - start_time
        
        # Large portfolio (repeat inforces 5 times)
        large_inforces = list(sample_portfolio) * 5  # 50 policies
        large_portfolio = Portfolio(large_inforces)
        pm_large = PricingModel(large_portfolio, strategy=strategy_jit)
        
        # Warm up
        pm_large.simulate(n_sims=100)
        
        start_time = time.time()
        result_large = pm_large.simulate(n_sims=1000)
        large_time = time.time() - start_time
        
        # Time should scale roughly linearly with portfolio size
        time_ratio = large_time / small_time
        portfolio_ratio = len(large_portfolio) / len(small_portfolio)
        
        print(f"Small portfolio time: {small_time:.3f}s, Large portfolio time: {large_time:.3f}s")
        print(f"Time ratio: {time_ratio:.2f}, Portfolio ratio: {portfolio_ratio:.2f}")
        
        # Verify results are valid
        assert result_small.mean > 0
        assert result_large.mean > 0
    
    @pytest.mark.parametrize("use_jit,qmc_method", [
        (False, None),
        (True, None),
        (False, "sobol"),
        (True, "sobol"),
        (False, "halton"),
        (True, "halton"),
    ])
    def test_optimization_combinations(self, sample_portfolio, use_jit, qmc_method):
        """Test various combinations of optimization settings."""
        n_sims = 2048 if qmc_method else 2000
        
        strategy = ClassicalPricingStrategy(use_jit=use_jit)
        pm = PricingModel(sample_portfolio, strategy=strategy)
        
        # Warm up JIT if enabled
        if use_jit:
            pm.simulate(n_sims=100)
        
        # Run simulation with settings
        kwargs = {
            'n_sims': n_sims,
            'mean': True,
            'variance': True
        }
        
        if qmc_method:
            kwargs.update({
                'qmc_method': qmc_method,
                'qmc_scramble': True,
                'qmc_seed': 42
            })
        
        result = pm.simulate(**kwargs)
        
        # Verify results are valid
        assert result.mean > 0
        assert result.variance > 0
    
    def test_optimization_with_edge_cases(self):
        """Test optimizations with edge case portfolios."""
        # Portfolio with zero frequency
        zero_freq_policy = Inforce(
            n_policies=1,
            name="P_ZERO",
            frequency=Poisson(mu=0.0),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
            terms=PolicyTerms(
                effective_date=date(2024, 1, 1),
                expiration_date=date(2024, 12, 31)
            )
        )
        
        portfolio = Portfolio([zero_freq_policy])
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        pm = PricingModel(portfolio, strategy=strategy_jit)
        result = pm.simulate(n_sims=1000)
        
        # Should handle zero frequency gracefully
        assert result.mean == 0.0
        assert result.variance == 0.0
        
        # Portfolio with very high severity
        high_sev_policy = Inforce(
            n_policies=1,
            name="P_HIGH",
            frequency=Poisson(mu=1.0),
            severity=Lognormal(shape=2.0, scale=np.exp(12.0)),
            terms=PolicyTerms(
                effective_date=date(2024, 1, 1),
                expiration_date=date(2024, 12, 31),
                per_occ_limit=1000000.0
            )
        )
        
        portfolio_high = Portfolio([high_sev_policy])
        pm_high = PricingModel(portfolio_high, strategy=strategy_jit)
        result_high = pm_high.simulate(n_sims=1000)
        
        # Should handle high severity
        assert result_high.mean > 0
        # Check samples are finite
        if hasattr(result_high, '_samples') and result_high._samples is not None:
            assert np.all(np.isfinite(result_high._samples))