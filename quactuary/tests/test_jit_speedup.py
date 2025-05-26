"""
Test JIT speedup vs baseline.
"""

import time
import numpy as np
import pandas as pd
from quactuary.book import Portfolio, Inforce, PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal, Exponential
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy


def test_jit_speedup():
    """Compare JIT vs baseline performance."""
    
    # Create test portfolios of different sizes
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31'),
        per_occ_retention=1000.0,
        per_occ_limit=100000.0
    )
    
    portfolios = {
        'small': Portfolio([
            Inforce(
                n_policies=10,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                name="Small"
            )
        ]),
        'medium': Portfolio([
            Inforce(
                n_policies=50,
                terms=terms,
                frequency=Poisson(mu=1.5),
                severity=Lognormal(shape=1.2, scale=np.exp(8.5)),
                name="Med1"
            ),
            Inforce(
                n_policies=50,
                terms=terms,
                frequency=Poisson(mu=0.8),
                severity=Exponential(scale=5000.0),
                name="Med2"
            )
        ]),
        'large': Portfolio([
            Inforce(
                n_policies=100,
                terms=terms,
                frequency=Poisson(mu=1.0 + i*0.1),
                severity=Lognormal(shape=0.8 + i*0.05, scale=np.exp(8.0 + i*0.1)),
                name=f"Large{i}"
            ) for i in range(5)
        ])
    }
    
    simulation_counts = {
        'small': [1000, 10000],
        'medium': [1000, 10000],
        'large': [1000, 5000]
    }
    
    print("JIT SPEEDUP BENCHMARK")
    print("=" * 80)
    
    for portfolio_name, portfolio in portfolios.items():
        total_policies = sum(bucket.n_policies for bucket in portfolio)
        print(f"\n{portfolio_name.upper()} PORTFOLIO ({total_policies} policies)")
        print("-" * 60)
        
        for n_sims in simulation_counts[portfolio_name]:
            print(f"\nSimulations: {n_sims:,}")
            
            # Baseline (no JIT)
            model_baseline = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=False))
            start = time.perf_counter()
            result_baseline = model_baseline.simulate(n_sims=n_sims)
            baseline_time = time.perf_counter() - start
            
            # JIT enabled
            model_jit = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
            # Warm up JIT
            _ = model_jit.simulate(n_sims=10)
            
            start = time.perf_counter()
            result_jit = model_jit.simulate(n_sims=n_sims)
            jit_time = time.perf_counter() - start
            
            # Calculate speedup
            speedup = baseline_time / jit_time
            
            # Verify results are similar
            baseline_mean = result_baseline.estimates.get('mean', 0)
            jit_mean = result_jit.estimates.get('mean', 0)
            relative_diff = abs(jit_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0
            
            print(f"  Baseline: {baseline_time:.3f}s")
            print(f"  JIT:      {jit_time:.3f}s")
            print(f"  Speedup:  {speedup:.1f}x")
            print(f"  Mean diff: {relative_diff:.1%}")
            
            if relative_diff > 0.05:
                print(f"  WARNING: Results differ by more than 5%!")


def test_jit_compilation_overhead():
    """Test JIT compilation overhead."""
    print("\n\nJIT COMPILATION OVERHEAD TEST")
    print("=" * 80)
    
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31')
    )
    
    portfolio = Portfolio([
        Inforce(
            n_policies=100,
            terms=terms,
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0))
        )
    ])
    
    model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
    
    # First run (includes compilation)
    start = time.perf_counter()
    _ = model.simulate(n_sims=100)
    first_run_time = time.perf_counter() - start
    
    # Second run (already compiled)
    start = time.perf_counter()
    _ = model.simulate(n_sims=100)
    second_run_time = time.perf_counter() - start
    
    # Third run with more simulations
    start = time.perf_counter()
    _ = model.simulate(n_sims=10000)
    large_run_time = time.perf_counter() - start
    
    print(f"First run (with compilation):  {first_run_time:.3f}s")
    print(f"Second run (pre-compiled):     {second_run_time:.3f}s")
    print(f"Compilation overhead:          {first_run_time - second_run_time:.3f}s")
    print(f"Large run (10k sims):          {large_run_time:.3f}s")
    print(f"Per-sim time after compilation: {large_run_time/10000*1000:.3f}ms")


if __name__ == "__main__":
    test_jit_speedup()
    test_jit_compilation_overhead()