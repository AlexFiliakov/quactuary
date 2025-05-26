"""
Profile baseline implementation to identify bottlenecks.
"""

import cProfile
import pstats
import io
from pstats import SortKey
import pandas as pd
import numpy as np

from quactuary.book import Portfolio, Inforce, PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy


def profile_baseline_simulation():
    """Profile the baseline simulation to identify bottlenecks."""
    # Create test portfolio
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31'),
        per_occ_retention=1000.0,
        per_occ_limit=100000.0
    )
    
    portfolio = Portfolio([
        Inforce(
            n_policies=100,
            terms=terms,
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
            name="Profile Test Bucket"
        )
    ])
    
    # Create model with baseline (no JIT) strategy
    model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=False))
    
    # Profile the simulation
    pr = cProfile.Profile()
    pr.enable()
    
    # Run simulation
    result = model.simulate(
        mean=True,
        variance=True,
        value_at_risk=True,
        tail_value_at_risk=True,
        n_sims=5000
    )
    
    pr.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    
    print("BASELINE PROFILING RESULTS")
    print("=" * 80)
    print(s.getvalue())
    
    # Also print by time
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats(SortKey.TIME)
    ps2.print_stats(20)  # Top 20 by time
    
    print("\nTOP FUNCTIONS BY TIME")
    print("=" * 80)
    print(s2.getvalue())
    
    # Print call counts for numpy/scipy functions
    print("\nCALL COUNTS FOR KEY FUNCTIONS")
    print("=" * 80)
    stats_dict = ps.stats
    for func, (cc, nc, tt, ct, callers) in stats_dict.items():
        func_name = f"{func[2]}"
        if any(module in str(func[0]) for module in ['numpy', 'scipy', 'random', 'rvs', 'simulate']):
            print(f"{func_name:50} calls: {nc:8d} time: {tt:8.3f}s")
    
    return pr


def profile_distribution_sampling():
    """Profile distribution sampling specifically."""
    from quactuary.distributions.frequency import Poisson
    from quactuary.distributions.severity import Lognormal
    
    freq = Poisson(mu=1.5)
    sev = Lognormal(shape=1.0, scale=np.exp(8.0))
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Profile frequency sampling
    for _ in range(10000):
        n = freq.rvs()
    
    # Profile severity sampling  
    for _ in range(10000):
        losses = sev.rvs(size=5)
    
    pr.disable()
    
    print("\nDISTRIBUTION SAMPLING PROFILE")
    print("=" * 80)
    ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
    ps.print_stats(15)


def profile_inforce_simulate():
    """Profile the Inforce.simulate method specifically."""
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31')
    )
    
    inforce = Inforce(
        n_policies=100,
        terms=terms,
        frequency=Poisson(mu=1.5),
        severity=Lognormal(shape=1.0, scale=np.exp(8.0))
    )
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Profile the simulate method
    results = inforce.simulate(1000)
    
    pr.disable()
    
    print("\nINFORCE.SIMULATE PROFILE")
    print("=" * 80)
    ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
    ps.print_stats(20)


if __name__ == "__main__":
    # Run all profiling
    profile_baseline_simulation()
    profile_distribution_sampling()
    profile_inforce_simulate()