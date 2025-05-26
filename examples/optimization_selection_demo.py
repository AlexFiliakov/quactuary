"""
Demonstration of automatic optimization selection based on portfolio characteristics.
"""

import numpy as np
import pandas as pd
from quactuary.pricing import PricingModel
from quactuary.book import Portfolio
from quactuary.distributions import Poisson, Lognormal, Pareto, NegativeBinomial
from quactuary.optimization_selector import OptimizationSelector


def create_sample_portfolio(n_policies):
    """Create a sample portfolio for testing."""
    policies_data = []
    for i in range(n_policies):
        policies_data.append({
            'policy_id': f'POL_{i:06d}',
            'exposure': np.random.uniform(100000, 1000000),
            'limit': np.random.uniform(500000, 5000000),
            'deductible': np.random.uniform(5000, 50000),
            'line_of_business': np.random.choice(['Property', 'Casualty', 'Auto'])
        })
    
    df = pd.DataFrame(policies_data)
    return Portfolio(df)


def demonstrate_optimization_selection():
    """Demonstrate automatic optimization selection for different scenarios."""
    
    # Create optimization selector
    optimizer = OptimizationSelector(enable_ml=False)
    
    print("=== Automatic Optimization Selection Demo ===\n")
    
    # Scenario 1: Small portfolio
    print("1. Small Portfolio (100 policies, 1000 simulations)")
    small_portfolio = create_sample_portfolio(100)
    small_model = PricingModel(small_portfolio, optimization_selector=optimizer)
    
    # Analyze what optimizations would be selected
    profile = optimizer.analyze_portfolio(small_portfolio, n_simulations=1000)
    config = optimizer.predict_best_strategy(profile)
    
    print(f"   Total data points: {profile.total_data_points:,}")
    print(f"   Estimated memory: {profile.estimated_memory_gb:.2f} GB")
    print(f"   Estimated time: {profile.estimated_compute_time:.2f} seconds")
    print(f"   Selected optimizations:")
    print(f"   - Vectorization: {config.use_vectorization}")
    print(f"   - JIT: {config.use_jit}")
    print(f"   - Parallel: {config.use_parallel}")
    print(f"   - QMC: {config.use_qmc}")
    print(f"   - Memory optimization: {config.use_memory_optimization}")
    
    # Run simulation with auto optimization
    print("\n   Running simulation with auto-optimization...")
    result = small_model.simulate(
        auto_optimize=True,
        n_sims=1000
    )
    print(f"   Expected loss: ${result.mean:,.2f}\n")
    
    # Scenario 2: Medium portfolio
    print("2. Medium Portfolio (10,000 policies, 10,000 simulations)")
    medium_portfolio = create_sample_portfolio(10000)
    medium_model = PricingModel(medium_portfolio, optimization_selector=optimizer)
    
    profile = optimizer.analyze_portfolio(medium_portfolio, n_simulations=10000)
    config = optimizer.predict_best_strategy(profile)
    
    print(f"   Total data points: {profile.total_data_points:,}")
    print(f"   Estimated memory: {profile.estimated_memory_gb:.2f} GB")
    print(f"   Estimated time: {profile.estimated_compute_time:.2f} seconds")
    print(f"   Selected optimizations:")
    print(f"   - Vectorization: {config.use_vectorization}")
    print(f"   - JIT: {config.use_jit}")
    print(f"   - Parallel: {config.use_parallel}")
    print(f"   - QMC: {config.use_qmc} (method: {config.qmc_method})")
    print(f"   - Memory optimization: {config.use_memory_optimization}\n")
    
    # Scenario 3: Large portfolio with complex distributions
    print("3. Large Portfolio with Complex Distributions")
    large_portfolio = create_sample_portfolio(50000)
    
    # Add complex compound distribution
    large_model = PricingModel(large_portfolio, optimization_selector=optimizer)
    large_model.set_compound_distribution(
        frequency=NegativeBinomial(n=20, p=0.4),
        severity=Pareto(alpha=2.5, scale=10000)
    )
    
    profile = optimizer.analyze_portfolio(large_portfolio, n_simulations=50000)
    config = optimizer.predict_best_strategy(profile)
    
    print(f"   Total data points: {profile.total_data_points:,}")
    print(f"   Estimated memory: {profile.estimated_memory_gb:.2f} GB")
    print(f"   Distribution complexity: {profile.distribution_complexity:.2f}")
    print(f"   Has dependencies: {profile.has_dependencies}")
    print(f"   Selected optimizations:")
    print(f"   - Vectorization: {config.use_vectorization}")
    print(f"   - JIT: {config.use_jit} (disabled due to complexity)")
    print(f"   - Parallel: {config.use_parallel} (workers: {config.n_workers})")
    print(f"   - QMC: {config.use_qmc}")
    print(f"   - Memory optimization: {config.use_memory_optimization}")
    if config.batch_size:
        print(f"   - Batch size: {config.batch_size}\n")
    
    # Scenario 4: Memory-constrained environment
    print("4. Memory-Constrained Scenario")
    
    # Simulate low memory availability
    print("   Simulating environment with limited memory...")
    
    # This would normally detect actual memory, but for demo we'll show the logic
    huge_portfolio = create_sample_portfolio(100000)
    profile = optimizer.analyze_portfolio(huge_portfolio, n_simulations=100000)
    
    # Manually set low available memory for demo
    profile.available_memory_gb = 4.0
    profile.estimated_memory_gb = 8.0
    
    config = optimizer.predict_best_strategy(profile)
    
    print(f"   Available memory: {profile.available_memory_gb:.1f} GB")
    print(f"   Required memory: {profile.estimated_memory_gb:.1f} GB")
    print(f"   Memory optimization enabled: {config.use_memory_optimization}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Parallel disabled to save memory: {not config.use_parallel}\n")
    
    # Runtime adaptation demo
    print("5. Runtime Adaptation Demo")
    print("   Enabling runtime monitoring...")
    
    optimizer.enable_monitoring()
    
    # Simulate runtime metrics showing high memory usage
    runtime_metrics = {
        'memory_usage': 0.92,  # 92% memory used
        'cpu_usage': 0.75,
        'time_elapsed': 30.0,
        'progress_rate': 0.8
    }
    
    print(f"   Current memory usage: {runtime_metrics['memory_usage']:.0%}")
    adapted_config = optimizer.monitor_and_adapt(runtime_metrics)
    
    if adapted_config:
        print("   ⚠️  High memory detected! Adapting strategy:")
        print(f"   - Memory optimization: {adapted_config.use_memory_optimization}")
        print(f"   - Parallel processing: {adapted_config.use_parallel}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_optimization_selection()