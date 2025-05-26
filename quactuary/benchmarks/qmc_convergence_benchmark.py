"""
QMC Convergence Benchmark

Measures convergence rates of QMC vs standard Monte Carlo for various portfolio configurations.
Uses the correct API: PolicyTerms, Inforce, Portfolio (not Book/Policy).
"""

import numpy as np
import pandas as pd
import time
from datetime import date
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from quactuary.backend import set_backend
from quactuary.book import LOB, PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson, NegativeBinomial, Binomial
from quactuary.distributions.severity import Exponential, Lognormal, Pareto, Gamma
from quactuary.pricing import PricingModel
from quactuary.sobol import set_qmc_simulator, reset_qmc_simulator


def create_test_portfolio(size: str = "medium") -> Portfolio:
    """Create test portfolio of specified size using correct API."""
    
    # Define policy terms
    gl_terms = PolicyTerms(
        effective_date=date(2026, 1, 1),
        expiration_date=date(2027, 1, 1),
        lob=LOB.GLPL,
        exposure_base="SALES",
        exposure_amount=10_000_000,
        retention_type="deductible",
        per_occ_retention=100_000,
        coverage="occ"
    )
    
    wc_terms = PolicyTerms(
        effective_date=date(2026, 1, 1),
        expiration_date=date(2027, 1, 1),
        lob=LOB.WC,
        exposure_base="PAYROLL",
        exposure_amount=5_000_000,
        retention_type="deductible",
        per_occ_retention=50_000,
        coverage="occ"
    )
    
    auto_terms = PolicyTerms(
        effective_date=date(2026, 1, 1),
        expiration_date=date(2027, 1, 1),
        lob=LOB.CAuto,
        exposure_base="VEHICLES",
        exposure_amount=100,
        retention_type="deductible",
        per_occ_retention=25_000,
        coverage="occ"
    )
    
    if size == "small":
        # Small portfolio: 3 inforce buckets
        inforces = [
            Inforce(
                n_policies=50,
                terms=gl_terms,
                frequency=Poisson(mu=2.5),
                severity=Exponential(scale=30_000),
                name="GL Small Accounts"
            ),
            Inforce(
                n_policies=30,
                terms=wc_terms,
                frequency=Poisson(mu=5.0),
                severity=Lognormal(mu=10, sigma=1.2),
                name="WC Small Accounts"
            ),
            Inforce(
                n_policies=20,
                terms=auto_terms,
                frequency=Binomial(n=10, p=0.3),
                severity=Gamma(alpha=2, scale=15_000),
                name="Auto Fleet"
            )
        ]
    
    elif size == "medium":
        # Medium portfolio: 6 inforce buckets
        inforces = [
            Inforce(
                n_policies=100,
                terms=gl_terms,
                frequency=Poisson(mu=3.5),
                severity=Exponential(scale=50_000),
                name="GL Medium Accounts"
            ),
            Inforce(
                n_policies=80,
                terms=gl_terms,
                frequency=NegativeBinomial(n=10, p=0.4),
                severity=Lognormal(mu=11, sigma=1.5),
                name="GL Large Accounts"
            ),
            Inforce(
                n_policies=60,
                terms=wc_terms,
                frequency=Poisson(mu=8.0),
                severity=Pareto(alpha=2.5, scale=20_000),
                name="WC Medium Accounts"
            ),
            Inforce(
                n_policies=40,
                terms=wc_terms,
                frequency=NegativeBinomial(n=15, p=0.3),
                severity=Lognormal(mu=10.5, sigma=1.8),
                name="WC Large Accounts"
            ),
            Inforce(
                n_policies=50,
                terms=auto_terms,
                frequency=Poisson(mu=4.0),
                severity=Gamma(alpha=2, scale=20_000),
                name="Auto Small Fleet"
            ),
            Inforce(
                n_policies=30,
                terms=auto_terms,
                frequency=Binomial(n=20, p=0.25),
                severity=Exponential(scale=35_000),
                name="Auto Large Fleet"
            )
        ]
    
    else:  # large
        # Large portfolio: 12 inforce buckets
        inforces = []
        
        # GL buckets
        for i in range(4):
            freq = Poisson(mu=2 + i) if i % 2 == 0 else NegativeBinomial(n=10+i*2, p=0.4-i*0.05)
            sev = Exponential(scale=30_000 + i*10_000) if i % 2 == 0 else Lognormal(mu=10+i*0.5, sigma=1+i*0.2)
            inforces.append(Inforce(
                n_policies=100 + i*20,
                terms=gl_terms,
                frequency=freq,
                severity=sev,
                name=f"GL Bucket {i+1}"
            ))
        
        # WC buckets
        for i in range(4):
            freq = Poisson(mu=5 + i*2) if i % 2 == 0 else NegativeBinomial(n=8+i*3, p=0.35-i*0.05)
            sev = Gamma(alpha=2+i*0.5, scale=15_000+i*5_000) if i % 2 == 0 else Pareto(alpha=2+i*0.3, scale=20_000+i*5_000)
            inforces.append(Inforce(
                n_policies=80 + i*15,
                terms=wc_terms,
                frequency=freq,
                severity=sev,
                name=f"WC Bucket {i+1}"
            ))
        
        # Auto buckets
        for i in range(4):
            freq = Poisson(mu=3 + i) if i % 2 == 0 else Binomial(n=10+i*5, p=0.3-i*0.02)
            sev = Exponential(scale=25_000 + i*5_000) if i % 2 == 0 else Gamma(alpha=2.5+i*0.3, scale=18_000+i*3_000)
            inforces.append(Inforce(
                n_policies=40 + i*10,
                terms=auto_terms,
                frequency=freq,
                severity=sev,
                name=f"Auto Bucket {i+1}"
            ))
    
    # Create portfolio by summing inforces
    portfolio = inforces[0]
    for inf in inforces[1:]:
        portfolio = portfolio + inf
    
    return portfolio


def run_convergence_test(portfolio: Portfolio, 
                        sample_sizes: List[int],
                        n_replications: int = 10,
                        confidence_level: float = 0.99) -> Dict[str, pd.DataFrame]:
    """Run convergence test comparing MC and QMC."""
    
    results = {
        'mc': [],
        'qmc_unscrambled': [],
        'qmc_scrambled': []
    }
    
    true_value = None  # Will estimate with very large MC run
    
    # Estimate "true" value with large MC simulation
    print("Estimating true value with large MC simulation...")
    set_backend('classical')
    reset_qmc_simulator()  # Ensure standard MC
    model = PricingModel(portfolio)
    large_result = model.simulate(n_sims=100_000, tail_alpha=1-confidence_level)
    true_value = large_result.estimates['VaR']
    print(f"Estimated true VaR: ${true_value:,.2f}")
    
    for n_sims in sample_sizes:
        print(f"\nTesting with {n_sims:,} simulations...")
        
        # Standard Monte Carlo
        mc_estimates = []
        mc_times = []
        for rep in range(n_replications):
            reset_qmc_simulator()
            set_backend('classical')
            
            start_time = time.time()
            model = PricingModel(portfolio)
            result = model.simulate(n_sims=n_sims, tail_alpha=1-confidence_level)
            mc_times.append(time.time() - start_time)
            mc_estimates.append(result.estimates['VaR'])
        
        results['mc'].append({
            'n_sims': n_sims,
            'mean': np.mean(mc_estimates),
            'std': np.std(mc_estimates),
            'rmse': np.sqrt(np.mean((np.array(mc_estimates) - true_value)**2)),
            'rel_error': np.abs(np.mean(mc_estimates) - true_value) / true_value,
            'time': np.mean(mc_times)
        })
        
        # QMC Unscrambled
        qmc_unscr_estimates = []
        qmc_unscr_times = []
        for rep in range(n_replications):
            set_qmc_simulator(method='sobol', scramble=False, skip=1024)
            set_backend('classical')
            
            start_time = time.time()
            model = PricingModel(portfolio)
            result = model.simulate(n_sims=n_sims, tail_alpha=1-confidence_level)
            qmc_unscr_times.append(time.time() - start_time)
            qmc_unscr_estimates.append(result.estimates['VaR'])
        
        results['qmc_unscrambled'].append({
            'n_sims': n_sims,
            'mean': np.mean(qmc_unscr_estimates),
            'std': np.std(qmc_unscr_estimates),
            'rmse': np.sqrt(np.mean((np.array(qmc_unscr_estimates) - true_value)**2)),
            'rel_error': np.abs(np.mean(qmc_unscr_estimates) - true_value) / true_value,
            'time': np.mean(qmc_unscr_times)
        })
        
        # QMC Scrambled (Owen)
        qmc_scr_estimates = []
        qmc_scr_times = []
        for rep in range(n_replications):
            set_qmc_simulator(method='sobol', scramble=True, seed=rep, skip=1024)
            set_backend('classical')
            
            start_time = time.time()
            model = PricingModel(portfolio)
            result = model.simulate(n_sims=n_sims, tail_alpha=1-confidence_level)
            qmc_scr_times.append(time.time() - start_time)
            qmc_scr_estimates.append(result.estimates['VaR'])
        
        results['qmc_scrambled'].append({
            'n_sims': n_sims,
            'mean': np.mean(qmc_scr_estimates),
            'std': np.std(qmc_scr_estimates),
            'rmse': np.sqrt(np.mean((np.array(qmc_scr_estimates) - true_value)**2)),
            'rel_error': np.abs(np.mean(qmc_scr_estimates) - true_value) / true_value,
            'time': np.mean(qmc_scr_times)
        })
    
    # Convert to DataFrames
    return {method: pd.DataFrame(data) for method, data in results.items()}


def plot_convergence_results(results: Dict[str, pd.DataFrame], title: str = ""):
    """Plot convergence results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE plot
    ax = axes[0, 0]
    for method, df in results.items():
        ax.loglog(df['n_sims'], df['rmse'], 'o-', label=method, markersize=8)
    
    # Add theoretical convergence lines
    n = results['mc']['n_sims'].values
    mc_theory = results['mc']['rmse'].iloc[0] * np.sqrt(n[0] / n)
    qmc_theory = results['qmc_scrambled']['rmse'].iloc[0] * (n[0] / n)
    
    ax.loglog(n, mc_theory, 'k--', alpha=0.5, label='MC O(n^-0.5)')
    ax.loglog(n, qmc_theory, 'k:', alpha=0.5, label='QMC O(n^-1)')
    
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Relative Error plot
    ax = axes[0, 1]
    for method, df in results.items():
        ax.loglog(df['n_sims'], df['rel_error'], 'o-', label=method, markersize=8)
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Standard Deviation plot
    ax = axes[1, 0]
    for method, df in results.items():
        ax.loglog(df['n_sims'], df['std'], 'o-', label=method, markersize=8)
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Estimator Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Efficiency plot (RMSE * Time)
    ax = axes[1, 1]
    for method, df in results.items():
        efficiency = df['rmse'] * df['time']
        ax.loglog(df['n_sims'], efficiency, 'o-', label=method, markersize=8)
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('RMSE × Time')
    ax.set_title('Computational Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'QMC vs MC Convergence Comparison{" - " + title if title else ""}')
    plt.tight_layout()
    return fig


def main():
    """Run convergence benchmarks."""
    print("QMC Convergence Benchmark")
    print("=" * 60)
    
    # Test parameters
    sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
    portfolio_sizes = ['small', 'medium', 'large']
    
    all_results = {}
    
    for size in portfolio_sizes:
        print(f"\n\nTesting {size} portfolio...")
        print("-" * 40)
        
        # Create portfolio
        portfolio = create_test_portfolio(size)
        n_policies = sum(inf.n_policies for inf in portfolio.inforces)
        print(f"Portfolio has {len(portfolio.inforces)} inforce buckets with {n_policies} total policies")
        
        # Run convergence test
        results = run_convergence_test(portfolio, sample_sizes, n_replications=10)
        all_results[size] = results
        
        # Print summary
        print("\nConvergence Summary:")
        print("-" * 40)
        for method, df in results.items():
            print(f"\n{method}:")
            print(df[['n_sims', 'rmse', 'rel_error', 'time']].to_string(index=False))
        
        # Calculate improvement factors
        mc_rmse = results['mc']['rmse'].values
        qmc_scr_rmse = results['qmc_scrambled']['rmse'].values
        improvement = mc_rmse / qmc_scr_rmse
        
        print(f"\nQMC Improvement Factor (RMSE ratio):")
        for i, n in enumerate(sample_sizes):
            print(f"  n={n:,}: {improvement[i]:.2f}x")
        
        # Plot results
        fig = plot_convergence_results(results, f"{size.capitalize()} Portfolio")
        plt.savefig(f'qmc_convergence_{size}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save detailed results
    for size, results in all_results.items():
        for method, df in results.items():
            df.to_csv(f'qmc_convergence_{size}_{method}.csv', index=False)
    
    print("\n\nBenchmark complete! Results saved to CSV and PNG files.")
    
    # Clean up
    reset_qmc_simulator()


if __name__ == "__main__":
    main()