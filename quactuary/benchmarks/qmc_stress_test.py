"""
QMC Stress Test

Tests QMC performance under challenging conditions and edge cases.
Uses the correct API: PolicyTerms, Inforce, Portfolio (not Book/Policy).
"""

import numpy as np
import pandas as pd
import time
import gc
import psutil
import os
from datetime import date
from typing import Dict, List, Tuple, Optional
import warnings

from quactuary.backend import set_backend
from quactuary.book import LOB, PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import (
    Poisson, NegativeBinomial, Binomial, Geometric
)
from quactuary.distributions.zero_inflated import (
    ZeroInflatedCompound
)
from quactuary.distributions.severity import (
    Exponential, Lognormal, Pareto, Gamma, Weibull,
    ContinuousUniformSeverity
)
from quactuary.distributions.compound import CompoundDistribution
from quactuary.pricing import PricingModel
from quactuary.sobol import set_qmc_simulator, reset_qmc_simulator, SobolEngine


class QMCStressTester:
    """Comprehensive stress testing for QMC implementation."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def create_high_dimension_portfolio(self, n_buckets: int) -> Portfolio:
        """Create portfolio with many inforce buckets (high dimension)."""
        print(f"Creating portfolio with {n_buckets} inforce buckets...")
        
        inforces = []
        
        # Vary all parameters to stress test dimension allocation
        for i in range(n_buckets):
            # Rotate through LOBs
            lobs = [LOB.GLPL, LOB.WC, LOB.CAuto, LOB.Prop]
            lob = lobs[i % len(lobs)]
            
            # Create terms
            terms = PolicyTerms(
                effective_date=date(2026, 1, 1),
                expiration_date=date(2027, 1, 1),
                lob=lob,
                exposure_base="SALES" if lob == LOB.GLPL else "PAYROLL",
                exposure_amount=1_000_000 * (i + 1),
                retention_type="deductible",
                per_occ_retention=10_000 * (i + 1),
                coverage="occ"
            )
            
            # Vary frequency distributions
            freq_idx = i % 6
            if freq_idx == 0:
                freq = Poisson(mu=2 + i * 0.1)
            elif freq_idx == 1:
                freq = NegativeBinomial(n=10 + i, p=0.3)
            elif freq_idx == 2:
                freq = Binomial(n=20 + i, p=0.1)
            elif freq_idx == 3:
                freq = Geometric(p=0.1 / (1 + i * 0.1))
            elif freq_idx == 4:
                # Use regular Poisson as ZI frequency doesn't exist
                freq = Poisson(mu=3 + i * 0.2)
            else:
                # Use regular NegativeBinomial as ZI frequency doesn't exist
                freq = NegativeBinomial(r=5 + i, p=0.4)
            
            # Vary severity distributions
            sev_idx = i % 5
            if sev_idx == 0:
                sev = Exponential(scale=10_000 + i * 1_000)
            elif sev_idx == 1:
                sev = Lognormal(mu=9 + i * 0.1, sigma=1 + i * 0.05)
            elif sev_idx == 2:
                sev = Gamma(alpha=2 + i * 0.1, scale=5_000 + i * 500)
            elif sev_idx == 3:
                sev = Pareto(alpha=2 + i * 0.05, scale=8_000 + i * 800)
            else:
                sev = Weibull(c=1.5 + i * 0.05, scale=12_000 + i * 1_000)
            
            # Create inforce
            n_policies = max(10, 100 - i * 2)  # Decreasing number of policies
            inforce = Inforce(
                n_policies=n_policies,
                terms=terms,
                frequency=freq,
                severity=sev,
                name=f"Bucket_{i+1}"
            )
            inforces.append(inforce)
        
        # Create portfolio
        portfolio = inforces[0]
        for inf in inforces[1:]:
            portfolio = portfolio + inf
        
        return portfolio
    
    def create_heavy_tail_portfolio(self) -> Portfolio:
        """Create portfolio with heavy-tailed distributions."""
        print("Creating heavy-tail portfolio...")
        
        # Very heavy-tailed severity distributions
        inforces = [
            Inforce(
                n_policies=100,
                terms=PolicyTerms(
                    effective_date=date(2026, 1, 1),
                    expiration_date=date(2027, 1, 1),
                    lob=LOB.GLPL,
                    exposure_base="SALES",
                    exposure_amount=50_000_000,
                    retention_type="deductible",
                    per_occ_retention=500_000,
                    coverage="occ"
                ),
                frequency=NegativeBinomial(n=5, p=0.2),  # High variance
                severity=Pareto(alpha=1.5, scale=100_000),  # Very heavy tail
                name="Catastrophic GL"
            ),
            Inforce(
                n_policies=50,
                terms=PolicyTerms(
                    effective_date=date(2026, 1, 1),
                    expiration_date=date(2027, 1, 1),
                    lob=LOB.Prop,
                    exposure_base="TIV",
                    exposure_amount=100_000_000,
                    retention_type="deductible",
                    per_occ_retention=1_000_000,
                    coverage="occ"
                ),
                frequency=CompoundDistribution(
                    primary=Poisson(mu=10),
                    secondary=NegativeBinomial(n=3, p=0.3)
                ),
                severity=Lognormal(mu=12, sigma=2.5),  # High volatility
                name="Property Cat"
            ),
            Inforce(
                n_policies=200,
                terms=PolicyTerms(
                    effective_date=date(2026, 1, 1),
                    expiration_date=date(2027, 1, 1),
                    lob=LOB.WC,
                    exposure_base="PAYROLL",
                    exposure_amount=10_000_000,
                    retention_type="deductible",
                    per_occ_retention=250_000,
                    coverage="occ"
                ),
                frequency=ZeroInflatedNegativeBinomial(pi=0.4, n=8, p=0.25),
                severity=CompoundDistribution(
                    primary=Gamma(alpha=1.5, scale=50_000),
                    secondary=Pareto(alpha=1.8, scale=200_000),
                    mixing_prob=0.9
                ),
                name="WC Severe Claims"
            )
        ]
        
        portfolio = inforces[0]
        for inf in inforces[1:]:
            portfolio = portfolio + inf
        
        return portfolio
    
    def create_sparse_portfolio(self) -> Portfolio:
        """Create portfolio with many zero-inflated distributions."""
        print("Creating sparse portfolio...")
        
        inforces = []
        for i in range(10):
            inforce = Inforce(
                n_policies=50 + i * 10,
                terms=PolicyTerms(
                    effective_date=date(2026, 1, 1),
                    expiration_date=date(2027, 1, 1),
                    lob=LOB.GLPL,
                    exposure_base="SALES",
                    exposure_amount=1_000_000 * (i + 1),
                    retention_type="deductible",
                    per_occ_retention=50_000,
                    coverage="occ"
                ),
                frequency=Poisson(mu=(1 - (0.7 + i * 0.02)) * (1 + i * 0.2)),  # Adjusted for zero inflation
                severity=Lognormal(shape=0.5, scale=np.exp(np.log(20_000) - 0.5**2/2)),  # Similar to truncated normal
                name=f"Sparse_Bucket_{i+1}"
            )
            inforces.append(inforce)
        
        portfolio = inforces[0]
        for inf in inforces[1:]:
            portfolio = portfolio + inf
        
        return portfolio
    
    def run_dimension_stress_test(self) -> pd.DataFrame:
        """Test QMC with increasing dimensions."""
        print("\n" + "="*60)
        print("DIMENSION STRESS TEST")
        print("="*60)
        
        bucket_counts = [5, 10, 20, 50, 100]
        results = []
        
        for n_buckets in bucket_counts:
            print(f"\nTesting with {n_buckets} buckets...")
            
            # Create portfolio
            portfolio = self.create_high_dimension_portfolio(n_buckets)
            n_policies = sum(inf.n_policies for inf in portfolio.inforces)
            
            # Estimate dimensions needed
            total_dims = 0
            for inf in portfolio.inforces:
                expected_claims = inf.frequency.mean() * inf.n_policies
                max_claims = int(expected_claims * 3)  # Conservative estimate
                total_dims += 1 + max_claims  # 1 for freq + severity dims
            
            print(f"  Total policies: {n_policies}")
            print(f"  Estimated dimensions: {total_dims:,}")
            
            # Test with different methods
            methods = [
                ('MC', lambda: reset_qmc_simulator()),
                ('QMC_1024', lambda: set_qmc_simulator(method='sobol', scramble=True, skip=1024)),
                ('QMC_4096', lambda: set_qmc_simulator(method='sobol', scramble=True, skip=4096)),
                ('QMC_16384', lambda: set_qmc_simulator(method='sobol', scramble=True, skip=16384))
            ]
            
            for method_name, setup_func in methods:
                print(f"\n  Testing {method_name}...")
                
                # Memory before
                gc.collect()
                mem_before = self.get_memory_usage()
                
                try:
                    # Setup and run
                    setup_func()
                    set_backend('classical')
                    
                    start_time = time.time()
                    model = PricingModel(portfolio)
                    result = model.simulate(n_sims=5000, tail_alpha=0.01)
                    elapsed = time.time() - start_time
                    
                    # Memory after
                    mem_after = self.get_memory_usage()
                    mem_used = mem_after - mem_before
                    
                    results.append({
                        'n_buckets': n_buckets,
                        'n_policies': n_policies,
                        'dimensions': total_dims,
                        'method': method_name,
                        'time': elapsed,
                        'memory_mb': mem_used,
                        'mean': result.estimates['mean'],
                        'var_99': result.estimates['VaR'],
                        'tvar_99': result.estimates['TVaR'],
                        'success': True,
                        'error': None
                    })
                    
                    print(f"    Time: {elapsed:.2f}s, Memory: {mem_used:.1f}MB")
                    print(f"    VaR(99%): ${result.estimates['VaR']:,.0f}")
                    
                except Exception as e:
                    results.append({
                        'n_buckets': n_buckets,
                        'n_policies': n_policies,
                        'dimensions': total_dims,
                        'method': method_name,
                        'time': np.nan,
                        'memory_mb': np.nan,
                        'mean': np.nan,
                        'var_99': np.nan,
                        'tvar_99': np.nan,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    FAILED: {e}")
                
                finally:
                    # Clean up
                    del model
                    gc.collect()
        
        return pd.DataFrame(results)
    
    def run_distribution_stress_test(self) -> pd.DataFrame:
        """Test QMC with challenging distributions."""
        print("\n" + "="*60)
        print("DISTRIBUTION STRESS TEST")
        print("="*60)
        
        test_cases = [
            ('Heavy Tail', self.create_heavy_tail_portfolio),
            ('Sparse', self.create_sparse_portfolio),
        ]
        
        results = []
        
        for test_name, portfolio_func in test_cases:
            print(f"\nTesting {test_name} portfolio...")
            
            portfolio = portfolio_func()
            n_policies = sum(inf.n_policies for inf in portfolio.inforces)
            print(f"  Total policies: {n_policies}")
            
            # Test with different sample sizes
            sample_sizes = [1000, 5000, 10000, 20000]
            
            for n_sims in sample_sizes:
                print(f"\n  Testing with {n_sims:,} simulations...")
                
                # Test MC and QMC
                for method in ['MC', 'QMC']:
                    if method == 'MC':
                        reset_qmc_simulator()
                    else:
                        set_qmc_simulator(method='sobol', scramble=True, skip=1024)
                    
                    set_backend('classical')
                    
                    # Run multiple replications
                    estimates = []
                    times = []
                    
                    for rep in range(5):
                        start_time = time.time()
                        model = PricingModel(portfolio)
                        result = model.simulate(n_sims=n_sims, tail_alpha=0.01)
                        times.append(time.time() - start_time)
                        estimates.append(result.estimates['VaR'])
                        
                        # Change seed for QMC
                        if method == 'QMC':
                            set_qmc_simulator(method='sobol', scramble=True, 
                                            skip=1024, seed=rep+1)
                    
                    results.append({
                        'test_case': test_name,
                        'n_sims': n_sims,
                        'method': method,
                        'mean_time': np.mean(times),
                        'var_99_mean': np.mean(estimates),
                        'var_99_std': np.std(estimates),
                        'var_99_cv': np.std(estimates) / np.mean(estimates) if np.mean(estimates) > 0 else np.nan
                    })
                    
                    print(f"    {method}: Time={np.mean(times):.2f}s, "
                          f"VaR=${np.mean(estimates):,.0f} "
                          f"(CV={np.std(estimates)/np.mean(estimates):.3f})")
        
        return pd.DataFrame(results)
    
    def run_memory_stress_test(self) -> pd.DataFrame:
        """Test memory usage patterns."""
        print("\n" + "="*60)
        print("MEMORY STRESS TEST")
        print("="*60)
        
        results = []
        
        # Create a large portfolio
        portfolio = self.create_high_dimension_portfolio(50)
        
        # Test with increasing sample sizes
        sample_sizes = [1000, 5000, 10000, 25000, 50000]
        
        for n_sims in sample_sizes:
            print(f"\nTesting with {n_sims:,} simulations...")
            
            for method in ['MC', 'QMC']:
                print(f"  {method}...")
                
                # Force garbage collection
                gc.collect()
                initial_mem = self.get_memory_usage()
                
                try:
                    # Setup
                    if method == 'MC':
                        reset_qmc_simulator()
                    else:
                        set_qmc_simulator(method='sobol', scramble=True, skip=1024)
                    
                    set_backend('classical')
                    
                    # Create model
                    model = PricingModel(portfolio)
                    model_mem = self.get_memory_usage()
                    
                    # Run simulation
                    start_time = time.time()
                    result = model.simulate(n_sims=n_sims)
                    elapsed = time.time() - start_time
                    
                    # Peak memory
                    peak_mem = self.get_memory_usage()
                    
                    results.append({
                        'n_sims': n_sims,
                        'method': method,
                        'initial_mb': initial_mem,
                        'model_mb': model_mem - initial_mem,
                        'peak_mb': peak_mem - initial_mem,
                        'time': elapsed,
                        'success': True,
                        'error': None
                    })
                    
                    print(f"    Memory: {peak_mem - initial_mem:.1f}MB, Time: {elapsed:.2f}s")
                    
                except Exception as e:
                    results.append({
                        'n_sims': n_sims,
                        'method': method,
                        'initial_mb': initial_mem,
                        'model_mb': np.nan,
                        'peak_mb': np.nan,
                        'time': np.nan,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    FAILED: {e}")
                
                finally:
                    # Clean up
                    if 'model' in locals():
                        del model
                    if 'result' in locals():
                        del result
                    gc.collect()
        
        return pd.DataFrame(results)
    
    def run_edge_case_tests(self) -> pd.DataFrame:
        """Test edge cases and potential failure modes."""
        print("\n" + "="*60)
        print("EDGE CASE TESTS")
        print("="*60)
        
        results = []
        
        # Test 1: Single policy portfolio
        print("\nTest 1: Single inforce bucket...")
        single_inforce = Inforce(
            n_policies=1,
            terms=PolicyTerms(
                effective_date=date(2026, 1, 1),
                expiration_date=date(2027, 1, 1),
                lob=LOB.GLPL,
                exposure_base="SALES",
                exposure_amount=1_000_000,
                retention_type="deductible",
                per_occ_retention=10_000,
                coverage="occ"
            ),
            frequency=Poisson(mu=2),
            severity=Exponential(scale=50_000),
            name="Single"
        )
        portfolio = Portfolio(single_inforce)
        
        for method in ['MC', 'QMC']:
            if method == 'MC':
                reset_qmc_simulator()
            else:
                set_qmc_simulator(method='sobol', scramble=True)
            
            try:
                set_backend('classical')
                model = PricingModel(portfolio)
                result = model.simulate(n_sims=1000)
                results.append({
                    'test': 'Single Inforce',
                    'method': method,
                    'success': True,
                    'mean': result.estimates['mean'],
                    'error': None
                })
            except Exception as e:
                results.append({
                    'test': 'Single Inforce',
                    'method': method,
                    'success': False,
                    'mean': np.nan,
                    'error': str(e)
                })
        
        # Test 2: Very high frequency
        print("\nTest 2: Very high frequency...")
        high_freq_inforce = Inforce(
            n_policies=10,
            terms=PolicyTerms(
                effective_date=date(2026, 1, 1),
                expiration_date=date(2027, 1, 1),
                lob=LOB.WC,
                exposure_base="PAYROLL",
                exposure_amount=10_000_000,
                retention_type="deductible",
                per_occ_retention=1000,
                coverage="occ"
            ),
            frequency=Poisson(mu=1000),  # Very high frequency
            severity=Exponential(scale=5000),
            name="High Frequency"
        )
        portfolio = Portfolio(high_freq_inforce)
        
        for method in ['MC', 'QMC']:
            if method == 'MC':
                reset_qmc_simulator()
            else:
                set_qmc_simulator(method='sobol', scramble=True)
            
            try:
                set_backend('classical')
                model = PricingModel(portfolio)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.simulate(n_sims=1000)
                results.append({
                    'test': 'High Frequency',
                    'method': method,
                    'success': True,
                    'mean': result.estimates['mean'],
                    'error': None
                })
            except Exception as e:
                results.append({
                    'test': 'High Frequency',
                    'method': method,
                    'success': False,
                    'mean': np.nan,
                    'error': str(e)
                })
        
        # Test 3: Zero claims expected
        print("\nTest 3: Zero claims expected...")
        zero_claims_inforce = Inforce(
            n_policies=100,
            terms=PolicyTerms(
                effective_date=date(2026, 1, 1),
                expiration_date=date(2027, 1, 1),
                lob=LOB.GLPL,
                exposure_base="SALES",
                exposure_amount=1_000_000,
                retention_type="deductible",
                per_occ_retention=10_000_000,  # Very high deductible
                coverage="occ"
            ),
            frequency=Poisson(mu=0.001),  # Very low rate (0.99 zero inflation * 0.1 rate)
            severity=Exponential(scale=1000),
            name="Zero Claims"
        )
        portfolio = Portfolio(zero_claims_inforce)
        
        for method in ['MC', 'QMC']:
            if method == 'MC':
                reset_qmc_simulator()
            else:
                set_qmc_simulator(method='sobol', scramble=True)
            
            try:
                set_backend('classical')
                model = PricingModel(portfolio)
                result = model.simulate(n_sims=1000)
                results.append({
                    'test': 'Zero Claims',
                    'method': method,
                    'success': True,
                    'mean': result.estimates['mean'],
                    'error': None
                })
            except Exception as e:
                results.append({
                    'test': 'Zero Claims',
                    'method': method,
                    'success': False,
                    'mean': np.nan,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def run_all_tests(self) -> Dict[str, pd.DataFrame]:
        """Run all stress tests."""
        all_results = {}
        
        # Dimension stress test
        print("\nRunning dimension stress test...")
        all_results['dimension'] = self.run_dimension_stress_test()
        
        # Distribution stress test
        print("\nRunning distribution stress test...")
        all_results['distribution'] = self.run_distribution_stress_test()
        
        # Memory stress test
        print("\nRunning memory stress test...")
        all_results['memory'] = self.run_memory_stress_test()
        
        # Edge case tests
        print("\nRunning edge case tests...")
        all_results['edge_cases'] = self.run_edge_case_tests()
        
        return all_results


def main():
    """Run QMC stress tests."""
    print("QMC STRESS TEST SUITE")
    print("=" * 80)
    print("Testing QMC implementation under various stress conditions")
    print("=" * 80)
    
    tester = QMCStressTester()
    results = tester.run_all_tests()
    
    # Save results
    print("\n\nSaving results...")
    for test_name, df in results.items():
        filename = f'qmc_stress_test_{test_name}.csv'
        df.to_csv(filename, index=False)
        print(f"  Saved {filename}")
    
    # Print summary
    print("\n\nSTRESS TEST SUMMARY")
    print("=" * 80)
    
    # Dimension test summary
    dim_df = results['dimension']
    print("\nDimension Scaling:")
    print("-" * 40)
    success_rate = dim_df.groupby('method')['success'].mean()
    print("Success rates:")
    print(success_rate.to_string())
    
    # Distribution test summary
    dist_df = results['distribution']
    print("\n\nDistribution Challenges:")
    print("-" * 40)
    cv_comparison = dist_df.pivot_table(
        values='var_99_cv',
        index='test_case',
        columns='method',
        aggfunc='mean'
    )
    print("Average CV by test case:")
    print(cv_comparison.to_string())
    
    # Memory test summary
    mem_df = results['memory']
    print("\n\nMemory Usage:")
    print("-" * 40)
    mem_comparison = mem_df[mem_df['success']].pivot_table(
        values='peak_mb',
        index='n_sims',
        columns='method',
        aggfunc='mean'
    )
    print("Peak memory (MB) by sample size:")
    print(mem_comparison.to_string())
    
    # Edge case summary
    edge_df = results['edge_cases']
    print("\n\nEdge Cases:")
    print("-" * 40)
    edge_summary = edge_df.groupby('test')['success'].mean()
    print("Success rates by test:")
    print(edge_summary.to_string())
    
    print("\n\nStress test complete!")
    
    # Clean up
    reset_qmc_simulator()


if __name__ == "__main__":
    main()