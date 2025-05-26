"""
QMC Convergence Diagnostics for quActuary.

This module provides convergence diagnostics and quality metrics for
Quasi-Monte Carlo simulations in actuarial pricing applications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass


@dataclass
class ConvergenceDiagnostics:
    """Container for QMC convergence diagnostics."""
    
    # Basic statistics
    mean_estimate: float
    std_error: float
    coefficient_of_variation: float
    
    # Convergence metrics
    effective_sample_size: float
    variance_reduction_factor: float
    convergence_rate: float
    
    # Uniformity metrics
    discrepancy: Optional[float] = None
    uniformity_test_pvalue: Optional[float] = None
    
    # Method info
    method: str = "qmc"
    n_simulations: int = 0
    n_dimensions: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert diagnostics to dictionary for integration with pricing results."""
        return {
            'qmc_mean_estimate': self.mean_estimate,
            'qmc_std_error': self.std_error,
            'qmc_cv': self.coefficient_of_variation,
            'qmc_ess': self.effective_sample_size,
            'qmc_vrf': self.variance_reduction_factor,
            'qmc_convergence_rate': self.convergence_rate,
            'qmc_method': self.method,
            'qmc_n_simulations': self.n_simulations,
            'qmc_n_dimensions': self.n_dimensions
        }


def calculate_effective_sample_size(
    estimates: np.ndarray,
    mc_variance: Optional[float] = None
) -> float:
    """
    Calculate effective sample size for QMC estimates.
    
    The effective sample size represents how many i.i.d. Monte Carlo samples
    would be needed to achieve the same variance as the QMC method.
    
    Args:
        estimates: Array of estimates from multiple QMC runs with different scrambles
        mc_variance: Known Monte Carlo variance (if available)
    
    Returns:
        Effective sample size
    """
    qmc_variance = np.var(estimates)
    n_actual = len(estimates)
    
    if mc_variance is not None:
        # ESS = n * (MC variance / QMC variance)
        ess = n_actual * (mc_variance / qmc_variance) if qmc_variance > 0 else n_actual
    else:
        # Estimate based on theoretical MC rate
        # Assume QMC achieves O(n^{-1}) vs MC's O(n^{-0.5})
        ess = n_actual ** 2
    
    return float(ess)


def calculate_variance_reduction_factor(
    qmc_estimates: np.ndarray,
    mc_estimates: Optional[np.ndarray] = None,
    theoretical_mc_var: Optional[float] = None
) -> float:
    """
    Calculate variance reduction factor of QMC over standard MC.
    
    Args:
        qmc_estimates: Array of QMC estimates
        mc_estimates: Array of MC estimates (if available)
        theoretical_mc_var: Theoretical MC variance (if known)
    
    Returns:
        Variance reduction factor (> 1 means QMC is better)
    """
    qmc_var = np.var(qmc_estimates)
    
    if mc_estimates is not None:
        mc_var = np.var(mc_estimates)
    elif theoretical_mc_var is not None:
        mc_var = theoretical_mc_var
    else:
        # Estimate based on sample size
        # MC variance scales as 1/n
        n = len(qmc_estimates)
        sample_var = np.var(qmc_estimates) * n
        mc_var = sample_var / n
    
    vrf = mc_var / qmc_var if qmc_var > 0 else 1.0
    return float(vrf)


def estimate_convergence_rate(
    sample_sizes: List[int],
    rmse_values: List[float]
) -> float:
    """
    Estimate empirical convergence rate from RMSE values.
    
    Fits log(RMSE) ~ alpha * log(n) to estimate convergence rate.
    
    Args:
        sample_sizes: List of sample sizes used
        rmse_values: Corresponding RMSE values
    
    Returns:
        Estimated convergence rate (e.g., -0.5 for MC, -1.0 for QMC)
    """
    if len(sample_sizes) < 2:
        return 0.0
    
    log_n = np.log(sample_sizes)
    log_rmse = np.log(rmse_values)
    
    # Linear regression on log scale
    slope, _ = np.polyfit(log_n, log_rmse, 1)
    
    return float(slope)


def calculate_star_discrepancy(points: np.ndarray, max_dim: int = 10) -> float:
    """
    Calculate star discrepancy for low-dimensional projections.
    
    Note: Full star discrepancy is computationally expensive, so we
    calculate it only for 2D projections and average.
    
    Args:
        points: QMC points array of shape (n_points, n_dims)
        max_dim: Maximum number of dimensions to consider
    
    Returns:
        Average star discrepancy across 2D projections
    """
    n_points, n_dims = points.shape
    n_dims = min(n_dims, max_dim)
    
    discrepancies = []
    
    # Check 2D projections
    for i in range(min(n_dims - 1, 5)):
        for j in range(i + 1, min(n_dims, 6)):
            proj = points[:, [i, j]]
            
            # Simple discrepancy approximation
            # Count points in grid cells and compare to expected
            n_bins = int(np.sqrt(n_points))
            hist, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=n_bins)
            expected = n_points / (n_bins * n_bins)
            disc = np.mean(np.abs(hist - expected)) / expected
            discrepancies.append(disc)
    
    return float(np.mean(discrepancies)) if discrepancies else 0.0


def uniformity_test(points: np.ndarray, test: str = 'ks') -> float:
    """
    Test uniformity of QMC points using statistical tests.
    
    Args:
        points: QMC points to test
        test: Test type ('ks' for Kolmogorov-Smirnov, 'chi2' for chi-squared)
    
    Returns:
        p-value of uniformity test
    """
    # Test each dimension independently
    p_values = []
    
    for dim in range(min(points.shape[1], 10)):
        if test == 'ks':
            # Kolmogorov-Smirnov test against uniform distribution
            _, p_value = stats.kstest(points[:, dim], 'uniform')
        elif test == 'chi2':
            # Chi-squared test
            n_bins = int(np.sqrt(len(points)))
            hist, _ = np.histogram(points[:, dim], bins=n_bins, range=(0, 1))
            expected = len(points) / n_bins
            chi2_stat = np.sum((hist - expected)**2 / expected)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_bins-1)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        p_values.append(p_value)
    
    # Return minimum p-value (most significant deviation)
    return float(np.min(p_values))


def analyze_qmc_convergence(
    estimates: List[float],
    sample_sizes: List[int],
    method: str = "sobol",
    n_dimensions: int = 0,
    mc_estimates: Optional[List[float]] = None,
    points_sample: Optional[np.ndarray] = None
) -> ConvergenceDiagnostics:
    """
    Comprehensive QMC convergence analysis.
    
    Args:
        estimates: List of estimates from QMC runs
        sample_sizes: Corresponding sample sizes
        method: QMC method used
        n_dimensions: Number of dimensions
        mc_estimates: Monte Carlo estimates for comparison
        points_sample: Sample of QMC points for uniformity testing
    
    Returns:
        ConvergenceDiagnostics object with all metrics
    """
    estimates = np.array(estimates)
    
    # Basic statistics
    mean_est = np.mean(estimates)
    std_err = np.std(estimates)
    cv = std_err / abs(mean_est) if mean_est != 0 else float('inf')
    
    # Effective sample size
    ess = calculate_effective_sample_size(estimates)
    
    # Variance reduction factor
    vrf = calculate_variance_reduction_factor(
        estimates,
        np.array(mc_estimates) if mc_estimates else None
    )
    
    # Convergence rate
    if len(sample_sizes) > 1 and len(estimates) == len(sample_sizes):
        # Calculate RMSE for each sample size
        rmse_values = []
        for i, n in enumerate(sample_sizes):
            # Use rolling estimate as "true" value
            true_val = np.mean(estimates[max(0, i-2):i+3])
            rmse = abs(estimates[i] - true_val)
            rmse_values.append(rmse)
        
        conv_rate = estimate_convergence_rate(sample_sizes, rmse_values)
    else:
        conv_rate = -1.0  # Assume optimal QMC rate
    
    # Uniformity metrics (optional)
    discrepancy = None
    uniformity_pval = None
    
    if points_sample is not None:
        discrepancy = calculate_star_discrepancy(points_sample)
        uniformity_pval = uniformity_test(points_sample)
    
    return ConvergenceDiagnostics(
        mean_estimate=float(mean_est),
        std_error=float(std_err),
        coefficient_of_variation=float(cv),
        effective_sample_size=float(ess),
        variance_reduction_factor=float(vrf),
        convergence_rate=float(conv_rate),
        discrepancy=discrepancy,
        uniformity_test_pvalue=uniformity_pval,
        method=method,
        n_simulations=int(np.mean(sample_sizes)) if sample_sizes else 0,
        n_dimensions=n_dimensions
    )


def visualize_convergence(
    diagnostics: ConvergenceDiagnostics,
    estimates_by_size: Optional[Dict[int, List[float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of convergence diagnostics.
    
    Args:
        diagnostics: Convergence diagnostics to visualize
        estimates_by_size: Dictionary mapping sample sizes to estimate lists
        save_path: Path to save figure (if provided)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Convergence rate plot
    if estimates_by_size:
        ax = axes[0, 0]
        sizes = sorted(estimates_by_size.keys())
        means = [np.mean(estimates_by_size[s]) for s in sizes]
        stds = [np.std(estimates_by_size[s]) for s in sizes]
        
        ax.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5, label='QMC')
        ax.axhline(diagnostics.mean_estimate, color='r', linestyle='--', label='Final estimate')
        ax.set_xscale('log')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Estimate')
        ax.set_title('Convergence of Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Variance reduction plot
    ax = axes[0, 1]
    categories = ['Monte Carlo', 'Quasi-MC']
    variances = [1.0, 1.0 / diagnostics.variance_reduction_factor]
    colors = ['blue', 'green']
    
    bars = ax.bar(categories, variances, color=colors, alpha=0.7)
    ax.set_ylabel('Relative Variance')
    ax.set_title(f'Variance Reduction Factor: {diagnostics.variance_reduction_factor:.1f}x')
    
    # Add value labels on bars
    for bar, val in zip(bars, variances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Effective sample size
    ax = axes[1, 0]
    actual_n = diagnostics.n_simulations
    ess = diagnostics.effective_sample_size
    
    ax.bar(['Actual', 'Effective'], [actual_n, ess], color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Sample Size')
    ax.set_title('Effective Sample Size')
    ax.text(0.5, 0.95, f'Efficiency: {ess/actual_n:.1%}',
            transform=ax.transAxes, ha='center', va='top')
    
    # 4. Summary metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = f"""
    QMC Convergence Diagnostics
    
    Method: {diagnostics.method}
    Dimensions: {diagnostics.n_dimensions}
    Simulations: {diagnostics.n_simulations:,}
    
    Mean Estimate: {diagnostics.mean_estimate:.4f}
    Std Error: {diagnostics.std_error:.4f}
    CV: {diagnostics.coefficient_of_variation:.2%}
    
    Convergence Rate: {diagnostics.convergence_rate:.2f}
    ESS: {diagnostics.effective_sample_size:,.0f}
    VRF: {diagnostics.variance_reduction_factor:.1f}x
    """
    
    if diagnostics.discrepancy is not None:
        metrics_text += f"\nDiscrepancy: {diagnostics.discrepancy:.4f}"
    if diagnostics.uniformity_test_pvalue is not None:
        metrics_text += f"\nUniformity p-value: {diagnostics.uniformity_test_pvalue:.4f}"
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def enhance_pricing_result_with_diagnostics(
    pricing_result: Dict,
    qmc_estimates: List[float],
    sample_sizes: List[int],
    method: str = "sobol",
    n_dimensions: int = 0
) -> Dict:
    """
    Enhance a pricing result dictionary with QMC convergence diagnostics.
    
    This function integrates QMC diagnostics into the standard pricing result format.
    
    Args:
        pricing_result: Original pricing result dictionary
        qmc_estimates: List of QMC estimates from multiple runs
        sample_sizes: Sample sizes used
        method: QMC method
        n_dimensions: Total dimensions used
    
    Returns:
        Enhanced pricing result with QMC diagnostics
    """
    # Calculate diagnostics
    diagnostics = analyze_qmc_convergence(
        estimates=qmc_estimates,
        sample_sizes=sample_sizes,
        method=method,
        n_dimensions=n_dimensions
    )
    
    # Add diagnostics to result
    enhanced_result = pricing_result.copy()
    enhanced_result.update(diagnostics.to_dict())
    
    # Add confidence intervals based on QMC std error
    for key in ['mean', 'var_95%', 'var_99%', 'tvar_95%', 'tvar_99%']:
        if key in enhanced_result:
            value = enhanced_result[key]
            # 95% confidence interval
            ci_lower = value - 1.96 * diagnostics.std_error
            ci_upper = value + 1.96 * diagnostics.std_error
            enhanced_result[f'{key}_ci'] = (ci_lower, ci_upper)
    
    return enhanced_result