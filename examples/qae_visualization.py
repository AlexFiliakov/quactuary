"""Generate key visualizations from QAE analysis for documentation."""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def plot_convergence_comparison():
    """Create convergence rate comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convergence rates
    qubits = np.arange(4, 11)
    quantum_error = 1.0 / (2**qubits)
    classical_error = 1.0 / np.sqrt(2**qubits)
    
    # Error comparison
    ax1.semilogy(qubits, quantum_error, 'b-o', label='Quantum AE (O(1/N))', 
                 linewidth=2, markersize=8)
    ax1.semilogy(qubits, classical_error, 'r-s', label='Classical MC (O(1/√N))', 
                 linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.set_title('Convergence Rate: Quantum vs Classical', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Speedup factor
    speedup = classical_error / quantum_error
    ax2.plot(qubits, speedup, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Quantum Speedup Factor', fontsize=12)
    ax2.set_title('Quantum Advantage Growth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (q, s) in enumerate(zip(qubits, speedup)):
        if q % 2 == 0:  # Annotate even qubits
            ax2.annotate(f'{s:.0f}x', (q, s), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('qae_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved convergence comparison plot")

def plot_error_scaling():
    """Create error scaling visualization."""
    plt.figure(figsize=(12, 8))
    
    # Resource ranges
    resources = np.logspace(1, 6, 100)
    classical_error = 1.0 / np.sqrt(resources)
    quantum_error = 1.0 / resources
    
    # Plot
    plt.loglog(resources, classical_error, 'r-', label='Classical MC (O(1/√N))', 
               linewidth=2.5, alpha=0.8)
    plt.loglog(resources, quantum_error, 'b-', label='Quantum AE (O(1/N))', 
               linewidth=2.5, alpha=0.8)
    
    # Add vertical lines for specific qubit counts
    for n_qubits in [6, 8, 10, 12]:
        n_resources = 2**n_qubits
        plt.axvline(x=n_resources, color='gray', linestyle=':', alpha=0.5)
        plt.text(n_resources, 1e-6, f'{n_qubits} qubits', 
                rotation=90, verticalalignment='bottom', alpha=0.7)
    
    # Highlight advantage region
    plt.fill_between(resources, classical_error, quantum_error, 
                    where=(quantum_error < classical_error), 
                    color='green', alpha=0.2, label='Quantum Advantage Region')
    
    plt.xlabel('Computational Resources (Samples/Oracle Calls)', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('Error Scaling: Quantum vs Classical Methods', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim(10, 1e6)
    plt.ylim(1e-7, 1)
    
    plt.tight_layout()
    plt.savefig('qae_error_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved error scaling plot")

def plot_risk_measures_comparison():
    """Create risk measures comparison visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example data (typical results)
    measures = ['Mean', 'Variance', 'VaR (95%)', 'TVaR (95%)']
    quantum_values = [50000, 2500000, 82000, 95000]
    classical_values = [51000, 2600000, 80000, 93000]
    
    # Normalize to percentage difference
    differences = [abs(q - c) / c * 100 for q, c in zip(quantum_values, classical_values)]
    
    x = np.arange(len(measures))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, differences, width, label='Quantum vs Classical Difference %', 
                    color='purple', alpha=0.7)
    
    # Add value labels
    for bar, diff in zip(bars1, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Risk Measure', fontsize=12)
    ax.set_ylabel('Relative Difference (%)', fontsize=12)
    ax.set_title('Risk Measures: Quantum vs Classical Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(measures)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qae_risk_measures_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved risk measures comparison plot")

def plot_summary_infographic():
    """Create summary infographic."""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Quantum Amplitude Estimation for Actuarial Science', 
                 fontsize=18, fontweight='bold')
    
    # Key advantages (top row)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    advantages = [
        "Quadratic Speedup\nO(1/N) vs O(1/√N)",
        "Exponential Resource\nReduction",
        "High Precision\nRisk Analysis"
    ]
    for i, adv in enumerate(advantages):
        ax1.text(0.2 + i*0.3, 0.5, adv, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7),
                fontsize=12, fontweight='bold')
    
    # Speedup chart (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    qubits = np.arange(4, 12)
    speedup = np.sqrt(2**qubits)
    ax2.plot(qubits, speedup, 'g-o', linewidth=2)
    ax2.set_xlabel('Qubits')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Quantum Advantage', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Applications (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    applications = ["• Portfolio VaR/TVaR\n• Reserve Estimation\n• Capital Requirements\n• Pricing Models"]
    ax3.text(0.5, 0.5, "Applications:\n" + applications[0], ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7),
            fontsize=11)
    
    # Requirements (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    requirements = ["• Quantum Computer\n• or Simulator\n• 10-20 qubits\n• Error mitigation"]
    ax4.text(0.5, 0.5, "Requirements:\n" + requirements[0], ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7),
            fontsize=11)
    
    # Performance comparison (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    resources = np.array([100, 1000, 10000, 100000])
    classical_time = resources / 1e6  # Assuming 1M samples/sec
    quantum_time = np.sqrt(resources) / 1e3  # Assuming 1k oracle calls/sec
    
    ax5.loglog(resources, classical_time, 'r-o', label='Classical MC', linewidth=2)
    ax5.loglog(resources, quantum_time, 'b-s', label='Quantum AE', linewidth=2)
    ax5.set_xlabel('Target Precision (1/error)', fontsize=12)
    ax5.set_ylabel('Time (seconds)', fontsize=12)
    ax5.set_title('Time to Solution Comparison', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qae_summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved summary infographic")

def main():
    """Generate all visualizations."""
    print("Generating QAE visualization suite...")
    print("="*50)
    
    plot_convergence_comparison()
    plot_error_scaling()
    plot_risk_measures_comparison()
    plot_summary_infographic()
    
    print("="*50)
    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - qae_convergence_comparison.png")
    print("  - qae_error_scaling.png")
    print("  - qae_risk_measures_comparison.png")
    print("  - qae_summary_infographic.png")

if __name__ == "__main__":
    main()