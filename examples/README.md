# QuActuary Examples

This directory contains Jupyter notebooks demonstrating various features and capabilities of the quActuary framework.

## Notebooks

### Core Features

1. **`demo_front_page_examples.ipynb`** - Basic examples showcasing the main features of quActuary, perfect for getting started.

2. **`portfolio_example.ipynb`** - Comprehensive demonstration of portfolio construction and risk analysis.

3. **`sobol_sequence_example.ipynb`** - Quasi-Monte Carlo methods using Sobol sequences for variance reduction.

### Distribution Examples

4. **`extended_distributions_examples.ipynb`** - Extensive examples of all available probability distributions and their applications.

### Quantum Computing

5. **`pilot_quantum_excess_evaluation_algorithm.ipynb`** - Implementation of quantum algorithms for excess loss evaluation in reinsurance, based on recent research papers.

6. **`demo_qae_simulation.ipynb`** ⭐ NEW - Comprehensive demonstration of Quantum Amplitude Estimation (QAE) for actuarial risk measures:
   - Mean loss estimation with quadratic speedup
   - Variance calculation using quantum algorithms
   - Value at Risk (VaR) computation
   - Tail Value at Risk (TVaR) analysis
   - Detailed benchmarks comparing quantum vs classical Monte Carlo methods
   - Error rate performance analysis showing quantum advantage
   - Practical considerations and trade-offs

### Performance and Optimization

7. **`performance_benchmarks_vs_scipy_numpy.ipynb`** - Benchmarks comparing quActuary performance against standard scientific Python libraries.

8. **`qmc_diagnostics_demo.ipynb`** - Diagnostics and visualization tools for Quasi-Monte Carlo convergence analysis.

9. **`monitor_performance_baseline.ipynb`** - Tools for monitoring and tracking performance baselines across different hardware.

10. **`optimization_selection_demo.py`** - Script demonstrating automatic optimization strategy selection based on problem characteristics.

## Running the Examples

### Prerequisites

```bash
# Install quActuary with all dependencies
pip install -e .

# For quantum examples, ensure quantum dependencies are installed
pip install -e .[quantum]

# For visualization support
pip install -e .[quantum,viz]
```

### Jupyter Setup

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

### Testing Examples

To verify that the examples work correctly:

```bash
# Test the QAE demo functionality
python test_qae_demo.py
```

## Key Concepts Demonstrated

- **Portfolio Construction**: Building portfolios with multiple lines of business
- **Risk Measures**: Computing VaR, TVaR, and other risk metrics
- **Distributions**: Using various frequency and severity distributions
- **Quantum Algorithms**: Leveraging quantum computing for actuarial calculations
- **Variance Reduction**: Implementing QMC techniques for improved convergence
- **Performance Optimization**: Selecting optimal computation strategies

## Contributing

If you have ideas for new examples or improvements to existing ones, please feel free to contribute! See the main CONTRIBUTING.md for guidelines.