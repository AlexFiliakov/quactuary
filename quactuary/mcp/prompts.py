"""Prompt definitions for MCP server."""

import logging

logger = logging.getLogger(__name__)


def register_prompts(mcp):
    """Register prompts with the MCP server."""
    
    @mcp.prompt(
        name="actuarial_concepts",
        description="Explains key actuarial concepts and terminology"
    )
    async def actuarial_concepts():
        return """
# Actuarial Concepts Guide

## Key Terms

- **Loss Ratio**: Incurred losses divided by earned premium
- **Frequency**: Number of claims per exposure unit
- **Severity**: Average cost per claim
- **Pure Premium**: Frequency × Severity
- **VaR (Value at Risk)**: Loss amount at a given percentile
- **TVaR (Tail Value at Risk)**: Average loss beyond VaR

## Distribution Types

### Frequency Distributions
- Poisson: Standard for claim counts
- Negative Binomial: Overdispersed claim counts
- Zero-Inflated: When many policies have no claims

### Severity Distributions
- Lognormal: Common for claim amounts
- Gamma: Alternative for claim amounts
- Pareto: Heavy-tailed losses

## Portfolio Analysis
- Aggregate Loss = Sum of individual claim amounts
- Uses compound distributions (frequency � severity)
- Monte Carlo simulation for complex cases
"""

    @mcp.prompt(
        name="portfolio_setup",
        description="Guide for setting up insurance portfolios"
    )
    async def portfolio_setup():
        return """
# Portfolio Setup Guide

## Required Policy Fields

```python
{
    "policy_id": "unique_identifier",
    "effective_date": "2024-01-01",  # ISO format
    "expiration_date": "2025-01-01",
    "lob": "GL",  # Line of business
    "exposure_amount": 1000000,  # Limit or exposure
    "premium": 5000,
    
    # Optional retention features
    "per_occ_retention": 1000,  # Deductible
    "agg_retention": 5000,  # Aggregate deductible
    "per_occ_limit": 100000,  # Per occurrence limit
    "agg_limit": 500000  # Aggregate limit
}
```

## Example Portfolio Creation

```python
portfolio_data = {
    "policies": [
        {
            "policy_id": "POL001",
            "effective_date": "2024-01-01",
            "expiration_date": "2025-01-01",
            "lob": "GL",
            "exposure_amount": 1000000,
            "premium": 5000
        },
        # More policies...
    ]
}
```
"""

    @mcp.prompt(
        name="optimization_guide",
        description="Guide for performance optimization options"
    )
    async def optimization_guide():
        return """
# Performance Optimization Guide

## Available Optimizations

### 1. Quasi-Monte Carlo (QMC)
- Better convergence than standard MC
- Use for large portfolios (>1000 policies)
- Enable with: `use_qmc=True`

### 2. JIT Compilation
- Speeds up numerical calculations
- Automatic for supported operations
- Best for repeated simulations

### 3. Parallel Processing
- Utilizes multiple CPU cores
- Automatic for large simulations
- Scales with portfolio size

## Recommendations by Portfolio Size

- **Small (<100 policies)**: Standard MC, no special options
- **Medium (100-1000)**: Consider QMC for better accuracy
- **Large (>1000)**: Use QMC and ensure parallel processing

## Performance Tips
1. Start with fewer simulations for testing
2. Use appropriate tail_alpha (0.01 or 0.05 typical)
3. Cache results when running multiple analyses
"""

    logger.info("Registered 3 prompts with MCP server")