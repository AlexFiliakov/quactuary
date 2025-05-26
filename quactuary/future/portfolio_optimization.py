"""
Portfolio optimization tools (Phase 3 features).

This module provides interfaces for optimizing risk portfolios using classical solvers (e.g., PyPortfolioOpt)
and optional quantum optimization backends (e.g. QAOA).

Examples:
    >>> from quactuary.future.portfolio_optimization import PortfolioOptimization
    >>> opt = PortfolioOptimization(portfolio, constraints)
    >>> solution = opt.solve(method='classical')
"""
