"""
Dependence and copula modeling (Phase 2 features).

This module provides tools for modeling dependencies between insurance risks,
including copula-based simulations and correlation structures.

Examples:
    >>> from quactuary.future.dependence import CopulaModel
    >>> copula = CopulaModel('t', df=4, rho=0.5)
    >>> samples = copula.sample(n=1000)
"""
