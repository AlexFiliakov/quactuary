"""
Copula and dependence functions.
(Phase 2 features)

CopulaModel (or dependency modeling): Provides tools to model dependency between risks, e.g. using copulas.
This can wrap functionality from `gemact`, which includes popular copulas with advanced features
(e.g. a Student-t copula with CDF approximation).

quActuary could allow users to fit a copula to data or specify parameters, then simulate correlated losses across lines.
When quantum random number generation or quantum copula sampling becomes viable, it can be added as a backend option here.
"""
