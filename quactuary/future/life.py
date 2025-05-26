"""
Life insurance actuarial module (Phase 4 features).

This subpackage supports life contingency calculations, such as present value of benefits,
survival models, and policy projections. It may integrate standard mortality tables and
use libraries like `lifelib` for mortality data.

Examples:
    >>> from quactuary.future.life import LifeTable
    >>> lt = LifeTable(tx=30, table='US_1990')
    >>> px = lt.probability_of_survival(5)
"""