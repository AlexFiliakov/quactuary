"""
(Phase 2 features)
Claims reserving models.

Purpose: Implement actuarial reserving methods (deterministic and stochastic)
by wrapping the established chainladder Python library.
This allows robust functionality for loss triangles, IBNR,
and reserve risk without writing new algorithms from scratch.

Contents:
------------------
ChainLadderReserve (deterministic chain-ladder): Wraps chainladder.Chainladder.
The class will accept loss development data as a pandas DataFrame
(with accident periods as index and development periods as columns)
or as a chainladder.Triangle object. Internally, it uses chainladder.Chainladder().fit() to perform the calculation.
We then expose results through quActuary in a user-friendly way: e.g. reserve_model.ultimates_ could be
a pandas Series of ultimate losses per origin year (extracted from model.ultimate_ in chainladder),
and reserve_model.ibnr_ a Series of IBNR reserves. By default, this uses classical computation
(there is no known quantum speedup for basic chain-ladder), but the consistent interface means
the user doesn’t need to interact with chainladder objects directly.

MackChainLadder (stochastic reserves): Wraps chainladder.MackChainladder to provide mean and variance of reserves.
The quActuary class could provide attributes like prediction_error_ in addition to ultimates.

Other reserving methods (Bornhuetter-Ferguson, Cape-Cod, etc.): These can also wrap chainladder’s
implementations by exposing a similar .fit(data) interface and result attributes (e.g. bf_model.ultimate_).
We will align with chainladder’s naming conventions for familiarity; for example, chainladder uses attributes
like .ultimate_ and .full_triangle_ after fitting. quActuary can adopt this convention,
which is also consistent with scikit-learn’s practice of attributes ending in underscore for fitted results.

Quantum considerations: Reserving models primarily rely on historical data and regression-like techniques.
Quantum computing does not yet offer an established advantage for chain-ladder style algorithms,
so Phase 2 development will focus on classical integration (i.e. chainladder) to deliver value.
However, we will prepare the design to potentially incorporate quantum methods for related tasks
(for example, a “Quantum Copula” module is planned in Phase 2 to model dependency between
lines of business in reserving scenarios). A future idea is using quantum tools for stochastic reserving
(e.g. quantum-accelerated bootstrap simulations), but initially the reserving module will default to classical engines.
All reserving outputs will be pandas DataFrames/Series so that users can easily join results (e.g. actual vs expected runoff)
and further analyze in Python/Pandas (similar to how chainladder returns pandas-friendly objects).
"""
