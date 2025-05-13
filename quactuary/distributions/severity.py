from typing import Protocol


class SeverityModel(Protocol):
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def rvs(self, size: int = 1) -> np.ndarray: ...

# TODO: Provide convenience wrappers so an actuary can pass in either:
# - a scalar (2 → “exactly 2 claims per year”),
# - a ready‑made scipy.stats distribution,
# - or a tiny helper like qa.freq.Poisson(mu=1.8) that just forwards to scipy.stats.poisson.
# Internally we normalize everything to an adapter object that satisfies the protocol.
