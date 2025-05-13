from typing import Protocol


class FrequencyModel(Protocol):
    def pmf(self, k: int) -> float: ...          # needed for exact quantum load
    def rvs(self, size: int = 1) -> np.ndarray: ...  # classical sampling fallback

# TODO: Provide convenience wrappers so an actuary can pass in either:
# - a scalar (2 → “exactly 2 claims per year”),
# - a ready‑made scipy.stats distribution,
# - or a tiny helper like qa.freq.Poisson(mu=1.8) that just forwards to scipy.stats.poisson.
# Internally we normalize everything to an adapter object that satisfies the protocol.
