import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from scipy.stats import binom as sp_binom
from scipy.stats import geom as sp_geom
from scipy.stats import hypergeom as sp_hypergeom
from scipy.stats import nbinom as sp_nbinom
from scipy.stats import poisson as sp_poisson
from scipy.stats import randint as randint
from scipy.stats import triang as sp_triang

import quactuary as qa
from quactuary.distributions.frequency import (Binomial,
                                               DeterministicFrequency,
                                               DiscreteUniformFrequency,
                                               EmpiricalFrequency,
                                               FrequencyModel, Geometric,
                                               Hypergeometric, MixedFrequency,
                                               NegativeBinomial, PanjerABk,
                                               Poisson, TriangularFrequency)

epsilon = 1e-12


def test_frequency_model():
    class DummyFrequencyModel(FrequencyModel):
        def __init__(self):
            return super().__init__()

        def pmf(self, k):
            return super().pmf(k)  # type: ignore[attr-defined]

        def cdf(self, k):
            return super().cdf(k)

        def rvs(self, size=1):
            return super().rvs(size=size)  # type: ignore[attr-defined]

    model = DummyFrequencyModel()
    with pytest.raises(NotImplementedError):
        model.pmf(0)
    with pytest.raises(NotImplementedError):
        model.cdf(0)
    with pytest.raises(NotImplementedError):
        model.rvs()
    with pytest.raises(NotImplementedError):
        model.rvs(size=10)


def test_scipy_frequency_adapter():
    from scipy.stats._distn_infrastructure import rv_frozen

    mu = 2
    frozen_model = sp_poisson(mu=mu)
    assert isinstance(frozen_model, rv_frozen)
    model = qa.distributions.frequency.to_frequency_model(frozen_model)
    k = 0
    assert model.pmf(k) == pytest.approx(
        sp_poisson(mu).pmf(k))  # type: ignore[attr-defined]
    assert model.cdf(3) == pytest.approx(sp_poisson(mu).cdf(3))
    samples = model.rvs(size=50)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.min() >= 0
    sample = model.rvs()
    assert isinstance(sample, np.integer)


def test_binomial():
    model = Binomial(n=5, p=0.4)
    k = 2
    assert model.pmf(k) == pytest.approx(
        sp_binom(5, 0.4).pmf(k))  # type: ignore[attr-defined]
    assert model.cdf(3) == pytest.approx(sp_binom(5, 0.4).cdf(3))
    samples = model.rvs(size=10)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.shape == (10,)
    assert np.all((samples >= 0) & (samples <= 5))
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = "Binomial(n=5, p=0.4)"
    assert str(model) == expected_str


def test_deterministic():
    model = DeterministicFrequency(3)  # type: ignore[attr-defined]
    assert model.pmf(3) == 1.0
    assert model.pmf(2) == 0.0
    assert model.cdf(2) == 0.0
    assert model.cdf(3) == 1.0
    samples = model.rvs(size=5)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.shape == (5,)
    assert np.all(samples == 3)
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = "DeterministicFrequency(value=3)"
    assert str(model) == expected_str


def test_discrete_uniform_freq():
    low, high = 0, 10
    model = DiscreteUniformFrequency(low, high)
    assert model.pmf(3) == pytest.approx(
        randint(low, high).pmf(3))  # type: ignore[attr-defined]
    assert model.cdf(3) == pytest.approx(randint(low, high).cdf(3))
    samples = model.rvs(size=100)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"DiscreteUniformFrequency(low={low}, high={high})"
    assert str(model) == expected_str


def test_empirical_freq():
    pmf_vals = {0: 0.2, 1: 0.8}
    model = EmpiricalFrequency(pmf_vals)
    assert model.pmf(0) == pytest.approx(0.2)
    assert model.pmf(2) == 0.0
    assert model.cdf(0) == pytest.approx(0.2)
    assert model.cdf(1) == pytest.approx(1.0)
    samples = model.rvs(size=1000)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert set(np.unique(samples)).issubset({0, 1})
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"EmpiricalFrequency(pmf_values={pmf_vals})"
    assert str(model) == expected_str


def test_geometric():
    p = 0.3
    model = Geometric(p)
    assert model.pmf(1) == pytest.approx(
        sp_geom(p).pmf(1))  # type: ignore[attr-defined]
    assert model.cdf(2) == pytest.approx(sp_geom(p).cdf(2))
    samples = model.rvs(size=100)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.min() >= 1
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = "Geometric(p=0.3)"
    assert str(model) == expected_str


def test_hypergeometric():
    M, n, N = 10, 3, 5
    model = Hypergeometric(M, n, N)
    k = 2
    assert model.pmf(k) == pytest.approx(sp_hypergeom(
        M, n, N).pmf(k))  # type: ignore[attr-defined]
    assert model.cdf(3) == pytest.approx(sp_hypergeom(M, n, N).cdf(3))
    samples = model.rvs(size=50)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    low = max(0, N - (M - n))
    high = min(n, N)
    assert samples.min() >= low
    assert samples.max() <= high
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"Hypergeometric(M={M}, n={n}, N={N})"
    assert str(model) == expected_str


def test_mixed_frequency():
    comps = [DeterministicFrequency(1),
             DeterministicFrequency(3)]  # type: ignore[attr-defined]
    weights = [0.4, 0.6]
    model = MixedFrequency(comps, weights)  # type: ignore[attr-defined]
    assert model.pmf(1) == pytest.approx(0.4)
    assert model.pmf(3) == pytest.approx(0.6)
    assert model.pmf(4) == pytest.approx(0.0)
    assert model.cdf(0) == pytest.approx(0.0)
    assert model.cdf(1) == pytest.approx(0.4)
    assert model.cdf(2) == pytest.approx(0.4)
    assert model.cdf(3) == pytest.approx(1.0)
    assert model.cdf(4) == pytest.approx(1.0)
    samples = model.rvs(size=100)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert set(np.unique(samples)).issubset({1, 3})
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_comps_str = [str(dist) for dist in comps]
    expected_str = f"MixedFrequency(components={expected_comps_str}, weights={weights})"
    assert str(model) == expected_str


def test_negative_binomial():
    r, p = 2, 0.5
    model = NegativeBinomial(r, p)
    k = 1
    assert model.pmf(k) == pytest.approx(
        sp_nbinom(r, p).pmf(k))  # type: ignore[attr-defined]
    assert model.cdf(2) == pytest.approx(sp_nbinom(r, p).cdf(2))
    samples = model.rvs(size=20)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.min() >= 0
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"NegativeBinomial(r={r}, p={p})"
    assert str(model) == expected_str


def test_panjer_abk():
    # Test zero-modified (k=0)
    zt_model = PanjerABk(a=-1/3, b=2, k=0)
    for i in range(100):
        assert zt_model.pmf(i) >= 0
    assert zt_model.pmf(0) == pytest.approx(243/1024)

    zt_model = PanjerABk(a=-1/4, b=7/4, k=0)
    assert zt_model.cdf(2) == zt_model.pmf(
        0) + zt_model.pmf(1) + zt_model.pmf(2)
    assert zt_model.cdf(2) == pytest.approx(1 - 0.09888)

    # Test zero-truncated (k=1)
    zm_model = PanjerABk(a=4/5, b=4/5, k=1)
    assert zm_model.pmf(0) == 0
    assert zm_model.pmf(1) == pytest.approx(8/120)
    assert zm_model.pmf(2) == pytest.approx(48/600)
    assert zm_model.pmf(3) == pytest.approx(256/3000)
    assert zm_model.cdf(3) == pytest.approx(
        zm_model.pmf(1) + zm_model.pmf(2) + zm_model.pmf(3))

    # Test general case (k>1)
    zm_model = PanjerABk(a=3/4, b=2/3, k=2)
    assert zt_model.cdf(2) == zt_model.pmf(
        0) + zt_model.pmf(1) + zt_model.pmf(2)

    # Test general functionality
    model = PanjerABk(a=2, b=3, k=0)
    samples = model.rvs(size=100)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.shape == (100,)
    assert np.all(samples >= 0)
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    # For k=1, values can be 0
    model2 = PanjerABk(a=3/4, b=2/3, k=1)
    samples2 = model2.rvs(size=100)
    assert np.all(samples2 >= 0)

    # Edge cases
    assert model2.pmf(-1) == 0
    assert model2.cdf(-1) == 0
    assert model2.cdf(1_000_000_000_000) == pytest.approx(1.0)

    expected_str = "PanjerABk(a=2, b=3, k=0)"
    assert str(model) == expected_str


def test_poisson():
    mu = 2.0
    model = Poisson(mu)
    k = 0
    assert model.pmf(k) == pytest.approx(
        sp_poisson(mu).pmf(k))  # type: ignore[attr-defined]
    assert model.cdf(3) == pytest.approx(sp_poisson(mu).cdf(3))
    samples = model.rvs(size=50)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    assert samples.min() >= 0
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"Poisson(mu={mu})"
    assert str(model) == expected_str


def test_triangular_freq():
    c, loc, scale = 0.5, 0, 1
    model = TriangularFrequency(c, loc, scale)
    assert model.pmf(1) == pytest.approx(
        sp_triang(c, loc=loc, scale=scale).cdf(1.5 - epsilon) -
        sp_triang(c, loc=loc, scale=scale).cdf(0.5))
    assert model.cdf(0) == pytest.approx(
        sp_triang(c, loc=loc, scale=scale).cdf(0.5 - epsilon))
    samples = model.rvs(size=100)
    assert isinstance(samples, pd.Series)
    assert is_integer_dtype(samples.dtype)
    sample = model.rvs()
    assert isinstance(sample, np.integer)

    expected_str = f"TriangularFrequency(c={c}, loc={loc}, scale={scale})"
    assert str(model) == expected_str


def test_to_frequency_model_scalar():
    from quactuary.distributions.frequency import (DeterministicFrequency,
                                                   to_frequency_model)

    model = to_frequency_model(5)
    assert isinstance(model, DeterministicFrequency)
    assert model.pmf(5) == 1.0


def test_to_frequency_model_scalar_npint():
    from quactuary.distributions.frequency import (DeterministicFrequency,
                                                   to_frequency_model)

    model = to_frequency_model(np.int64(5))
    assert isinstance(model, DeterministicFrequency)
    assert model.pmf(5) == 1.0


def test_to_frequency_model_list():
    from quactuary.distributions.frequency import (EmpiricalFrequency,
                                                   to_frequency_model)

    model = to_frequency_model([1, 2, 3])
    assert isinstance(model, EmpiricalFrequency)
    assert model.pmf(1) == pytest.approx(1/3)
    assert model.pmf(2) == pytest.approx(1/3)
    assert model.pmf(3) == pytest.approx(1/3)
    assert model.pmf(4) == 0.0


def test_to_frequency_model_nparray():
    from quactuary.distributions.frequency import (EmpiricalFrequency,
                                                   to_frequency_model)

    model = to_frequency_model(np.array([1, 2, 3]))
    assert isinstance(model, EmpiricalFrequency)
    assert model.pmf(1) == pytest.approx(1/3)
    assert model.pmf(2) == pytest.approx(1/3)
    assert model.pmf(3) == pytest.approx(1/3)
    assert model.pmf(4) == 0.0


def test_to_frequency_model_series():
    import pandas as pd

    from quactuary.distributions.frequency import (EmpiricalFrequency,
                                                   to_frequency_model)
    data = pd.Series([1, 2, 3])
    model = to_frequency_model(data)
    assert isinstance(model, EmpiricalFrequency)
    assert model.pmf(1) == pytest.approx(1/3)
    assert model.pmf(2) == pytest.approx(1/3)
    assert model.pmf(3) == pytest.approx(1/3)
    assert model.pmf(4) == 0.0


def test_to_frequency_model_frozen():
    from scipy.stats import poisson

    from quactuary.distributions.frequency import to_frequency_model

    frozen = poisson(mu=2)
    model = to_frequency_model(frozen)
    # Should proxy to frozen
    assert model.pmf(0) == pytest.approx(
        frozen.pmf(0))  # type: ignore[attr-defined]
    samples = model.rvs(size=10)
    assert samples.shape == (10,)


def test_to_frequency_model_model_returned():
    from quactuary.distributions.frequency import Poisson, to_frequency_model

    orig = Poisson(mu=1.5)
    model = to_frequency_model(orig)
    assert model is orig


def test_to_frequency_model_empirical_list():
    from quactuary.distributions.frequency import (EmpiricalFrequency,
                                                   to_frequency_model)

    data = [1, 2, 3]
    model = to_frequency_model(data)
    assert isinstance(model, EmpiricalFrequency)
    for i in data:
        assert model.pmf(i) == pytest.approx(1/3)


def test_to_frequency_model_empty_list_error():
    from quactuary.distributions.frequency import to_frequency_model
    with pytest.raises(ValueError):
        to_frequency_model([])


def test_to_frequency_model_float_error():
    from quactuary.distributions.frequency import to_frequency_model
    with pytest.raises(TypeError):
        to_frequency_model(0.5)


def test_to_frequency_model_list_with_float_error():
    from quactuary.distributions.frequency import to_frequency_model
    with pytest.raises(TypeError):
        to_frequency_model([1, 2, 3, 0.5])


def test_to_frequency_model_invalid_list_error():
    from quactuary.distributions.frequency import to_frequency_model
    with pytest.raises(TypeError):
        to_frequency_model([1, 'a', 3])
