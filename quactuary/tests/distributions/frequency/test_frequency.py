import numpy as np
import pytest
from scipy.stats import binom as sp_binom
from scipy.stats import geom as sp_geom
from scipy.stats import hypergeom as sp_hypergeom
from scipy.stats import nbinom as sp_nbinom
from scipy.stats import poisson as sp_poisson
from scipy.stats import randint as randint
from scipy.stats import triang as sp_triang

from quactuary.distributions.frequency import (Binomial, DeterministicFreq,
                                               DiscreteUniformFreq,
                                               EmpiricalFreq, Geometric,
                                               Hypergeometric, MixFreq,
                                               NegativeBinomial, Poisson,
                                               TriangularFreq)


def test_binomial():
    model = Binomial(n=5, p=0.4)
    k = 2
    assert model.pmf(k) == pytest.approx(sp_binom(5, 0.4).pmf(k))
    assert model.cdf(3) == pytest.approx(sp_binom(5, 0.4).cdf(3))
    samples = model.rvs(size=10)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (10,)
    assert np.all((samples >= 0) & (samples <= 5))


def test_deterministic():
    model = DeterministicFreq(3)
    assert model.pmf(3) == 1.0
    assert model.pmf(2) == 0.0
    assert model.cdf(2) == 0.0
    assert model.cdf(3) == 1.0
    samples = model.rvs(size=5)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5,)
    assert np.all(samples == 3)


def test_discrete_uniform_freq():
    low, high = 0, 10
    model = DiscreteUniformFreq(low, high)
    assert model.pmf(3) == pytest.approx(randint(low, high).pmf(3))
    assert model.cdf(3) == pytest.approx(randint(low, high).cdf(3))
    samples = model.rvs(size=100)
    assert isinstance(samples, np.ndarray)
    assert np.issubdtype(samples.dtype, np.integer)


def test_empirical_freq():
    pmf_vals = {0: 0.2, 1: 0.8}
    model = EmpiricalFreq(pmf_vals)
    assert model.pmf(0) == pytest.approx(0.2)
    assert model.pmf(2) == 0.0
    assert model.cdf(0) == pytest.approx(0.2)
    assert model.cdf(1) == pytest.approx(1.0)
    samples = model.rvs(size=1000)
    assert set(np.unique(samples)).issubset({0, 1})


def test_geometric():
    p = 0.3
    model = Geometric(p)
    assert model.pmf(1) == pytest.approx(sp_geom(p).pmf(1))
    assert model.cdf(2) == pytest.approx(sp_geom(p).cdf(2))
    samples = model.rvs(size=100)
    assert samples.min() >= 1


def test_hypergeometric():
    M, n, N = 10, 3, 5
    model = Hypergeometric(M, n, N)
    k = 2
    assert model.pmf(k) == pytest.approx(sp_hypergeom(M, n, N).pmf(k))
    assert model.cdf(3) == pytest.approx(sp_hypergeom(M, n, N).cdf(3))
    samples = model.rvs(size=50)
    low = max(0, N - (M - n))
    high = min(n, N)
    assert samples.min() >= low
    assert samples.max() <= high


def test_mixfreq():
    comps = [DeterministicFreq(1), DeterministicFreq(3)]
    weights = [0.4, 0.6]
    model = MixFreq(comps, weights)
    assert model.pmf(1) == pytest.approx(0.4)
    assert model.pmf(3) == pytest.approx(0.6)
    samples = model.rvs(size=100)
    assert set(np.unique(samples)).issubset({1, 3})


def test_negative_binomial():
    r, p = 2, 0.5
    model = NegativeBinomial(r, p)
    k = 1
    assert model.pmf(k) == pytest.approx(sp_nbinom(r, p).pmf(k))
    assert model.cdf(2) == pytest.approx(sp_nbinom(r, p).cdf(2))
    samples = model.rvs(size=20)
    assert samples.min() >= 0


def test_poisson():
    mu = 2.0
    model = Poisson(mu)
    k = 0
    assert model.pmf(k) == pytest.approx(sp_poisson(mu).pmf(k))
    assert model.cdf(3) == pytest.approx(sp_poisson(mu).cdf(3))
    samples = model.rvs(size=50)
    assert samples.min() >= 0


def test_triangular_freq():
    c, loc, scale = 0.5, 0, 1
    model = TriangularFreq(c, loc, scale)
    assert model.pmf(1) == 0.0
    assert model.cdf(0) == pytest.approx(
        sp_triang(c, loc=loc, scale=scale).cdf(0))
    samples = model.rvs(size=100)
    assert isinstance(samples, np.ndarray)
    assert np.issubdtype(samples.dtype, np.integer)


def test_to_frequency_model_scalar():
    from quactuary.distributions.frequency import (DeterministicFreq,
                                                   to_frequency_model)

    model = to_frequency_model(5)
    assert isinstance(model, DeterministicFreq)
    assert model.pmf(5) == 1.0


def test_to_frequency_model_scalar_npint():
    from quactuary.distributions.frequency import (DeterministicFreq,
                                                   to_frequency_model)

    model = to_frequency_model(np.int64(5))
    assert isinstance(model, DeterministicFreq)
    assert model.pmf(5) == 1.0


def test_to_frequency_model_list():
    from quactuary.distributions.frequency import (EmpiricalFreq,
                                                   to_frequency_model)

    model = to_frequency_model([1, 2, 3])
    assert isinstance(model, EmpiricalFreq)
    assert model.pmf(1) == pytest.approx(1/3)
    assert model.pmf(2) == pytest.approx(1/3)
    assert model.pmf(3) == pytest.approx(1/3)
    assert model.pmf(4) == 0.0


def test_to_frequency_model_nparray():
    from quactuary.distributions.frequency import (EmpiricalFreq,
                                                   to_frequency_model)

    model = to_frequency_model(np.array([1, 2, 3]))
    assert isinstance(model, EmpiricalFreq)
    assert model.pmf(1) == pytest.approx(1/3)
    assert model.pmf(2) == pytest.approx(1/3)
    assert model.pmf(3) == pytest.approx(1/3)
    assert model.pmf(4) == 0.0


def test_to_frequency_model_series():
    import pandas as pd

    from quactuary.distributions.frequency import (EmpiricalFreq,
                                                   to_frequency_model)
    data = pd.Series([1, 2, 3])
    model = to_frequency_model(data)
    assert isinstance(model, EmpiricalFreq)
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
    assert model.pmf(0) == pytest.approx(frozen.pmf(0))
    samples = model.rvs(size=10)
    assert samples.shape == (10,)


def test_to_frequency_model_model_returned():
    from quactuary.distributions.frequency import Poisson, to_frequency_model

    orig = Poisson(mu=1.5)
    model = to_frequency_model(orig)
    assert model is orig


def test_to_frequency_model_empirical_list():
    from quactuary.distributions.frequency import (EmpiricalFreq,
                                                   to_frequency_model)

    data = [1, 2, 3]
    model = to_frequency_model(data)
    assert isinstance(model, EmpiricalFreq)
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
