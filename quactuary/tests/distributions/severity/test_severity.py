import numpy as np
import pytest
from scipy.stats import beta as sp_beta
from scipy.stats import chi2 as sp_chi2
from scipy.stats import expon as sp_expon
from scipy.stats import gamma as sp_gamma
from scipy.stats import lognorm as sp_lognorm
from scipy.stats import pareto as sp_pareto
from scipy.stats import triang as sp_triang
from scipy.stats import uniform as sp_uniform
from scipy.stats import weibull_min as sp_weibull

from quactuary.distributions.severity import (Beta, ChiSquared, ConstantSev,
                                              ContinuousUniformSev,
                                              EmpiricalSev, Exponential, Gamma,
                                              Lognormal, MixSev, Pareto,
                                              TriangularSev, Weibull)


def test_beta_severity():
    a, b, loc, scale = 2.0, 5.0, 0.0, 1.0
    model = Beta(a, b, loc=loc, scale=scale)
    x = 0.3
    assert model.pdf(x) == pytest.approx(
        sp_beta(a, b, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_beta(a, b, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=5)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5,)
    assert np.all((samples >= loc) & (samples <= loc + scale))


def test_chi_squared_severity():
    df, loc, scale = 3.0, 0.0, 2.0
    model = ChiSquared(df, loc=loc, scale=scale)
    x = 1.5
    assert model.pdf(x) == pytest.approx(
        sp_chi2(df, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_chi2(df, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=5)
    assert samples.shape == (5,)
    assert np.all(samples >= loc)


def test_constant_severity():
    value = 7.5
    model = ConstantSev(value)
    assert model.pdf(value) == 1.0
    assert model.pdf(value + 1) == 0.0
    assert model.cdf(value - 0.1) == 0.0
    assert model.cdf(value) == 1.0
    samples = model.rvs(size=3)
    assert np.all(samples == value)


def test_continous_uniform_severity():
    loc, scale = 2.0, 5.0
    model = ContinuousUniformSev(loc=loc, scale=scale)
    x = 4.0
    assert model.pdf(x) == pytest.approx(
        sp_uniform(loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_uniform(loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=9)
    assert samples.shape == (9,)
    assert np.all((samples >= loc) & (samples <= loc + scale))


def test_empirical_severity():
    values = [0.0, 5.0, 10.0]
    probs = [0.2, 0.3, 0.5]
    model = EmpiricalSev(values, probs)
    # PDF exact matches
    assert model.pdf(5.0) == pytest.approx(0.3)
    assert model.pdf(1.0) == pytest.approx(0.0)
    # CDF cumulative
    assert model.cdf(0.0) == pytest.approx(0.2)
    assert model.cdf(6.0) == pytest.approx(0.2 + 0.3)
    # RVS draws from provided values
    samples = model.rvs(size=100)
    assert set(np.unique(samples)).issubset(set(values))


def test_exponential_severity():
    loc, scale = 1.0, 2.0
    model = Exponential(scale=scale, loc=loc)
    x = 2.5
    assert model.pdf(x) == pytest.approx(sp_expon(loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(sp_expon(loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=10)
    assert samples.shape == (10,)
    assert np.all(samples >= loc)


def test_gamma_severity():
    shape, loc, scale = 2.0, 0.0, 3.0
    model = Gamma(shape, loc=loc, scale=scale)
    x = 4.0
    assert model.pdf(x) == pytest.approx(
        sp_gamma(shape, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_gamma(shape, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=8)
    assert samples.shape == (8,)
    assert np.all(samples >= loc)


def test_lognormal_severity():
    s, loc, scale = 0.9, 0.0, 1.5
    model = Lognormal(s, loc=loc, scale=scale)
    x = 1.2
    assert model.pdf(x) == pytest.approx(
        sp_lognorm(s, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_lognorm(s, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=6)
    assert samples.shape == (6,)
    assert np.all(samples >= 0)


def test_mix_severity():
    comps = [ConstantSev(2.0), ConstantSev(5.0)]
    weights = [0.4, 0.6]
    model = MixSev(comps, weights)
    assert model.pdf(2.0) == pytest.approx(0.4)
    assert model.pdf(5.0) == pytest.approx(0.6)
    samples = model.rvs(size=50)
    assert set(np.unique(samples)).issubset({2.0, 5.0})


def test_pareto_severity():
    b, loc, scale = 2.5, 0.0, 1.0
    model = Pareto(b, loc=loc, scale=scale)
    x = 3.0
    assert model.pdf(x) == pytest.approx(
        sp_pareto(b, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_pareto(b, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=7)
    assert samples.shape == (7,)
    assert np.all(samples >= loc)


def test_triangular_severity():
    c, loc, scale = 0.3, 1.0, 4.0
    model = TriangularSev(c, loc=loc, scale=scale)
    x = 2.0
    assert model.pdf(x) == pytest.approx(
        sp_triang(c, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_triang(c, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=12)
    assert samples.shape == (12,)
    assert np.all((samples >= loc) & (samples <= loc + scale))


def test_weibull_severity():
    c, loc, scale = 1.7, 0.0, 2.0
    model = Weibull(c, loc=loc, scale=scale)
    x = 1.0
    assert model.pdf(x) == pytest.approx(
        sp_weibull(c, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_weibull(c, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=11)
    assert samples.shape == (11,)
    assert np.all(samples >= loc)


def test_to_severity_model_scalar():
    from quactuary.distributions.severity import ConstantSev, to_severity_model

    model = to_severity_model(3.5)
    assert isinstance(model, ConstantSev)
    assert model.pdf(3.5) == 1.0


def test_to_severity_model_frozen():
    from scipy.stats import expon

    from quactuary.distributions.severity import to_severity_model

    frozen = expon(scale=2.0)
    model = to_severity_model(frozen)
    assert model.pdf(1.0) == pytest.approx(frozen.pdf(1.0))
    samples = model.rvs(size=5)
    assert isinstance(samples, np.ndarray)
    assert samples.min() >= 0


def test_to_severity_model_model_returned():
    from quactuary.distributions.severity import Exponential, to_severity_model

    orig = Exponential(scale=2.0)
    model = to_severity_model(orig)
    assert model is orig


def test_to_severity_model_empirical_list():
    from quactuary.distributions.severity import (EmpiricalSev,
                                                  to_severity_model)

    data = [2.5, 3.5, 4.5]
    model = to_severity_model(data)
    assert isinstance(model, EmpiricalSev)
    for i in data:
        assert model.pdf(i) == pytest.approx(1/3)


def test_to_severity_model_empty_list_error():
    from quactuary.distributions.severity import to_severity_model
    with pytest.raises(ValueError):
        to_severity_model([])


def test_to_severity_model_invalid_list_error():
    from quactuary.distributions.severity import to_severity_model
    with pytest.raises(TypeError):
        to_severity_model([1.0, 'a', 2.0])
