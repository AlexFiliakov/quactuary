import numpy as np
import pytest
from scipy.stats import beta as sp_beta
from scipy.stats import chi2 as sp_chi2
from scipy.stats import expon as sp_expon
from scipy.stats import gamma as sp_gamma
from scipy.stats import invgamma as sp_invgamma
from scipy.stats import invgauss as sp_invgauss
from scipy.stats import invweibull as sp_invweibull
from scipy.stats import lognorm as sp_lognorm
from scipy.stats import pareto as sp_pareto
from scipy.stats import t as sp_t
from scipy.stats import triang as sp_triang
from scipy.stats import uniform as sp_uniform
from scipy.stats import weibull_min as sp_weibull

from quactuary.distributions.severity import (Beta, ChiSquared, ConstantSev,
                                              ContinuousUniformSev,
                                              DiscretizedSeverity,
                                              EmpiricalSev, Exponential, Gamma,
                                              InverseGamma, InverseGaussian,
                                              InverseWeibull, Lognormal,
                                              MixSev, Pareto, StudentsT,
                                              TriangularSev, Weibull)


def test_discreteized_severity_model():
    # Test the DiscretizedSeverityModel with a ChiSquared distribution
    df, loc, scale = 4.0, 0.0, 500.0
    model = ChiSquared(df, loc=loc, scale=scale)
    discretized_model = DiscretizedSeverity(
        model, min_val=0, max_val=8000, bins=100)

    # Check the properties of the discretized model
    assert isinstance(discretized_model.sev_dist, ChiSquared)
    assert discretized_model.step > 0
    assert len(discretized_model.mid_x_vals) == 100
    assert len(discretized_model.bin_mean) == 100
    assert len(discretized_model._probs) == 100
    assert discretized_model.pmf(9000) == pytest.approx(0.0)

    # Test the DiscretizedSeverityModel with an EmpiricalSev distribution
    values = [0.0, 5.0, 10.0]
    probs = [0.2, 0.3, 0.5]
    model = EmpiricalSev(values, probs)
    discretized_model = DiscretizedSeverity(
        model, min_val=0, max_val=10, bins=3)
    # The distribution gets transformed to bin midpoints
    # and the probabilities are assigned to those midpoints
    edges = np.linspace(0, 10, 4)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    step = midpoints[1] - midpoints[0]
    assert discretized_model.step == pytest.approx(step)
    for i, x in enumerate(midpoints):
        assert discretized_model.pmf(x) == pytest.approx(
            probs[i]), f"PMF at {x} should be {probs[i]}, but got {discretized_model.pmf(x)}"
    assert discretized_model.pmf(0.0) == pytest.approx(0.0)
    assert discretized_model.pmf(5.0) == pytest.approx(0.3)
    assert discretized_model.pmf(10.0) == pytest.approx(0.0)
    assert discretized_model.cdf(0.1) == pytest.approx(0.0)
    assert discretized_model.cdf(9.5) == pytest.approx(1.0)

    # Test continuous distribution discretization via uniform
    loc, scale = 2.0, 5.0
    test_values = [1.0, 2.5, 3.5, 8.0]
    model = ContinuousUniformSev(loc=loc, scale=scale)
    discretized_model = DiscretizedSeverity(
        model, min_val=loc, max_val=loc + scale, bins=5)
    # Test PMF
    test_pmf_expected = [0.0, 1.0/5.0, 1.0/5.0, 0.0]
    for i, x in enumerate(test_values):
        assert discretized_model.pmf(x) == pytest.approx(
            test_pmf_expected[i]), f"PMF at {x} should be {test_pmf_expected[i]}, but got {discretized_model.pmf(x)}"
    # Test CDF
    test_cdf_expected = [0.0, 1.0/5.0, 2.0/5.0, 1.0]
    for i, x in enumerate(test_values):
        assert discretized_model.cdf(x) == pytest.approx(
            test_cdf_expected[i]), f"CDF at {x} should be {test_cdf_expected[i]}, but got {discretized_model.pmf(x)}"


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


def test_inverse_gamma_severity():
    shape, loc, scale = 2.0, 0.0, 3.0
    model = InverseGamma(shape, loc=loc, scale=scale)
    x = 4.0
    assert model.pdf(x) == pytest.approx(
        sp_invgamma(shape, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_invgamma(shape, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=8)
    assert samples.shape == (8,)
    assert np.all(samples >= loc)


def test_inverse_gaussian_severity():
    shape, loc, scale = 10.0, 3.0, 1.5
    model = InverseGaussian(shape, loc=loc, scale=scale)
    x = 6.0
    assert model.pdf(x) == pytest.approx(
        sp_invgauss(shape, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_invgauss(shape, loc=loc, scale=scale).cdf(x))
    samples = model.rvs(size=8)
    assert samples.shape == (8,)
    assert np.all(samples >= 0)


def test_inverse_weibull_severity():
    shape, loc, scale = 2.0, 0.0, 3.0
    model = InverseWeibull(shape, loc=loc, scale=scale)
    x = 4.0
    assert model.pdf(x) == pytest.approx(
        sp_invweibull(shape, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_invweibull(shape, loc=loc, scale=scale).cdf(x))
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


def test_t_severity():
    df, loc, scale = 5.1, 3.0, 1.5
    model = StudentsT(df, loc=loc, scale=scale)
    x = 2.0
    assert model.pdf(x) == pytest.approx(
        sp_t(df, loc=loc, scale=scale).pdf(x))
    assert model.cdf(x) == pytest.approx(
        sp_t(df, loc=loc, scale=scale).cdf(x))
    np.random.seed(42)
    samples = model.rvs(size=10)
    assert samples.shape == (10,)
    assert np.all((samples >= loc - 4 * scale) & (samples <= loc + 4 * scale))


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
    assert model.pdf(4.0) == 0


def test_to_severity_model_scalar_npint():
    from quactuary.distributions.severity import ConstantSev, to_severity_model

    model = to_severity_model(np.float64(3.5))
    assert isinstance(model, ConstantSev)
    assert model.pdf(3.5) == 1.0
    assert model.pdf(4.0) == 0


def test_to_severity_model_list():
    from quactuary.distributions.severity import (EmpiricalSev,
                                                  to_severity_model)

    model = to_severity_model([1, 3.5])
    assert isinstance(model, EmpiricalSev)
    assert model.pdf(1.0) == pytest.approx(1/2)
    assert model.pdf(3.5) == pytest.approx(1/2)
    assert model.pdf(4.0) == 0.0


def test_to_severity_model_nparray():
    from quactuary.distributions.severity import (EmpiricalSev,
                                                  to_severity_model)

    model = to_severity_model(np.array([1, 3.5]))
    assert isinstance(model, EmpiricalSev)
    assert model.pdf(1.0) == pytest.approx(1/2)
    assert model.pdf(3.5) == pytest.approx(1/2)
    assert model.pdf(4.0) == 0.0


def test_to_severity_model_series():
    import pandas as pd

    from quactuary.distributions.severity import (EmpiricalSev,
                                                  to_severity_model)

    model = to_severity_model(pd.Series([1, 3.5]))
    assert isinstance(model, EmpiricalSev)
    assert model.pdf(1) == pytest.approx(1/2)
    assert model.pdf(3.5) == pytest.approx(1/2)
    assert model.pdf(4.0) == 0.0


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
