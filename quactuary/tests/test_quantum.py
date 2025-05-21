import pytest

from quactuary.quantum import QuantumPricingModel


def test_quantum_pricing_model_mixin():
    qpmm = QuantumPricingModel()
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics()
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            mean=False,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=False,
            tail_alpha=0.95)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            mean=True,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=False)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            mean=False,
            variance=True,
            value_at_risk=False,
            tail_value_at_risk=False)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            mean=False,
            variance=False,
            value_at_risk=True,
            tail_value_at_risk=False,
            tail_alpha=0.80)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            mean=False,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=True,
            tail_alpha=0.80)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics()
    with pytest.raises(NotImplementedError):
        qpmm.mean_loss()
    with pytest.raises(NotImplementedError):
        qpmm.variance()
    with pytest.raises(NotImplementedError):
        qpmm.value_at_risk(alpha=0.95)
    with pytest.raises(NotImplementedError):
        qpmm.tail_value_at_risk(alpha=0.95)
