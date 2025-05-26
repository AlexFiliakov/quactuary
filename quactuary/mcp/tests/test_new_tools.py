"""Tests for newly implemented MCP tools."""

import pytest
import numpy as np
from quactuary.mcp.tools import calculate_var, create_portfolio
from quactuary.mcp.formats import MCPExecutionError


class TestCalculateVar:
    """Test pricing_calculate_var tool."""
    
    def test_empirical_var(self):
        """Test empirical VaR calculation."""
        losses = list(np.random.normal(100, 20, 1000))
        result = calculate_var(losses, confidence_level=0.95)
        
        assert "var" in result
        assert "tvar" in result
        assert "statistics" in result
        assert result["confidence_level"] == 0.95
        assert result["method"] == "empirical"
        assert result["statistics"]["n_samples"] == 1000
    
    def test_normal_var(self):
        """Test parametric VaR calculation."""
        losses = list(np.random.normal(100, 20, 1000))
        result = calculate_var(losses, confidence_level=0.99, method="normal")
        
        assert "var" in result
        assert "tvar" in result
        assert result["method"] == "normal"
        assert result["confidence_level"] == 0.99
    
    def test_empty_losses(self):
        """Test error handling for empty losses."""
        with pytest.raises(MCPExecutionError, match="Losses array cannot be empty"):
            calculate_var([])
    
    def test_invalid_confidence(self):
        """Test error handling for invalid confidence level."""
        with pytest.raises(MCPExecutionError, match="Confidence level must be between 0 and 1"):
            calculate_var([1, 2, 3], confidence_level=1.5)


class TestCreatePortfolio:
    """Test portfolio_create tool."""
    
    def test_create_basic_portfolio(self):
        """Test basic portfolio creation."""
        policies = [
            {"policy_id": "P001", "sum_insured": 100000, "premium": 1000},
            {"policy_id": "P002", "sum_insured": 200000, "premium": 2000},
            {"policy_id": "P003", "sum_insured": 150000, "premium": 1500}
        ]
        
        result = create_portfolio(policies)
        
        assert "portfolio_id" in result
        assert "statistics" in result
        assert result["statistics"]["total_policies"] == 3
        assert result["statistics"]["total_sum_insured"] == 450000
        assert result["statistics"]["total_premium"] == 4500
        assert result["statistics"]["average_sum_insured"] == 150000
        assert result["statistics"]["average_premium"] == 1500
    
    def test_create_portfolio_with_distributions(self):
        """Test portfolio creation with distribution info."""
        policies = [
            {
                "policy_id": "P001", 
                "sum_insured": 100000, 
                "premium": 1000,
                "frequency_type": "poisson",
                "severity_type": "lognormal"
            },
            {
                "policy_id": "P002", 
                "sum_insured": 200000, 
                "premium": 2000,
                "frequency_type": "binomial",
                "severity_type": "gamma"
            }
        ]
        
        result = create_portfolio(policies)
        
        assert "frequency_distribution" in result["statistics"]
        assert "severity_distribution" in result["statistics"]
    
    def test_empty_policies(self):
        """Test error handling for empty policies."""
        with pytest.raises(MCPExecutionError, match="Policies list cannot be empty"):
            create_portfolio([])
    
    def test_missing_required_columns(self):
        """Test error handling for missing columns."""
        policies = [{"policy_id": "P001", "sum_insured": 100000}]  # Missing premium
        
        with pytest.raises(MCPExecutionError, match="Missing required columns"):
            create_portfolio(policies)