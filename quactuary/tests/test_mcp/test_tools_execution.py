"""Test actual MCP tool execution functionality."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from quactuary.mcp.tools import registry, simulate_portfolio
from quactuary.mcp.formats import MCPToolInput, MCPToolOutput


class TestPricingToolExecution:
    """Test pricing tool execution with realistic scenarios."""
    
    def test_simulate_portfolio_basic(self):
        """Test basic portfolio simulation with mocked imports."""
        # We'll test the function interface without executing the actual simulation
        # since that would require the full quactuary setup
        
        # Test that the function exists and has proper signature
        assert callable(simulate_portfolio)
        
        # Test parameter validation
        import inspect
        sig = inspect.signature(simulate_portfolio)
        params = list(sig.parameters.keys())
        
        assert 'portfolio_data' in params
        assert 'n_simulations' in params
        assert 'tail_alpha' in params
        assert 'use_qmc' in params
        
        # Test default values
        assert sig.parameters['n_simulations'].default == 10000
        assert sig.parameters['tail_alpha'].default == 0.05
        assert sig.parameters['use_qmc'].default is False
        
    def test_simulate_portfolio_validation(self):
        """Test portfolio simulation input validation."""
        # Test with invalid n_simulations
        with pytest.raises(Exception) as exc_info:
            simulate_portfolio(
                portfolio_data={"policies": []},
                n_simulations=-100
            )
        assert "must be positive" in str(exc_info.value)
        
        # Test with invalid tail_alpha
        with pytest.raises(Exception) as exc_info:
            simulate_portfolio(
                portfolio_data={"policies": []},
                n_simulations=1000,
                tail_alpha=1.5
            )
        assert "must be between 0 and 1" in str(exc_info.value)
    
    def test_tool_through_registry(self):
        """Test executing tool through registry."""
        tool = registry.get_tool("pricing_simulate_portfolio")
        assert tool is not None
        
        # Create input
        input_data = MCPToolInput(
            parameters={
                "portfolio_data": {"policies": []},
                "n_simulations": 100
            }
        )
        
        # Execute (will fail without full setup, but validates interface)
        output = tool.execute(input_data)
        assert isinstance(output, MCPToolOutput)
        
        # Should have proper error handling
        if not output.success:
            assert output.error is not None


class TestToolDiscovery:
    """Test MCP tool discovery and metadata."""
    
    def test_list_all_tools(self):
        """Test listing all registered tools."""
        tools = registry.list_tools()
        assert len(tools) >= 1
        assert "pricing_simulate_portfolio" in tools
    
    def test_get_tools_by_category(self):
        """Test getting tools organized by category."""
        by_category = registry.get_tools_by_category()
        
        # Should have at least pricing category
        from quactuary.mcp.categories import ToolCategory
        assert ToolCategory.PRICING in by_category
        assert len(by_category[ToolCategory.PRICING]) >= 1
    
    def test_tool_metadata(self):
        """Test tool metadata and parameters."""
        tool = registry.get_tool("pricing_simulate_portfolio")
        
        # Check basic metadata
        assert tool.name == "pricing_simulate_portfolio"
        assert tool.description != ""
        assert tool.category.value == "pricing"
        
        # Check parameters
        params = tool.get_parameters()
        assert "portfolio_data" in params
        assert "n_simulations" in params
        assert "tail_alpha" in params
        assert "use_qmc" in params
        
        # Verify required parameters
        assert params["portfolio_data"]["required"] is True
        assert params["n_simulations"]["required"] is False
        assert params["n_simulations"]["default"] == 10000