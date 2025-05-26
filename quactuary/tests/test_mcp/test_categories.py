"""Test MCP category definitions."""

import pytest
from quactuary.mcp.categories import (
    ToolCategory, 
    CATEGORY_INFO, 
    get_category_prefix,
    get_tool_category,
    PLANNED_TOOLS
)


class TestToolCategory:
    """Test ToolCategory enum and related functions."""
    
    def test_tool_category_enum(self):
        """Test that all expected categories exist."""
        assert ToolCategory.PRICING.value == "pricing"
        assert ToolCategory.DISTRIBUTIONS.value == "distributions"
        assert ToolCategory.PORTFOLIO.value == "portfolio"
        assert ToolCategory.UTILITIES.value == "utilities"
        assert ToolCategory.BENCHMARKS.value == "benchmarks"
    
    def test_category_info_complete(self):
        """Test that all categories have complete info."""
        for category in ToolCategory:
            assert category in CATEGORY_INFO
            info = CATEGORY_INFO[category]
            assert "name" in info
            assert "description" in info
            assert "prefix" in info
            assert info["prefix"].endswith("_")
    
    def test_get_category_prefix(self):
        """Test prefix retrieval for categories."""
        assert get_category_prefix(ToolCategory.PRICING) == "pricing_"
        assert get_category_prefix(ToolCategory.DISTRIBUTIONS) == "dist_"
        assert get_category_prefix(ToolCategory.PORTFOLIO) == "portfolio_"
        assert get_category_prefix(ToolCategory.UTILITIES) == "util_"
        assert get_category_prefix(ToolCategory.BENCHMARKS) == "bench_"
    
    def test_get_tool_category(self):
        """Test category detection from tool names."""
        assert get_tool_category("pricing_simulate_portfolio") == ToolCategory.PRICING
        assert get_tool_category("dist_create_frequency") == ToolCategory.DISTRIBUTIONS
        assert get_tool_category("portfolio_validate") == ToolCategory.PORTFOLIO
        assert get_tool_category("util_convert_format") == ToolCategory.UTILITIES
        assert get_tool_category("bench_run_test") == ToolCategory.BENCHMARKS
    
    def test_get_tool_category_invalid(self):
        """Test that invalid tool names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool category"):
            get_tool_category("invalid_tool_name")
        
        with pytest.raises(ValueError, match="Unknown tool category"):
            get_tool_category("notaprefix_something")
    
    def test_planned_tools_structure(self):
        """Test that planned tools are properly structured."""
        # All categories should have planned tools
        assert len(PLANNED_TOOLS) == len(ToolCategory)
        
        for category in ToolCategory:
            assert category in PLANNED_TOOLS
            tools = PLANNED_TOOLS[category]
            assert isinstance(tools, list)
            assert len(tools) > 0
            
            # All tool names should start with correct prefix
            prefix = get_category_prefix(category)
            for tool_name in tools:
                assert tool_name.startswith(prefix)