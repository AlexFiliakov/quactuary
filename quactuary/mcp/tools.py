# mcp/tools.py
"""Tool definitions for MCP server."""

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from .base import ToolRegistry, create_tool_decorator
from .categories import ToolCategory
from .formats import MCPToolInput, MCPToolOutput, DataFormats, MCPExecutionError, validate_positive

# Initialize logging
logger = logging.getLogger(__name__)

# Create tool registry
registry = ToolRegistry()

# Create decorators for each category
pricing_tool = create_tool_decorator(registry, ToolCategory.PRICING)
dist_tool = create_tool_decorator(registry, ToolCategory.DISTRIBUTIONS)
portfolio_tool = create_tool_decorator(registry, ToolCategory.PORTFOLIO)
util_tool = create_tool_decorator(registry, ToolCategory.UTILITIES)
bench_tool = create_tool_decorator(registry, ToolCategory.BENCHMARKS)


# Pricing Tools
@pricing_tool(
    name="pricing_simulate_portfolio",
    description="Run Monte Carlo simulation on a portfolio to calculate risk measures",
    parameters={
        "portfolio_data": {
            "type": "object",
            "description": "Portfolio data in standard format or dict",
            "required": True
        },
        "n_simulations": {
            "type": "number",
            "description": "Number of Monte Carlo simulations",
            "required": False,
            "default": 10000
        },
        "tail_alpha": {
            "type": "number",
            "description": "Tail probability for VaR/TVaR calculation (e.g., 0.05 for 95%)",
            "required": False,
            "default": 0.05
        },
        "use_qmc": {
            "type": "boolean",
            "description": "Use Quasi-Monte Carlo instead of standard MC",
            "required": False,
            "default": False
        }
    }
)
def simulate_portfolio(portfolio_data: Dict[str, Any], 
                      n_simulations: int = 10000,
                      tail_alpha: float = 0.05,
                      use_qmc: bool = False) -> Dict[str, Any]:
    """Simulate a portfolio and calculate risk measures."""
    try:
        from ..pricing import PricingModel
        from ..book import Portfolio
        import pandas as pd
        
        # Validate inputs
        validate_positive(n_simulations, "n_simulations")
        if not 0 < tail_alpha < 1:
            raise MCPExecutionError("tail_alpha must be between 0 and 1")
        
        # Create portfolio
        if isinstance(portfolio_data, dict) and "policies" in portfolio_data:
            # Standard format
            policies_df = pd.DataFrame(portfolio_data["policies"])
        else:
            # Direct dataframe data
            policies_df = pd.DataFrame(portfolio_data)
        
        portfolio = Portfolio(policies_df)
        
        # Create pricing model
        model = PricingModel(portfolio)
        
        # Configure optimization
        if use_qmc:
            model.optimization_selector.set_optimizations(use_qmc_sampling=True)
        
        # Run simulation
        result = model.simulate(n_sims=n_simulations, tail_alpha=tail_alpha)
        
        # Format output
        return DataFormats.simulation_result(
            estimates={
                "mean": float(result.estimates["mean"]),
                "std": float(result.estimates["std"]),
                "var": float(result.estimates["VaR"]),
                "tvar": float(result.estimates["TVaR"]),
                "quantiles": {
                    str(q): float(v) 
                    for q, v in result.estimates.get("quantiles", {}).items()
                }
            },
            metadata={
                "n_simulations": n_simulations,
                "tail_alpha": tail_alpha,
                "use_qmc": use_qmc,
                "portfolio_size": len(portfolio.get_valid_policies())
            }
        )
        
    except ImportError as e:
        raise MCPExecutionError(f"Failed to import required modules: {str(e)}")
    except Exception as e:
        logger.error(f"Error in simulate_portfolio: {str(e)}")
        raise MCPExecutionError(f"Simulation failed: {str(e)}")


def register_tools(mcp):
    """Register all tools with the MCP server.
    
    This function is called by the server to register tools.
    The actual tool implementations are defined above using decorators.
    """
    # Tools are automatically registered via decorators
    # Just need to wire them to the MCP server
    
    for tool_name in registry.list_tools():
        tool = registry.get_tool(tool_name)
        if tool:
            # Create MCP tool wrapper
            @mcp.tool(name=tool_name, description=tool.description)
            async def mcp_tool_wrapper(**kwargs):
                # Get the actual tool from registry
                actual_tool = registry.get_tool(mcp_tool_wrapper.__name__)
                
                # Create input wrapper
                input_data = MCPToolInput(parameters=kwargs)
                
                # Execute tool
                output = actual_tool.execute(input_data)
                
                # Return result
                if output.success:
                    return output.data
                else:
                    raise Exception(output.error)
            
            # Set the wrapper name to match the tool
            mcp_tool_wrapper.__name__ = tool_name
    
    logger.info(f"Registered {len(registry.list_tools())} tools with MCP server")
