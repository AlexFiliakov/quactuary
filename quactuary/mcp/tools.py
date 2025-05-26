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


@pricing_tool(
    name="pricing_calculate_var",
    description="Calculate Value at Risk (VaR) for a given loss distribution",
    parameters={
        "losses": {
            "type": "array",
            "description": "Array of simulated loss values",
            "required": True
        },
        "confidence_level": {
            "type": "number",
            "description": "Confidence level for VaR (e.g., 0.95 for 95% VaR)",
            "required": False,
            "default": 0.95
        },
        "method": {
            "type": "string",
            "description": "Method for VaR calculation: 'empirical' or 'normal'",
            "required": False,
            "default": "empirical",
            "enum": ["empirical", "normal"]
        }
    }
)
def calculate_var(losses: List[float], 
                 confidence_level: float = 0.95,
                 method: str = "empirical") -> Dict[str, Any]:
    """Calculate Value at Risk (VaR) for a loss distribution."""
    try:
        losses_array = np.array(losses)
        
        # Validate inputs
        if len(losses_array) == 0:
            raise MCPExecutionError("Losses array cannot be empty")
        if not 0 < confidence_level < 1:
            raise MCPExecutionError("Confidence level must be between 0 and 1")
        
        if method == "empirical":
            # Empirical VaR (percentile method)
            var_value = float(np.percentile(losses_array, confidence_level * 100))
        elif method == "normal":
            # Parametric VaR assuming normal distribution
            mean = float(np.mean(losses_array))
            std = float(np.std(losses_array))
            from scipy.stats import norm
            var_value = float(mean + std * norm.ppf(confidence_level))
        else:
            raise MCPExecutionError(f"Unknown method: {method}")
        
        # Calculate additional statistics
        mean_loss = float(np.mean(losses_array))
        std_loss = float(np.std(losses_array))
        max_loss = float(np.max(losses_array))
        min_loss = float(np.min(losses_array))
        
        # Calculate TVaR (Tail Value at Risk / Conditional VaR)
        tail_losses = losses_array[losses_array >= var_value]
        tvar_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value
        
        return {
            "var": var_value,
            "tvar": tvar_value,
            "confidence_level": confidence_level,
            "method": method,
            "statistics": {
                "mean": mean_loss,
                "std": std_loss,
                "max": max_loss,
                "min": min_loss,
                "n_samples": len(losses_array)
            }
        }
        
    except ImportError as e:
        raise MCPExecutionError(f"Failed to import required modules: {str(e)}")
    except Exception as e:
        logger.error(f"Error in calculate_var: {str(e)}")
        raise MCPExecutionError(f"VaR calculation failed: {str(e)}")


@portfolio_tool(
    name="portfolio_create",
    description="Create a portfolio from policy data",
    parameters={
        "policies": {
            "type": "array",
            "description": "Array of policy objects with required fields",
            "required": True
        },
        "validate": {
            "type": "boolean",
            "description": "Whether to validate the portfolio data",
            "required": False,
            "default": True
        }
    }
)
def create_portfolio(policies: List[Dict[str, Any]], 
                    validate: bool = True) -> Dict[str, Any]:
    """Create a portfolio from policy data."""
    try:
        import pandas as pd
        
        # Validate input
        if not policies:
            raise MCPExecutionError("Policies list cannot be empty")
        
        # Convert to DataFrame
        policies_df = pd.DataFrame(policies)
        
        # Check required columns
        required_columns = ["policy_id", "sum_insured", "premium"]
        missing_columns = [col for col in required_columns if col not in policies_df.columns]
        if missing_columns:
            raise MCPExecutionError(f"Missing required columns: {missing_columns}")
        
        # Simple validation of data
        invalid_count = 0
        if validate:
            # Check for invalid values
            invalid_sum_insured = policies_df["sum_insured"] <= 0
            invalid_premium = policies_df["premium"] < 0
            invalid_mask = invalid_sum_insured | invalid_premium
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                logger.warning(f"Portfolio contains {invalid_count} policies with invalid values")
        
        # Get portfolio statistics
        stats = {
            "total_policies": len(policies_df),
            "valid_policies": len(policies_df) - invalid_count,
            "total_sum_insured": float(policies_df["sum_insured"].sum()),
            "total_premium": float(policies_df["premium"].sum()),
            "average_sum_insured": float(policies_df["sum_insured"].mean()),
            "average_premium": float(policies_df["premium"].mean())
        }
        
        # Additional statistics if frequency/severity are available
        if "frequency_type" in policies_df.columns:
            freq_dist = policies_df["frequency_type"].value_counts().to_dict()
            stats["frequency_distribution"] = freq_dist
        
        if "severity_type" in policies_df.columns:
            sev_dist = policies_df["severity_type"].value_counts().to_dict()
            stats["severity_distribution"] = sev_dist
        
        return DataFormats.portfolio_result(
            portfolio_id=f"portfolio_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            statistics=stats,
            metadata={
                "created_at": pd.Timestamp.now().isoformat(),
                "validated": validate
            }
        )
        
    except ImportError as e:
        raise MCPExecutionError(f"Failed to import required modules: {str(e)}")
    except Exception as e:
        logger.error(f"Error in create_portfolio: {str(e)}")
        raise MCPExecutionError(f"Portfolio creation failed: {str(e)}")


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
