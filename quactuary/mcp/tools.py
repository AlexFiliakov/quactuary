# mcp/tools.py
"""Tool definitions for MCP server"""


def register_tools(mcp):
    # TODO: Define the tools here
    @mcp.tool()
    def calculate_aggregate_statistics(triangle_data: list) -> dict:
        """Calculate aggregate statistics using Classical Monte Carlo methods"""
        from .. import pricing
        return pricing.PricingModel.calculate_aggregate_statistics(...)

    # Add tools...
