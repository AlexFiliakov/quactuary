# mcp/config.py
"""MCP-specific configuration"""

try:
    from .. import __version__
    version = __version__
except ImportError:
    version = "0.0.1"

MCP_CONFIG = {
    "server_name": "quActuary MCP Server",
    "version": version,
    "dependencies": [
        "numpy>=2.2.5",
        "pandas>=2.2.3", 
        "scipy>=1.14.1",
        "numba>=0.56.0"
    ],
    "max_workers": 4,  # For async operations
    "timeout_seconds": 300,  # For long-running simulations
}

# Tool-specific limits
TOOL_LIMITS = {
    "monte_carlo_simulation": {
        "max_iterations": 10_000_000,
        "default_iterations": 10_000
    },
    "batch_operations": {
        "max_batch_size": 1000
    }
}
