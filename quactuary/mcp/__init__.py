# mcp/__init__.py
"""Model Context Protocol server for quActuary package."""

try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.1"

MCP_API_VERSION = "2025-05-26"  # Date-based API versioning

# Make the server accessible
from .server import mcp

__all__ = ["mcp", "__version__", "MCP_API_VERSION"]
