# mcp/server.py
"""Main MCP server that imports and wraps quactuary package functionality."""

import logging
from mcp.server.fastmcp import FastMCP

from .config import MCP_CONFIG
from .prompts import register_prompts
from .resources import register_resources
from .tools import register_tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name=MCP_CONFIG["server_name"],
    version=MCP_CONFIG["version"]
)

# Register all components
register_tools(mcp)
register_resources(mcp)
register_prompts(mcp)

# Log initialization
logger.info(f"Initialized {MCP_CONFIG['server_name']} v{MCP_CONFIG['version']}")

def main():
    """Main entry point for the MCP server."""
    import asyncio
    
    # Run the server
    asyncio.run(mcp.run(transport="stdio"))

if __name__ == "__main__":
    main()
