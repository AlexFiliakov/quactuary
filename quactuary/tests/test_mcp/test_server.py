# tests/test_mcp/test_server.py
"""Test MCP server integration"""

import json

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_mcp_version_agreement():
    """Test that MCP version matches package version"""

    from quactuary.mcp import __version__ as mcp_version
    from quactuary.mcp.config import MCP_CONFIG

    assert mcp_version == MCP_CONFIG["version"], \
        f"MCP version mismatch: {mcp_version} != {MCP_CONFIG['version']}"


@pytest.mark.asyncio
def test_mcp_server_version():
    """Test that MCP server version matches expected format"""

    server_params = StdioServerParameters(
        command="python",
        args=["mcp/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Check server version
            assert session.server_version is not None
            assert isinstance(session.server_version, str)
            assert session.server_version.startswith("0.0.")


@pytest.mark.asyncio
async def test_mcp_server_tools():
    """Test that MCP server exposes expected tools"""

    server_params = StdioServerParameters(
        command="python",
        args=["mcp/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test tool discovery
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools]

            assert "calculate_loss_ratio" in tool_names
            assert "run_chain_ladder" in tool_names
            assert "price_policy" in tool_names

            # Test tool execution
            result = await session.call_tool(
                "calculate_loss_ratio",
                arguments={
                    "earned_premium": 1000000,
                    "incurred_losses": 600000,
                    "alae_ratio": 0.1
                }
            )

            # Parse result
            result_data = json.loads(result[0].text)
            assert result_data["loss_ratio"] == 0.6
            assert "alae_amount" in result_data


@pytest.mark.asyncio
async def test_mcp_resources():
    """Test MCP resource access"""

    server_params = StdioServerParameters(
        command="python",
        args=["mcp/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test resource listing
            resources = await session.list_resources()
            assert len(resources) > 0

            # Test resource access
            content, mime_type = await session.read_resource("rates://base/gl")
            assert content is not None
            assert "base_rate" in content
