"""Pytest configuration for MCP tests."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_mcp_server():
    """Fixture for creating a mock MCP server."""
    server = Mock()
    server.name = "Test MCP Server"
    server.version = "0.0.1"
    server.tools = {}
    server.resources = {}
    server.prompts = {}
    return server


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()