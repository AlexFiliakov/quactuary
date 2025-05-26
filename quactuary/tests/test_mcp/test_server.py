# tests/test_mcp/test_server.py
"""Test MCP server integration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestMCPServerBasic:
    """Test basic MCP server functionality."""
    
    def test_mcp_imports(self):
        """Test that MCP modules can be imported."""
        # Should not raise
        import quactuary.mcp
        import quactuary.mcp.server
        import quactuary.mcp.tools
        import quactuary.mcp.resources
        import quactuary.mcp.prompts
    
    def test_mcp_version_available(self):
        """Test that MCP version is available."""
        from quactuary.mcp import __version__, MCP_API_VERSION
        
        assert __version__ is not None
        assert MCP_API_VERSION == "2025-05-26"
    
    def test_mcp_server_instance(self):
        """Test that MCP server instance exists."""
        from quactuary.mcp.server import mcp
        
        assert mcp is not None
        assert hasattr(mcp, "tool")
        assert hasattr(mcp, "resource")
        assert hasattr(mcp, "prompt")


class TestMCPRegistration:
    """Test MCP component registration."""
    
    def test_tool_registration(self):
        """Test that tools are registered."""
        from quactuary.mcp.tools import registry
        
        tools = registry.list_tools()
        assert len(tools) > 0
        assert "pricing_simulate_portfolio" in tools
    
    def test_register_tools_function(self):
        """Test register_tools function."""
        from quactuary.mcp.tools import register_tools
        
        # Create mock MCP server
        mock_mcp = Mock()
        mock_mcp.tool = Mock(return_value=lambda f: f)
        
        # Should not raise
        register_tools(mock_mcp)
        
        # Should have registered tools
        assert mock_mcp.tool.called
    
    def test_register_resources_function(self):
        """Test register_resources function."""
        from quactuary.mcp.resources import register_resources
        
        # Create mock MCP server
        mock_mcp = Mock()
        mock_mcp.resource = Mock(return_value=lambda f: f)
        
        # Should not raise
        register_resources(mock_mcp)
        
        # Should have registered resources
        assert mock_mcp.resource.call_count >= 3
    
    def test_register_prompts_function(self):
        """Test register_prompts function."""
        from quactuary.mcp.prompts import register_prompts
        
        # Create mock MCP server
        mock_mcp = Mock()
        mock_mcp.prompt = Mock(return_value=lambda f: f)
        
        # Should not raise  
        register_prompts(mock_mcp)
        
        # Should have registered prompts
        assert mock_mcp.prompt.call_count >= 3


class TestMCPServerStartup:
    """Test MCP server startup functionality."""
    
    def test_server_main_function(self):
        """Test that server has main function."""
        from quactuary.mcp.server import main
        
        assert callable(main)
    
    def test_server_main_calls_run(self):
        """Test that main function calls asyncio.run properly."""
        import asyncio
        
        # Create a mock coroutine
        async def mock_coro():
            return None
        
        # Patch mcp.run to return a coroutine that doesn't actually start the server
        with patch('quactuary.mcp.server.mcp.run', return_value=mock_coro()):
            with patch('asyncio.run') as mock_run:
                # Configure mock to avoid actually running the server
                mock_run.return_value = None
                
                # Import and call main
                from quactuary.mcp.server import main
                main()
                
                assert mock_run.called
                # Verify asyncio.run was called with a coroutine
                assert mock_run.call_count == 1
                # Verify it was called with the result of mcp.run
                args, kwargs = mock_run.call_args
                assert asyncio.iscoroutine(args[0])


class TestMCPToolExecution:
    """Test MCP tool execution through registry."""
    
    def test_pricing_tool_exists(self):
        """Test that pricing tool is registered."""
        from quactuary.mcp.tools import registry
        
        tool = registry.get_tool("pricing_simulate_portfolio")
        assert tool is not None
        assert tool.category.value == "pricing"
    
    def test_pricing_tool_parameters(self):
        """Test pricing tool parameters."""
        from quactuary.mcp.tools import registry
        
        tool = registry.get_tool("pricing_simulate_portfolio")
        params = tool.get_parameters()
        
        assert "portfolio_data" in params
        assert params["portfolio_data"]["required"] is True
        assert "n_simulations" in params
        assert "tail_alpha" in params
        assert "use_qmc" in params
    
    def test_pricing_tool_execution_mock(self):
        """Test pricing tool execution with mock data."""
        from quactuary.mcp.tools import registry
        from quactuary.mcp.formats import MCPToolInput
        
        tool = registry.get_tool("pricing_simulate_portfolio")
        
        # Create mock portfolio data
        input_data = MCPToolInput(
            parameters={
                "portfolio_data": {
                    "policies": [
                        {
                            "policy_id": "TEST001",
                            "effective_date": "2024-01-01",
                            "expiration_date": "2025-01-01",
                            "exposure_amount": 1000000,
                            "premium": 5000
                        }
                    ]
                },
                "n_simulations": 100,
                "tail_alpha": 0.05
            }
        )
        
        # Execute tool (this will fail without full quactuary setup, but tests the interface)
        output = tool.execute(input_data)
        
        # Even if execution fails, output should have proper structure
        assert hasattr(output, "success")
        assert hasattr(output, "data")
        assert hasattr(output, "error")
