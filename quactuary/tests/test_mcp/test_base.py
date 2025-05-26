"""Test MCP base classes and tool registration."""

import pytest
from unittest.mock import Mock
from quactuary.mcp.base import (
    MCPTool,
    ToolRegistry,
    SimpleTool,
    create_tool_decorator
)
from quactuary.mcp.categories import ToolCategory
from quactuary.mcp.formats import MCPToolInput, MCPToolOutput


class MockTool(MCPTool):
    """Mock tool for testing."""
    
    def __init__(self):
        super().__init__(
            name="pricing_test_tool",
            description="Test tool",
            category=ToolCategory.PRICING
        )
    
    def get_parameters(self):
        return {
            "param1": {
                "type": "string",
                "description": "Test parameter",
                "required": True
            }
        }
    
    def execute(self, input_data: MCPToolInput) -> MCPToolOutput:
        return MCPToolOutput.success_response({"result": "test"})


class TestMCPTool:
    """Test MCPTool base class."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        tool = MockTool()
        assert tool.name == "pricing_test_tool"
        assert tool.description == "Test tool"
        assert tool.category == ToolCategory.PRICING
    
    def test_tool_name_validation_success(self):
        """Test tool name validation - success."""
        # Should not raise
        tool = MockTool()
        assert tool.name.startswith("pricing_")
    
    def test_tool_name_validation_failure(self):
        """Test tool name validation - failure."""
        with pytest.raises(ValueError, match="Unknown tool category"):
            class BadTool(MCPTool):
                def __init__(self):
                    super().__init__(
                        name="wrong_prefix_tool",
                        description="Bad tool",
                        category=ToolCategory.PRICING
                    )
                
                def get_parameters(self):
                    return {}
                
                def execute(self, input_data):
                    return None
            
            BadTool()
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            MCPTool("test", "test", ToolCategory.PRICING)


class TestToolRegistry:
    """Test ToolRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = ToolRegistry()
        assert len(registry.list_tools()) == 0
        assert all(len(tools) == 0 for tools in registry.get_tools_by_category().values())
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        assert len(registry.list_tools()) == 1
        assert "pricing_test_tool" in registry.list_tools()
        assert registry.get_tool("pricing_test_tool") == tool
    
    def test_register_duplicate_tool(self):
        """Test registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)
    
    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None
    
    def test_list_tools_by_category(self):
        """Test listing tools by category."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        
        # List all tools
        all_tools = registry.list_tools()
        assert len(all_tools) == 1
        
        # List by specific category
        pricing_tools = registry.list_tools(ToolCategory.PRICING)
        assert len(pricing_tools) == 1
        assert "pricing_test_tool" in pricing_tools
        
        # Empty category
        dist_tools = registry.list_tools(ToolCategory.DISTRIBUTIONS)
        assert len(dist_tools) == 0
    
    def test_get_tools_by_category(self):
        """Test getting tools organized by category."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        
        by_category = registry.get_tools_by_category()
        
        # Should only have pricing category with tools
        assert ToolCategory.PRICING in by_category
        assert "pricing_test_tool" in by_category[ToolCategory.PRICING]
        
        # Other categories should not be in result (empty)
        assert ToolCategory.DISTRIBUTIONS not in by_category


class TestSimpleTool:
    """Test SimpleTool implementation."""
    
    def test_simple_tool_creation(self):
        """Test creating a simple tool."""
        def my_func(param1: str, param2: int = 10):
            return {"result": f"{param1}-{param2}"}
        
        params = {
            "param1": {"type": "string", "required": True},
            "param2": {"type": "number", "required": False, "default": 10}
        }
        
        tool = SimpleTool(
            name="util_test",
            description="Test utility",
            category=ToolCategory.UTILITIES,
            func=my_func,
            parameters=params
        )
        
        assert tool.name == "util_test"
        assert tool.func == my_func
        assert tool.get_parameters() == params
    
    def test_simple_tool_execution_success(self):
        """Test successful tool execution."""
        def add_numbers(a: float, b: float):
            return {"sum": a + b}
        
        tool = SimpleTool(
            name="util_add",
            description="Add numbers",
            category=ToolCategory.UTILITIES,
            func=add_numbers,
            parameters={
                "a": {"type": "number", "required": True},
                "b": {"type": "number", "required": True}
            }
        )
        
        input_data = MCPToolInput(parameters={"a": 5, "b": 3})
        output = tool.execute(input_data)
        
        assert output.success is True
        assert output.data["sum"] == 8
        assert output.metadata["tool"] == "util_add"
    
    def test_simple_tool_with_defaults(self):
        """Test tool execution with default parameters."""
        def greet(name: str, greeting: str = "Hello"):
            return {"message": f"{greeting}, {name}!"}
        
        tool = SimpleTool(
            name="util_greet",
            description="Greeting tool",
            category=ToolCategory.UTILITIES,
            func=greet,
            parameters={
                "name": {"type": "string", "required": True},
                "greeting": {"type": "string", "required": False, "default": "Hello"}
            }
        )
        
        # Without optional parameter
        input_data = MCPToolInput(parameters={"name": "World"})
        output = tool.execute(input_data)
        assert output.data["message"] == "Hello, World!"
        
        # With optional parameter
        input_data = MCPToolInput(parameters={"name": "World", "greeting": "Hi"})
        output = tool.execute(input_data)
        assert output.data["message"] == "Hi, World!"
    
    def test_simple_tool_missing_required(self):
        """Test tool execution with missing required parameter."""
        def dummy_func(required_param: str):
            return {"result": required_param}
        
        tool = SimpleTool(
            name="util_dummy",
            description="Dummy tool",
            category=ToolCategory.UTILITIES,
            func=dummy_func,
            parameters={
                "required_param": {"type": "string", "required": True}
            }
        )
        
        input_data = MCPToolInput(parameters={})
        output = tool.execute(input_data)
        
        assert output.success is False
        # The error is caught as unexpected due to ValueError not being MCPValidationError
        assert "Missing required parameters" in output.error


class TestCreateToolDecorator:
    """Test create_tool_decorator function."""
    
    def test_decorator_creation(self):
        """Test creating a tool decorator."""
        registry = ToolRegistry()
        pricing_tool = create_tool_decorator(registry, ToolCategory.PRICING)
        
        @pricing_tool(
            description="Test pricing tool",
            parameters={"value": {"type": "number", "required": True}}
        )
        def calculate_something(value: float):
            return {"result": value * 2}
        
        # Tool should be registered
        assert len(registry.list_tools()) == 1
        tool_name = "pricing_calculate_something"
        assert tool_name in registry.list_tools()
        
        # Test execution
        tool = registry.get_tool(tool_name)
        input_data = MCPToolInput(parameters={"value": 5})
        output = tool.execute(input_data)
        
        assert output.success is True
        assert output.data["result"] == 10
    
    def test_decorator_with_custom_name(self):
        """Test decorator with custom tool name."""
        registry = ToolRegistry()
        dist_tool = create_tool_decorator(registry, ToolCategory.DISTRIBUTIONS)
        
        @dist_tool(
            name="dist_custom_name",
            description="Custom named tool"
        )
        def my_distribution_func():
            return {"status": "ok"}
        
        assert "dist_custom_name" in registry.list_tools()
        assert "dist_my_distribution_func" not in registry.list_tools()
    
    def test_decorator_uses_docstring(self):
        """Test that decorator uses function docstring."""
        registry = ToolRegistry()
        util_tool = create_tool_decorator(registry, ToolCategory.UTILITIES)
        
        @util_tool(parameters={})
        def documented_function():
            """This is a well-documented function."""
            return {"done": True}
        
        # Tool name should use the prefix from utilities category
        tool = registry.get_tool("util_documented_function")
        assert tool.description == "This is a well-documented function."