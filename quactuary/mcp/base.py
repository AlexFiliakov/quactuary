"""Base classes for MCP tool implementation."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import inspect
import logging

from .categories import ToolCategory, get_tool_category
from .formats import MCPToolInput, MCPToolOutput, handle_tool_error

logger = logging.getLogger(__name__)


class MCPTool(ABC):
    """Base class for MCP tools."""
    
    def __init__(self, name: str, description: str, category: ToolCategory):
        """Initialize the tool."""
        self.name = name
        self.description = description
        self.category = category
        self._validate_name()
    
    def _validate_name(self):
        """Validate that the tool name matches its category."""
        expected_category = get_tool_category(self.name)
        if expected_category != self.category:
            raise ValueError(
                f"Tool name '{self.name}' doesn't match category {self.category}"
            )
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter definitions for the tool.
        
        Returns:
            Dict mapping parameter names to their specifications::
            
                {
                    "param_name": {
                        "type": "string|number|boolean|object|array",
                        "description": "Parameter description",
                        "required": True/False,
                        "default": <default_value> (optional)
                    }
                }
        
        """
        pass
    
    @abstractmethod
    @handle_tool_error
    def execute(self, input_data: MCPToolInput) -> MCPToolOutput:
        """Execute the tool with given input.
        
        Args:
            input_data: Standardized input data
            
        Returns:
            MCPToolOutput with results or error
        """
        pass


class ToolRegistry:
    """Registry for managing MCP tools."""
    
    def __init__(self):
        """Initialize the registry."""
        self._tools: Dict[str, MCPTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
    
    def register(self, tool: MCPTool) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List available tools, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def get_tools_by_category(self) -> Dict[ToolCategory, List[str]]:
        """Get all tools organized by category."""
        return {
            cat: tools.copy() 
            for cat, tools in self._categories.items() 
            if tools
        }


class SimpleTool(MCPTool):
    """Simple tool implementation for function-based tools."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 category: ToolCategory,
                 func: Callable,
                 parameters: Dict[str, Dict[str, Any]]):
        """Initialize a simple function-based tool."""
        super().__init__(name, description, category)
        self.func = func
        self.parameters = parameters
        self._validate_function_signature()
    
    def _validate_function_signature(self):
        """Validate that function parameters match declared parameters."""
        sig = inspect.signature(self.func)
        func_params = set(sig.parameters.keys())
        declared_params = set(self.parameters.keys())
        
        # Check for missing parameters
        missing = func_params - declared_params
        if missing:
            logger.warning(
                f"Function '{self.func.__name__}' has undeclared parameters: {missing}"
            )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter definitions."""
        return self.parameters.copy()
    
    @handle_tool_error
    def execute(self, input_data: MCPToolInput) -> MCPToolOutput:
        """Execute the tool function."""
        # Validate required parameters
        required = [
            name for name, spec in self.parameters.items()
            if spec.get("required", False)
        ]
        input_data.validate(required)
        
        # Extract parameters for function call
        kwargs = {}
        for param_name in self.parameters:
            if param_name in input_data.parameters:
                kwargs[param_name] = input_data.parameters[param_name]
            elif "default" in self.parameters[param_name]:
                kwargs[param_name] = self.parameters[param_name]["default"]
        
        # Execute function
        result = self.func(**kwargs)
        
        # Return standardized output
        return MCPToolOutput.success_response(
            data=result if isinstance(result, dict) else {"result": result},
            metadata={"tool": self.name}
        )


def create_tool_decorator(registry: ToolRegistry, category: ToolCategory):
    """Create a decorator for registering tools in a specific category."""
    
    def tool_decorator(name: str = None, 
                      description: str = None,
                      parameters: Dict[str, Dict[str, Any]] = None):
        """Decorator for registering a tool function."""
        
        def decorator(func: Callable) -> Callable:
            # Use function name if tool name not provided
            # Get the correct prefix for the category
            from .categories import get_category_prefix
            prefix = get_category_prefix(category)
            tool_name = name or f"{prefix}{func.__name__}"
            
            # Use docstring if description not provided
            tool_description = description or (func.__doc__ or "").strip()
            
            # Create and register the tool
            tool = SimpleTool(
                name=tool_name,
                description=tool_description,
                category=category,
                func=func,
                parameters=parameters or {}
            )
            registry.register(tool)
            
            # Return the original function
            return func
        
        return decorator
    
    return tool_decorator