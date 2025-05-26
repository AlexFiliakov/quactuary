"""Standard input/output formats and error handling for MCP tools."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import traceback


@dataclass
class MCPToolInput:
    """Standard input format for MCP tools."""
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self, required_params: List[str]) -> None:
        """Validate that all required parameters are present."""
        missing = [p for p in required_params if p not in self.parameters]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.parameters.get(name, default)


@dataclass
class MCPToolOutput:
    """Standard output format for MCP tools."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def success_response(cls, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> "MCPToolOutput":
        """Create a successful response."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> "MCPToolOutput":
        """Create an error response."""
        return cls(success=False, error=error, metadata=metadata)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPValidationError(MCPError):
    """Error raised when input validation fails."""
    pass


class MCPExecutionError(MCPError):
    """Error raised during tool execution."""
    pass


def handle_tool_error(func):
    """Decorator for consistent error handling in tools."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MCPValidationError as e:
            return MCPToolOutput.error_response(
                f"Validation error: {str(e)}",
                metadata={"error_type": "validation"}
            )
        except MCPExecutionError as e:
            return MCPToolOutput.error_response(
                f"Execution error: {str(e)}",
                metadata={"error_type": "execution"}
            )
        except Exception as e:
            return MCPToolOutput.error_response(
                f"Unexpected error: {str(e)}",
                metadata={
                    "error_type": "unexpected",
                    "traceback": traceback.format_exc()
                }
            )
    return wrapper


# Standard data formats
class DataFormats:
    """Standard data format specifications."""
    
    @staticmethod
    def distribution_spec(dist_type: str, params: Dict[str, float]) -> Dict[str, Any]:
        """Standard format for distribution specifications."""
        return {
            "type": dist_type,
            "parameters": params,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "format_version": "1.0"
            }
        }
    
    @staticmethod
    def portfolio_spec(policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard format for portfolio specifications."""
        return {
            "policies": policies,
            "count": len(policies),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "format_version": "1.0"
            }
        }
    
    @staticmethod
    def simulation_result(estimates: Dict[str, float], 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standard format for simulation results."""
        result = {
            "estimates": estimates,
            "metadata": metadata or {}
        }
        result["metadata"]["created_at"] = datetime.now().isoformat()
        result["metadata"]["format_version"] = "1.0"
        return result


# Input validation helpers
def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise MCPValidationError(f"{name} must be positive, got {value}")


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability."""
    if not 0 <= value <= 1:
        raise MCPValidationError(f"{name} must be between 0 and 1, got {value}")


def validate_date_format(date_str: str) -> None:
    """Validate ISO date format."""
    try:
        datetime.fromisoformat(date_str)
    except ValueError:
        raise MCPValidationError(f"Invalid date format: {date_str}, expected ISO format (YYYY-MM-DD)")