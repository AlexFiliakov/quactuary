"""Test MCP format definitions and error handling."""

import pytest
import json
from datetime import datetime
from quactuary.mcp.formats import (
    MCPToolInput,
    MCPToolOutput,
    MCPError,
    MCPValidationError,
    MCPExecutionError,
    handle_tool_error,
    DataFormats,
    validate_positive,
    validate_probability,
    validate_date_format
)


class TestMCPToolInput:
    """Test MCPToolInput class."""
    
    def test_input_creation(self):
        """Test creating tool input."""
        params = {"param1": "value1", "param2": 42}
        metadata = {"source": "test"}
        
        input_data = MCPToolInput(parameters=params, metadata=metadata)
        assert input_data.parameters == params
        assert input_data.metadata == metadata
    
    def test_input_validation_success(self):
        """Test successful parameter validation."""
        input_data = MCPToolInput(parameters={"required1": 1, "required2": "test"})
        
        # Should not raise
        input_data.validate(["required1", "required2"])
    
    def test_input_validation_missing(self):
        """Test validation with missing parameters."""
        input_data = MCPToolInput(parameters={"param1": 1})
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            input_data.validate(["param1", "param2"])
    
    def test_get_param(self):
        """Test parameter retrieval."""
        input_data = MCPToolInput(parameters={"param1": "value1"})
        
        assert input_data.get_param("param1") == "value1"
        assert input_data.get_param("param2") is None
        assert input_data.get_param("param2", "default") == "default"


class TestMCPToolOutput:
    """Test MCPToolOutput class."""
    
    def test_success_response(self):
        """Test creating successful response."""
        data = {"result": 42}
        metadata = {"tool": "test"}
        
        output = MCPToolOutput.success_response(data, metadata)
        assert output.success is True
        assert output.data == data
        assert output.metadata == metadata
        assert output.error is None
    
    def test_error_response(self):
        """Test creating error response."""
        error_msg = "Something went wrong"
        metadata = {"error_type": "test"}
        
        output = MCPToolOutput.error_response(error_msg, metadata)
        assert output.success is False
        assert output.error == error_msg
        assert output.metadata == metadata
        assert output.data is None
    
    def test_to_json(self):
        """Test JSON serialization."""
        output = MCPToolOutput.success_response({"result": 42})
        json_str = output.to_json()
        
        parsed = json.loads(json_str)
        assert parsed["success"] is True
        assert parsed["data"]["result"] == 42


class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_mcp_error_hierarchy(self):
        """Test error class hierarchy."""
        assert issubclass(MCPValidationError, MCPError)
        assert issubclass(MCPExecutionError, MCPError)
    
    def test_handle_tool_error_decorator_validation(self):
        """Test error handler with validation error."""
        @handle_tool_error
        def tool_with_validation_error():
            raise MCPValidationError("Invalid input")
        
        output = tool_with_validation_error()
        assert output.success is False
        assert "Validation error" in output.error
        assert output.metadata["error_type"] == "validation"
    
    def test_handle_tool_error_decorator_execution(self):
        """Test error handler with execution error."""
        @handle_tool_error
        def tool_with_execution_error():
            raise MCPExecutionError("Failed to execute")
        
        output = tool_with_execution_error()
        assert output.success is False
        assert "Execution error" in output.error
        assert output.metadata["error_type"] == "execution"
    
    def test_handle_tool_error_decorator_unexpected(self):
        """Test error handler with unexpected error."""
        @handle_tool_error
        def tool_with_unexpected_error():
            raise RuntimeError("Unexpected!")
        
        output = tool_with_unexpected_error()
        assert output.success is False
        assert "Unexpected error" in output.error
        assert output.metadata["error_type"] == "unexpected"
        assert "traceback" in output.metadata


class TestDataFormats:
    """Test standard data format functions."""
    
    def test_distribution_spec(self):
        """Test distribution specification format."""
        spec = DataFormats.distribution_spec("Poisson", {"lambda": 5.0})
        
        assert spec["type"] == "Poisson"
        assert spec["parameters"]["lambda"] == 5.0
        assert "metadata" in spec
        assert "created_at" in spec["metadata"]
        assert spec["metadata"]["format_version"] == "1.0"
    
    def test_portfolio_spec(self):
        """Test portfolio specification format."""
        policies = [
            {"policy_id": "P1", "premium": 1000},
            {"policy_id": "P2", "premium": 2000}
        ]
        
        spec = DataFormats.portfolio_spec(policies)
        assert spec["policies"] == policies
        assert spec["count"] == 2
        assert "metadata" in spec
        assert spec["metadata"]["format_version"] == "1.0"
    
    def test_simulation_result(self):
        """Test simulation result format."""
        estimates = {"mean": 1000, "std": 100}
        metadata = {"n_sims": 10000}
        
        result = DataFormats.simulation_result(estimates, metadata)
        assert result["estimates"] == estimates
        assert result["metadata"]["n_sims"] == 10000
        assert "created_at" in result["metadata"]


class TestValidationHelpers:
    """Test validation helper functions."""
    
    def test_validate_positive_success(self):
        """Test positive value validation - success."""
        validate_positive(1, "test_value")
        validate_positive(0.1, "test_value")
        validate_positive(100, "test_value")
    
    def test_validate_positive_failure(self):
        """Test positive value validation - failure."""
        with pytest.raises(MCPValidationError, match="must be positive"):
            validate_positive(0, "test_value")
        
        with pytest.raises(MCPValidationError, match="must be positive"):
            validate_positive(-1, "test_value")
    
    def test_validate_probability_success(self):
        """Test probability validation - success."""
        validate_probability(0.0, "test_prob")
        validate_probability(0.5, "test_prob")
        validate_probability(1.0, "test_prob")
    
    def test_validate_probability_failure(self):
        """Test probability validation - failure."""
        with pytest.raises(MCPValidationError, match="between 0 and 1"):
            validate_probability(-0.1, "test_prob")
        
        with pytest.raises(MCPValidationError, match="between 0 and 1"):
            validate_probability(1.1, "test_prob")
    
    def test_validate_date_format_success(self):
        """Test date format validation - success."""
        validate_date_format("2024-01-01")
        validate_date_format("2024-12-31")
        validate_date_format("2024-01-01T12:00:00")
    
    def test_validate_date_format_failure(self):
        """Test date format validation - failure."""
        with pytest.raises(MCPValidationError, match="Invalid date format"):
            validate_date_format("01/01/2024")
        
        with pytest.raises(MCPValidationError, match="Invalid date format"):
            validate_date_format("2024-13-01")
        
        with pytest.raises(MCPValidationError, match="Invalid date format"):
            validate_date_format("not-a-date")