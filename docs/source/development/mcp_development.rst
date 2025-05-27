=======================
MCP Tool Development
=======================

This guide covers developing new tools for the quActuary MCP (Model Context Protocol) Server, which enables LLM assistants like Claude to use actuarial modeling capabilities.

Overview
--------

The MCP server is built using FastMCP and provides a standardized interface for exposing quActuary functionality to AI assistants. Tools are organized into categories and follow consistent patterns for parameter validation and error handling.

Architecture
------------

The MCP implementation consists of several key components:

**Core Components**

- ``server.py`` - Main server implementation using FastMCP
- ``tools.py`` - Tool implementations
- ``base.py`` - Base classes and interfaces
- ``categories.py`` - Tool categorization system
- ``formats.py`` - Data formatting utilities
- ``config.py`` - Configuration management

**Tool Categories**

Tools are organized into four categories:

1. **PRICING** - Pricing simulations and calculations
2. **DISTRIBUTIONS** - Distribution creation and manipulation
3. **PORTFOLIO** - Portfolio management and analysis
4. **UTILITIES** - General utilities and metrics

Creating a New Tool
-------------------

Step 1: Define the Tool Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new tool by subclassing ``QuactuaryTool``:

.. code-block:: python

   from quactuary.mcp.base import QuactuaryTool
   from quactuary.mcp.categories import ToolCategory
   from typing import Dict, Any

   class CalculateReserveTool(QuactuaryTool):
       """Calculate reserve requirements for a portfolio."""
       
       category = ToolCategory.PRICING
       name = "calculate_reserve"
       description = "Calculate reserve requirements based on confidence level"
       
       # Define parameter schema
       parameters_schema = {
           "type": "object",
           "properties": {
               "portfolio_id": {
                   "type": "string",
                   "description": "ID of the portfolio"
               },
               "confidence_level": {
                   "type": "number",
                   "description": "Confidence level (0-1)",
                   "minimum": 0,
                   "maximum": 1
               },
               "time_horizon": {
                   "type": "integer",
                   "description": "Time horizon in years",
                   "minimum": 1
               }
           },
           "required": ["portfolio_id", "confidence_level"]
       }

Step 2: Implement the Execute Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
       """Execute the reserve calculation."""
       try:
           # Validate parameters
           portfolio_id = params["portfolio_id"]
           confidence_level = params["confidence_level"]
           time_horizon = params.get("time_horizon", 1)
           
           # Load portfolio (example)
           portfolio = self.load_portfolio(portfolio_id)
           
           # Perform calculation
           from quactuary.pricing import PricingModel
           model = PricingModel(portfolio)
           
           # Run simulation with specified parameters
           result = model.simulate(
               n_simulations=100_000,
               time_horizon=time_horizon
           )
           
           # Calculate reserve at confidence level
           import numpy as np
           quantile = np.quantile(
               result.total_losses, 
               confidence_level
           )
           
           return {
               "status": "success",
               "reserve_amount": float(quantile),
               "confidence_level": confidence_level,
               "time_horizon": time_horizon,
               "expected_loss": float(result.estimates["mean"]),
               "portfolio_id": portfolio_id
           }
           
       except Exception as e:
           return {
               "status": "error",
               "error": str(e),
               "error_type": type(e).__name__
           }

Step 3: Register the Tool
~~~~~~~~~~~~~~~~~~~~~~~~~

Add your tool to the server in ``tools.py``:

.. code-block:: python

   # In tools.py
   from .calculate_reserve import CalculateReserveTool

   # Register with the server
   def register_tools(server):
       """Register all tools with the MCP server."""
       tools = [
           # Existing tools...
           CalculateReserveTool(),
       ]
       
       for tool in tools:
           server.register_tool(tool)

Best Practices
--------------

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

Always validate parameters thoroughly:

.. code-block:: python

   def validate_params(self, params: Dict[str, Any]) -> None:
       """Validate tool parameters."""
       # Check required fields
       if "portfolio_id" not in params:
           raise ValueError("portfolio_id is required")
       
       # Validate types and ranges
       confidence = params.get("confidence_level")
       if not isinstance(confidence, (int, float)):
           raise TypeError("confidence_level must be numeric")
       
       if not 0 <= confidence <= 1:
           raise ValueError("confidence_level must be between 0 and 1")

Error Handling
~~~~~~~~~~~~~~

Implement comprehensive error handling:

.. code-block:: python

   def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
       try:
           # Validate first
           self.validate_params(params)
           
           # Perform operation
           result = self.perform_calculation(params)
           
           return {
               "status": "success",
               "result": result
           }
           
       except ValueError as e:
           # Parameter validation errors
           return {
               "status": "error",
               "error": str(e),
               "error_type": "validation_error"
           }
           
       except MemoryError:
           # Resource limitations
           return {
               "status": "error",
               "error": "Insufficient memory for calculation",
               "error_type": "resource_error",
               "suggestion": "Try reducing simulation size"
           }
           
       except Exception as e:
           # Unexpected errors
           import logging
           logging.error(f"Tool error: {e}", exc_info=True)
           
           return {
               "status": "error",
               "error": "An unexpected error occurred",
               "error_type": "internal_error"
           }

Consistent Output Format
~~~~~~~~~~~~~~~~~~~~~~~~

Follow consistent patterns for output:

.. code-block:: python

   # Success response
   {
       "status": "success",
       "result": {
           "primary_metric": value,
           "secondary_metrics": {...},
           "metadata": {
               "calculation_time": 1.23,
               "parameters_used": {...}
           }
       }
   }
   
   # Error response
   {
       "status": "error",
       "error": "Description of what went wrong",
       "error_type": "category_of_error",
       "suggestion": "How to fix it (optional)"
   }

Documentation
~~~~~~~~~~~~~

Document your tools thoroughly:

.. code-block:: python

   class MyTool(QuactuaryTool):
       """One-line summary of the tool.
       
       Detailed description of what the tool does, when to use it,
       and any important considerations.
       
       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type, optional
           Description of param2 (default: value)
           
       Returns
       -------
       dict
           Dictionary containing:
           - key1: Description
           - key2: Description
           
       Examples
       --------
       >>> tool = MyTool()
       >>> result = tool.execute({"param1": value})
       >>> print(result["key1"])
       """

Testing MCP Tools
-----------------

Unit Tests
~~~~~~~~~~

Write comprehensive unit tests for each tool:

.. code-block:: python

   # tests/test_calculate_reserve.py
   import pytest
   from quactuary.mcp.tools import CalculateReserveTool

   class TestCalculateReserveTool:
       def setup_method(self):
           self.tool = CalculateReserveTool()
       
       def test_valid_calculation(self):
           params = {
               "portfolio_id": "test_portfolio",
               "confidence_level": 0.95,
               "time_horizon": 1
           }
           result = self.tool.execute(params)
           
           assert result["status"] == "success"
           assert "reserve_amount" in result
           assert result["confidence_level"] == 0.95
       
       def test_missing_required_param(self):
           params = {"confidence_level": 0.95}
           result = self.tool.execute(params)
           
           assert result["status"] == "error"
           assert "portfolio_id" in result["error"]
       
       def test_invalid_confidence_level(self):
           params = {
               "portfolio_id": "test",
               "confidence_level": 1.5
           }
           result = self.tool.execute(params)
           
           assert result["status"] == "error"
           assert "confidence_level" in result["error"]

Integration Tests
~~~~~~~~~~~~~~~~~

Test tool integration with the MCP server:

.. code-block:: python

   # tests/test_mcp_integration.py
   import asyncio
   from quactuary.mcp.server import create_server

   async def test_tool_registration():
       server = create_server()
       
       # Check tool is registered
       tools = await server.list_tools()
       tool_names = [t.name for t in tools]
       
       assert "calculate_reserve" in tool_names

Manual Testing
~~~~~~~~~~~~~~

Test your tool in Claude Desktop:

1. Start the MCP server with your new tool
2. Open Claude Desktop and ensure it connects
3. Test the tool with various inputs:

   .. code-block:: text

      User: Calculate the reserve requirement for portfolio ABC123 
            at 95% confidence level over 2 years.

4. Verify error handling:

   .. code-block:: text

      User: Calculate reserve for portfolio XYZ at 150% confidence.

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

For tools that process large datasets:

.. code-block:: python

   def execute(self, params):
       # Use context managers for large objects
       with self.load_large_dataset(params["dataset_id"]) as data:
           # Process in chunks if needed
           chunk_size = 10000
           results = []
           
           for i in range(0, len(data), chunk_size):
               chunk = data[i:i + chunk_size]
               results.append(self.process_chunk(chunk))
           
           # Aggregate results
           return self.aggregate_results(results)

Caching
~~~~~~~

Implement caching for expensive operations:

.. code-block:: python

   from functools import lru_cache
   
   class MyTool(QuactuaryTool):
       @lru_cache(maxsize=100)
       def load_distribution(self, dist_type, **params):
           """Cache distribution objects."""
           return create_distribution(dist_type, **params)

Async Operations
~~~~~~~~~~~~~~~~

For I/O-bound operations, consider async:

.. code-block:: python

   import asyncio
   
   async def execute_async(self, params):
       """Async version for I/O operations."""
       # Async database query
       portfolio = await self.load_portfolio_async(params["id"])
       
       # Run CPU-bound work in thread pool
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(
           None, 
           self.calculate_metrics, 
           portfolio
       )
       
       return result

Debugging Tools
---------------

Logging
~~~~~~~

Use structured logging for debugging:

.. code-block:: python

   import logging
   import json
   
   logger = logging.getLogger(__name__)
   
   def execute(self, params):
       logger.info("Tool execution started", extra={
           "tool": self.name,
           "params": json.dumps(params)
       })
       
       try:
           result = self.perform_calculation(params)
           logger.info("Tool execution completed", extra={
               "tool": self.name,
               "status": "success"
           })
           return result
       
       except Exception as e:
           logger.error("Tool execution failed", extra={
               "tool": self.name,
               "error": str(e),
               "params": json.dumps(params)
           }, exc_info=True)
           raise

Development Workflow
--------------------

1. **Create Feature Branch**

   .. code-block:: bash

      git checkout -b feature/new-mcp-tool

2. **Implement Tool**
   
   - Write tool class
   - Add tests
   - Update documentation

3. **Test Locally**

   .. code-block:: bash

      # Run unit tests
      pytest tests/test_new_tool.py
      
      # Test with MCP server
      python -m quactuary.mcp.server

4. **Test in Claude Desktop**
   
   - Configure Claude Desktop
   - Test various scenarios
   - Verify error handling

5. **Submit PR**
   
   - Ensure tests pass
   - Update CHANGELOG
   - Submit pull request

Next Steps
----------

- Review existing tools in ``quactuary/mcp/tools.py`` for examples
- Check ``quactuary/mcp/tests/`` for testing patterns
- Read the FastMCP documentation for advanced features
- Join discussions on GitHub for tool ideas and feedback