quactuary.mcp package
=====================

.. automodule:: quactuary.mcp
   :members:
   :undoc-members:
   :show-inheritance:

The quActuary MCP (Model Context Protocol) Server provides integration with Claude and other LLM assistants, exposing actuarial tools through a standardized interface.

Overview
--------

The MCP server allows you to use quActuary's actuarial modeling capabilities directly within Claude conversations. It provides tools for:

- Running pricing simulations
- Working with probability distributions
- Managing portfolios and policy terms
- Performing actuarial calculations

Usage
-----

Starting the MCP Server
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using the command-line interface
   quactuary-mcp

   # Or as a Python module
   python -m quactuary.mcp.server

Configuration
~~~~~~~~~~~~~

The server can be configured in Claude Desktop's settings:

.. code-block:: json

   {
     "mcpServers": {
       "quactuary": {
         "command": "python",
         "args": ["-m", "quactuary.mcp.server"],
         "env": {}
       }
     }
   }

Submodules
----------

quactuary.mcp.server module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.server
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.tools module
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.tools
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.resources module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.resources
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.prompts module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.prompts
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.categories module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.categories
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.formats module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.formats
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.config module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.config
   :members:
   :undoc-members:
   :show-inheritance:

quactuary.mcp.base module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quactuary.mcp.base
   :members:
   :undoc-members:
   :show-inheritance:

Tool Categories
---------------

The MCP server organizes tools into the following categories:

**Pricing Tools**
   - run_pricing_simulation: Execute pricing model simulations
   - calculate_premium: Calculate premiums for policies
   - estimate_reserves: Estimate reserves for portfolios

**Distribution Tools**
   - create_distribution: Create frequency or severity distributions
   - fit_distribution: Fit distributions to data
   - sample_distribution: Generate samples from distributions

**Portfolio Tools**
   - create_portfolio: Build insurance portfolios
   - analyze_portfolio: Analyze portfolio metrics
   - optimize_portfolio: Optimize portfolio allocations

**Utility Tools**
   - calculate_metrics: Compute actuarial metrics
   - generate_reports: Generate actuarial reports
   - validate_inputs: Validate input parameters

Examples
--------

Using MCP Tools in Claude
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the MCP server is running and configured, you can use it in Claude conversations:

.. code-block:: text

   User: Run a pricing simulation for a GL policy with Poisson(3.5) frequency 
         and Exponential(1000) severity, with a $1M deductible.

   Claude: I'll run a pricing simulation for your GL policy using the MCP tools...
           [Uses run_pricing_simulation tool]

Creating Custom Tools
~~~~~~~~~~~~~~~~~~~~~

You can extend the MCP server with custom tools:

.. code-block:: python

   from quactuary.mcp.base import QuactuaryTool
   from quactuary.mcp.categories import ToolCategory

   class CustomActuarialTool(QuactuaryTool):
       category = ToolCategory.UTILITIES
       
       def execute(self, params):
           # Your custom implementation
           return {"result": "Custom calculation"}

Module Contents
---------------