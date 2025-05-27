============================
MCP Server Integration Guide
============================

The quActuary MCP (Model Context Protocol) Server enables seamless integration with Large Language Model assistants like Claude, providing direct access to actuarial modeling tools through conversational interfaces.

What is MCP?
------------

Model Context Protocol (MCP) is an open standard that enables AI assistants to securely access tools and data sources. The quActuary MCP Server implements this protocol, exposing our actuarial tools in a way that LLMs can understand and use effectively.

Installation
------------

To use the MCP Server, install quActuary with the MCP extras:

.. code-block:: bash

   pip install quactuary[mcp]

This installs the required dependencies including ``mcp>=0.9.0`` and ``fastmcp``.

Starting the Server
-------------------

There are two ways to start the MCP server:

1. **Using the command-line interface:**

   .. code-block:: bash

      quactuary-mcp

2. **As a Python module:**

   .. code-block:: bash

      python -m quactuary.mcp.server

The server runs on stdio transport by default, making it compatible with Claude Desktop and other MCP clients.

Configuring Claude Desktop
--------------------------

To use quActuary in Claude Desktop, add the following to your Claude Desktop configuration file:

**Windows:** ``%APPDATA%\Claude\claude_desktop_config.json``

**macOS:** ``~/Library/Application Support/Claude/claude_desktop_config.json``

**Linux:** ``~/.config/Claude/claude_desktop_config.json``

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

Alternative configuration using the CLI command:

.. code-block:: json

   {
     "mcpServers": {
       "quactuary": {
         "command": "quactuary-mcp",
         "args": [],
         "env": {}
       }
     }
   }

Available Tools
---------------

The MCP server provides tools organized into four categories:

Pricing Tools
~~~~~~~~~~~~~

**run_pricing_simulation**
   Execute full pricing model simulations with specified parameters.

   Example usage in Claude:
   
   .. code-block:: text

      User: Run a pricing simulation for a GL policy with Poisson(3.5) frequency,
            Exponential(1000) severity, and $1M deductible.

**calculate_premium**
   Calculate premiums for specific policy configurations.

**estimate_reserves**
   Estimate required reserves for portfolios.

Distribution Tools
~~~~~~~~~~~~~~~~~~

**create_distribution**
   Create frequency or severity distributions with specified parameters.

   Example:
   
   .. code-block:: text

      User: Create a compound distribution with NegativeBinomial(10, 0.3) frequency
            and Gamma(2, 1000) severity.

**fit_distribution**
   Fit distributions to empirical data.

**sample_distribution**
   Generate samples from distributions for analysis.

Portfolio Tools
~~~~~~~~~~~~~~~

**create_portfolio**
   Build insurance portfolios from policy specifications.

**analyze_portfolio**
   Analyze portfolio metrics including risk measures.

**optimize_portfolio**
   Optimize portfolio allocations and reinsurance structures.

Utility Tools
~~~~~~~~~~~~~

**calculate_metrics**
   Compute various actuarial metrics (VaR, TVaR, etc.).

**generate_reports**
   Generate formatted actuarial reports.

**validate_inputs**
   Validate input parameters for actuarial calculations.

Usage Examples
--------------

Basic Pricing Simulation
~~~~~~~~~~~~~~~~~~~~~~~~

In a Claude conversation:

.. code-block:: text

   User: I need to price a general liability policy for a client. The expected 
         claim frequency follows a Poisson distribution with mean 3.5 claims 
         per year. Claim severities are exponentially distributed with mean $1,000. 
         The policy has a $1M per-occurrence deductible. Can you run a simulation?

   Claude: I'll run a pricing simulation for your general liability policy using 
           the specified parameters.

           [Claude uses run_pricing_simulation tool]

           The simulation results show:
           - Expected annual loss: $2,450,000
           - 95% VaR: $2,566,000
           - 95% TVaR: $2,596,000

Portfolio Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User: Analyze a portfolio with three lines of business:
         1. GL: 700 policies, Poisson(3.5), Exponential(1000)
         2. Auto: 500 policies, NegBinom(10, 0.3), Gamma(2, 500)
         3. Property: 300 policies, Poisson(1.2), Lognormal(9, 1.5)

   Claude: I'll create and analyze this multi-line portfolio for you.

           [Claude uses create_portfolio and analyze_portfolio tools]

           Portfolio Analysis Results:
           - Total expected loss: $5,250,000
           - Portfolio 99% VaR: $5,890,000
           - Diversification benefit: 12.3%

Custom Calculations
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User: Calculate the limited expected value for a Pareto(3, 5000) 
         distribution with a limit of $10,000.

   Claude: I'll calculate the limited expected value for your Pareto distribution.

           [Claude uses calculate_metrics tool]

           The limited expected value LEV(10,000) = $2,291.67

Best Practices
--------------

1. **Clear Parameter Specification**
   Always specify distribution parameters clearly. Claude will ask for clarification if parameters are ambiguous.

2. **Incremental Analysis**
   Build complex analyses step by step. Start with simple distributions and add complexity.

3. **Validation**
   Use the validate_inputs tool to ensure parameters are reasonable before running simulations.

4. **Performance Considerations**
   Large simulations may take time. Claude will inform you of expected runtime for complex calculations.

Troubleshooting
---------------

Server Won't Start
~~~~~~~~~~~~~~~~~~

1. Ensure MCP dependencies are installed:

   .. code-block:: bash

      pip install quactuary[mcp]

2. Check Python version (3.10+ required):

   .. code-block:: bash

      python --version

3. Verify quactuary is installed:

   .. code-block:: bash

      python -c "import quactuary.mcp; print('MCP module found')"

Claude Can't Connect
~~~~~~~~~~~~~~~~~~~~

1. Restart Claude Desktop after updating configuration
2. Check configuration file syntax (must be valid JSON)
3. Ensure no other process is using the MCP server
4. Check Claude Desktop logs for connection errors

Tool Errors
~~~~~~~~~~~

If tools return errors:

1. Validate input parameters using the validate_inputs tool
2. Check distribution parameter constraints
3. Ensure sufficient memory for large simulations
4. Review error messages for specific parameter issues

Advanced Usage
--------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

You can configure the MCP server using environment variables:

.. code-block:: json

   {
     "mcpServers": {
       "quactuary": {
         "command": "python",
         "args": ["-m", "quactuary.mcp.server"],
         "env": {
           "QUACTUARY_BACKEND": "classical",
           "QUACTUARY_MAX_WORKERS": "4"
         }
       }
     }
   }

Custom Tool Development
~~~~~~~~~~~~~~~~~~~~~~~

Extend the MCP server with custom tools by creating a new tool class:

.. code-block:: python

   from quactuary.mcp.base import QuactuaryTool
   from quactuary.mcp.categories import ToolCategory

   class CustomRiskTool(QuactuaryTool):
       """Custom tool for specialized risk calculations."""
       
       category = ToolCategory.PRICING
       name = "calculate_custom_risk"
       description = "Calculate custom risk metrics"
       
       def execute(self, params):
           # Implementation
           return {"risk_metric": calculated_value}

Security Considerations
-----------------------

The MCP server:

- Runs locally on your machine
- Does not send data to external services
- Operates within Claude's security sandbox
- Only exposes explicitly defined tools

Performance Tips
----------------

1. **Use Appropriate Simulation Sizes**
   Start with smaller simulations (10,000 trials) for exploration, then increase for final results.

2. **Leverage Caching**
   The server caches distribution objects to improve performance across multiple calculations.

3. **Batch Operations**
   When analyzing multiple scenarios, describe them all at once so Claude can optimize the workflow.

Further Resources
-----------------

- :doc:`/quactuary.mcp` - MCP API Reference
- `MCP Documentation <https://modelcontextprotocol.io>`_ - Official MCP specification
- `Claude Desktop <https://claude.ai/desktop>`_ - Download Claude Desktop
- :doc:`/user_guide/index` - General user guide