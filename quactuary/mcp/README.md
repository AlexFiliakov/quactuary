# MCP Interface for quActuary

This directory contains the Model Context Protocol (MCP) server implementation for the quActuary package.

## Overview

The MCP server exposes quActuary functionality through a standardized protocol that allows Claude Code and other MCP clients to:
- Execute actuarial calculations and simulations
- Work with probability distributions
- Analyze insurance portfolios
- Run performance benchmarks

## Installation

```bash
# Install with MCP support
pip install -e .[mcp]

# Or just install the MCP dependency
pip install mcp>=0.9.0
```

## Running the Server

```bash
# Run directly
python -m quactuary.mcp.server

# Or use the installed command
quactuary-mcp
```

## Development

### Testing

```bash
# Run MCP-specific tests
pytest quactuary/tests/test_mcp/

# Run with coverage
pytest --cov=quactuary.mcp quactuary/tests/test_mcp/
```

### Adding New Tools

1. Add tool implementation in `tools.py`
2. Follow the naming convention: `category_action` (e.g., `pricing_calculate_var`)
3. Include proper documentation and error handling
4. Add corresponding tests in `test_tools.py`

### Tool Categories

- **pricing**: Portfolio pricing and risk measures
- **distributions**: Frequency and severity distributions
- **portfolio**: Portfolio construction and analysis
- **utilities**: Helper functions and data processing

### Data Format Expectations

- **Dates**: ISO format strings (YYYY-MM-DD)
- **Money**: Float values in base currency
- **Distributions**: JSON objects with type and parameters
- **Results**: Structured JSON with estimates and metadata
