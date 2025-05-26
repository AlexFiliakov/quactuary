"""Resource definitions for MCP server."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def register_resources(mcp):
    """Register resources with the MCP server."""
    
    @mcp.resource("schema://portfolio")
    async def portfolio_schema():
        """JSON schema for portfolio data format."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Insurance Portfolio",
            "type": "object",
            "properties": {
                "policies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["policy_id", "effective_date", "expiration_date", "exposure_amount"],
                        "properties": {
                            "policy_id": {"type": "string"},
                            "effective_date": {"type": "string", "format": "date"},
                            "expiration_date": {"type": "string", "format": "date"},
                            "lob": {"type": "string", "enum": ["GL", "AL", "PL", "WC", "Property"]},
                            "exposure_amount": {"type": "number", "minimum": 0},
                            "premium": {"type": "number", "minimum": 0},
                            "per_occ_retention": {"type": "number", "minimum": 0},
                            "agg_retention": {"type": "number", "minimum": 0},
                            "per_occ_limit": {"type": "number", "minimum": 0},
                            "agg_limit": {"type": "number", "minimum": 0}
                        }
                    }
                }
            }
        }, indent=2)
    
    @mcp.resource("example://small_portfolio")
    async def example_small_portfolio():
        """Example small portfolio for testing."""
        return json.dumps({
            "policies": [
                {
                    "policy_id": "TEST001",
                    "effective_date": "2024-01-01",
                    "expiration_date": "2025-01-01",
                    "lob": "GL",
                    "exposure_amount": 1000000,
                    "premium": 5000,
                    "per_occ_retention": 1000,
                    "per_occ_limit": 100000
                },
                {
                    "policy_id": "TEST002",
                    "effective_date": "2024-01-01",
                    "expiration_date": "2025-01-01",
                    "lob": "AL",
                    "exposure_amount": 2000000,
                    "premium": 15000,
                    "per_occ_retention": 2500,
                    "per_occ_limit": 250000
                },
                {
                    "policy_id": "TEST003",
                    "effective_date": "2024-01-01",
                    "expiration_date": "2025-01-01",
                    "lob": "Property",
                    "exposure_amount": 5000000,
                    "premium": 25000,
                    "per_occ_retention": 5000,
                    "per_occ_limit": 1000000
                }
            ],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Small test portfolio with 3 policies"
            }
        }, indent=2)
    
    @mcp.resource("distributions://catalog")
    async def distribution_catalog():
        """Catalog of available distributions."""
        return json.dumps({
            "frequency": {
                "Poisson": {
                    "parameters": ["lambda"],
                    "description": "Standard distribution for claim counts",
                    "use_cases": ["Low variance claim frequency"]
                },
                "NegativeBinomial": {
                    "parameters": ["n", "p"],
                    "description": "Overdispersed claim counts",
                    "use_cases": ["High variance claim frequency"]
                },
                "ZeroInflatedPoisson": {
                    "parameters": ["lambda", "p0"],
                    "description": "Poisson with excess zeros",
                    "use_cases": ["Many policies with no claims"]
                }
            },
            "severity": {
                "Lognormal": {
                    "parameters": ["mu", "sigma"],
                    "description": "Log-transformed normal distribution",
                    "use_cases": ["General claim amounts"]
                },
                "Gamma": {
                    "parameters": ["alpha", "beta"],
                    "description": "Flexible positive distribution",
                    "use_cases": ["Moderate claim amounts"]
                },
                "Pareto": {
                    "parameters": ["alpha", "xm"],
                    "description": "Heavy-tailed distribution",
                    "use_cases": ["Large/catastrophic losses"]
                }
            }
        }, indent=2)
    
    logger.info("Registered 3 resources with MCP server")