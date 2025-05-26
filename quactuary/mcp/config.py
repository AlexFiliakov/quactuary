# mcp/config.py
"""MCP-specific configuration"""

try:
    from .. import __version__
    version = __version__
except ImportError:
    version = "0.0.1"

MCP_CONFIG = {
    "server_name": "quActuary MCP Server",
    "version": version,
    "dependencies": [
        "numpy>=2.2.5",
        "pandas>=2.2.3", 
        "scipy>=1.14.1",
        "numba>=0.56.0"
    ],
    "max_workers": 4,  # For async operations
    "timeout_seconds": 300,  # For long-running simulations
}

# Tool-specific limits
TOOL_LIMITS = {
    "monte_carlo_simulation": {
        "max_iterations": 10_000_000,
        "default_iterations": 10_000
    },
    "batch_operations": {
        "max_batch_size": 1000
    },
    "distributions": {
        "max_sample_size": 1_000_000,
        "default_sample_size": 10_000
    },
    "portfolio": {
        "max_policies": 100_000,
        "max_file_size_mb": 100
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    "enable_parallel": True,
    "enable_jit": True,
    "cache_results": True,
    "cache_ttl_seconds": 3600,  # 1 hour
    "max_cache_size_mb": 500
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # Set to path for file logging
    "max_file_size_mb": 100,
    "backup_count": 5
}


class ConfigManager:
    """Manages MCP server configuration."""
    
    def __init__(self):
        self._config = {
            "mcp": MCP_CONFIG.copy(),
            "limits": TOOL_LIMITS.copy(),
            "performance": PERFORMANCE_CONFIG.copy(),
            "logging": LOGGING_CONFIG.copy()
        }
        self._overrides = {}
    
    def get(self, key: str, default=None):
        """Get a configuration value using dot notation."""
        keys = key.split(".")
        value = self._overrides
        
        # Check overrides first
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                value = None
                break
        
        # Fall back to default config
        if value is None:
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
        
        return value
    
    def set(self, key: str, value):
        """Set a configuration override using dot notation."""
        keys = key.split(".")
        current = self._overrides
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def get_tool_limit(self, tool_category: str, limit_name: str, default=None):
        """Get a specific tool limit."""
        return self.get(f"limits.{tool_category}.{limit_name}", default)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a performance feature is enabled."""
        return self.get(f"performance.{feature}", False)
    
    def reset_overrides(self):
        """Reset all configuration overrides."""
        self._overrides = {}


# Global configuration instance
config = ConfigManager()
