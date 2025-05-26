"""Test MCP configuration management."""

import pytest
from quactuary.mcp.config import (
    MCP_CONFIG,
    TOOL_LIMITS,
    PERFORMANCE_CONFIG,
    LOGGING_CONFIG,
    ConfigManager,
    config
)


class TestConfigConstants:
    """Test configuration constants."""
    
    def test_mcp_config_structure(self):
        """Test MCP_CONFIG has required fields."""
        assert "server_name" in MCP_CONFIG
        assert "version" in MCP_CONFIG
        assert "dependencies" in MCP_CONFIG
        assert "max_workers" in MCP_CONFIG
        assert "timeout_seconds" in MCP_CONFIG
        
        assert isinstance(MCP_CONFIG["dependencies"], list)
        assert MCP_CONFIG["max_workers"] > 0
        assert MCP_CONFIG["timeout_seconds"] > 0
    
    def test_tool_limits_structure(self):
        """Test TOOL_LIMITS has expected categories."""
        assert "monte_carlo_simulation" in TOOL_LIMITS
        assert "batch_operations" in TOOL_LIMITS
        assert "distributions" in TOOL_LIMITS
        assert "portfolio" in TOOL_LIMITS
        
        # Check monte carlo limits
        mc_limits = TOOL_LIMITS["monte_carlo_simulation"]
        assert "max_iterations" in mc_limits
        assert "default_iterations" in mc_limits
        assert mc_limits["max_iterations"] > mc_limits["default_iterations"]
    
    def test_performance_config_structure(self):
        """Test PERFORMANCE_CONFIG has expected fields."""
        assert "enable_parallel" in PERFORMANCE_CONFIG
        assert "enable_jit" in PERFORMANCE_CONFIG
        assert "cache_results" in PERFORMANCE_CONFIG
        assert "cache_ttl_seconds" in PERFORMANCE_CONFIG
        assert "max_cache_size_mb" in PERFORMANCE_CONFIG
        
        assert isinstance(PERFORMANCE_CONFIG["enable_parallel"], bool)
        assert isinstance(PERFORMANCE_CONFIG["enable_jit"], bool)
        assert PERFORMANCE_CONFIG["cache_ttl_seconds"] > 0
    
    def test_logging_config_structure(self):
        """Test LOGGING_CONFIG has expected fields."""
        assert "level" in LOGGING_CONFIG
        assert "format" in LOGGING_CONFIG
        assert "file" in LOGGING_CONFIG
        assert "max_file_size_mb" in LOGGING_CONFIG
        assert "backup_count" in LOGGING_CONFIG


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_creation(self):
        """Test creating a config manager."""
        cm = ConfigManager()
        
        # Should have default configs loaded
        assert cm.get("mcp.server_name") == MCP_CONFIG["server_name"]
        assert cm.get("limits.monte_carlo_simulation.max_iterations") == 10_000_000
    
    def test_get_nested_config(self):
        """Test getting nested configuration values."""
        cm = ConfigManager()
        
        # Top level
        assert cm.get("mcp") == cm._config["mcp"]
        
        # Nested
        assert cm.get("mcp.version") == MCP_CONFIG["version"]
        assert cm.get("limits.distributions.max_sample_size") == 1_000_000
        assert cm.get("performance.enable_jit") is True
    
    def test_get_with_default(self):
        """Test getting config with default value."""
        cm = ConfigManager()
        
        assert cm.get("nonexistent.key", "default_value") == "default_value"
        assert cm.get("mcp.nonexistent", 42) == 42
    
    def test_set_override(self):
        """Test setting configuration overrides."""
        cm = ConfigManager()
        
        # Set override
        cm.set("performance.enable_jit", False)
        assert cm.get("performance.enable_jit") is False
        
        # Original should be unchanged
        assert PERFORMANCE_CONFIG["enable_jit"] is True
        
        # Nested override
        cm.set("limits.custom.max_value", 1000)
        assert cm.get("limits.custom.max_value") == 1000
    
    def test_override_precedence(self):
        """Test that overrides take precedence."""
        cm = ConfigManager()
        
        # Original value
        original = cm.get("mcp.max_workers")
        
        # Set override
        cm.set("mcp.max_workers", 8)
        assert cm.get("mcp.max_workers") == 8
        assert cm._config["mcp"]["max_workers"] == original
    
    def test_get_tool_limit(self):
        """Test getting tool-specific limits."""
        cm = ConfigManager()
        
        assert cm.get_tool_limit("monte_carlo_simulation", "max_iterations") == 10_000_000
        assert cm.get_tool_limit("distributions", "default_sample_size") == 10_000
        assert cm.get_tool_limit("nonexistent", "limit", 999) == 999
    
    def test_is_feature_enabled(self):
        """Test checking if features are enabled."""
        cm = ConfigManager()
        
        assert cm.is_feature_enabled("enable_parallel") is True
        assert cm.is_feature_enabled("enable_jit") is True
        assert cm.is_feature_enabled("nonexistent_feature") is False
        
        # With override
        cm.set("performance.enable_parallel", False)
        assert cm.is_feature_enabled("enable_parallel") is False
    
    def test_reset_overrides(self):
        """Test resetting configuration overrides."""
        cm = ConfigManager()
        
        # Set some overrides
        cm.set("mcp.max_workers", 16)
        cm.set("performance.enable_jit", False)
        assert cm.get("mcp.max_workers") == 16
        assert cm.get("performance.enable_jit") is False
        
        # Reset
        cm.reset_overrides()
        assert cm.get("mcp.max_workers") == MCP_CONFIG["max_workers"]
        assert cm.get("performance.enable_jit") is True


class TestGlobalConfig:
    """Test global config instance."""
    
    def test_global_config_exists(self):
        """Test that global config instance exists."""
        assert config is not None
        assert isinstance(config, ConfigManager)
    
    def test_global_config_isolation(self):
        """Test that tests don't affect global config."""
        # Save original state
        original_value = config.get("mcp.max_workers")
        
        # Modify in test
        config.set("mcp.max_workers", 99)
        assert config.get("mcp.max_workers") == 99
        
        # Reset for other tests
        config.reset_overrides()
        assert config.get("mcp.max_workers") == original_value