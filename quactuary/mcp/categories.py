"""Tool category definitions for MCP server."""

from enum import Enum
from typing import Dict, List


class ToolCategory(Enum):
    """Enumeration of tool categories."""
    PRICING = "pricing"
    DISTRIBUTIONS = "distributions"
    PORTFOLIO = "portfolio"
    UTILITIES = "utilities"
    BENCHMARKS = "benchmarks"


# Tool category metadata
CATEGORY_INFO: Dict[ToolCategory, Dict[str, str]] = {
    ToolCategory.PRICING: {
        "name": "Pricing and Risk",
        "description": "Tools for portfolio pricing, risk measures, and simulations",
        "prefix": "pricing_",
    },
    ToolCategory.DISTRIBUTIONS: {
        "name": "Probability Distributions",
        "description": "Tools for working with frequency, severity, and compound distributions",
        "prefix": "dist_",
    },
    ToolCategory.PORTFOLIO: {
        "name": "Portfolio Management",
        "description": "Tools for portfolio construction, validation, and analysis",
        "prefix": "portfolio_",
    },
    ToolCategory.UTILITIES: {
        "name": "Utility Functions",
        "description": "Helper tools for data processing, validation, and conversion",
        "prefix": "util_",
    },
    ToolCategory.BENCHMARKS: {
        "name": "Performance Benchmarks",
        "description": "Tools for running and analyzing performance benchmarks",
        "prefix": "bench_",
    },
}


def get_category_prefix(category: ToolCategory) -> str:
    """Get the tool name prefix for a category."""
    return CATEGORY_INFO[category]["prefix"]


def get_tool_category(tool_name: str) -> ToolCategory:
    """Determine the category of a tool based on its name."""
    for category, info in CATEGORY_INFO.items():
        if tool_name.startswith(info["prefix"]):
            return category
    raise ValueError(f"Unknown tool category for: {tool_name}")


# Planned tools by category
PLANNED_TOOLS: Dict[ToolCategory, List[str]] = {
    ToolCategory.PRICING: [
        "pricing_simulate_portfolio",
        "pricing_calculate_var",
        "pricing_calculate_tvar",
        "pricing_run_simulation",
        "pricing_compare_backends",
        "pricing_optimize_portfolio",
    ],
    ToolCategory.DISTRIBUTIONS: [
        "dist_create_frequency",
        "dist_create_severity",
        "dist_create_compound",
        "dist_fit_parameters",
        "dist_compare",
        "dist_goodness_of_fit",
        "dist_generate_samples",
    ],
    ToolCategory.PORTFOLIO: [
        "portfolio_create",
        "portfolio_validate",
        "portfolio_statistics",
        "portfolio_analyze_policy",
        "portfolio_import_data",
        "portfolio_export_results",
    ],
    ToolCategory.UTILITIES: [
        "util_validate_data",
        "util_convert_format",
        "util_generate_test_data",
        "util_check_convergence",
        # Workflow tools
        "util_workflow_create_analysis",
        "util_workflow_batch_process",
        "util_workflow_schedule",
        # Visualization tools
        "util_viz_create_chart",
        "util_report_generate",
        "util_dashboard_data",
        # Analytics tools
        "util_analytics_scenario_analysis",
        "util_analytics_stress_test",
        "util_analytics_predictive_model",
        # Data quality tools
        "util_data_quality_check",
        "util_data_lineage_trace",
        "util_data_reconciliation",
    ],
    ToolCategory.BENCHMARKS: [
        "bench_run_performance_test",
        "bench_compare_methods",
        "bench_profile_simulation",
        # Performance monitoring
        "bench_perf_benchmark_suite",
        "bench_perf_optimization_advisor",
        "bench_perf_resource_monitor",
    ],
}