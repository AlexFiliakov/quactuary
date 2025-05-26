#!/usr/bin/env python3
"""
Script to check for performance regressions in CI/CD.

This script analyzes performance test results and determines if there are
any regressions that should block a PR or trigger alerts.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from quactuary.performance_baseline import AdaptiveBaselineManager


def check_regressions(
    baseline_dir: str,
    threshold_factor: float = 1.2,
    allowed_regressions: int = 0
) -> Tuple[bool, List[Dict], Dict]:
    """
    Check for performance regressions across all tests.
    
    Args:
        baseline_dir: Directory containing baseline data
        threshold_factor: Regression threshold multiplier
        allowed_regressions: Number of regressions allowed before failing
        
    Returns:
        Tuple of (has_blocking_regressions, regression_list, summary_stats)
    """
    manager = AdaptiveBaselineManager(baseline_dir)
    manager.regression_threshold = threshold_factor
    
    regressions = []
    total_tests = 0
    
    # Check each test with recent data
    for test_name, baselines in manager.baselines.items():
        if not baselines:
            continue
            
        latest = baselines[-1]
        total_tests += 1
        
        # Check if this is a regression
        is_regression, expected_time, message = manager.check_regression(
            test_name=test_name,
            current_time=latest.raw_time,
            sample_size=latest.sample_size
        )
        
        if is_regression:
            regressions.append({
                'test': test_name,
                'current_time': latest.raw_time,
                'expected_time': expected_time,
                'slowdown_factor': latest.raw_time / expected_time if expected_time else 0,
                'message': message
            })
    
    # Calculate summary statistics
    summary = {
        'total_tests': total_tests,
        'regression_count': len(regressions),
        'regression_rate': len(regressions) / total_tests if total_tests > 0 else 0,
        'max_slowdown': max((r['slowdown_factor'] for r in regressions), default=0),
        'hardware_profile': manager.current_profile.to_dict()
    }
    
    # Determine if regressions are blocking
    has_blocking_regressions = len(regressions) > allowed_regressions
    
    return has_blocking_regressions, regressions, summary


def generate_github_summary(regressions: List[Dict], summary: Dict) -> str:
    """Generate a GitHub Actions summary markdown."""
    lines = []
    lines.append("# Performance Test Summary\n")
    
    # Overall status
    status_emoji = "✅" if not regressions else "⚠️"
    lines.append(f"{status_emoji} **{summary['total_tests']} tests run, "
                 f"{summary['regression_count']} regressions detected**\n")
    
    # Hardware info
    hw = summary['hardware_profile']
    lines.append("## Test Environment")
    lines.append(f"- **CPU**: {hw['cpu_model']}")
    lines.append(f"- **Cores**: {hw['cpu_count']}")
    lines.append(f"- **Performance Score**: {hw['performance_score']:.2f}\n")
    
    # Regression details
    if regressions:
        lines.append("## Performance Regressions")
        lines.append("| Test | Current | Expected | Slowdown | Status |")
        lines.append("|------|---------|----------|----------|---------|")
        
        for reg in sorted(regressions, key=lambda x: x['slowdown_factor'], reverse=True):
            slowdown_pct = (reg['slowdown_factor'] - 1) * 100
            status = "🔴" if slowdown_pct > 50 else "🟡"
            lines.append(
                f"| {reg['test']} | {reg['current_time']:.3f}s | "
                f"{reg['expected_time']:.3f}s | +{slowdown_pct:.1f}% | {status} |"
            )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check for performance regressions"
    )
    parser.add_argument(
        '--baseline-dir',
        default='./performance_baselines',
        help='Directory containing baseline data'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.2,
        help='Regression threshold factor (default: 1.2 = 20% slower)'
    )
    parser.add_argument(
        '--allowed-regressions',
        type=int,
        default=0,
        help='Number of regressions allowed before failing'
    )
    parser.add_argument(
        '--output-json',
        help='Output detailed results to JSON file'
    )
    parser.add_argument(
        '--github-summary',
        action='store_true',
        help='Output GitHub Actions summary'
    )
    
    args = parser.parse_args()
    
    # Check for regressions
    has_blocking, regressions, summary = check_regressions(
        baseline_dir=args.baseline_dir,
        threshold_factor=args.threshold,
        allowed_regressions=args.allowed_regressions
    )
    
    # Output JSON report if requested
    if args.output_json:
        report = {
            'has_blocking_regressions': has_blocking,
            'regressions': regressions,
            'summary': summary
        }
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Output GitHub summary if requested
    if args.github_summary:
        summary_text = generate_github_summary(regressions, summary)
        
        # Write to GitHub step summary if available
        github_summary_path = Path(os.environ.get('GITHUB_STEP_SUMMARY', ''))
        if github_summary_path and github_summary_path.parent.exists():
            with open(github_summary_path, 'w') as f:
                f.write(summary_text)
        else:
            print(summary_text)
    
    # Print results to console
    if regressions:
        print(f"\n⚠️  Found {len(regressions)} performance regressions:")
        for reg in regressions:
            print(f"  - {reg['test']}: {reg['message']}")
    else:
        print("\n✅ No performance regressions detected")
    
    # Exit with appropriate code
    if has_blocking:
        print(f"\n❌ Build failed: {len(regressions)} regressions exceed allowed limit of {args.allowed_regressions}")
        sys.exit(1)
    else:
        print(f"\n✅ Performance check passed")
        sys.exit(0)


if __name__ == '__main__':
    import os
    main()