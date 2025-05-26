#!/usr/bin/env python3
"""
CLI tool for managing performance baselines.

This tool provides commands for:
- Viewing current baselines
- Updating baselines
- Comparing performance
- Exporting/importing baselines
- Generating reports
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from quactuary.performance_baseline import AdaptiveBaselineManager, HardwareProfile
from quactuary.benchmarks import PerformanceBenchmark


def cmd_show(args):
    """Show baseline information."""
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    if args.hardware:
        # Show current hardware profile
        profile = HardwareProfile.get_current()
        print("Current Hardware Profile:")
        print("-" * 40)
        print(f"CPU Model: {profile.cpu_model}")
        print(f"CPU Count: {profile.cpu_count}")
        print(f"CPU Frequency: {profile.cpu_freq_mhz:.0f} MHz")
        print(f"Total Memory: {profile.total_memory_gb:.1f} GB")
        print(f"Platform: {profile.platform_system} {profile.platform_machine}")
        print(f"Python Version: {profile.python_version}")
        print(f"Performance Score: {profile.performance_score:.2f}")
        print(f"Profile Hash: {profile.profile_hash}")
        return
    
    if args.test:
        # Show specific test baseline
        report = manager.get_performance_report(args.test)
        if "error" in report:
            print(f"Error: {report['error']}")
            return
        
        print(f"Performance Report for: {report['test_name']}")
        print("=" * 60)
        print(f"Total Runs: {report['total_runs']}")
        print(f"Hardware Profiles: {report['hardware_profiles']}")
        print()
        
        print("Performance by Hardware:")
        print("-" * 60)
        for hw_hash, hw_data in report['performance_by_hardware'].items():
            print(f"\nHardware {hw_hash} ({hw_data['cpu_model']}):")
            print(f"  Runs: {hw_data['runs']}")
            print(f"  Performance Score: {hw_data['performance_score']:.2f}")
            print(f"  Raw Time: {hw_data['raw_time_stats']['median']:.3f}s "
                  f"(±{hw_data['raw_time_stats']['std']:.3f}s)")
            print(f"  Normalized Time: {hw_data['normalized_time_stats']['median']:.3f}s "
                  f"(±{hw_data['normalized_time_stats']['std']:.3f}s)")
        
        print("\nOverall Normalized Statistics:")
        print("-" * 40)
        stats = report['overall_normalized_stats']
        print(f"Mean: {stats['mean']:.3f}s")
        print(f"Median: {stats['median']:.3f}s")
        print(f"Std Dev: {stats['std']:.3f}s")
        print(f"Trend: {stats['trend']}")
    else:
        # Show all tests
        if not manager.baselines:
            print("No baselines recorded yet.")
            return
        
        print("Available Performance Baselines:")
        print("-" * 60)
        for test_name, baselines in manager.baselines.items():
            latest = baselines[-1]
            print(f"\n{test_name}:")
            print(f"  Runs: {len(baselines)}")
            print(f"  Latest: {latest.timestamp}")
            print(f"  Latest Time: {latest.raw_time:.3f}s (normalized: {latest.normalized_time:.3f}s)")
            print(f"  Sample Size: {latest.sample_size}")


def cmd_update(args):
    """Update baselines by running benchmarks."""
    print("Running benchmarks to update baselines...")
    print("-" * 60)
    
    benchmark = PerformanceBenchmark(output_dir=args.output_dir)
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    # Get test portfolios
    portfolios = benchmark.create_test_portfolios()
    
    if args.portfolio:
        if args.portfolio not in portfolios:
            print(f"Error: Portfolio '{args.portfolio}' not found. Available: {list(portfolios.keys())}")
            return
        portfolios = {args.portfolio: portfolios[args.portfolio]}
    
    # Run benchmarks
    for portfolio_name, portfolio in portfolios.items():
        total_policies = sum(bucket.n_policies for bucket in portfolio)
        print(f"\nBenchmarking {portfolio_name} portfolio ({total_policies} policies)...")
        
        n_sims = args.simulations or (10000 if portfolio_name in ['small', 'medium'] else 1000)
        
        # Run different methods
        methods = []
        if args.baseline or not (args.jit or args.qmc):
            methods.append(('baseline', lambda p, n: benchmark.benchmark_baseline(p, n, portfolio_name)))
        if args.jit:
            methods.append(('jit', lambda p, n: benchmark.benchmark_jit(p, n, portfolio_name)))
        if args.qmc:
            methods.append(('qmc', lambda p, n: benchmark.benchmark_qmc(p, n, portfolio_name)))
        
        for method_name, method_func in methods:
            try:
                print(f"  Running {method_name}...", end='', flush=True)
                result = method_func(portfolio, n_sims)
                
                # Record in baseline system
                test_name = f"{method_name}_{portfolio_name}_{n_sims}"
                baseline = manager.record_performance(
                    test_name=test_name,
                    execution_time=result.execution_time,
                    sample_size=total_policies * n_sims,
                    metadata={
                        'method': method_name,
                        'portfolio': portfolio_name,
                        'n_simulations': n_sims
                    }
                )
                
                print(f" {result.execution_time:.3f}s")
            except Exception as e:
                print(f" FAILED: {str(e)}")
    
    print("\nBaselines updated successfully!")


def cmd_compare(args):
    """Compare current performance with baselines."""
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    if args.test:
        # Check specific test
        is_regression, expected_time, message = manager.check_regression(
            test_name=args.test,
            current_time=args.time,
            sample_size=args.sample_size or 1
        )
        
        print(f"Test: {args.test}")
        print(f"Current Time: {args.time:.3f}s")
        if expected_time:
            print(f"Expected Time: {expected_time:.3f}s")
        print(f"Status: {'REGRESSION' if is_regression else 'OK'}")
        print(f"Message: {message}")
    else:
        print("Error: --test is required for compare command")


def cmd_export(args):
    """Export baselines to file."""
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    output_file = args.output or f"baselines_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manager.export_baselines(output_file)
    print(f"Baselines exported to: {output_file}")


def cmd_import(args):
    """Import baselines from file."""
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    manager.import_baselines(args.input, merge=not args.replace)
    print(f"Baselines imported from: {args.input}")
    print(f"Mode: {'replace' if args.replace else 'merge'}")


def cmd_clear(args):
    """Clear baselines."""
    manager = AdaptiveBaselineManager(args.baseline_dir)
    
    if args.test:
        # Clear specific test
        if args.test in manager.baselines:
            count = len(manager.baselines[args.test])
            del manager.baselines[args.test]
            manager._save_baselines()
            print(f"Cleared {count} baselines for test: {args.test}")
        else:
            print(f"No baselines found for test: {args.test}")
    else:
        # Clear all with confirmation
        if not args.force:
            response = input("Clear all baselines? This cannot be undone. [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        count = sum(len(b) for b in manager.baselines.values())
        manager.baselines.clear()
        manager._save_baselines()
        print(f"Cleared {count} baselines.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Performance baseline management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current hardware profile
  %(prog)s show --hardware
  
  # Show all baselines
  %(prog)s show
  
  # Show specific test baseline
  %(prog)s show --test baseline_small_10000
  
  # Update baselines by running benchmarks
  %(prog)s update
  
  # Update only JIT baselines for medium portfolio
  %(prog)s update --jit --portfolio medium
  
  # Compare performance
  %(prog)s compare --test my_test --time 1.234 --sample-size 1000
  
  # Export baselines
  %(prog)s export -o baselines_backup.json
  
  # Import baselines
  %(prog)s import -i baselines_backup.json
        """
    )
    
    parser.add_argument(
        '--baseline-dir',
        default='./performance_baselines',
        help='Directory for baseline storage (default: ./performance_baselines)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Show command
    parser_show = subparsers.add_parser('show', help='Show baseline information')
    parser_show.add_argument('--hardware', action='store_true', help='Show current hardware profile')
    parser_show.add_argument('--test', help='Show specific test baseline')
    
    # Update command
    parser_update = subparsers.add_parser('update', help='Update baselines by running benchmarks')
    parser_update.add_argument('--output-dir', default='./benchmark_results', help='Output directory for results')
    parser_update.add_argument('--portfolio', choices=['small', 'medium', 'large', 'xlarge'], help='Run only specific portfolio')
    parser_update.add_argument('--simulations', type=int, help='Number of simulations')
    parser_update.add_argument('--baseline', action='store_true', help='Run baseline method')
    parser_update.add_argument('--jit', action='store_true', help='Run JIT method')
    parser_update.add_argument('--qmc', action='store_true', help='Run QMC method')
    
    # Compare command
    parser_compare = subparsers.add_parser('compare', help='Compare performance with baseline')
    parser_compare.add_argument('--test', required=True, help='Test name')
    parser_compare.add_argument('--time', type=float, required=True, help='Execution time to compare')
    parser_compare.add_argument('--sample-size', type=int, help='Sample size')
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export baselines to file')
    parser_export.add_argument('-o', '--output', help='Output file name')
    
    # Import command
    parser_import = subparsers.add_parser('import', help='Import baselines from file')
    parser_import.add_argument('-i', '--input', required=True, help='Input file name')
    parser_import.add_argument('--replace', action='store_true', help='Replace existing baselines instead of merging')
    
    # Clear command
    parser_clear = subparsers.add_parser('clear', help='Clear baselines')
    parser_clear.add_argument('--test', help='Clear specific test only')
    parser_clear.add_argument('--force', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    command_map = {
        'show': cmd_show,
        'update': cmd_update,
        'compare': cmd_compare,
        'export': cmd_export,
        'import': cmd_import,
        'clear': cmd_clear
    }
    
    command_map[args.command](args)


if __name__ == '__main__':
    main()