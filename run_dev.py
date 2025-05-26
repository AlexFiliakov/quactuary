#!/usr/bin/env python3
"""
Development script for quActuary project.

Provides centralized access to all development operations.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def run_command(cmd: List[str], check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)


def ensure_venv() -> None:
    """Ensure we're running in a virtual environment."""
    if not hasattr(sys, 'real_prefix') and sys.base_prefix == sys.prefix:
        # Check for common virtual environment indicators
        venv_indicators = [
            Path("venv/bin/activate"),
            Path(".venv/bin/activate"),
            Path("env/bin/activate"),
        ]
        
        for venv_path in venv_indicators:
            if venv_path.exists():
                print(f"Virtual environment found at {venv_path.parent.parent}")
                print("Please activate it with:")
                print(f"  source {venv_path}")
                sys.exit(1)
        
        # Check if we're in claude-env
        if "claude-env" not in sys.prefix:
            print("Warning: No virtual environment detected")
            print("It's recommended to work in a virtual environment")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)


def install_dev(args: argparse.Namespace) -> None:
    """Install package in development mode with dependencies."""
    print("Installing quActuary in development mode...")
    
    # Install package in editable mode
    run_command([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Install development dependencies
    if Path("requirements-dev.txt").exists():
        print("Installing development dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    
    print("Installation complete!")


def run_tests(args: argparse.Namespace) -> None:
    """Run tests with various options."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if args.coverage:
        cmd.extend(["--cov=quactuary", "--cov-report=term", "--cov-report=html"])
        if args.coverage_branch:
            cmd.append("--cov-branch")
    
    if args.file:
        cmd.append(args.file)
    elif args.pattern:
        cmd.extend(["-k", args.pattern])
    
    if args.verbose:
        cmd.append("-v")
    
    if args.failfast:
        cmd.append("-x")
    
    print(f"Running: {' '.join(cmd)}")
    run_command(cmd)
    
    if args.coverage:
        print("\nCoverage report generated in htmlcov/")


def run_lint(args: argparse.Namespace) -> None:
    """Run linting tools."""
    tools = []
    
    # Check which tools are available
    ruff_available = run_command(["which", "ruff"], check=False, capture_output=True).returncode == 0
    mypy_available = run_command(["which", "mypy"], check=False, capture_output=True).returncode == 0
    black_available = run_command(["which", "black"], check=False, capture_output=True).returncode == 0
    
    if ruff_available:
        tools.append((["ruff", "check", "."], "Ruff linting"))
    
    if mypy_available:
        tools.append(([sys.executable, "-m", "mypy", "quactuary"], "Type checking"))
    
    if black_available:
        tools.append(([sys.executable, "-m", "black", "--check", "."], "Code formatting check"))
    
    if not tools:
        print("No linting tools found. Install them with:")
        print("  pip install ruff mypy black")
        sys.exit(1)
    
    all_passed = True
    
    for cmd, description in tools:
        print(f"\n{description}...")
        result = run_command(cmd, check=False)
        if result.returncode != 0:
            all_passed = False
    
    if not all_passed:
        sys.exit(1)
    else:
        print("\nAll linting checks passed!")


def run_format(args: argparse.Namespace) -> None:
    """Auto-format code."""
    black_available = run_command(["which", "black"], check=False, capture_output=True).returncode == 0
    
    if not black_available:
        print("Black not found. Install it with:")
        print("  pip install black")
        sys.exit(1)
    
    print("Formatting code with Black...")
    run_command([sys.executable, "-m", "black", "."])
    print("Code formatting complete!")


def build_docs(args: argparse.Namespace) -> None:
    """Build Sphinx documentation."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("Documentation directory not found!")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    if args.clean:
        print("Cleaning documentation build...")
        run_command(["make", "clean"])
    
    print("Building HTML documentation...")
    run_command(["make", "html"])
    
    build_dir = Path("build/html")
    if build_dir.exists():
        print(f"\nDocumentation built successfully!")
        print(f"View at: file://{build_dir.absolute()}/index.html")
        
        if args.serve:
            print("\nStarting documentation server...")
            os.chdir(build_dir)
            run_command([sys.executable, "-m", "http.server", "8000"])


def run_coverage(args: argparse.Namespace) -> None:
    """Generate and display coverage report."""
    print("Running tests with coverage...")
    cmd = [sys.executable, "-m", "pytest", "--cov=quactuary", "--cov-report=html", "--cov-report=term"]
    
    if args.branch:
        cmd.append("--cov-branch")
    
    run_command(cmd)
    
    print("\nCoverage report generated in htmlcov/")
    
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{Path('htmlcov/index.html').absolute()}")


def clean_build(args: argparse.Namespace) -> None:
    """Clean build artifacts."""
    patterns = [
        "build/",
        "dist/",
        "*.egg-info/",
        "htmlcov/",
        ".coverage",
        ".pytest_cache/",
        "**/__pycache__/",
        "**/*.pyc",
        "**/*.pyo",
    ]
    
    print("Cleaning build artifacts...")
    
    for pattern in patterns:
        if "*" in pattern:
            # Use glob for patterns
            for path in Path(".").glob(pattern):
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"Removed file: {path}")
        else:
            # Direct path
            path = Path(pattern)
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"Removed file: {path}")
    
    print("Clean complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="quActuary development commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dev.py test --coverage          # Run tests with coverage
  python run_dev.py lint                      # Run all linters
  python run_dev.py install                   # Install dev environment
  python run_dev.py docs --serve              # Build and serve docs
  python run_dev.py test --file test_pricing.py  # Run specific test
  python run_dev.py clean                     # Clean build artifacts
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    test_parser.add_argument('--coverage-branch', action='store_true', help='Include branch coverage')
    test_parser.add_argument('--file', help='Run specific test file')
    test_parser.add_argument('--pattern', '-k', help='Run tests matching pattern')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    test_parser.add_argument('--failfast', '-x', action='store_true', help='Stop on first failure')
    test_parser.set_defaults(func=run_tests)
    
    # Lint command
    lint_parser = subparsers.add_parser('lint', help='Run linting tools')
    lint_parser.set_defaults(func=run_lint)
    
    # Format command
    format_parser = subparsers.add_parser('format', help='Auto-format code')
    format_parser.set_defaults(func=run_format)
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install development environment')
    install_parser.set_defaults(func=install_dev)
    
    # Build command (alias for install)
    build_parser = subparsers.add_parser('build', help='Build package')
    build_parser.set_defaults(func=install_dev)
    
    # Coverage command
    coverage_parser = subparsers.add_parser('coverage', help='Generate coverage report')
    coverage_parser.add_argument('--branch', action='store_true', help='Include branch coverage')
    coverage_parser.add_argument('--open', action='store_true', help='Open report in browser')
    coverage_parser.set_defaults(func=run_coverage)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean build artifacts')
    clean_parser.set_defaults(func=clean_build)
    
    # Docs command
    docs_parser = subparsers.add_parser('docs', help='Build documentation')
    docs_parser.add_argument('--serve', action='store_true', help='Serve docs locally')
    docs_parser.add_argument('--clean', action='store_true', help='Clean before building')
    docs_parser.set_defaults(func=build_docs)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Ensure we're in the right directory
    if not Path("quactuary").exists() and not Path("setup.py").exists():
        print("Error: run_dev.py must be run from the project root directory")
        sys.exit(1)
    
    # Check virtual environment
    ensure_venv()
    
    # Run the selected command
    args.func(args)


if __name__ == "__main__":
    main()