#!/usr/bin/env python3
"""
Development script for quActuary project.

Provides centralized access to all development operations.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional


class ProgressIndicator:
    """Simple progress indicator for long-running commands."""
    
    def __init__(self, message: str = "Working"):
        self.message = message
        self.running = False
        self.thread = None
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
    
    def _spin(self):
        """Spin the progress indicator."""
        while self.running:
            print(f"\r{self.message}... {self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]}", end="", flush=True)
            self.spinner_idx += 1
            time.sleep(0.1)
    
    def start(self):
        """Start the progress indicator."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._spin)
            self.thread.start()
    
    def stop(self, success: bool = True):
        """Stop the progress indicator."""
        if self.running:
            self.running = False
            self.thread.join()
            # Clear the line
            print(f"\r{' ' * (len(self.message) + 10)}\r", end="", flush=True)
            if success:
                print(f"✓ {self.message} complete")
            else:
                print(f"✗ {self.message} failed")


def run_command(cmd: List[str], check: bool = True, capture_output: bool = False, show_progress: bool = False, progress_message: str = "Running command") -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    progress = None
    if show_progress:
        progress = ProgressIndicator(progress_message)
        progress.start()
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        if progress:
            progress.stop(success=True)
        return result
    except subprocess.CalledProcessError as e:
        if progress:
            progress.stop(success=False)
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
    
    # Determine progress message
    if args.file:
        progress_msg = f"Running tests in {args.file}"
    elif args.pattern:
        progress_msg = f"Running tests matching '{args.pattern}'"
    else:
        progress_msg = "Running all tests"
    
    if args.coverage:
        progress_msg += " with coverage"
    
    run_command(cmd, show_progress=True, progress_message=progress_msg)
    
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
    
    run_command(["make", "html"], show_progress=True, progress_message="Building HTML documentation")
    
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
    cmd = [sys.executable, "-m", "pytest", "--cov=quactuary", "--cov-report=html", "--cov-report=term"]
    
    if args.branch:
        cmd.append("--cov-branch")
    
    progress_msg = "Generating coverage report"
    if args.branch:
        progress_msg += " with branch coverage"
    
    run_command(cmd, show_progress=True, progress_message=progress_msg)
    
    print("\nCoverage report generated in htmlcov/")
    
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{Path('htmlcov/index.html').absolute()}")


def run_profile(args: argparse.Namespace) -> None:
    """Run performance profiling on specified module/function."""
    print("Running performance profiling...")
    
    # Default to profiling the main test suite if no specific target
    target = args.target or "pytest quactuary/tests -v"
    
    # Prepare the profiling command
    profile_cmd = [
        sys.executable, "-m", "cProfile",
        "-s", "cumulative",  # Sort by cumulative time
        "-o", "profile_output.prof"  # Output file
    ]
    
    # If target is a command string, split it
    if isinstance(target, str) and " " in target:
        import shlex
        profile_cmd.extend(["-m"] + shlex.split(target)[0:1])
        profile_cmd.extend(shlex.split(target)[1:])
    else:
        profile_cmd.extend(["-m", target])
    
    # Run profiling
    run_command(profile_cmd)
    
    # Analyze results
    print("\nProfile results saved to profile_output.prof")
    print("\nTop 20 functions by cumulative time:")
    
    import pstats
    stats = pstats.Stats("profile_output.prof")
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    
    # Generate visual output if snakeviz is available
    try:
        run_command([sys.executable, "-m", "snakeviz", "profile_output.prof"], check=False)
    except:
        print("\nInstall snakeviz for visual profiling: pip install snakeviz")


def setup_environment(args: argparse.Namespace) -> None:
    """Set up initial development environment from scratch."""
    print("Setting up development environment...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 9):
        print(f"Error: Python 3.9+ required, found {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} OK")
    
    # Check if in a virtual environment
    if not hasattr(sys, 'real_prefix') and sys.base_prefix == sys.prefix:
        print("\nNo virtual environment detected.")
        
        # Create virtual environment
        venv_path = Path("venv")
        if not venv_path.exists():
            print("Creating virtual environment...")
            run_command([sys.executable, "-m", "venv", "venv"])
            print(f"✓ Virtual environment created at {venv_path}")
            print(f"\nActivate it with:")
            print(f"  source venv/bin/activate  # Linux/Mac")
            print(f"  venv\\Scripts\\activate     # Windows")
            sys.exit(0)
        else:
            print(f"Virtual environment exists at {venv_path}")
            print("Please activate it and run this command again.")
            sys.exit(1)
    
    # Install development dependencies
    print("\nInstalling development dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    print("✓ Dependencies installed")
    
    # Run initial tests to verify setup
    print("\nRunning basic tests to verify setup...")
    result = run_command(
        [sys.executable, "-m", "pytest", "--tb=short", "-v", "-k", "test_basic"],
        check=False,
        capture_output=True
    )
    
    if result.returncode == 0:
        print("✓ Basic tests passed")
    else:
        print("⚠ Some tests failed - this is normal for initial setup")
    
    print("\n✅ Development environment setup complete!")
    print("\nNext steps:")
    print("  1. Run 'python run_dev.py test' to run all tests")
    print("  2. Run 'python run_dev.py lint' to check code style")
    print("  3. Run 'python run_dev.py docs' to build documentation")


def setup_completion(args: argparse.Namespace) -> None:
    """Generate and display shell completion setup instructions."""
    script_path = Path(__file__).absolute()
    
    print("Tab Completion Setup for run_dev.py")
    print("=" * 50)
    
    # Bash completion script
    bash_completion = f"""
# Add this to your ~/.bashrc or ~/.bash_profile:

_run_dev_complete() {{
    local cur prev opts base
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Main commands
    opts="test lint format install build coverage clean docs profile setup version"
    
    # Complete main commands
    if [[ ${{COMP_CWORD}} == 1 ]] ; then
        COMPREPLY=( $(compgen -W "${{opts}}" -- ${{cur}}) )
        return 0
    fi
    
    # Complete test options
    if [[ "${{prev}}" == "test" ]] ; then
        opts="--coverage --coverage-branch --file --pattern --verbose --failfast"
        COMPREPLY=( $(compgen -W "${{opts}}" -- ${{cur}}) )
        return 0
    fi
    
    # Complete coverage options
    if [[ "${{prev}}" == "coverage" ]] ; then
        opts="--branch --open"
        COMPREPLY=( $(compgen -W "${{opts}}" -- ${{cur}}) )
        return 0
    fi
    
    # Complete docs options
    if [[ "${{prev}}" == "docs" ]] ; then
        opts="--serve --clean"
        COMPREPLY=( $(compgen -W "${{opts}}" -- ${{cur}}) )
        return 0
    fi
}}

complete -F _run_dev_complete python {script_path}
complete -F _run_dev_complete ./run_dev.py
"""

    # Zsh completion script
    zsh_completion = f"""
# Add this to your ~/.zshrc:

_run_dev_complete() {{
    local -a commands
    commands=(
        'test:Run tests'
        'lint:Run linting tools'
        'format:Auto-format code'
        'install:Install development environment'
        'build:Build package'
        'coverage:Generate coverage report'
        'clean:Clean build artifacts'
        'docs:Build documentation'
        'profile:Run performance profiling'
        'setup:Set up development environment'
        'version:Show version and environment info'
    )
    
    if (( CURRENT == 2 )); then
        _describe -t commands 'run_dev.py commands' commands
    elif (( CURRENT == 3 )); then
        case ${{words[2]}} in
            test)
                _arguments \\
                    '--coverage[Run with coverage]' \\
                    '--coverage-branch[Include branch coverage]' \\
                    '--file[Run specific test file]:file:_files' \\
                    '--pattern[Run tests matching pattern]:pattern:' \\
                    '--verbose[Verbose output]' \\
                    '--failfast[Stop on first failure]'
                ;;
            coverage)
                _arguments \\
                    '--branch[Include branch coverage]' \\
                    '--open[Open report in browser]'
                ;;
            docs)
                _arguments \\
                    '--serve[Serve docs locally]' \\
                    '--clean[Clean before building]'
                ;;
        esac
    fi
}}

compdef _run_dev_complete {script_path}
compdef _run_dev_complete ./run_dev.py
"""

    print("\nFor Bash:")
    print("-" * 40)
    print(bash_completion)
    
    print("\nFor Zsh:")
    print("-" * 40)
    print(zsh_completion)
    
    print("\nQuick Setup:")
    print("-" * 40)
    print("1. Choose the script for your shell (bash or zsh)")
    print("2. Copy the script to your shell config file")
    print("3. Reload your shell: source ~/.bashrc (or ~/.zshrc)")
    print("4. Tab completion should now work!")
    
    print("\nAlternative: Using argcomplete (requires pip install argcomplete)")
    print("-" * 40)
    print("1. pip install argcomplete")
    print("2. Add to ~/.bashrc: eval \"$(register-python-argcomplete run_dev.py)\"")
    print("3. Reload shell: source ~/.bashrc")


def show_version(args: argparse.Namespace) -> None:
    """Display package version and environment info."""
    print("quActuary Development Environment")
    print("=" * 40)
    
    # Package version
    try:
        from quactuary._version import __version__
        print(f"quActuary version: {__version__}")
    except ImportError:
        print("quActuary version: Development")
    
    # Python version
    import sys
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    
    # Virtual environment
    if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
        print(f"Virtual environment: Active")
    else:
        print(f"Virtual environment: Not active")
    
    # Key dependencies
    print("\nKey dependencies:")
    deps = ["numpy", "scipy", "pandas", "qiskit", "pytest", "sphinx"]
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "Unknown")
            print(f"  {dep}: {version}")
        except ImportError:
            print(f"  {dep}: Not installed")
    
    # Git information
    try:
        git_result = run_command(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False
        )
        if git_result.returncode == 0:
            print(f"\nGit commit: {git_result.stdout.strip()}")
            
            # Check for uncommitted changes
            status_result = run_command(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=False
            )
            if status_result.stdout.strip():
                print("Git status: Uncommitted changes")
            else:
                print("Git status: Clean")
    except:
        pass


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
  python run_dev.py profile                   # Profile test performance
  python run_dev.py setup                     # Set up environment from scratch
  python run_dev.py version                   # Show version info
  python run_dev.py completion                # Show tab completion setup
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
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Run performance profiling')
    profile_parser.add_argument('target', nargs='?', help='Module or command to profile')
    profile_parser.set_defaults(func=run_profile)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up development environment')
    setup_parser.set_defaults(func=setup_environment)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version and environment info')
    version_parser.set_defaults(func=show_version)
    
    # Completion command
    completion_parser = subparsers.add_parser('completion', help='Show tab completion setup instructions')
    completion_parser.set_defaults(func=setup_completion)
    
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