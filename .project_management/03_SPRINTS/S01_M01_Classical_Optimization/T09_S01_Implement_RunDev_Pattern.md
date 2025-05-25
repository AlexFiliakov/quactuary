---
task_id: T09_S01
sprint: S01
sequence: 9
status: open
title: Implement run_dev.py Pattern
assigned_to: TBD
estimated_hours: 4
actual_hours: 0
priority: medium
risk: low
dependencies: []
last_updated: 2025-01-25
---

# T09_S01: Implement run_dev.py Pattern

## Description
Create a centralized development script pattern using run_dev.py to standardize all development operations and eliminate the workflow compliance violation identified in the project review. This provides a single entry point for all development tasks.

## Acceptance Criteria
- [ ] Create functional run_dev.py script in project root
- [ ] Support all common development operations
- [ ] Provide clear help documentation
- [ ] Handle virtual environment activation automatically
- [ ] Include error handling and user-friendly messages
- [ ] Document usage patterns for team

## Subtasks

### 1. Design Command Structure
- [ ] Plan command categories and hierarchy:
  ```
  python run_dev.py test [options]
  python run_dev.py lint [options]  
  python run_dev.py install [options]
  python run_dev.py build [options]
  python run_dev.py coverage [options]
  python run_dev.py clean [options]
  python run_dev.py docs [options]
  ```
- [ ] Define command aliases and shortcuts
- [ ] Plan help system and documentation

### 2. Implement Core Framework
- [ ] Create run_dev.py with argument parsing:
  ```python
  #!/usr/bin/env python3
  """
  Development script for quActuary project.
  
  Provides centralized access to all development operations.
  """
  import argparse
  import subprocess
  import sys
  from pathlib import Path
  
  def main():
      parser = argparse.ArgumentParser(description="quActuary development commands")
      subparsers = parser.add_subparsers(dest='command', help='Available commands')
      
      # Add subcommands...
  ```
- [ ] Add virtual environment detection and activation
- [ ] Implement error handling and user-friendly messages
- [ ] Add verbose/quiet modes

### 3. Implement Test Commands
- [ ] Add test command with options:
  ```python
  test_parser = subparsers.add_parser('test', help='Run tests')
  test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
  test_parser.add_argument('--file', help='Run specific test file')
  test_parser.add_argument('--pattern', help='Run tests matching pattern')
  test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
  ```
- [ ] Support pytest with various options
- [ ] Add coverage reporting integration
- [ ] Handle test result reporting

### 4. Implement Build and Install Commands
- [ ] Add install command for development setup:
  ```python
  def install_dev():
      """Install package in development mode with dependencies."""
      run_command(["pip", "install", "-e", "."])
      run_command(["pip", "install", "-r", "requirements-dev.txt"])
  ```
- [ ] Add build command for package building
- [ ] Add clean command to remove build artifacts
- [ ] Handle dependency installation and updates

### 5. Implement Code Quality Commands
- [ ] Add lint command with multiple linters:
  ```python
  def lint():
      """Run linting tools."""
      tools = [
          (["ruff", "check", "."], "Ruff linting"),
          (["mypy", "quactuary"], "Type checking"),
          (["black", "--check", "."], "Code formatting check")
      ]
      # Run each tool and collect results
  ```
- [ ] Add format command for auto-formatting
- [ ] Support multiple code quality tools
- [ ] Provide summary reporting

### 6. Implement Documentation Commands
- [ ] Add docs command for building documentation:
  ```python
  def build_docs():
      """Build Sphinx documentation."""
      os.chdir("docs")
      run_command(["make", "html"])
      print("Documentation built in docs/build/html/")
  ```
- [ ] Add docs-serve command for local preview
- [ ] Support documentation cleanup
- [ ] Handle Sphinx build process

### 7. Add Utility Commands
- [ ] Add coverage command with reporting:
  ```python
  def coverage():
      """Generate and display coverage report."""
      run_command(["pytest", "--cov=quactuary", "--cov-report=html", "--cov-report=term"])
      print("Coverage report generated in htmlcov/")
  ```
- [ ] Add profile command for performance profiling
- [ ] Add setup command for initial environment setup
- [ ] Add version command for package information

### 8. Virtual Environment Management
- [ ] Auto-detect virtual environment:
  ```python
  def ensure_venv():
      """Ensure we're running in a virtual environment."""
      if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
          venv_path = Path("venv/bin/activate")
          if venv_path.exists():
              print("Activating virtual environment...")
              # Handle venv activation
          else:
              print("Warning: No virtual environment detected")
  ```
- [ ] Support different virtual environment tools (venv, conda, etc.)
- [ ] Provide setup instructions when venv missing

### 9. Error Handling and User Experience
- [ ] Add comprehensive error handling
- [ ] Provide helpful error messages with suggestions
- [ ] Add progress indicators for long-running commands
- [ ] Support --help for all commands and subcommands
- [ ] Add command aliases for common operations

### 10. Documentation and Testing
- [ ] Create comprehensive README section for run_dev.py usage
- [ ] Add examples for common workflows
- [ ] Test all commands work correctly
- [ ] Add tab completion support if possible
- [ ] Document integration with IDEs and editors

## Implementation Example
```python
# Example usage patterns
python run_dev.py test --coverage                    # Run tests with coverage
python run_dev.py lint                              # Run all linting tools
python run_dev.py install                           # Set up development environment
python run_dev.py docs --serve                      # Build and serve docs
python run_dev.py test --file test_pricing.py       # Run specific test file
python run_dev.py clean                             # Clean build artifacts
```

## Integration Points
- Integrate with existing scripts/coverage_to_json.py
- Support existing pytest configuration
- Work with current virtual environment setup
- Maintain compatibility with existing workflows

## Output Log
<!-- Add timestamped entries for each subtask completion -->