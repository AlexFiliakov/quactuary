"""
Validation script for integration test infrastructure.

This script validates that the integration test structure is correctly set up
without requiring full dependencies.
"""

import os
import sys
import importlib.util
from pathlib import Path


def validate_file_structure():
    """Validate that all required files exist."""
    base_path = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "conftest.py",
        "test_optimization_combinations.py",
        "test_end_to_end_scenarios.py", 
        "test_performance_validation.py",
        "test_accuracy_validation.py",
        "benchmarks/__init__.py",
        "benchmarks/baseline_results.json",
        "benchmarks/performance_tracking.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(file_path))
    
    return missing_files


def validate_pytest_config():
    """Validate pytest configuration exists."""
    pytest_ini = Path(__file__).parent.parent.parent / "pytest.ini"
    return pytest_ini.exists()


def validate_imports():
    """Validate that test files have correct structure."""
    base_path = Path(__file__).parent
    test_files = [
        "test_optimization_combinations.py",
        "test_end_to_end_scenarios.py",
        "test_performance_validation.py", 
        "test_accuracy_validation.py"
    ]
    
    validation_results = {}
    
    for test_file in test_files:
        file_path = base_path / test_file
        try:
            # Read file and check for key components
            with open(file_path, 'r') as f:
                content = f.read()
            
            checks = {
                'has_imports': 'import pytest' in content,
                'has_test_classes': 'class Test' in content,
                'has_integration_markers': '@pytest.mark.integration' in content,
                'has_docstring': '"""' in content,
                'has_parametrize': '@pytest.mark.parametrize' in content
            }
            
            validation_results[test_file] = checks
            
        except Exception as e:
            validation_results[test_file] = {'error': str(e)}
    
    return validation_results


def main():
    """Run all validations."""
    print("=== Integration Test Infrastructure Validation ===\n")
    
    # File structure validation
    print("1. File Structure Validation:")
    missing_files = validate_file_structure()
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    else:
        print("   ✅ All required files present")
    
    # Pytest config validation
    print("\n2. Pytest Configuration:")
    if validate_pytest_config():
        print("   ✅ pytest.ini found")
    else:
        print("   ❌ pytest.ini missing")
        return False
    
    # Import structure validation
    print("\n3. Test File Structure:")
    import_results = validate_imports()
    all_valid = True
    
    for test_file, checks in import_results.items():
        if 'error' in checks:
            print(f"   ❌ {test_file}: {checks['error']}")
            all_valid = False
        else:
            passed_checks = sum(1 for check in checks.values() if check)
            total_checks = len(checks)
            print(f"   ✅ {test_file}: {passed_checks}/{total_checks} checks passed")
            
            # Show details of failed checks
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                print(f"      ⚠️  Failed: {failed_checks}")
    
    print("\n=== Validation Summary ===")
    if all_valid and not missing_files:
        print("✅ Integration test infrastructure is properly set up!")
        print("\nTo run tests (when dependencies are available):")
        print("  pytest tests/integration/ -m integration")
        print("  pytest tests/integration/ -m 'integration and not slow'")
        print("  pytest tests/integration/ -m 'integration and performance'")
        return True
    else:
        print("❌ Some issues found in integration test infrastructure")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)