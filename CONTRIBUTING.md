# Contributing to quactuary

Thank you for your interest in contributing to quactuary! We welcome contributions from the community and are excited to work with you.

## Quick Start

1. **Browse Issues**: Check out [open issues](https://github.com/AlexFiliakov/quactuary/issues) to find something that interests you
2. **Fork & Clone**: Fork the repository and clone your fork locally
3. **Set Up Environment**: Follow our [development environment setup guide](https://docs.quactuary.com/development/setting_up_environment.html)
4. **Make Changes**: Create a feature branch and implement your changes
5. **Test**: Ensure your code has ≥90% test coverage with `pytest --cov`
6. **Submit**: Open a pull request with a clear description of your changes

## Development Guidelines

For comprehensive information on contributing to quactuary, please see our detailed development documentation:

**📚 [Complete Development Guide](https://docs.quactuary.com/development/)**

Key sections include:

- **[Contributing Guide](https://docs.quactuary.com/development/contributing.html)** - Complete workflow from fork to merge
- **[Setting Up Environment](https://docs.quactuary.com/development/setting_up_environment.html)** - Development environment setup
- **[Code Standards](https://docs.quactuary.com/development/code_standards.html)** - Python PEP 8, type hints, and project conventions
- **[Testing Guidelines](https://docs.quactuary.com/development/testing_guidelines.html)** - Comprehensive testing with pytest
- **[Documentation Guidelines](https://docs.quactuary.com/development/documentation_guidelines.html)** - Google docstring format and examples
- **[Submitting Changes](https://docs.quactuary.com/development/submitting_changes.html)** - Pull request process and review
- **[Issue Reporting](https://docs.quactuary.com/development/issue_reporting.html)** - Bug reports and feature requests
- **[Community Guidelines](https://docs.quactuary.com/development/community_guidelines.html)** - Code of conduct and community standards

## Requirements Summary

- **Python**: 3.8+ (3.8, 3.9, 3.10, 3.11, 3.12)
- **Code Style**: [PEP 8](https://peps.python.org/pep-0008/) with Black formatting
- **Docstrings**: [Google format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Test Coverage**: ≥90% for all new code
- **Testing Framework**: pytest with coverage reporting

## Code Standards

- Follow [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- Use [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints for all public functions
- Add comprehensive examples to docstrings
- Write tests with ≥90% coverage

## Quick Commands

```bash
# Set up development environment
python -m venv quactuary-dev
source quactuary-dev/bin/activate  # On Windows: quactuary-dev\Scripts\activate
pip install -e .[dev]

# Code formatting and checking
black .
flake8 quactuary/
mypy quactuary/

# Run tests with coverage
pytest --cov=quactuary --cov-report=html

# Build documentation
cd docs/
make html
```

## Getting Help

- **GitHub Issues**: [Ask questions or report bugs](https://github.com/AlexFiliakov/quactuary/issues)
- **Documentation**: [Complete guides and API reference](https://docs.quactuary.com)
- **Discussions**: [Community discussions on GitHub](https://github.com/AlexFiliakov/quactuary/discussions)

## Types of Contributions

We welcome:

- **Bug fixes** and improvements
- **New features** and algorithms
- **Documentation** improvements
- **Tests** and code quality improvements
- **Performance** optimizations
- **Examples** and tutorials
- **Quantum algorithms** research and implementation

## Recognition

Contributors are recognized through:

- Contributors list in repository and documentation
- Acknowledgment in release notes
- Co-authorship opportunities for research contributions
- Community recognition and thanks

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read our [Community Guidelines](https://docs.quactuary.com/development/community_guidelines.html) for details on our code of conduct.

## License

By contributing to quactuary, you agree that your contributions will be licensed under the same license as the project (BSD 3-Clause License).

---

**Ready to contribute?** Start by reading our [complete development guide](https://docs.quactuary.com/development/) and browsing [open issues](https://github.com/AlexFiliakov/quactuary/issues). We're here to help make your contribution experience successful!