.. _setting_up_environment:

***************************
Setting Up Your Environment
***************************

This guide will help you set up a development environment for contributing to quactuary. We'll cover everything from installing Python to running your first tests.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
=============

Python Version
--------------

quactuary supports Python 3.8+ (3.8, 3.9, 3.10, 3.11, 3.12). We recommend using the latest stable version for development.

Check your Python version:

.. code-block:: bash

   python --version

If you need to install or upgrade Python:

* **Windows**: Download from `python.org <https://www.python.org/downloads/>`_ or use Microsoft Store
* **macOS**: Use `Homebrew <https://brew.sh/>`_: ``brew install python@3.11``
* **Linux**: Use your distribution's package manager: ``sudo apt install python3.11``

Git
---

You'll need Git for version control:

* **Windows**: Download from `git-scm.com <https://git-scm.com/>`_ or use GitHub Desktop
* **macOS**: ``brew install git`` or use Xcode Command Line Tools
* **Linux**: ``sudo apt install git`` (Ubuntu/Debian) or equivalent

Configure Git with your information:

.. code-block:: bash

   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"

Setting Up the Repository
=========================

Fork and Clone
--------------

1. **Fork the repository** on GitHub: https://github.com/AlexFiliakov/quactuary
2. **Clone your fork** locally:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/quactuary.git
   cd quactuary

3. **Add the upstream remote**:

.. code-block:: bash

   git remote add upstream https://github.com/AlexFiliakov/quactuary.git

4. **Verify your remotes**:

.. code-block:: bash

   git remote -v
   # Should show:
   # origin    https://github.com/YOUR_USERNAME/quactuary.git (fetch)
   # origin    https://github.com/YOUR_USERNAME/quactuary.git (push)
   # upstream  https://github.com/AlexFiliakov/quactuary.git (fetch)
   # upstream  https://github.com/AlexFiliakov/quactuary.git (push)

Python Environment Setup
=========================

Virtual Environment (Recommended)
----------------------------------

Create an isolated Python environment for quactuary development:

**Using venv (built-in):**

.. code-block:: bash

   # Create virtual environment
   python -m venv quactuary-dev
   
   # Activate it
   # On macOS/Linux:
   source quactuary-dev/bin/activate
   
   # On Windows:
   quactuary-dev\Scripts\activate
   
   # Verify activation (should show your venv path)
   which python

**Using conda (if you prefer):**

.. code-block:: bash

   # Create environment with specific Python version
   conda create -n quactuary-dev python=3.11
   
   # Activate it
   conda activate quactuary-dev

Installing Dependencies
=======================

Development Installation
-------------------------

Install quactuary in development mode with all dependencies:

.. code-block:: bash

   # Basic development installation
   pip install -e .[dev]
   
   # Or with quantum support
   pip install -e .[dev,quantum]

This installs:

* **Core dependencies**: numpy, pandas, scipy
* **Testing tools**: pytest, pytest-cov, pytest-xdist
* **Development tools**: black, flake8, mypy, pre-commit
* **Documentation tools**: sphinx, sphinx-rtd-theme
* **Quantum dependencies** (optional): qiskit, qiskit-aer

Manual Installation (Alternative)
----------------------------------

If you prefer to install components separately:

.. code-block:: bash

   # Core dependencies
   pip install numpy pandas scipy matplotlib
   
   # Testing
   pip install pytest pytest-cov pytest-xdist
   
   # Code quality
   pip install black flake8 mypy pre-commit
   
   # Documentation
   pip install sphinx sphinx-rtd-theme nbsphinx
   
   # Quantum (optional)
   pip install qiskit qiskit-aer
   
   # Install quactuary in development mode
   pip install -e .

Verify Installation
===================

Test Basic Functionality
-------------------------

.. code-block:: bash

   # Test imports
   python -c "import quactuary; print(f'quactuary version: {quactuary.__version__}')"
   
   # Run quick test
   python -c "
   from quactuary.distributions import Poisson
   p = Poisson(lambda_=2.0)
   print(f'Poisson(2.0) mean: {p.pmf(1):.3f}')
   "

Run Test Suite
--------------

.. code-block:: bash

   # Run all tests (may take a few minutes)
   pytest
   
   # Run with coverage report
   pytest --cov=quactuary
   
   # Run specific test file
   pytest tests/test_pricing.py -v

Expected output should show all tests passing. If you see failures, check that all dependencies are properly installed.

Development Tools Setup
=======================

Pre-commit Hooks (Recommended)
-------------------------------

Pre-commit hooks automatically format code and catch issues before you commit:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install
   
   # Test the hooks
   pre-commit run --all-files

Code Formatting
---------------

We use `black` for code formatting:

.. code-block:: bash

   # Format all Python files
   black .
   
   # Check what would be formatted
   black --check .

Linting
-------

We use `flake8` for style checking:

.. code-block:: bash

   # Check code style
   flake8 quactuary/
   
   # Check specific file
   flake8 quactuary/pricing.py

Type Checking
-------------

We use `mypy` for static type checking:

.. code-block:: bash

   # Type check the package
   mypy quactuary/
   
   # Type check with more detail
   mypy quactuary/ --strict

IDE/Editor Setup
================

VS Code
-------

Recommended extensions:

* **Python** (Microsoft): Core Python support
* **Pylance** (Microsoft): Enhanced Python language server
* **Python Docstring Generator**: Auto-generate docstring templates
* **GitLens**: Enhanced Git integration
* **Black Formatter**: Automatic code formatting

Settings (add to ``.vscode/settings.json``):

.. code-block:: json

   {
       "python.defaultInterpreterPath": "./quactuary-dev/bin/python",
       "python.formatting.provider": "black",
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.linting.mypyEnabled": true,
       "python.testing.pytestEnabled": true,
       "python.testing.pytestArgs": ["tests/"],
       "editor.formatOnSave": true
   }

PyCharm
-------

Configuration:

1. **Interpreter**: Set to your virtual environment
2. **Code Style**: Set to follow PEP 8
3. **Inspections**: Enable type checking and style warnings
4. **Testing**: Configure pytest as the default test runner

Jupyter Notebooks
------------------

For working with examples and tutorials:

.. code-block:: bash

   # Install Jupyter
   pip install jupyter notebook
   
   # Start notebook server
   jupyter notebook

Docker Environment (Optional)
==============================

If you prefer using Docker:

.. code-block:: bash

   # Build development image
   docker build -t quactuary-dev .
   
   # Run interactive development container
   docker run -it -v $(pwd):/workspace quactuary-dev bash

Common Issues and Solutions
===========================

Import Errors
-------------

**Problem**: ``ModuleNotFoundError`` when importing quactuary

**Solutions**:
1. Ensure virtual environment is activated
2. Reinstall in development mode: ``pip install -e .``
3. Check Python path: ``python -c "import sys; print(sys.path)"``

Test Failures
--------------

**Problem**: Tests fail on clean installation

**Solutions**:
1. Update dependencies: ``pip install --upgrade -e .[dev]``
2. Clear pytest cache: ``pytest --cache-clear``
3. Check for conflicting packages: ``pip list | grep quactuary``

Permission Errors
------------------

**Problem**: Permission denied when installing packages

**Solutions**:
1. Use virtual environment (recommended)
2. Add ``--user`` flag: ``pip install --user -e .``
3. On macOS/Linux, avoid ``sudo pip`` (use virtual environment instead)

Quantum Dependencies
--------------------

**Problem**: Qiskit installation fails

**Solutions**:
1. Try installing without quantum first: ``pip install -e .[dev]``
2. Install Qiskit separately: ``pip install qiskit qiskit-aer``
3. For M1 Macs, you may need: ``pip install qiskit --no-deps`` then install dependencies manually

Performance Issues
-------------------

**Problem**: Tests run very slowly

**Solutions**:
1. Use parallel testing: ``pytest -n auto``
2. Run subset of tests: ``pytest tests/test_pricing.py``
3. Skip slow tests: ``pytest -m "not slow"``

Updating Your Environment
=========================

Keeping Dependencies Updated
----------------------------

.. code-block:: bash

   # Update all packages
   pip install --upgrade -e .[dev]
   
   # Update pre-commit hooks
   pre-commit autoupdate

Syncing with Upstream
---------------------

.. code-block:: bash

   # Fetch latest changes from upstream
   git fetch upstream
   
   # Update your main branch
   git checkout main
   git merge upstream/main
   
   # Push updates to your fork
   git push origin main

Environment Variables
=====================

For quantum development, you may want to set:

.. code-block:: bash

   # For IBM Quantum access
   export QISKIT_IBM_TOKEN="your_token_here"
   
   # For development settings
   export QUACTUARY_DEV_MODE=1

Add these to your shell profile (``.bashrc``, ``.zshrc``, etc.) to make them persistent.

Next Steps
==========

Once your environment is set up:

1. **Read the code standards**: :doc:`code_standards`
2. **Understand testing**: :doc:`testing_guidelines`
3. **Learn about documentation**: :doc:`documentation_guidelines`
4. **Pick an issue**: Browse `GitHub issues <https://github.com/AlexFiliakov/quactuary/issues>`_
5. **Make your first contribution**: Follow our :doc:`contributing` guide

You're now ready to contribute to quactuary!