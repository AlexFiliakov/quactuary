import os
import subprocess
import sys

# -- Path setup --------------------------------------------------------------
# Add the parent directory of quactuary to sys.path so Sphinx can find the package
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

try:
    # e.g. on GitHub Actions GITHUB_REF_NAME == “v1.2.3”
    release = os.environ.get("GITHUB_REF_NAME") \
           or subprocess.check_output(
               ["git","describe","--tags","--always"], 
               stderr=subprocess.DEVNULL
             ).decode().strip()
except Exception:
    from quactuary._version import __version__
    release = __version__

# the short X.Y version
version = release.lstrip("v")

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quActuary'
copyright = '2025, Alex Filiakov, ACAS'
author = 'Alex Filiakov, ACAS'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_sitemap'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Sitemap settings
html_baseurl = 'https://docs.quactuary.com/'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']
