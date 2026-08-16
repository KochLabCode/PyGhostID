# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('W:/GitHub/GhostID/src'))  # adjust path to point at your package root

project = 'PyGhostID'
copyright = '2026, Daniel Koch'
author = 'Daniel Koch'
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # pulls docstrings from your code
    'sphinx.ext.napoleon',     # lets you write docstrings in NumPy or Google style (readable, not RST)
    'sphinx.ext.viewcode',     # adds links to source code
    'sphinx.ext.mathjax',      # renders LaTeX math in docstrings, if you have any
    'myst_parser',
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
