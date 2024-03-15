# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

sys.path.insert(0, "../src")

project = "Featurize"
copyright = "2024, Martin Eastwood"
author = "Martin Eastwood"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

master_doc = "index"

html_logo = "_static/logo.png"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_sidebars = {
    "**": [
        "globaltoc.html",
    ],
}

html_theme_options = {
    "pygment_light_style": "tango",
    "pygment_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/martineastwood/featurize",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

# autosummary_generate = ["api_reference.rst"]

html_static_path = ["_static"]

html_permalinks_icon = "<span>#</span>"
html_theme = "pydata_sphinx_theme"

pygments_style = "sphinx"
