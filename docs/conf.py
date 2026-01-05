# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

sys.path.insert(0, "../src")

project = "Featuristic"
copyright = "2025, Martin Eastwood"
author = "Martin Eastwood"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "nbsphinx",  # Removed: migrating to RST format
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",  # Added: for mathematical formulas
    "sphinx_design",  # Added: for modern UI components (cards, grids, dropdowns)
]

# Autosummary configuration
# Disabled: generating stubs for classes that Sphinx can't import
autosummary_generate = []  # Empty list = don't generate stubs automatically
autosummary_imported_members = True

# Intersphinx mapping for external links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "*.ipynb",
]

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
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/martineastwood/featuristic",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    # "navbar_end": ["theme-switcher", "search-field"],
    # "navbar_start": ["logo"],
    "navbar_center": [],
    "navbar_align": "right",
    # "external_links": [],
    # "header_links_before_dropdown": [],
}


# autosummary_generate = ["api_reference.rst"]

html_static_path = ["_static"]

html_permalinks_icon = "<span>#</span>"
html_theme = "pydata_sphinx_theme"

pygments_style = "sphinx"
