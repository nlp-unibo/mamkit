# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

import mamkit  # noqa: E402

project = "MAMKit"
copyright = "2024, LT-NLP-lab UniBO"
author = "Eleonora Mancini, Federico Ruggeri"
release = mamkit.__version__

extensions = [
    "sphinx.ext.autodoc",
    # Cinnamon documents its API in Google style, and these pages render
    # Cinnamon symbols imported into the configuration modules.
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # Builds the Cinnamon registry at documentation-build time and renders
    # every registration as a card, so the page cannot drift from the keys.
    "registry_cards",
]

autodoc_typehints = "description"
autodoc_inherit_docstrings = False

html_theme = "pydata_sphinx_theme"
html_title = "MAMKit"
html_logo = "img/mamkit-logo-image.png"
html_static_path = ["_static"]
html_css_files = ["width.css"]
html_theme_options = {
    "github_url": "https://github.com/nlp-unibo/mamkit",
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {"default_mode": "auto"}
html_sidebars = {
    "index": [],
    "concepts/argument-mining": [],
    "project/contributing": [],
}

exclude_patterns = []
