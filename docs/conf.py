project = "Tenseur"
author = "Vincent Choqueuse"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "Tenseur"
html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#569cd6",
        "color-brand-content": "#9cdcfe",
    },
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
