# docs/source/conf.py
import os, sys
sys.path.insert(0, os.path.abspath("../.."))

# Safer version fetch (works on RTD even if deps aren’t built yet)
try:
    import importlib.metadata as importlib_metadata
    release = importlib_metadata.version("GPmix")
except Exception:
    release = "0.0.0"

project = "GPmix"
author = "E. Akeweje"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # optional but useful:
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    # "show-inheritance": False,
    "special-members": False,
    # "exclude-members": "__init__",
    "private-members": False,
}

autodoc_typehints = "description"   # put type hints in the text
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Do NOT pull __init__ docstring into the class docs
autoclass_content = "class"
# Napoleon: don't merge __init__ doc into class doc
napoleon_include_init_with_doc = False

def _skip_attrs_and_dunder_init(app, what, name, obj, skip, options):
    # Skip all attributes
    if what == "attribute":
        return True
    # Skip all dunder methods
    if name.startswith("__") and name.endswith("__"):
        return True
    # Skip all private members (leading underscore)
    if name.startswith("_"):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", _skip_attrs_and_dunder_init)

# If you have external refs (e.g., NumPy, SciPy):
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# to remove these later I just want to build the docs first
# nitpicky = False
# suppress_warnings = ["autodoc", "autosummary"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]