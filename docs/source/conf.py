# # Configuration file for the Sphinx documentation builder.
# import os
# import sys
# import types  # for hard mocks

# # --- Make 'import GPmix' work on RTD & locally
# # THIS_DIR = os.path.dirname(__file__)
# # REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
# # PKG_DIR = os.path.join(REPO_ROOT, "GPmix")
# # sys.path.insert(0, REPO_ROOT)
# # sys.path.insert(0, PKG_DIR)

# # --- Hard-mock heavy/optional deps so autosummary imports don't fail
# MOCK_MODULES = [
#     "numpy", "scipy", "matplotlib", "seaborn",
#     "sklearn", "skfda", "pywt", "joblib", "aeon",
# ]
# for mod in MOCK_MODULES:
#     if mod not in sys.modules:
#         sys.modules[mod] = types.ModuleType(mod)
        
# Safer version fetch (works on RTD even if deps aren’t built yet)
from importlib.metadata import version, PackageNotFoundError
project = "GPmix"
try:
    release = version("GPmix")
except PackageNotFoundError:
    release = "0.0.0"

project = "GPmix"
author = "E. Akeweje"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]
autosummary_generate = True

# # If imports are heavy, mock them so autodoc can import GPmix:
# autodoc_mock_imports = ["sklearn", "matplotlib", "PyWavelets", "skfda", "pandas"]

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