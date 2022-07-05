# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../vital_sqi/'))

# -- Project information -----------------------------------------------------

project = 'vital_sqi'
copyright = '2022, Oucru'
author = 'Oucru'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',        # source source link next to docs
    'sphinx.ext.githubpages',     # gh-pages needs a .nojekyll file
    'sphinx_gallery.gen_gallery'  # galleries with examples
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. In order to work with 'sphinx_rtd_theme' need
# to install it: $ python -m pip install sphinx-rtd-theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Substitute project name into .rst files when |project_name| is used
rst_epilog = '.. |project_name| replace:: %s' % project


# -- Extensions configuration ------------------------------------------------

# -----------------------
# Napoleon settings
# -----------------------
# The following variables include all the possible settings for the napoleon
# sphinx extension. In addition, the default value is specified in a comment
# for those entries that have been modified.
napoleon_google_docstring = False # Default True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# ----------------------------------------
# Plotly outcomes
# ----------------------------------------
# Include plotly outputs in sphinx-gallery
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery'


# -----------------------
# Sphinx-gallery settings
# -----------------------
# Information about the sphinx gallery configuration
# https://sphinx-gallery.github.io/stable/configuration.html

# Import library
from sphinx_gallery.sorting import FileNameSortKey

# Configuration for sphinx_gallery
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': [
        '../../examples/tutorials',
        '../../examples/preprocess',
        '../../examples/sqi',
    ],
    # path to save gallery generated output
    'gallery_dirs': [
        '../source/_examples/tutorials',
        '../source/_examples/preprocess',
        '../source/_examples/sqi',
    ],
    # Other
    'line_numbers': True,
    'download_all_examples': False,
    'within_subsection_order': FileNameSortKey
}