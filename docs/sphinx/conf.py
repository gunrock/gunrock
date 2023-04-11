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
import textwrap

sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'gunrock'
copyright = ''

# The full version, including alpha/beta/rc tags
release = '2.0.0'


# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath('_extensions'))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax', 'sphinx.ext.intersphinx', 'sphinx.ext.viewcode', 'sphinx.ext.graphviz',
    'sphinxcontrib.bibtex',
    'sphinx_toolbox.more_autodoc.overloads',
    # 'sphinx_book_theme', 
    # 'sphinx_rtd_theme',
    'breathe', 
    'nw_exhale',
    'myst_parser'
]

source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', '_layouts', '_includes']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '_api', '_bpi'
]

highlight_language = 'c++'
pygments_style = 'emacs'

# Enable numref
numfig = True

# -- Options for MathJax -----------------------------------------------------

mathjax_options = {
}
mathjax3_config = {
    'TeX': {
    'Macros': {
      'RR': "{\\mathbb R}",
      'Real': "{\\mathbb R}",
      'Complex': "{\\mathbb C}",        
      'mat' :["{\\mathbf{#1}}",1],
      'vec' :["{\\mathbf{#1}}",1],       
      'bold': ["{\\bf #1}",1],
        'Spc': ["\\mathbb{#1}",1],
        'norm': ["|| #1 ||",1]        
    }
  }
}


# -- Options for HTML output -------------------------------------------------

html_theme_path = [ "_themes" ]
# html_theme = 'sphinx13'
html_theme = 'sphinx_book_theme'
html_title = "Gunrock Documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css'
]

html_theme_options = {
    "repository_url": "https://github.com/gunrock/gunrock",
    "use_repository_button": True,
    # "home_page_in_toc": True,
    # "show_navbar_depth": 3,
    # "show_toc_level": 3
}
html_logo = "https://raw.githubusercontent.com/gunrock/docs/develop/docs/_media/logo.png"
html_favicon = "https://raw.githubusercontent.com/gunrock/docs/develop/docs/_media/logo.png"
# html_additional_pages = []
# html_copy_source = True
# html_show_source_link = True

# -- Options for the C++ Domain ----------------------------------------------

cpp_index_common_prefix = ['gunrock', 'gunrock::', 'gunrock::graph::']


# -- Options for Breathe -----------------------------------------------------

#sys.path.append('_breathe')

breathe_projects = { "gunrock": "./_doxygen/xml"}
breathe_default_project = "gunrock"

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

todo_include_todos = False


# -- Options for Exhale ------------------------------------------------------

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./_api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "../..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "Gunrock API",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":  textwrap.dedent('''
    INPUT = ../../include
    CLANG_ASSISTED_PARSING = YES
    CLANG_OPTIONS          = -std=c++17
    CLANG_DATABASE_PATH    = ../../build/
    BUILTIN_STL_SUPPORT    = YES
    EXTRACT_ALL            = NO
    GENERATE_HTML          = YES
    HIDE_UNDOC_CLASSES     = YES
    HIDE_UNDOC_MEMBERS     = YES
    LAYOUT_FILE            = "./layout.xml"
    HTML_COLORSTYLE_HUE    = 75
    HTML_DYNAMIC_SECTIONS  = YES
    HTML_INDEX_NUM_ENTRIES = 0
    DISABLE_INDEX          = YES
    GENERATE_TREEVIEW      = YES
    TREEVIEW_WIDTH         = 270
    '''),
    # "verboseBuild": True,
    "listingExclude": [ r'.*_tag_invoke*' ],
    "unabridgedOrphanKinds": { "namespace", "page" }
}


# -- Options for bibtex -- ---------------------------------------------------

bibtex_bibfiles = ['userguide/refs.bib']
