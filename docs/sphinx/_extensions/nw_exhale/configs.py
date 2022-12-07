# -*- coding: utf8 -*-
########################################################################################
# This file is part of exhale.  Copyright (c) 2017-2022, Stephen McDowell.             #
# Full BSD 3-Clause license available here:                                            #
#                                                                                      #
#                https://github.com/svenevs/exhale/blob/master/LICENSE                 #
########################################################################################
'''
The ``configs`` module exists to contain the Sphinx Application configurations specific
to this extension.  Almost every ``global`` variable defined in this file can be
modified using the ``exhale_args`` in ``conf.py``.  The convention for this file is as
follows:

1. Things that are **not** supposed to change, because their value is expected to be
   constant, are declared in ``ALL_CAPS``.  See

   - :data:`~exhale.configs.SECTION_HEADING_CHAR`
   - :data:`~exhale.configs.SUB_SECTION_HEADING_CHAR`
   - :data:`~exhale.configs.SUB_SUB_SECTION_HEADING_CHAR`
   - :data:`~exhale.configs.DEFAULT_DOXYGEN_STDIN_BASE`

2. Internal / private variables that are **not** supposed to changed except for by this
   extension are declared as ``_lower_case_with_single_leading_underscore`` as is common
   in Python ;).

3. Every other variable is declared as ``camelCase``, indicating that it can be
   configured **indirectly** by using it as a key in the arguments to ``exhale_args``
   present in your ``conf.py``.  For example, one of the *required* arguments for this
   extension is :data:`~exhale.configs.containmentFolder`.  This means that the key
   ``"containmentFolder"`` is *expected* to be present in ``exhale_args``.

   .. code-block:: py

      exhale_args = {
         "containmentFolder": "./api",
         # ...
      }

   Read the documentation for the various configs present to see what the various
   options are to modify the behavior of Exhale.
'''

from __future__ import unicode_literals

import os
import six
import textwrap
from pathlib import Path

from sphinx.errors import ConfigError, ExtensionError
from sphinx.util import logging
from types import FunctionType, ModuleType

try:
    # Python 2 StringIO
    from cStringIO import StringIO
except ImportError:
    # Python 3 StringIO
    from io import StringIO


logger = logging.getLogger(__name__)
"""
The |SphinxLoggerAdapter| for communicating with the sphinx build process.

.. |SphinxLoggerAdapter| replace:: :class:`sphinx:sphinx.util.SphinxLoggerAdapter`
"""


########################################################################################
##                                                                                     #
## Required configurations, these get set indirectly via the dictionary argument       #
## given to exhale in your conf.py.                                                    #
##                                                                                     #
########################################################################################
containmentFolder = None
'''
**Required**
    The location where Exhale is going to generate all of the reStructuredText documents.

**Value in** ``exhale_args`` (str)
    The value of key ``"containmentFolder"`` should be a string representing the
    (relative or absolute) path to the location where Exhale will be creating all of the
    files.  **Relative paths are relative to the Sphinx application source directory**,
    which is almost always wherever the file ``conf.py`` is.

    .. note::

       To better help you the user know what Exhale is generating (and therefore safe
       to delete), it is a **hard requirement** that ``containmentFolder`` is a
       **subdirectory** of the Sphinx Source Directory.  AKA the path ``"."`` will be
       rejected, but the path ``"./api"`` will be accepted.

       The suggested value for ``"containmentFolder"`` is ``"./api"``, or
       ``"./source/api"`` if you have separate source and build directories with Sphinx.
       When the html is eventually generated, this will make for a more human friendly
       url being generated.
'''

rootFileName = None
'''
**Required**
    The name of the file that **you** will be linking to from your reStructuredText
    documents.  Do **not** include the ``containmentFolder`` path in this file name,
    Exhale will create the file ``"{contaimentFolder}/{rootFileName}"`` for you.

**Value in** ``exhale_args`` (str)
    The value of key ``"rootFileName"`` should be a string representing the name of
    the file you will be including in your top-level ``toctree`` directive.  In order
    for Sphinx to be happy, you should include a ``.rst`` suffix.  All of the generated
    API uses reStructuredText, and that will not ever change.

    For example, if you specify

    - ``"containmentFolder" = "./api"``, and
    - ``"rootFileName" = "library_root.rst"``

    Then exhale will generate the file ``./api/library_root.rst``.  You would then
    include this file in a ``toctree`` directive (say in ``index.rst``) with:

    .. raw:: html

       <div class="highlight-rest">
         <div class="highlight">
           <pre>
       .. toctree::
          :maxdepth: 2

          about
          <b>api/library_root</b></pre>
         </div>
       </div>
'''

doxygenStripFromPath = None
'''
**Required**
    When building on Read the Docs, there seem to be issues regarding the Doxygen
    variable ``STRIP_FROM_PATH`` when built remotely.  That is, it isn't stripped at
    all.  This value enables Exhale to manually strip the path.

**Value in** ``exhale_args`` (str)
    The value of the key ``"doxygenStripFromPath"`` should be a string representing the
    (relative or absolute) path to be stripped from the final documentation.  As with
    :data:`~exhale.configs.containmentFolder`, relative paths are relative to the Sphinx
    source directory (where ``conf.py`` is).  Consider the following directory structure::

        my_project/
        ├───docs/
        │       conf.py
        │
        └───include/
            └───my_project/
                    common.hpp

    In this scenario, if you supplied ``"doxygenStripFromPath" = ".."``, then the file
    page for ``common.hpp`` would list its declaration as
    ``include/my_project/common.hpp``.  If you instead set it to be ``"../include"``,
    then the file page for ``common.hpp`` would list its declaration as just
    ``my_project/common.hpp``.

    As a consequence, modification of this variable directly affects what shows up in
    the file view hierarchy.  In the previous example, the difference would really just
    be whether or not all files are nestled underneath a global ``include`` folder or
    not.

    .. warning::

       It is **your** responsibility to ensure that the value you provide for this
       configuration is valid.  The file view hierarchy will almost certainly break if
       you give nonsense.

    .. note::

       Depending on your project layout, some links may be broken in the above example
       if you use ``"../include"`` that work when you use ``".."``.  To get your docs
       working, revert to ``".."``.  If you're feeling nice, raise an issue on GitHub
       and let me know --- I haven't been able to track this one down yet :/

       Particularly, this seems to happen with projects that have duplicate filenames
       in different folders, e.g.::

           include/
           └───my_project/
               │    common.hpp
               │
               └───viewing/
                       common.hpp
'''

########################################################################################
##                                                                                     #
## Additional configurations available to further customize the output of exhale.      #
##                                                                                     #
########################################################################################
# Heavily Encouraged Optional Configuration                                            #
########################################################################################
rootFileTitle = None
r'''
**Optional**
    The title to be written at the top of ``rootFileName``, which will appear in your
    file including it in the ``toctree`` directive.

**Value in** ``exhale_args`` (str)
    The value of the key ``"rootFileTitle"`` should be a string that has the title of
    the main library root document folder Exhale will be generating. For example, if
    you are including the Exhale generated library root file in your ``index.rst``
    top-level ``toctree`` directive, the title you supply here will show up on both
    your main page, as well as in the navigation menus.

    An example value could be ``"Library API"``.

.. danger::

    If you are **not** using doxygen pages (``\mainpage``, ``\page``, and/or
    ``\subpage`` commands), then you need to include this argument!  Exhale does not
    have the ability to detect whether or not your project needs this.

    **If** ``\mainpage`` **is used:**
        The title is set to the ``\mainpage`` title unconditionally.
    **Otherwise:**
        The title is set to ``"rootFileTitle"`` (this config).

    Since :data:`~exhale.configs.rootFileName` is ultimately going to be included in a
    ``.. toctree::`` directive, this document needs a title in some way.  Projects
    utilizing the ``\mainpage`` command should not be required to duplicate this title,
    projects **not** using this command **need to supply a title**.
'''
########################################################################################
# Build Process Logging, Colors, and Debugging                                         #
########################################################################################
verboseBuild = False
'''
**Optional**
    If you are having a hard time getting documentation to build, or say hierarchies are
    not appearing as they should be, set this to ``True``.

**Value in** ``exhale_args`` (bool)
    Set the boolean value to be ``True`` to include colorized printing at various stages
    of the build process.

    .. warning::

       There is only one level of verbosity: excessively verbose.  **All logging is
       written to** ``sys.stderr``.  See :data:`~exhale.configs.alwaysColorize`.

    .. tip::

       Looking at the actual code of Exhale trying to figure out what is going on?  All
       logging sections have a comment ``# << verboseBuild`` just before the logging
       section.  So you can ``grep -r '# << verboseBuild' exhale/`` if you're working
       with the code locally.
'''

alwaysColorize = True
'''
**Optional**
    Exhale prints various messages throughout the build process to both ``sys.stdout``
    and ``sys.stderr``.  The default behavior is to colorize output always, regardless
    of if the output is being directed to a file.  This is because you can simply use
    ``cat`` or ``less -R``.  By setting this to ``False``, when redirecting output to
    a file the color will not be included.

**Value in** ``exhale_args`` (bool)
    The default is ``True`` because I find color to be something developers should
    embrace.  Simply use ``less -R`` to view colorized output conveniently.  While I
    have a love of all things color, I understand you may not.  So just set this to
    ``False``.

    .. note::

       There is not and will never be a way to remove the colorized logging from the
       console.  This only controls when ``sys.stdout`` and ``sys.stderr`` are being
       redirected to a file.
'''

generateBreatheFileDirectives = False
'''
**Optional**
    Append the ``.. doxygenfile::`` directive from Breathe for *every* file page
    generated in the API.

**Value in** ``exhale_args`` (bool)
    If True, then the breathe directive (``doxygenfile``) will be incorporated at the
    bottom of the file.

    .. danger::

       **This feature is not intended for production release of pages, only debugging.**

       This feature is "deprecated" in lieu of minimal parsing of the input Doxygen xml
       for a given documented file.  This feature can be used to help determine if
       Exhale has made a mistake in parsing the file level documentation, but usage of
       this feature will create **many** duplicate id's and the Sphinx build process
       will be littered with complaints.

       **Usage of this feature will completely dismantle the links coordinated in all
       parts of Exhale**.  Because duplicate id's are generated, Sphinx chooses where
       to link to.  It seems to reliably choose the links generated by the Breathe File
       directive, meaning the majority of the navigational setup of Exhale is pretty
       much invalidated.
'''

########################################################################################
# Root API Document Customization and Treeview                                         #
########################################################################################
afterTitleDescription = None
'''
**Optional**
    Provide a description to appear just after :data:`~exhale.configs.rootFileTitle`.

**Value in** ``exhale_args`` (str)
    If you want to provide a brief summary of say the layout of the API, or call
    attention to specific classes, functions, etc, use this.  For example, if you had
    Python bindings but no explicit documentation for the Python side of the API, you
    could use something like

    .. code-block:: py

       exhale_args = {
           # ... other required arguments...
           "rootFileTitle": "Library API",
           "afterTitleDescription": textwrap.dedent(\'\'\'
              .. note::

              The following documentation presents the C++ API.  The Python API
              generally mirrors the C++ API, but some methods may not be available in
              Python or may perform different actions.
           \'\'\')
       }
'''

pageHierarchySubSectionTitle = "Page Hierarchy"
'''
**Optional**
    The title for the subsection that comes before the Page hierarchy.

**Value in** ``exhale_args`` (str)
    The default value is simply ``"Page Hierarchy"``.  Change this to be something else if you
    so desire.
'''

afterHierarchyDescription = None
'''
**Optional**
    Provide a description that appears after the Class and File hierarchies, but before
    the full (and usually very long) API listing.

**Value in** ``exhale_args`` (str)
    Similar to :data:`~exhale.configs.afterTitleDescription`, only it is included in the
    middle of the document.
'''

fullApiSubSectionTitle = "Full API"
'''
**Optional**
    The title for the subsection that comes after the Class and File hierarchies, just
    before the enumeration of the full API.

**Value in** ``exhale_args`` (str)
    The default value is simply ``"Full API"``.  Change this to be something else if you
    so desire.
'''

afterBodySummary = None
'''
**Optional**
    Provide a summary to be included at the bottom of the root library file.

**Value in** ``exhale_args`` (str)
    Similar to :data:`~exhale.configs.afterTitleDescription`, only it is included at the
    bottom of the document.

    .. note::

       The root library document generated can be quite long, depending on your
       framework.  Important notes to developers should be included at the top of the
       file using :data:`~exhale.configs.afterTitleDescription`, or after the hierarchies
       using :data:`~exhale.configs.afterHierarchyDescription`.
'''

fullToctreeMaxDepth = 5
'''
**Optional**
    The generated library root document performs ``.. include:: unabridged_api.rst`` at
    the bottom, after the Class and File hierarchies.  Inside ``unabridged_api.rst``,
    every generated file is included using a ``toctree`` directive to prevent Sphinx
    from getting upset about documents not being included.  This value controls the
    ``:maxdepth:`` for all of these ``toctree`` directives.

**Value in** ``exhale_args`` (int)
    The default value is ``5``, but you may want to give a smaller value depending on
    the framework being documented.

    .. warning::

       This value must be greater than or equal to ``1``.  You are advised not to use
       a value greater than ``5``.
'''

listingExclude = []
'''
**Optional**
    A list of regular expressions to exclude from both the class hierarchy and namespace
    page enumerations.  This can be useful when you want to keep the listings for the
    hierarchy / namespace pages more concise, but **do** ultimately want the excluded
    items documented somewhere.

    Nodes whose ``name`` (fully qualified, e.g., ``namespace::ClassName``) matches any
    regular expression supplied here will:

    1. Exclude this item from the class view hierarchy listing.
    2. Exclude this item from the defining namespace's listing (where applicable).
    3. The "excluded" item will still have it's own documentation **and** be linked in
       the "full API listing", as well as from the file page that defined the compound
       (if recovered).  Otherwise Sphinx will explode with warnings about documents not
       being included in any ``toctree`` directives.

    This configuration variable is **one size fits all**.  It was created as a band-aid
    fix for PIMPL frameworks.

    .. todo::

        More fine-grained control will be available in the pickleable writer API
        sometime in Exhale 1.x.

    .. note::

        If you want to skip documentation of a compound in your framework *entirely*,
        this configuration variable is **not** where you do it.  See
        :ref:`Doxygen PREDEFINED <doxygen_predefined>` for information on excluding
        compounds entirely using the doxygen preprocessor.

**Value in** ``exhale_args`` (list)
    The list can be of variable types, but each item will be compiled into an internal
    list using :func:`python:re.compile`.  The arguments for
    ``re.compile(pattern, flags=0)`` should be specified in order, but for convenience
    if no ``flags`` are needed for your use case you can just specify a string.  For
    example:

    .. code-block:: py

        exhale_args = {
            # These two patterns should be equitable for excluding PIMPL
            # objects in a framework that uses the ``XxxImpl`` naming scheme.
            "listingExclude": [r".*Impl$", (r".*impl$", re.IGNORECASE)]
        }

    Each item in ``listingExclude`` may either be a string (the regular expression
    pattern), or it may be a length two iterable ``(string pattern, int flags)``.
'''

# Compiled regular expressions from listingExclude
# TODO: moves into config object
_compiled_listing_exclude = []

unabridgedOrphanKinds = {"dir", "file", "page"}
"""
**Optional**
    The list of node kinds to **exclude** from the unabridged API listing beneath the
    class and file hierarchies.

**Value in** ``exhale_args`` (list or set of strings)
    The list of kinds (see :data:`~exhale.utils.AVAILABLE_KINDS`) that will **not** be
    included in the unabridged API listing.  The default is to exclude pages (which are
    already in the page hierarhcy), directories and files (which are already in the file
    hierarchy).  Note that if this variable is provided, it will overwrite the default
    ``{"dir", "file", "page"}``, meaning if you want to exclude something in addition
    you need to include ``"page"``,  ``"dir"``, and ``"file"``:

    .. code-block:: py

        # In conf.py
        exhale_args = {
            # Case 1: _only_ exclude union
            "unabridgedOrphanKinds": {"union"}
            # Case 2: exclude union in addition to dir / file / page.
            "unabridgedOrphanKinds": {"dir", "file", "page", union"}
        }

    .. tip::

        See :data:`~exhale.configs.fullToctreeMaxDepth`, users seeking to reduce the
        length of the unabridged API should set this value to ``1``.

    .. warning::

        If **either** ``"class"`` **or** ``"struct"`` appear in
        ``unabridgedOrphanKinds`` then **both** will be excluded.  The unabridged API
        will present classes and structs together.
"""

########################################################################################
# Clickable Hierarchies <3                                                             #
########################################################################################
createTreeView = False
'''
**Optional**
    When set to ``True``, clickable hierarchies for the Class and File views will be
    generated.  **Set this variable to** ``True`` **if you are generating html** output
    for much more attractive websites!

**Value in** ``exhale_args`` (bool)
    When set to ``False``, the Class and File hierarches are just reStructuredText
    bullet lists.  This is rather unattractive, but the default of ``False`` is to
    hopefully enable non-html writers to still be able to use ``exhale``.

    .. tip::

       Using ``html_theme = "bootstrap"`` (the `Sphinx Bootstrap Theme`__)?  Make sure
       you set :data:`~exhale.configs.treeViewIsBootstrap` to ``True``!

    __ https://ryan-roemer.github.io/sphinx-bootstrap-theme/
'''

minifyTreeView = True
'''
**Optional**
    When set to ``True``, the generated html and/or json for the class and file
    hierarchy trees will be minified.

**Value in** ``exhale_args`` (bool)
    The default value is ``True``, which should help page load times for larger APIs.
    Setting to ``False`` should only really be necessary if there is a problem -- the
    minified version will be hard to parse as a human.
'''

treeViewIsBootstrap = False
'''
**Optional**
    If the generated html website is using ``bootstrap``, make sure to set this to
    ``True``.  The `Bootstrap Treeview`__ library will be used.

    __ http://jonmiles.github.io/bootstrap-treeview/

**Value in** ``exhale_args`` (bool)
    When set to ``True``, the clickable hierarchies will be generated using a Bootstrap
    friendly library.
'''

treeViewBootstrapTextSpanClass = "text-muted"
'''
**Optional**
    What **span** class to use for the *qualifying* text after the icon, but before the
    hyperlink to the actual documentation page.  For example, ``Struct Foo`` in the
    hierarchy would have ``Struct`` as the *qualifying* text (controlled by this
    variable), and ``Foo`` will be a hyperlink to ``Foo``'s actual documentation.

**Value in** ``exhale_args`` (str)
    A valid class to apply to a ``span``.  The actual HTML being generated is something
    like:

    .. code-block:: html

       <span class="{span_cls}">{qualifier}</span> {hyperlink text}

    So if the value of this input was ``"text-muted"``, and it was the hierarchy element
    for ``Struct Foo``, it would be

    .. code-block:: html

       <span class="text-muted">Struct</span> Foo

    The ``Foo`` portion will receive the hyperlink styling elsewhere.

    .. tip::

       Easy choices to consider are the `contextual classes`__ provided by your
       bootstrap theme.  Alternatively, add your own custom stylesheet to Sphinx
       directly and create a class with the color you want there.

       __ https://getbootstrap.com/docs/3.3/css/#helper-classes-colors

    .. danger::

       No validity checks are performed.  If you supply a class that cannot be used,
       there is no telling what will happen.
'''

treeViewBootstrapIconMimicColor = "text-muted"
'''
**Optional**
    The **paragraph** CSS class to *mimic* for the icon color in the tree view.

**Value in** ``exhale_args`` (str)
    This value must be a valid CSS class for a **paragraph**.  The way that it is used
    is in JavaScript, on page-load, a "fake paragraph" is inserted with the class
    specified by this variable.  The color is extracted, and then a force-override is
    applied to the page's stylesheet.  This was necessary to override some aspects of
    what the ``bootstrap-treeview`` library does.  It's full usage looks like this:

    .. code-block:: js

       /* Inspired by very informative answer to get color of links:
          https://stackoverflow.com/a/2707837/3814202 */
       /*                         vvvvvvvvvv what you give */
       var $fake_p = $('<p class="icon_mimic"></p>').hide().appendTo("body");
       /*                         ^^^^^^^^^^               */
       var iconColor = $fake_p.css("color");
       $fake_p.remove();

       /* later on */
       // Part 2: override the style of the glyphicons by injecting some CSS
       $('<style type="text/css" id="exhaleTreeviewOverride">' +
         '    .treeview span[class~=icon] { '                  +
         '        color: ' + iconColor + ' ! important;'       +
         '    }'                                               +
         '</style>').appendTo('head');


    .. tip::

       Easy choices to consider are the `contextual classes`__ provided by your
       bootstrap theme.  Alternatively, add your own custom stylesheet to Sphinx
       directly and create a class with the color you want there.

       __ https://getbootstrap.com/docs/3.3/css/#helper-classes-colors

    .. danger::

       No validity checks are performed.  If you supply a class that cannot be used,
       there is no telling what will happen.
'''

treeViewBootstrapOnhoverColor = "#F5F5F5"
'''
**Optional**
    The hover color for elements in the hierarchy trees.  Default color is a light-grey,
    as specified by default value of ``bootstrap-treeview``'s `onhoverColor`_.

*Value in** ``exhale_args`` (str)
    Any valid color.  See `onhoverColor`_ for information.

.. _onhoverColor: https://github.com/jonmiles/bootstrap-treeview#onhovercolor
'''

treeViewBootstrapUseBadgeTags = True
'''
**Optional**
    When set to ``True`` (default), a Badge indicating the number of nested children
    will be included **when 1 or more children are present**.

    When enabled, each node in the json data generated has it's `tags`_ set, and the
    global `showTags`_ option is set to ``true``.

    .. _tags: https://github.com/jonmiles/bootstrap-treeview#tags

    .. _showTags: https://github.com/jonmiles/bootstrap-treeview#showtags

**Value in** ``exhale_args`` (bool)
    Set to ``False`` to exclude the badges.  Search for ``Tags as Badges`` on the
    `example bootstrap treeview page`__, noting that if a given node does not have any
    children, no badge will be added.  This is simply because a ``0`` badge is likely
    more confusing than helpful.

    __ http://jonmiles.github.io/bootstrap-treeview/
'''

treeViewBootstrapExpandIcon = "glyphicon glyphicon-plus"
'''
**Optional**
    Global setting for what the "expand" icon is for the bootstrap treeview.  The
    default value here is the default of the ``bootstrap-treeview`` library.

**Value in** ``exhale_args`` (str)
    See the `expandIcon`_ description of ``bootstrap-treeview`` for more information.

    .. _expandIcon: https://github.com/jonmiles/bootstrap-treeview#expandicon

    .. note::

       Exhale handles wrapping this in quotes, you just need to specify the class
       (making sure that it has spaces where it should).  Exhale does **not** perform
       any validity checks on the value of this variable.  For example, you could use
       something like:

       .. code-block:: py

          exhale_args = {
              # ... required / other optional args ...
              # you can set one, both, or neither. just showing both in same example
              # set the icon to show it can be expanded
              "treeViewBootstrapExpandIcon":   "glyphicon glyphicon-chevron-right",
              # set the icon to show it can be collapsed
              "treeViewBootstrapCollapseIcon": "glyphicon glyphicon-chevron-down"
          }
'''

treeViewBootstrapCollapseIcon = "glyphicon glyphicon-minus"
'''
**Optional**
    Global setting for what the "collapse" icon is for the bootstrap treeview.  The
    default value here is the default of the ``bootstrap-treeview`` library.

**Value in** ``exhale_args`` (str)
    See the `collapseIcon`_ description of ``bootstrap-treeview`` for more information.
    See :data:`~exhale.configs.treeViewBootstrapExpandIcon` for how to specify this
    CSS class value.

    .. _collapseIcon: https://github.com/jonmiles/bootstrap-treeview#collapseicon
'''

treeViewBootstrapLevels = 1
'''
**Optional**
    The default number of levels to expand on page load.  Note that the
    ``bootstrap-treeview`` default `levels`_ value is ``2``.  ``1`` seems like a safer
    default for Exhale since the value you choose here largely depends on how you have
    structured your code.

    .. _levels: https://github.com/jonmiles/bootstrap-treeview#levels

**Value in** ``exhale_args`` (int)
    An integer representing the number of levels to expand for **both** the Class and
    File hierarchies.  **This value should be greater than or equal to** ``1``, but
    **no validity checks are performed** on your input.  Buyer beware.
'''

_class_hierarchy_id = "class-treeView"
'''
The ``id`` attribute of the HTML element associated with the **Class** Hierarchy when
:data:`~exhale.configs.createTreeView` is ``True``.

1. When :data:`~exhale.configs.treeViewIsBootstrap` is ``False``, this ``id`` is attached
   to the outer-most ``ul``.
2. For bootstrap, an empty ``div`` is inserted with this ``id``, which will be the
   anchor point for the ``bootstrap-treeview`` library.
'''

_file_hierarchy_id = "file-treeView"
'''
The ``id`` attribute of the HTML element associated with the **Class** Hierarchy when
:data:`~exhale.configs.createTreeView` is ``True``.

1. When :data:`~exhale.configs.treeViewIsBootstrap` is ``False``, this ``id`` is attached
   to the outer-most ``ul``.
2. For bootstrap, an empty ``div`` is inserted with this ``id``, which will be the
   anchor point for the ``bootstrap-treeview`` library.
'''

_page_hierarchy_id = "page-treeView"
'''
The ``id`` attribute of the HTML element associated with the **Page** Hierarchy when
:data:`~exhale.configs.createTreeView` is ``True``.

1. When :data:`~exhale.configs.treeViewIsBootstrap` is ``False``, this ``id`` is attached
   to the outer-most ``ul``.
2. For bootstrap, an empty ``div`` is inserted with this ``id``, which will be the
   anchor point for the ``bootstrap-treeview`` library.
'''

_bstrap_class_hierarchy_fn_data_name = "getClassHierarchyTree"
'''
The name of the JavaScript function that returns the ``json`` data associated with the
**Class** Hierarchy when :data:`~exhale.configs.createTreeView` is ``True`` **and**
:data:`~exhale.configs.treeViewIsBootstrap` is ``True``.
'''

_bstrap_file_hierarchy_fn_data_name = "getFileHierarchyTree"
'''
The name of the JavaScript function that returns the ``json`` data associated with the
**File** Hierarchy when :data:`~exhale.configs.createTreeView` is ``True`` **and**
:data:`~exhale.configs.treeViewIsBootstrap` is ``True``.
'''

_bstrap_page_hierarchy_fn_data_name = "getPageHierarchyTree"
'''
The name of the JavaScript function that returns the ``json`` data associated with the
**Page** Hierarchy when :data:`~exhale.configs.createTreeView` is ``True`` **and**
:data:`~exhale.configs.treeViewIsBootstrap` is ``True``.
'''

########################################################################################
# Page Level Customization                                                             #
########################################################################################
includeTemplateParamOrderList = False
'''
**Optional**
    For Classes and Structs (only), Exhale can provide a numbered list enumeration
    displaying the template parameters in the order they should be specified.

**Value in** ``exhale_args`` (bool)
    This feature can be useful when you have template classes that have **many**
    template parameters.  The Breathe directives **will** include the parameters in the
    order they should be given.  However, if you have a template class with more than
    say 5 parameters, it can become a little hard to read.

    .. note::

       This configuration is all or nothing, and applies to every template Class /
       Struct.  Additionally, **no** ``tparam`` documentation is displayed with this
       listing.  Just the types / names they are declared as (and default values if
       provided).

       This feature really only exists as a historical accident.

.. warning::

   As a consequence of the (hacky) implementation, if you use this feature you commit
   to HTML output only.  Where applicable, template parameters that generate links to
   other items being documented **only** work in HTML.
'''

pageLevelConfigMeta = None
'''
**Optional**
    reStructuredText allows you to employ page-level configurations.  These are included
    at the top of the page, before the title.

**Value in** ``exhale_args`` (str)
    An example of one such feature would be ``":tocdepth: 5"``.  To be honest, I'm not
    sure why you would need this feature.  But it's easy to implement, you just need to
    make sure that you provide valid reStructuredText or *every* page will produce
    errors.

    See the `Field Lists`__ guide for more information.

    __ https://www.sphinx-doc.org/en/master/usage/restructuredtext/field-lists.html
'''

repoRedirectURL = None
'''
.. todo::

   **This feature is NOT implemented yet**!  Hopefully soon.  It definitely gets under
   my skin.  It's mostly documented just to show up in the ``todolist`` for me ;)

**Optional**
    When using the Sphinx RTD theme, there is a button placed in the top-right saying
    something like "Edit this on GitHub".  Since the documents are all being generated
    dynamically (and not supposed to be tracked by ``git``), the links all go nowhere.
    Set this so Exhale can try and fix this.

**Value in** ``exhale_args`` (str)
    The url of the repository your documentation is being generated from.

    .. warning::

       Seriously this isn't implemented.  I may not even need this from you.  The harder
       part is figuring out how to map a given nodes "``def_in_file``" to the correct
       URL.  I should be able to get the URL from ``git remote`` and construct the
       URL from that and ``git branch``.  Probably just some path hacking with
       ``git rev-parse --show-toplevel`` and comparing that to
       :data:`~exhale.configs.doxygenStripFromPath`?

       Please feel free to `add your input here`__.

       __ https://github.com/svenevs/exhale/issues/2
'''

# Using Contents Directives ############################################################
contentsDirectives = True
'''
**Optional**
    Include a ``.. contents::`` directive beneath the title on pages that have potential
    to link to a decent number of documents.

**Value in** ``exhale_args`` (bool)
    By default, Exhale will include a ``.. contents::`` directive on the individual
    generated pages for the types specified by
    :data:`~exhale.configs.kindsWithContentsDirectives`.  Set this to ``False`` to
    disable globally.

    See the :ref:`using_contents_directives` section for all pieces of the puzzle.
'''

contentsTitle = "Contents"
'''
**Optional**
    The title of the ``.. contents::`` directive for an individual file page, when it's
    ``kind`` is in the list specified by
    :data:`~exhale.configs.kindsWithContentsDirectives` **and**
    :data:`~exhale.configs.contentsDirectives` is ``True``.

**Value in** ``exhale_args`` (str)
    The default (for both Exhale and reStructuredText) is to label this as ``Contents``.
    You can choose whatever value you like.  If you prefer to have **no title** for the
    ``.. contents::`` directives, **specify the empty string**.

    .. note::

       Specifying the empty string only removes the title **when** ``":local:"`` **is
       present in** :data:`~exhale.configs.contentsSpecifiers`.  See the
       :ref:`using_contents_directives` section for more information.
'''

contentsSpecifiers = [":local:", ":backlinks: none"]
'''
**Optional**
    The specifications to apply to ``.. contents::`` directives for the individual file
    pages when it's ``kind`` is in the list specified by
    :data:`~exhale.configs.kindsWithContentsDirectives` **and**
    :data:`~exhale.configs.contentsDirectives` is ``True``.

**Value in** ``exhale_args`` (list)
    A (one-dimensional) list of strings that will be applied to any ``.. contents::``
    directives generated.  Provide the **empty list** if you wish to have no specifiers
    added to these directives.  See the :ref:`using_contents_directives` section for
    more information.
'''

kindsWithContentsDirectives = ["file", "namespace"]
'''
**Optional**
    The kinds of compounds that will include a ``.. contents::`` directive on their
    individual library page.  The default is to generate one for Files and Namespaces.
    Only takes meaning when :data:`~exhale.configs.contentsDirectives` is ``True``.

**Value in** ``exhale_args`` (list)
    Provide a (one-dimensional) ``list`` or ``tuple`` of strings of the kinds of
    compounds that should include a ``.. contents::`` directive.  Each kind given
    must one of the entries in :data:`~exhale.utils.AVAILABLE_KINDS`.

    For example, if you wanted to enable Structs and Classes as well you would do
    something like:

    .. code-block:: py

       # in conf.py
       exhale_args = {
           # ... required / optional args ...
           "kindsWithContentsDirectives": ["file", "namespace", "class", "struct"]
       }

    .. note::

       This is a "full override".  So if you want to still keep the defaults of
       ``"file"`` and ``"namespace"``, **you** must include them yourself.
'''

########################################################################################
# Breathe Customization                                                                #
########################################################################################
customSpecificationsMapping = None
'''
**Optional**
    See the :ref:`usage_customizing_breathe_output` section for how to use this.

**Value in** ``exhale_args`` (dict)
    The dictionary produced by calling
    :func:`~exhale.utils.makeCustomSpecificationsMapping` with your custom function.
'''

_closure_map_sanity_check = "blargh_BLARGH_blargh"
'''
See :func:`~exhale.utils.makeCustomSpecificationsMapping` implementation, this is
inserted to help enforce that Exhale made the dictionary going into
:data:`~exhale.configs.customSpecificationsMapping`.
'''

########################################################################################
# Doxygen Execution and Customization                                                  #
########################################################################################
_doxygen_xml_output_directory = None
'''
The absolute path the the root level of the doxygen xml output.  If the path to the
``index.xml`` file created by doxygen was ``./_doxygen/xml/index.xml``, then this would
simply be ``./_doxygen/xml``.

.. note::

   This is the exact same path as ``breathe_projects[breathe_default_project]``, only it
   is an absolute path.
'''

exhaleExecutesDoxygen = False
'''
**Optional**
    Have Exhale launch Doxygen when you execute ``make html``.

**Value in** ``exhale_args`` (bool)
    Set to ``True`` to enable launching Doxygen.  You must set either
    :data:`~exhale.configs.exhaleUseDoxyfile` or :data:`~exhale.configs.exhaleDoxygenStdin`.
'''

exhaleUseDoxyfile = False
'''
**Optional**
    If :data:`~exhale.configs.exhaleExecutesDoxygen` is ``True``, this tells Exhale to
    use your own ``Doxyfile``.  The encouraged approach is to use
    :data:`~exhale.configs.exhaleDoxygenStdin`.

**Value in** ``exhale_args`` (bool)
    Set to ``True`` to have Exhale use your ``Doxyfile``.

    .. note::

       The ``Doxyfile`` must be in the **same** directory as ``conf.py``.  Exhale will
       change directories to here before launching Doxygen when you have separate source
       and build directories for Sphinx configured.

    .. warning::

       No sanity checks on the ``Doxyfile`` are performed.  If you are using this option
       you need to verify two parameters in particular:

       1. ``OUTPUT_DIRECTORY`` is configured so that
          ``breathe_projects[breathe_default_project]`` agrees.  See the
          :ref:`Mapping of Project Names to Doxygen XML Output Paths <breathe_project>`
          section.

       2. ``STRIP_FROM_PATH`` is configured to be identical to what is specified with
          :data:`~exhale.configs.doxygenStripFromPath`.

       I have no idea what happens when these conflict, but it likely will never result
       in valid documentation.
'''

exhaleDoxygenStdin = None
'''
**Optional**
    If :data:`~exhale.configs.exhaleExecutesDoxygen` is ``True``, this tells Exhale to
    use the (multiline string) value specified in this argument *in addition to* the
    :data:`~exhale.configs.DEFAULT_DOXYGEN_STDIN_BASE`.

**Value in** ``exhale_args`` (str)
    This string describes your project's specific Doxygen configurations.  At the very
    least, it must provide ``INPUT``.  See the :ref:`usage_exhale_executes_doxygen`
    section for how to use this in conjunction with the default configurations, as well
    as how to override them.
'''

DEFAULT_DOXYGEN_STDIN_BASE = textwrap.dedent(r'''
    # If you need this to be YES, exhale will probably break.
    CREATE_SUBDIRS         = NO
    # So that only Doxygen does not trim paths, which affects the File hierarchy
    FULL_PATH_NAMES        = YES
    # Nested folders will be ignored without this.  You may not need it.
    RECURSIVE              = YES
    # Set to YES if you are debugging or want to compare.
    GENERATE_HTML          = NO
    # Unless you want it...
    GENERATE_LATEX         = NO
    # Both breathe and exhale need the xml.
    GENERATE_XML           = YES
    # Set to NO if you do not want the Doxygen program listing included.
    XML_PROGRAMLISTING     = YES
    # Allow for rst directives and advanced functions e.g. grid tables
    ALIASES                = "rst=\verbatim embed:rst:leading-asterisk"
    ALIASES               += "endrst=\endverbatim"
    # Enable preprocessing and related preprocessor necessities
    ENABLE_PREPROCESSING   = YES
    MACRO_EXPANSION        = YES
    EXPAND_ONLY_PREDEF     = NO
    SKIP_FUNCTION_MACROS   = NO
    # extra defs for to help with building the _right_ version of the docs
    PREDEFINED             = DOXYGEN_DOCUMENTATION_BUILD
    PREDEFINED            += DOXYGEN_SHOULD_SKIP_THIS
''')
'''
These are the default values sent to Doxygen along stdin when
:data:`~exhale.configs.exhaleExecutesDoxygen` is ``True``.  This is sent to Doxygen
immediately **before** the :data:`~exhale.configs.exhaleDoxygenStdin` provided to
``exhale_args`` in your ``conf.py``.  In this way, you can override any of the specific
defaults shown here.

.. tip::

   See the documentation for :data:`~exhale.configs.exhaleDoxygenStdin`, as well as
   :data:`~exhale.configs.exhaleUseDoxyfile`.  Only **one** may be provided to the
   ``exhale_args`` in your ``conf.py``.

.. include:: ../DEFAULT_DOXYGEN_STDIN_BASE_value.rst
'''

exhaleSilentDoxygen = False
'''
**Optional**
    When set to ``True``, the Doxygen output is omitted from the build.

**Value in** ``exhale_args`` (bool)
    Documentation generation can be quite verbose, especially when running both Sphinx
    and Doxygen in the same process.  Use this to silence Doxygen.

    .. danger::

       You are **heavily** discouraged from setting this to ``True``.  Many problems
       that may arise through either Exhale or Breathe are because the Doxygen
       documentation itself has errors.  It will be much more difficult to find these
       when you squelch the Doxygen output.

       The reason you would do this is for actual limitations on your specific
       ``stdout`` (e.g. you are getting a buffer maxed out).  The likelihood of this
       being a problem for you is exceptionally small.
'''

########################################################################################
# Programlisting Customization                                                         #
########################################################################################
lexerMapping = {}
r'''
**Optional**
    When specified, and ``XML_PROGRAMLISTING`` is set to ``YES`` in Doxygen (either via
    your ``Doxyfile`` or :data:`exhaleDoxygenStdin <exhale.configs.exhaleDoxygenStdin>`),
    this mapping can be used to customize / correct the Pygments lexer used for the
    program listing page generated for files.  Most projects will **not** need to use
    this setting.

**Value in** ``exhale_args`` (dict)
    The keys and values are both strings.  Each key is a regular expression that will be
    used to check with :func:`python:re.match`, noting that the primary difference
    between :func:`python:re.match` and :func:`python:re.search` that you should be
    aware of is that ``match`` searches from the **beginning** of the string.  Each
    value should be a **valid** `Pygments lexer <http://pygments.org/docs/lexers/>`_.

    Example usage:

    .. code-block:: py

       exhale_args {
           # ...
           "lexerMapping": {
               r".*\.cuh": "cuda",
               r"path/to/exact_filename\.ext": "c"
           }
       }

    .. note::

       The pattern is used to search the full path of a file, **as represented in
       Doxygen**.  This is so that duplicate file names in separate folders can be
       distinguished if needed.  The file path as represented in Doxygen is defined
       by the path to the file, with some prefix stripped out.  The prefix stripped out
       depends entirely on what you provided to
       :data:`doxygenStripFromPath <exhale.configs.doxygenStripFromPath>`.

    .. tip::

       This mapping is used in
       :func:`utils.doxygenLanguageToPygmentsLexer <exhale.utils.doxygenLanguageToPygmentsLexer>`,
       when provided it is queried first.  If you are trying to get program listings for
       a file that is otherwise not supported directly by Doxygen, you typically want to
       tell Doxygen to interpret the file as a different language.  Take the CUDA case.
       In my input to :data:`exhaleDoxygenStdin <exhale.configs.exhaleDoxygenStdin>`, I
       will want to set both ``FILE_PATTERNS`` and append to ``EXTENSION_MAPPING``:

       .. code-block:: make

          FILE_PATTERNS          = *.hpp *.cuh
          EXTENSION_MAPPING     += cuh=c++

       By setting ``FILE_PATTERNS``, Doxygen will now try and process ``*.cuh`` files.
       By *appending* to ``EXTENSION_MAPPING``, it will treat ``*.cuh`` as C++ files.
       For CUDA, this is a reasonable choice because Doxygen is generally able to parse
       the file as C++ and get everything right in terms of member definitions,
       docstrings, etc.  **However**, now the XML generated by doxygen looks like this:

       .. code-block:: xml

          <!-- >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> vvv -->
          <compounddef id="bilateral__filter_8cuh" kind="file" language="C++">

       So Exhale would be default put the program listing in a ``.. code-block:: cpp``.
       By setting this variable in ``exhale_args``, you can bypass this and get the
       desired lexer of your choice.

    Some important notes for those not particularly comfortable or familiar with regular
    expressions in python:

    1. Note that each key defines a *raw* string (prefix with ``r``): ``r"pattern"``.
       This is not entirely necessary for this case, but using raw strings makes it so
       that you do not have to escape as many things.  It's a good practice to adopt,
       but for these purposes should not matter all that much.

    2. Note the escaped ``.`` character.  This means find the literal ``.``, rather than
       the regular expression wildcard for *any character*.  Observe the difference
       with and without:

       .. code-block:: pycon

          >>> import re
          >>> if re.match(r".*.cuh", "some_filecuh.hpp"): print("Oops!")
          ...
          Oops!
          >>> if re.match(r".*\.cuh", "some_filecuh.hpp"): print("Oops!")
          ...
          >>>

       Without ``\.``, the ``.cuh`` matches ``ecuh`` since ``.`` is a wildcard for *any*
       character.  You may also want to use ``$`` at the end of the expression if there
       are multiple file extensions involved: ``r".*\.cuh$"``.  The ``$`` states
       "end-of-pattern", which in the usage of Exhale means end of line (the compiled
       regular expressions are not compiled with :data:`python:re.MULTILINE`).

    3. Take special care at the beginning of your regular expression.  The pattern
       ``r"*\.cuh"`` does **not** compile!  You need to use ``r".*\.cuh"``, with the
       leading ``.`` being required.
'''

_compiled_lexer_mapping = {}
'''
Internal mapping of compiled regular expression objects to Pygments lexer strings.  This
dictionary is created by compiling every key in
:data:`lexerMapping <exhale.configs.lexerMapping>`.  See implementation of
:func:`utils.doxygenLanguageToPygmentsLexer <exhale.utils.doxygenLanguageToPygmentsLexer>`
for usage.
'''

########################################################################################
##                                                                                     #
## Utility variables.                                                                  #
##                                                                                     #
########################################################################################
SECTION_HEADING_CHAR = "="
''' The restructured text H1 heading character used to underline sections. '''

SUB_SECTION_HEADING_CHAR = "-"
''' The restructured text H2 heading character used to underline subsections. '''

SUB_SUB_SECTION_HEADING_CHAR = "*"
''' The restructured text H3 heading character used to underline sub-subsections. '''

MAXIMUM_FILENAME_LENGTH = 255
'''
When a potential filename is longer than ``255``, a sha1 sum is used to shorten.  Note
that there is no ubiquitous and reliable way to query this information, as it depends
on both the operating system, filesystem, **and** even the location (directory path) the
file would be generated to (depending on the filesystem).  As such, a conservative value
of ``255`` should guarantee that the desired filename can always be created.
'''

MAXIMUM_WINDOWS_PATH_LENGTH = 260
r'''
The file path length on Windows cannot be greater than or equal to ``260`` characters.

Since Windows' pathetically antiquated filesystem cannot handle this, they have enabled
a "magic" prefix they call an *extended-length path*.  This is achieved by inserting
the prefix ``\\?\`` which allows you to go up to a maximum path of ``32,767`` characters
**but you may only do this for absolute paths**.  See `Maximum Path Length Limitation`__
for more information.

Dear Windows, did you know it is the 21st century?

__ https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
'''

_the_app = None
''' The Sphinx ``app`` object.  Currently unused, saved for availability in future. '''

_app_src_dir = None
'''
**Do not modify**.  The location of ``app.srcdir`` of the Sphinx application, once the
build process has begun to execute.  Saved to be able to run a few different sanity
checks in different places.
'''

_on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
'''
**Do not modify**.  Signals whether or not the build is taking place on ReadTheDocs.  If
it is, then colorization of output is disabled, as well as the Doxygen output (where
applicable) is directed to ``/dev/null`` as capturing it can cause the ``subprocess``
buffers to overflow.
'''

########################################################################################
##                                                                                     #
## Secondary Sphinx Entry Point                                                        #
## Called from exhale/__init__.py:environment_ready during the sphinx build process.   #
##                                                                                     #
########################################################################################
def apply_sphinx_configurations(app):
    '''
    This method applies the various configurations users place in their ``conf.py``, in
    the dictionary ``exhale_args``.  The error checking seems to be robust, and
    borderline obsessive, but there may very well be some glaring flaws.

    When the user requests for the ``treeView`` to be created, this method is also
    responsible for adding the various CSS / JavaScript to the Sphinx Application
    to support the hierarchical views.

    .. danger::

       This method is **not** supposed to be called directly.  See
       ``exhale/__init__.py`` for how this function is called indirectly via the Sphinx
       API.

    **Parameters**
        ``app`` (:class:`sphinx.application.Sphinx`)
            The Sphinx Application running the documentation build.
    '''
    # Import local to function to prevent circular imports elsewhere in the framework.
    from . import deploy
    from . import utils
    ####################################################################################
    # Make sure they have the `breathe` configs setup in a way that we can use them.   #
    ####################################################################################
    # Breathe allows users to have multiple projects to configure in one `conf.py`
    # A dictionary of keys := project names, values := path to Doxygen xml output dir
    breathe_projects = app.config.breathe_projects
    if not breathe_projects:
        raise ConfigError("You must set the `breathe_projects` in `conf.py`.")
    elif type(breathe_projects) is not dict:
        raise ConfigError("The type of `breathe_projects` in `conf.py` must be a dictionary.")
    # The breathe_default_project is required by `exhale` to determine where to look for
    # the doxygen xml.
    #
    # TODO: figure out how to allow multiple breathe projects?
    breathe_default_project = app.config.breathe_default_project
    if not breathe_default_project:
        raise ConfigError("You must set the `breathe_default_project` in `conf.py`.")
    elif not isinstance(breathe_default_project, six.string_types):
        raise ConfigError("The type of `breathe_default_project` must be a string.")

    if breathe_default_project not in breathe_projects:
        raise ConfigError(
            "The given breathe_default_project='{0}' was not a valid key in `breathe_projects`:\n{1}".format(
                breathe_default_project, breathe_projects
            )
        )

    # Grab where the Doxygen xml output is supposed to go, make sure it is a string,
    # defer validation of existence until after potentially running Doxygen based on
    # the configs given to exhale
    doxy_xml_dir = breathe_projects[breathe_default_project]
    if not isinstance(doxy_xml_dir, six.string_types):
        raise ConfigError(
            "The type of `breathe_projects[breathe_default_project]` from `conf.py` was not a string."
        )

    # Make doxy_xml_dir relative to confdir (where conf.py is)
    if not os.path.isabs(doxy_xml_dir):
        doxy_xml_dir = os.path.abspath(os.path.join(app.confdir, doxy_xml_dir))

    ####################################################################################
    # Initial sanity-check that we have the arguments needed.                          #
    ####################################################################################
    exhale_args = app.config.exhale_args
    if not exhale_args:
        raise ConfigError("You must set the `exhale_args` dictionary in `conf.py`.")
    elif type(exhale_args) is not dict:
        raise ConfigError("The type of `exhale_args` in `conf.py` must be a dictionary.")

    ####################################################################################
    # In order to be able to loop through things below, we want to grab the globals    #
    # dictionary (rather than needing to do `global containmentFolder` etc for every   #
    # setting that is being changed).                                                  #
    ####################################################################################
    configs_globals = globals()
    # Used for internal verification of available keys
    keys_available = []
    # At the end of input processing, fail out if unrecognized keys were found.
    keys_processed = []

    ####################################################################################
    # Gather the mandatory input for exhale.                                           #
    ####################################################################################
    key_error = "Did not find required key `{key}` in `exhale_args`."
    val_error = "The type of the value for key `{key}` must be `{exp}`, but was `{got}`."

    req_kv = [
        ("containmentFolder",    six.string_types,  True),
        ("rootFileName",         six.string_types, False),
        ("doxygenStripFromPath", six.string_types,  True)
    ]
    for key, expected_type, make_absolute in req_kv:
        # Used in error checking later
        keys_available.append(key)

        # Make sure we have the key
        if key not in exhale_args:
            raise ConfigError(key_error.format(key=key))
        # Make sure the value is at the very least the correct type
        val = exhale_args[key]
        if not isinstance(val, expected_type):
            val_t = type(val)
            raise ConfigError(val_error.format(key=key, exp=expected_type, got=val_t))
        # Make sure that a value was provided (e.g. no empty strings)
        if not val:
            raise ConfigError("Non-empty value for key [{0}] required.".format(key))
        # If the string represents a path, make it absolute
        if make_absolute:
            # Directories are made absolute relative to app.confdir (where conf.py is)
            if not os.path.isabs(val):
                val = os.path.abspath(os.path.join(os.path.abspath(app.confdir), val))
        # Set the config for use later
        try:
            configs_globals[key] = val
            keys_processed.append(key)
        except Exception as e:
            raise ExtensionError(
                "Critical error: unable to set `global {0}` to `{1}` in exhale.configs:\n{2}".format(
                    key, val, e
                )
            )

    ####################################################################################
    # Validate what can be checked from the required arguments at this time.           #
    ####################################################################################
    global _the_app
    _the_app = app

    # Require that containmentFolder is a subpath of the sphinx application source
    # directory (otherwise Sphinx will not process the generated documents).
    containment_folder_parent = Path(containmentFolder).absolute()
    app_srcdir = Path(app.srcdir).absolute()
    try:
        # relative_to will raise if it is not a subchild
        containment_folder_parent.relative_to(app_srcdir)
        # but if it is the same path (docs/ directory) relative_to succeeds
        if containment_folder_parent == app_srcdir:
            raise ValueError
    except:
        raise ConfigError(
            "The given `containmentFolder` [{0}] must be a *SUBDIRECTORY* of [{1}].".format(
                containmentFolder, app.srcdir
            )
        )

    global _app_src_dir
    _app_src_dir = os.path.abspath(app.srcdir)

    # We *ONLY* generate reStructuredText, make sure Sphinx is expecting this as well as
    # the to-be-generated library root file is correctly suffixed.
    if not rootFileName.endswith(".rst"):
        raise ConfigError(
            "The given `rootFileName` ({0}) did not end with '.rst'; Exhale is reStructuredText only.".format(
                rootFileName
            )
        )
    if ".rst" not in app.config.source_suffix:
        raise ConfigError(
            "Exhale is reStructuredText only, but '.rst' was not found in `source_suffix` list of `conf.py`."
        )

    # Make sure the doxygen strip path is an exclude-able path
    if not os.path.exists(doxygenStripFromPath):
        raise ConfigError(
            "The path given as `doxygenStripFromPath` ({0}) does not exist!".format(doxygenStripFromPath)
        )

    ####################################################################################
    # Gather the optional input for exhale.                                            #
    ####################################################################################
    # TODO: `list` -> `(list, tuple)`, update docs too.
    opt_kv = [
        ("rootFileTitle",                   six.string_types),
        # Build Process Logging, Colors, and Debugging
        ("verboseBuild",                                bool),
        ("alwaysColorize",                              bool),
        ("generateBreatheFileDirectives",               bool),
        # Root API Document Customization and Treeview
        ("afterTitleDescription",           six.string_types),
        ("pageHierarchySubSectionTitle",    six.string_types),
        ("afterHierarchyDescription",       six.string_types),
        ("fullApiSubSectionTitle",          six.string_types),
        ("afterBodySummary",                six.string_types),
        ("fullToctreeMaxDepth",                          int),
        ("listingExclude",                              list),
        ("unabridgedOrphanKinds",                (list, set)),
        # Clickable Hierarchies <3
        ("createTreeView",                              bool),
        ("minifyTreeView",                              bool),
        ("treeViewIsBootstrap",                         bool),
        ("treeViewBootstrapTextSpanClass",  six.string_types),
        ("treeViewBootstrapIconMimicColor", six.string_types),
        ("treeViewBootstrapOnhoverColor",   six.string_types),
        ("treeViewBootstrapUseBadgeTags",               bool),
        ("treeViewBootstrapExpandIcon",     six.string_types),
        ("treeViewBootstrapCollapseIcon",   six.string_types),
        ("treeViewBootstrapLevels",                      int),
        # Page Level Customization
        ("includeTemplateParamOrderList",               bool),
        ("pageLevelConfigMeta",             six.string_types),
        ("repoRedirectURL",                 six.string_types),
        ("contentsDirectives",                          bool),
        ("contentsTitle",                   six.string_types),
        ("contentsSpecifiers",                          list),
        ("kindsWithContentsDirectives",                 list),
        # Breathe Customization
        ("customSpecificationsMapping",                 dict),
        # Doxygen Execution and Customization
        ("exhaleExecutesDoxygen",                       bool),
        ("exhaleUseDoxyfile",                           bool),
        ("exhaleDoxygenStdin",              six.string_types),
        ("exhaleSilentDoxygen",                         bool),
        # Programlisting Customization
        ("lexerMapping",                                 dict)
    ]
    for key, expected_type in opt_kv:
        # Used in error checking later
        keys_available.append(key)

        # Override the default settings if the key was provided
        if key in exhale_args:
            # Make sure the value is at the very least the correct type
            val = exhale_args[key]
            if not isinstance(val, expected_type):
                val_t = type(val)
                raise ConfigError(val_error.format(key=key, exp=expected_type, got=val_t))
            # Set the config for use later
            try:
                configs_globals[key] = val
                keys_processed.append(key)
            except Exception as e:
                raise ExtensionError(
                    "Critical error: unable to set `global {0}` to `{1}` in exhale.configs:\n{2}".format(
                        key, val, e
                    )
                )

    # These two need to be lists of strings, check to make sure
    def _list_of_strings(lst, title):
        for spec in lst:
            if not isinstance(spec, six.string_types):
                raise ConfigError(
                    "`{title}` must be a list of strings.  `{spec}` was of type `{spec_t}`".format(
                        title=title,
                        spec=spec,
                        spec_t=type(spec)
                    )
                )

    _list_of_strings(         contentsSpecifiers,          "contentsSpecifiers")
    _list_of_strings(kindsWithContentsDirectives, "kindsWithContentsDirectives")
    _list_of_strings(      unabridgedOrphanKinds,       "unabridgedOrphanKinds")

    # Make sure the kinds they specified are valid
    unknown = "Unknown kind `{kind}` given in `{config}`.  See utils.AVAILABLE_KINDS."
    for kind in kindsWithContentsDirectives:
        if kind not in utils.AVAILABLE_KINDS:
            raise ConfigError(
                unknown.format(kind=kind, config="kindsWithContentsDirectives")
            )
    for kind in unabridgedOrphanKinds:
        if kind not in utils.AVAILABLE_KINDS:
            raise ConfigError(
                unknown.format(kind=kind, config="unabridgedOrphanKinds")
            )

    # Make sure the listingExlcude is usable
    if "listingExclude" in exhale_args:
        import re
        # TODO: remove this once config objects are in.  Reset needed for testing suite.
        configs_globals["_compiled_listing_exclude"] = []

        # used for error printing, tries to create string out of item otherwise
        # returns 'at index {idx}'
        def item_or_index(item, idx):
            try:
                return "`{item}`".format(item=item)
            except:
                return "at index {idx}".format(idx=idx)

        exclusions = exhale_args["listingExclude"]
        for idx in range(len(exclusions)):
            # Gather the `pattern` and `flags` parameters for `re.compile`
            item = exclusions[idx]
            if isinstance(item, six.string_types):
                pattern = item
                flags   = 0
            else:
                try:
                    pattern, flags = item
                except Exception as e:
                    raise ConfigError(
                        "listingExclude item {0} cannot be unpacked as `pattern, flags = item`:\n{1}".format(
                            item_or_index(item, idx), e
                        )
                    )
            # Compile the regular expression object.
            try:
                regex = re.compile(pattern, flags)
            except Exception as e:
                raise ConfigError(
                    "Unable to compile specified listingExclude {0}:\n{1}".format(
                        item_or_index(item, idx), e
                    )
                )
            configs_globals["_compiled_listing_exclude"].append(regex)

    # Make sure the lexerMapping is usable
    if "lexerMapping" in exhale_args:
        from pygments import lexers
        import re
        # TODO: remove this once config objects are in.  Reset needed for testing suite.
        configs_globals["_compiled_lexer_mapping"] = {}

        lexer_mapping = exhale_args["lexerMapping"]
        for key in lexer_mapping:
            val = lexer_mapping[key]
            # Make sure both are strings
            if not isinstance(key, six.string_types) or not isinstance(val, six.string_types):
                raise ConfigError("All keys and values in `lexerMapping` must be strings.")
            # Make sure the key is a valid regular expression
            try:
                regex = re.compile(key)
            except Exception as e:
                raise ConfigError(
                    "The `lexerMapping` key [{0}] is not a valid regular expression: {1}".format(key, e)
                )
            # Make sure the provided lexer is available
            try:
                lex = lexers.find_lexer_class_by_name(val)
            except Exception as e:
                raise ConfigError(
                    "The `lexerMapping` value of [{0}] for key [{1}] is not a valid Pygments lexer.".format(
                        val, key
                    )
                )
            # Everything works, stash for later processing
            configs_globals["_compiled_lexer_mapping"][regex] = val

    ####################################################################################
    # Internal consistency check to make sure available keys are accurate.             #
    ####################################################################################
    # See naming conventions described at top of file for why this is ok!
    keys_expected = []
    for key in configs_globals.keys():
        val = configs_globals[key]
        # Ignore modules and functions
        if not isinstance(val, FunctionType) and not isinstance(val, ModuleType):
            if key != "logger":  # band-aid for logging api with Sphinx prior to config objects
                # Ignore specials like __name__ and internal variables like _the_app
                if "_" not in key and len(key) > 0:  # don't think there can be zero length ones...
                    first = key[0]
                    if first.isalpha() and first.islower():
                        keys_expected.append(key)

    keys_expected  = set(keys_expected)
    keys_available = set(keys_available)
    if keys_expected != keys_available:
        err = StringIO()
        err.write(textwrap.dedent('''
            CRITICAL: Exhale encountered an internal error, please raise an Issue on GitHub:

                https://github.com/svenevs/exhale/issues

            Please paste the following in the issue report:

            Expected keys:

        '''))
        for key in keys_expected:
            err.write("- {0}\n".format(key))
        err.write(textwrap.dedent('''
            Available keys:

        '''))
        for key in keys_available:
            err.write("- {0}\n".format(key))
        err.write(textwrap.dedent('''
            The Mismatch(es):

        '''))
        for key in (keys_available ^ keys_expected):
            err.write("- {0}\n".format(key))

        err_msg = err.getvalue()
        err.close()
        raise ExtensionError(err_msg)

    ####################################################################################
    # See if unexpected keys were presented.                                           #
    ####################################################################################
    all_keys = set(exhale_args.keys())
    keys_processed = set(keys_processed)
    if all_keys != keys_processed:
        # Much love: https://stackoverflow.com/a/17388505/3814202
        from difflib import SequenceMatcher

        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio() * 100.0

        # If there are keys left over after taking the differences of keys_processed
        # (which is all keys Exhale expects to see), inform the user of keys they might
        # have been trying to provide.
        #
        # Convert everything to lower case for better matching success
        potential_keys = keys_available - keys_processed
        potential_keys_lower = {key.lower(): key for key in potential_keys}
        extras = all_keys - keys_processed
        extra_error = StringIO()
        extra_error.write("Exhale found unexpected keys in `exhale_args`:\n")
        for key in extras:
            extra_error.write("  - Extra key: {0}\n".format(key))
            potentials = []
            for mate in potential_keys_lower:
                similarity = similar(key, mate)
                if similarity > 50.0:
                    # Output results with the non-lower version they should put in exhale_args
                    potentials.append((similarity, potential_keys_lower[mate]))
            if potentials:
                potentials = reversed(sorted(potentials))
                for rank, mate in potentials:
                    extra_error.write("    - {0:2.2f}% match with: {1}\n".format(rank, mate))

        extra_error_str = extra_error.getvalue()
        extra_error.close()
        raise ConfigError(extra_error_str)

    ####################################################################################
    # Verify some potentially inconsistent or ignored settings.                        #
    ####################################################################################
    # treeViewIsBootstrap only takes meaning when createTreeView is True
    if not createTreeView and treeViewIsBootstrap:
        logger.warning("Exhale: `treeViewIsBootstrap=True` ignored since `createTreeView=False`")

    # fullToctreeMaxDepth > 5 may produce other sphinx issues unrelated to exhale
    if fullToctreeMaxDepth > 5:
        logger.warning(
            "Exhale: `fullToctreeMaxDepth={0}` is greater than 5 and may build errors for non-html.".format(
                fullToctreeMaxDepth
            )
        )

    # Make sure that we received a valid mapping created by utils.makeCustomSpecificationsMapping
    sanity = _closure_map_sanity_check
    insane = "`customSpecificationsMapping` *MUST* be made using  exhale.utils.makeCustomSpecificationsMapping"
    if customSpecificationsMapping:
        # Sanity check to make sure exhale made this mapping
        if sanity not in customSpecificationsMapping:
            raise ConfigError(insane)
        elif customSpecificationsMapping[sanity] != sanity:  # LOL
            raise ConfigError(insane)
        # Sanity check #2: enforce no new additions were made
        expected_keys = set([sanity]) | set(utils.AVAILABLE_KINDS)
        provided_keys = set(customSpecificationsMapping.keys())
        diff = provided_keys - expected_keys
        if diff:
            raise ConfigError("Found extra keys in `customSpecificationsMapping`: {0}".format(diff))
        # Sanity check #3: make sure the return values are all strings
        for key in customSpecificationsMapping:
            val_t = type(customSpecificationsMapping[key])
            if not isinstance(key, six.string_types):
                raise ConfigError(
                    "`customSpecificationsMapping` key `{key}` gave value type `{val_t}` (need `str`).".format(
                        key=key, val_t=val_t
                    )
                )

    # Specify where the doxygen output should be going
    global _doxygen_xml_output_directory
    _doxygen_xml_output_directory = doxy_xml_dir

    # If requested, the time is nigh for executing doxygen.  The strategy:
    # 1. Execute doxygen if requested
    # 2. Verify that the expected doxy_xml_dir (specified to `breathe`) was created
    # 3. Assuming everything went to plan, let exhale take over and create all of the .rst docs
    if exhaleExecutesDoxygen:
        # Cannot use both, only one or the other
        if exhaleUseDoxyfile and (exhaleDoxygenStdin is not None):
            raise ConfigError("You must choose one of `exhaleUseDoxyfile` or `exhaleDoxygenStdin`, not both.")

        # The Doxyfile *must* be at the same level as conf.py
        # This is done so that when separate source / build directories are being used,
        # we can guarantee where the Doxyfile is.
        if exhaleUseDoxyfile:
            doxyfile_path = os.path.abspath(os.path.join(app.confdir, "Doxyfile"))
            if not os.path.exists(doxyfile_path):
                raise ConfigError("The file [{0}] does not exist".format(doxyfile_path))

        here = os.path.abspath(os.curdir)
        if here == app.confdir:
            returnPath = None
        else:
            returnPath = here

        # All necessary information ready, go to where the Doxyfile is, run Doxygen
        # and then return back (where applicable) so sphinx can continue
        start = utils.get_time()
        if returnPath:
            logger.info(utils.info(
                "Exhale: changing directories to [{0}] to execute Doxygen.".format(app.confdir)
            ))
            os.chdir(app.confdir)
        logger.info(utils.info("Exhale: executing doxygen."))
        status = deploy.generateDoxygenXML()
        # Being overly-careful to put sphinx back where it was before potentially erroring out
        if returnPath:
            logger.info(utils.info(
                "Exhale: changing directories back to [{0}] after Doxygen.".format(returnPath)
            ))
            os.chdir(returnPath)
        if status:
            raise ExtensionError(status)
        else:
            end = utils.get_time()
            logger.info(utils.progress(
                "Exhale: doxygen ran successfully in {0}.".format(utils.time_string(start, end))
            ))
    else:
        if exhaleUseDoxyfile:
            logger.warning("Exhale: `exhaleUseDoxyfile` ignored since `exhaleExecutesDoxygen=False`")
        if exhaleDoxygenStdin is not None:
            logger.warning("Exhale: `exhaleDoxygenStdin` ignored since `exhaleExecutesDoxygen=False`")
        if exhaleSilentDoxygen:
            logger.warning("Exhale: `exhaleSilentDoxygen=True` ignored since `exhaleExecutesDoxygen=False`")

    # Either Doxygen was run prior to this being called, or we just finished running it.
    # Make sure that the files we need are actually there.
    if not os.path.isdir(doxy_xml_dir):
        raise ConfigError(
            "Exhale: the specified folder [{0}] does not exist.  Has Doxygen been run?".format(doxy_xml_dir)
        )
    index = os.path.join(doxy_xml_dir, "index.xml")
    if not os.path.isfile(index):
        raise ConfigError("Exhale: the file [{0}] does not exist.  Has Doxygen been run?".format(index))

    # Legacy / debugging feature, warn of its purpose
    if generateBreatheFileDirectives:
        logger.warning("Exhale: `generateBreatheFileDirectives` is a debugging feature not intended for production.")

    ####################################################################################
    # If using a fancy treeView, add the necessary frontend files.                     #
    ####################################################################################
    if createTreeView:
        if treeViewIsBootstrap:
            tree_data_static_base = "treeView-bootstrap"
            tree_data_css = [os.path.join("bootstrap-treeview", "bootstrap-treeview.min.css")]
            tree_data_js  = [
                os.path.join("bootstrap-treeview", "bootstrap-treeview.min.js"),
                # os.path.join("bootstrap-treeview", "apply-bootstrap-treview.js")
            ]
            tree_data_ext = []
        else:
            tree_data_static_base = "treeView"
            tree_data_css = [os.path.join("collapsible-lists", "css", "tree_view.css")]
            tree_data_js  = [
                os.path.join("collapsible-lists", "js", "CollapsibleLists.compressed.js"),
                os.path.join("collapsible-lists", "js", "apply-collapsible-lists.js")
            ]
            # The tree_view.css file uses these
            tree_data_ext = [
                os.path.join("collapsible-lists", "css", "button-closed.png"),
                os.path.join("collapsible-lists", "css", "button-open.png"),
                os.path.join("collapsible-lists", "css", "button.png"),
                os.path.join("collapsible-lists", "css", "list-item-contents.png"),
                os.path.join("collapsible-lists", "css", "list-item-last-open.png"),
                os.path.join("collapsible-lists", "css", "list-item-last.png"),
                os.path.join("collapsible-lists", "css", "list-item-open.png"),
                os.path.join("collapsible-lists", "css", "list-item.png"),
                os.path.join("collapsible-lists", "css", "list-item-root.png"),
            ]

        # Make sure we have everything we need
        collapse_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", tree_data_static_base)
        if not os.path.isdir(collapse_data):
            raise ExtensionError(
                "Exhale: the path to [{0}] was not found, possible installation error.".format(collapse_data)
            )
        else:
            all_files = tree_data_css + tree_data_js + tree_data_ext
            missing   = []
            for file in all_files:
                path = os.path.join(collapse_data, file)
                if not os.path.isfile(path):
                    missing.append(path)
            if missing:
                raise ExtensionError(
                    "Exhale: the path(s) {0} were not found, possible installation error.".format(missing)
                )

        # We have all the files we need, the extra files will be copied automatically by
        # sphinx to the correct _static/ location, but stylesheets and javascript need
        # to be added explicitly
        logger.info(utils.info("Exhale: adding tree view css / javascript."))

        # TODO: hack for multiproj
        if collapse_data not in app.config.html_static_path:
            app.config.html_static_path.append(collapse_data)

        # TODO: dubious hack on multiproj monkeypatch calling this method multiple times
        # resulting in the css / js files being added multiple times (which is a problem
        # so we have to bypass).  Probably it is an upstream bug that adding the same
        # file multiple times is allowed, but it's also definitely operator error.
        #
        # app.add_css_files -> look at the source, it registers first and then it adds
        # to the html builder, so we want to check if it is already there first before
        # trying to add it again.
        for css in tree_data_css:
            already_there = False
            for filename, attributes in app.registry.css_files:
                if css == filename:
                    already_there = True
                    break
            if not already_there:
                app.add_css_file(css)

        for js in tree_data_js:
            already_there = False
            for filename, attributes in app.registry.js_files:
                if js == filename:
                    already_there = True
                    break
            if not already_there:
                app.add_js_file(js)

        logger.info(utils.progress("Exhale: added tree view css / javascript."))
