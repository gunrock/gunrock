########################################################################################
# This file is part of exhale.  Copyright (c) 2017-2022, Stephen McDowell.             #
# Full BSD 3-Clause license available here:                                            #
#                                                                                      #
#                https://github.com/svenevs/exhale/blob/master/LICENSE                 #
########################################################################################

from __future__ import unicode_literals

from . import configs

import os
import sys
import six
import datetime
import time
import types
import codecs
import traceback
import textwrap

# Fancy error printing <3
try:
    import pygments
    from pygments import lexers, formatters
    __USE_PYGMENTS = True
except:
    __USE_PYGMENTS = False


def heading_mark(title, char):
    '''
    Given an input title and character, creates the reStructuredText underline
    according to the length of the title.

    **Parameters**
        **title** (str)
            The title that is being underlined, length assumed to be >= 1.

        **char** (str)
            The single character being used for this heading, e.g.
            :data:`~exhale.configs.SECTION_HEADING_CHAR`.

    **Return**
        **str**
            Returns ``len(title) * char``.
    '''
    return len(title) * char


def time_string(start, end):
    delta = datetime.timedelta(seconds=(end - start))
    # If the program took this long, the output has X Days, H:MM:S, which makes it
    # very clear what the units are between the colons.  Just display that directly
    if delta.days > 0:
        time_str = str(delta)
    # Otherwise, I like to make it clear about hours, minutes and seconds
    else:
        parts = str(delta).split(":")
        if len(parts) == 3:
            try:
                hours = int(parts[0])
                mins  = int(parts[1])
                secs  = round(float(parts[2]), 2)

                if hours == 0:
                    hours_str = ""
                elif hours == 1:
                    hours_str = "1 hour, "
                else:
                    hours_str = "{0} hours, ".format(hours)

                if mins == 0:
                    mins_str = ""
                elif mins == 1:
                    mins_str = "1 minute, and "
                else:
                    mins_str = "{0} minutes, and ".format(mins)

                if secs == 1.00:  # LOL I would love to see this happen
                    secs_str = "1.00 second"
                else:
                    secs_str = "{0} seconds".format(secs)

                time_str = "{0}{1}{2}".format(hours_str, mins_str, secs_str)
            except:
                time_str = str(delta)
        else:
            # Uhh. Time to give up pretty printing because this shouldn't happen
            time_str = str(delta)

    return time_str


def get_time():
    if sys.version_info > (3, 3):
        # monotonic introduced in 3.3
        return time.monotonic()
    else:
        return time.time()


AVAILABLE_KINDS = [
    "concept",
    "class",
    "define",
    "dir",
    "enum",
    "enumvalue",  # unused
    "file",
    "function",
    "group",  # unused
    "namespace",
    "struct",
    "typedef",
    "union",
    "variable",
    "page"
]
'''
All potential input ``kind`` values coming from Doxygen.

The ``"group"`` and ``"enumvalue"`` kinds are currently detected, but unused.
'''

LEAF_LIKE_KINDS = [
    "define",
    "enum",
    "function",
    "concept",
    "class",
    "struct",
    "typedef",
    "union",
    "variable"
]
'''
All kinds that can be generated using |generateSingleNodeRST|.

This more or less corresponds to the kinds that Exhale uses a Breathe directive.  This
is everything in :data:`AVAILABLE_KINDS`, except for

- ``"enumvalue"`` and ``"group"`` (unused in framework), and
- ``"dir"``, ``"file"``, and ``"namespace"`` since they require special treatment.

.. |generateSingleNodeRST| replace:: :class:`ExhaleRoot.generateSingleNodeRST <exhale.graph.ExhaleRoot.generateSingleNodeRST>`
'''

CLASS_LIKE_KINDS = [
    "class",
    "struct",
    "interface"  # TODO: not currently supported or used
]
"""All kinds that are "class-like"."""


def contentsDirectiveOrNone(kind):
    '''
    Generates a string ``.. contents::`` directives according to the rules outlined in
    the :ref:`using_contents_directives` section.

    **Parameters**
        ``kind`` (str)
            The ``kind`` of the compound (one of :data:`~exhale.utils.AVAILABLE_KINDS`).

    **Return**
        ``str`` or ``None``
            If this ``kind`` should have a ``.. contents::`` directive, it returns the
            string that can be written to file.  Otherwise, ``None`` is returned.
    '''
    if configs.contentsDirectives and kind in configs.kindsWithContentsDirectives:
        ret = "\n.. contents:: {contentsTitle}".format(
            contentsTitle=configs.contentsTitle
        )
        if configs.contentsSpecifiers:
            specs = "\n".join(s for s in configs.contentsSpecifiers)
            ret   = "{directive}\n{specs}".format(
                directive=ret,
                specs=prefix("   ", specs)
            )
        return "{full_directive}\n\n".format(full_directive=ret)
    else:
        return None


def makeCustomSpecificationsMapping(func):
    '''
    Creates the "pickleable" dictionary that will be used with
    :data:`~exhale.configs.customSpecificationsMapping` supplied to ``exhale_args`` in
    your ``conf.py``.

    **Parameters**
        ``func`` (types.FunctionType)
            A callable function that takes as input a string from
            :data:`~exhale.utils.AVAILABLE_KINDS` and returns a ``list`` of strings.

            The empty list ``[]`` indicates to use the Breathe defaults.

    **Return**
        ``dict``
            A dictionary where the keys are every value in
            :data:`~exhale.utils.AVAILABLE_KINDS`, and the values are the ``list``
            returns of the input ``func``.

    .. note::

       To help ensure the dictionary has everything it needs for the rest of Exhale to
       function, a "secret" key-value pair is inserted to the returned dictionary.
    '''
    # Make sure they gave us a function
    if not isinstance(func, types.FunctionType):
        raise ValueError(
            "The input to exhale.util.makeCustomSpecificationsMapping was *NOT* a function: {0}".format(
                type(func)
            )
        )

    # Stamp the return to ensure exhale created this function.
    ret = {configs._closure_map_sanity_check: configs._closure_map_sanity_check}
    try:
        # Because we cannot pickle a fully-fledged function object, we are going to go
        # through every kind and store its return value.
        for kind in AVAILABLE_KINDS:
            specs = func(kind)
            bad   = type(specs) is not list
            for s in specs:
                if not isinstance(s, six.string_types):
                    bad = True
                    break
            if bad:
                raise RuntimeError(textwrap.dedent('''
                    The specifications function did not return a valid list for input

                        `{kind}`

                    1. Make sure that every entry in the returned list is a string.
                    2. If you want to use the breathe defaults, you must return the
                       empty list `[]`.
                '''.format(kind=kind)))
            ret[kind] = specs
    except Exception as e:
        raise RuntimeError("Unable to create custom specifications:\n{0}".format(e))

    # Everything went according to plan, send it back to `conf.py` :)
    return ret


def nodeCompoundXMLContents(node):
    node_xml_path = os.path.join(configs._doxygen_xml_output_directory, "{0}.xml".format(node.refid))
    if os.path.isfile(node_xml_path):
        try:
            with codecs.open(node_xml_path, "r", "utf-8") as xml:
                node_xml_contents = xml.read()

            return node_xml_contents
        except:
            return None
    return None


def sanitize(name):
    """
    Sanitize the specified ``name`` for use with breathe directives.

    **Parameters**

    ``name`` (:class:`python:str`)
        The name to be sanitized.

    **Return**

    :class:`python:str`
        The input ``name`` sanitized to use with breathe directives (primarily for use
        with ``.. doxygenfunction::``).  Replacements such as ``"&lt;" -> "<"`` are
        performed, as well as removing spaces ``"< " -> "<"`` must be done.  Breathe is
        particularly sensitive with respect to whitespace.
    """
    return name.replace(
        "&lt;", "<"
    ).replace(
        "&gt;", ">"
    ).replace(
        "&amp;", "&"
    ).replace(
        "< ", "<"
    ).replace(
        " >", ">"
    ).replace(
        " &", "&"
    ).replace(
        "& ", "&"
    )


def sanitize_all(names):
    """
    Convenience function to :func:`~exhale.utils.sanitize` all provided names.

    **Parameters**
        ``names`` (:class:`python:list` of :class:`python:str`)
            A list of strings to sanitize.

    **Return**
        :class:`python:list` of :class:`python:str`
            Each name in ``names`` sanitized: ``[sanitize(n) for n in names]``.
    """
    return [sanitize(n) for n in names]


LANG_TO_LEX = {
    # Default Doxygen languages
    "IDL":          "idl",
    "Java":         "java",
    "Javascript":   "js",
    "C#":           "csharp",
    "C":            "c",
    "C++":          "cpp",
    "D":            "d",
    "PHP":          "php",
    "Objecive-C":   "objective-c",
    "Python":       "py",
    "Fortran":      "fortran",
    "FortranFree":  "fortran",
    "FortranFixed": "fortranfixed",
    "VHDL":         "vhdl",
    # Custom Doxygen languages
    "Verilog":      "verilog",
    "Markdown":     "markdown"
}
'''
.. include:: ../LANG_TO_LEX_value.rst

Mapping of ``language="xxx"`` from the Doxygen programlisting to Pygments Lexers.  This
mapping is used in :func:`doxygenLanguageToPygmentsLexer`.

From the Doxygen documentation on `EXTENSION_MAPPING <ext_map_>`_:

..

    IDL, Java, Javascript, C#, C, C++, D, PHP, Objective-C, Python,Fortran (fixed format
    Fortran: FortranFixed, free formatted Fortran: FortranFree, unknown formatted
    Fortran: Fortran. In the later case the parser tries to guess whether the code is
    fixed or free formatted code, this is the default for Fortran type files), VHDL.

We need to take each one of those, and map them to their corresponding
`Pygments Lexer <http://pygments.org/docs/lexers/>`_.

.. _ext_map: https://www.doxygen.nl/manual/config.html#cfg_extension_mapping

.. note::

   Feel free to make a pull request adding more lanugages here.  For example, the
   ``Verilog`` support comes from
   `doxygen-verilog <https://github.com/avelure/doxygen-verilog>`_.
'''


def doxygenLanguageToPygmentsLexer(location, language):
    '''
    Given an input location and language specification, acquire the Pygments lexer to
    use for this file.

    1. If :data:`configs.lexerMapping <exhale.configs.lexerMapping>` has been specified,
       then :data:`configs._compiled_lexer_mapping <exhale.configs._compiled_lexer_mapping>`
       will be queried first using the ``location`` parameter.
    2. If no matching was found, then the appropriate lexer defined in
       :data:`LANG_TO_LEX <exhale.utils.LANG_TO_LEX>` is used.
    3. If no matching language is found, ``"none"`` is returned (indicating to Pygments
       that no syntax highlighting should occur).
    '''
    if configs._compiled_lexer_mapping:
        for regex in configs._compiled_lexer_mapping:
            if regex.match(location):
                return configs._compiled_lexer_mapping[regex]

    if language in LANG_TO_LEX:
        return LANG_TO_LEX[language]

    return "none"


########################################################################################
#
##
###
####
##### Utility / helper functions.
####
###
##
#
########################################################################################
def qualifyKind(kind):
    '''
    Qualifies the breathe ``kind`` and returns an qualifier string describing this
    to be used for the text output (e.g. in generated file headings and link names).

    The output for a given kind is as follows:

    +-------------+------------------+
    | Input Kind  | Output Qualifier |
    +=============+==================+
    | "class"     | "Class"          |
    +-------------+------------------+
    | "concept"   | "Concept"        |
    +-------------+------------------+
    | "define"    | "Define"         |
    +-------------+------------------+
    | "enum"      | "Enum"           |
    +-------------+------------------+
    | "enumvalue" | "Enumvalue"      |
    +-------------+------------------+
    | "file"      | "File"           |
    +-------------+------------------+
    | "function"  | "Function"       |
    +-------------+------------------+
    | "group"     | "Group"          |
    +-------------+------------------+
    | "namespace" | "Namespace"      |
    +-------------+------------------+
    | "struct"    | "Struct"         |
    +-------------+------------------+
    | "typedef"   | "Typedef"        |
    +-------------+------------------+
    | "union"     | "Union"          |
    +-------------+------------------+
    | "variable"  | "Variable"       |
    +-------------+------------------+

    The following breathe kinds are ignored:

    - "autodoxygenfile"
    - "doxygenindex"
    - "autodoxygenindex"

    Note also that although a return value is generated, neither "enumvalue" nor
    "group" are actually used.

    :Parameters:
        ``kind`` (str)
            The return value of a Breathe ``compound`` object's ``get_kind()`` method.

    :Return (str):
        The qualifying string that will be used to build the reStructuredText titles and
        other qualifying names.  If the empty string is returned then it was not
        recognized.
    '''
    if kind == "dir":
        return "Directory"
    else:
        return kind.capitalize()


def kindAsBreatheDirective(kind):
    '''
    Returns the appropriate breathe restructured text directive for the specified kind.
    The output for a given kind is as follows:

    +-------------+--------------------+
    | Input Kind  | Output Directive   |
    +=============+====================+
    | "concept"   | "doxygenconcept"   |
    +-------------+--------------------+
    | "class"     | "doxygenclass"     |
    +-------------+--------------------+
    | "define"    | "doxygendefine"    |
    +-------------+--------------------+
    | "enum"      | "doxygenenum"      |
    +-------------+--------------------+
    | "enumvalue" | "doxygenenumvalue" |
    +-------------+--------------------+
    | "file"      | "doxygenfile"      |
    +-------------+--------------------+
    | "function"  | "doxygenfunction"  |
    +-------------+--------------------+
    | "group"     | "doxygengroup"     |
    +-------------+--------------------+
    | "namespace" | "doxygennamespace" |
    +-------------+--------------------+
    | "struct"    | "doxygenstruct"    |
    +-------------+--------------------+
    | "typedef"   | "doxygentypedef"   |
    +-------------+--------------------+
    | "union"     | "doxygenunion"     |
    +-------------+--------------------+
    | "variable"  | "doxygenvariable"  |
    +-------------+--------------------+
    | "page"      | "doxygenpage"      |
    +-------------+--------------------+

    The following breathe kinds are ignored:

    - "autodoxygenfile"
    - "doxygenindex"
    - "autodoxygenindex"

    Note also that although a return value is generated, neither "enumvalue" nor
    "group" are actually used.

    :Parameters:
        ``kind`` (str)
            The kind of the breathe compound / ExhaleNode object (same values).

    :Return (str):
        The directive to be used for the given ``kind``.  The empty string is returned
        for both unrecognized and ignored input values.
    '''
    return "doxygen{kind}".format(kind=kind)


def specificationsForKind(kind):
    '''
    .. todo:: update docs for new list version rather than string returns
    '''
    '''
    Returns the relevant modifiers for the restructured text directive associated with
    the input kind.  The only considered values for the default implementation are
    ``class`` and ``struct``, for which the return value is exactly::

        "   :members:\\n   :protected-members:\\n   :undoc-members:\\n"

    Formatting of the return is fundamentally important, it must include both the prior
    indentation as well as newlines separating any relevant directive modifiers.  The
    way the framework uses this function is very specific; if you do not follow the
    conventions then sphinx will explode.

    Consider a ``struct thing`` being documented.  The file generated for this will be::

        .. _struct_thing:

        Struct thing
        ================================================================================

        .. doxygenstruct:: thing
           :members:
           :protected-members:
           :undoc-members:

    Assuming the first two lines will be in a variable called ``link_declaration``, and
    the next three lines are stored in ``header``, the following is performed::

        directive = ".. {}:: {}\\n".format(kindAsBreatheDirective(node.kind), node.name)
        specifications = "{}\\n\\n".format(specificationsForKind(node.kind))
        gen_file.write("{}{}{}{}".format(link_declaration, header, directive, specifications))

    That is, **no preceding newline** should be returned from your custom function, and
    **no trailing newline** is needed.  Your indentation for each specifier should be
    **exactly three spaces**, and if you want more than one you need a newline in between
    every specification you want to include.  Whitespace control is handled internally
    because many of the directives do not need anything added.  For a full listing of
    what your specifier options are, refer to the breathe documentation:

        http://breathe.readthedocs.io/en/latest/directives.html

    :Parameters:
        ``kind`` (str)
            The kind of the node we are generating the directive specifications for.

    :Return (str):
        The correctly formatted specifier(s) for the given ``kind``.  If no specifier(s)
        are necessary or desired, the empty string is returned.
    '''
    # TODO: this is to support the monkeypatch
    # https://github.com/svenevs/exhale/issues/27
    ret = []

    # use the custom directives function
    if configs.customSpecificationsMapping:
        ret = configs.customSpecificationsMapping[kind]
    # otherwise, just provide class and struct
    elif kind == "class" or kind == "struct":
        ret = [":members:", ":protected-members:", ":undoc-members:"]

    # the monkeypatch re-configures breathe_default_project each time which was
    # foolishly relied on elsewhere and undoing that blunder requires undoing
    # all of the shenanigans that is configs.py...
    if not any(":project:" in spec for spec in ret):
        ret.insert(0, ":project: " + configs._the_app.config.breathe_default_project)
    return ret


class AnsiColors:
    '''
    A simple wrapper class for convenience definitions of common ANSI formats to enable
    colorizing output in various formats.  The definitions below only affect the
    foreground color of the text, but you can easily change the background color too.
    See `ANSI color codes <http://misc.flogisoft.com/bash/tip_colors_and_formatting>`_
    for a concise overview of how to use the ANSI color codes.
    '''
    BOLD          = "1m"
    ''' The ANSI bold modifier, see :ref:`utils.AnsiColors.BOLD_RED` for an example. '''
    DIM           = "2m"
    ''' The ANSI dim modifier, see :ref:`utils.AnsiColors.DIM_RED` for an example. '''
    UNDER         = "4m"
    ''' The ANSI underline modifier, see :ref:`utils.AnsiColors.UNDER_RED` for an example. '''
    INV           = "7m"
    ''' The ANSI inverted modifier, see :ref:`utils.AnsiColors.INV_RED` for an example. '''
    ####################################################################################
    BLACK         = "30m"
    ''' The ANSI black color. '''
    BOLD_BLACK    = "30;{bold}".format(bold=BOLD)
    ''' The ANSI bold black color. '''
    DIM_BLACK     = "30;{dim}".format(dim=DIM)
    ''' The ANSI dim black color. '''
    UNDER_BLACK   = "30;{under}".format(under=UNDER)
    ''' The ANSI underline black color. '''
    INV_BLACK     = "30;{inv}".format(inv=INV)
    ''' The ANSI inverted black color. '''
    ####################################################################################
    RED           = "31m"
    ''' The ANSI red color. '''
    BOLD_RED      = "31;{bold}".format(bold=BOLD)
    ''' The ANSI bold red color. '''
    DIM_RED       = "31;{dim}".format(dim=DIM)
    ''' The ANSI dim red color. '''
    UNDER_RED     = "31;{under}".format(under=UNDER)
    ''' The ANSI underline red color. '''
    INV_RED       = "31;{inv}".format(inv=INV)
    ''' The ANSI inverted red color. '''
    ####################################################################################
    GREEN         = "32m"
    ''' The ANSI green color. '''
    BOLD_GREEN    = "32;{bold}".format(bold=BOLD)
    ''' The ANSI bold green color. '''
    DIM_GREEN     = "32;{dim}".format(dim=DIM)
    ''' The ANSI dim green color. '''
    UNDER_GREEN   = "32;{under}".format(under=UNDER)
    ''' The ANSI underline green color. '''
    INV_GREEN     = "32;{inv}".format(inv=INV)
    ''' The ANSI inverted green color. '''
    ####################################################################################
    YELLOW        = "33m"
    ''' The ANSI yellow color. '''
    BOLD_YELLOW   = "33;{bold}".format(bold=BOLD)
    ''' The ANSI bold yellow color. '''
    DIM_YELLOW    = "33;{dim}".format(dim=DIM)
    ''' The ANSI dim yellow color. '''
    UNDER_YELLOW  = "33;{under}".format(under=UNDER)
    ''' The ANSI underline yellow color. '''
    INV_YELLOW    = "33;{inv}".format(inv=INV)
    ''' The ANSI inverted yellow color. '''
    ####################################################################################
    BLUE          = "34m"
    ''' The ANSI blue color. '''
    BOLD_BLUE     = "34;{bold}".format(bold=BOLD)
    ''' The ANSI bold blue color. '''
    DIM_BLUE      = "34;{dim}".format(dim=DIM)
    ''' The ANSI dim blue color. '''
    UNDER_BLUE    = "34;{under}".format(under=UNDER)
    ''' The ANSI underline blue color. '''
    INV_BLUE      = "34;{inv}".format(inv=INV)
    ''' The ANSI inverted blue color. '''
    ####################################################################################
    MAGENTA       = "35m"
    ''' The ANSI magenta (purple) color. '''
    BOLD_MAGENTA  = "35;{bold}".format(bold=BOLD)
    ''' The ANSI bold magenta (purple) color. '''
    DIM_MAGENTA   = "35;{dim}".format(dim=DIM)
    ''' The ANSI dim magenta (purple) color. '''
    UNDER_MAGENTA = "35;{under}".format(under=UNDER)
    ''' The ANSI underlined magenta (purple) color. '''
    INV_MAGENTA   = "35;{inv}".format(inv=INV)
    ''' The ANSI inverted magenta (purple) color. '''
    ####################################################################################
    CYAN          = "36m"
    ''' The ANSI cyan color. '''
    BOLD_CYAN     = "36;{bold}".format(bold=BOLD)
    ''' The ANSI bold cyan color. '''
    DIM_CYAN      = "36;{dim}".format(dim=DIM)
    ''' The ANSI dim cyan color. '''
    UNDER_CYAN    = "36;{under}".format(under=UNDER)
    ''' The ANSI underline cyan color. '''
    INV_CYAN      = "36;{inv}".format(inv=INV)
    ''' The ANSI inverted cyan color. '''
    ####################################################################################
    WHITE         = "37m"
    ''' The ANSI white color. '''
    BOLD_WHITE    = "37;{bold}".format(bold=BOLD)
    ''' The ANSI bold white color. '''
    DIM_WHITE     = "37;{dim}".format(dim=DIM)
    ''' The ANSI dim white color. '''
    UNDER_WHITE   = "37;{under}".format(under=UNDER)
    ''' The ANSI underline white color. '''
    INV_WHITE     = "37;{inv}".format(inv=INV)
    ''' The ANSI inverted white color. '''

    @classmethod
    def printAllColorsToConsole(cls):
        ''' A simple enumeration of the colors to the console to help decide :) '''
        for elem in cls.__dict__:
            # ignore specials such as __class__ or __module__
            if not elem.startswith("__"):
                color_fmt = cls.__dict__[elem]
                if isinstance(color_fmt, six.string_types) and color_fmt != "BOLD" and color_fmt != "DIM" and \
                        color_fmt != "UNDER" and color_fmt != "INV":
                    print("\033[{fmt}AnsiColors.{name}\033[0m".format(fmt=color_fmt, name=elem))


def indent(text, prefix, predicate=None):
    '''
    This is a direct copy of ``textwrap.indent`` for availability in Python 2.

    Their documentation:

    Adds 'prefix' to the beginning of selected lines in 'text'.
    If 'predicate' is provided, 'prefix' will only be added to the lines
    where 'predicate(line)' is True. If 'predicate' is not provided,
    it will default to adding 'prefix' to all non-empty lines that do not
    consist solely of whitespace characters.
    '''
    if predicate is None:
        def predicate(line):
            return line.strip()

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if predicate(line) else line)

    return ''.join(prefixed_lines())


def prefix(token, msg):
    '''
    Wrapper call to :func:`~exhale.utils.indent` with an always-true predicate so that
    empty lines (e.g. `\\n`) still get indented by the ``token``.

    :Parameters:
        ``token`` (str)
            What to indent the message by (e.g. ``"(!) "``).

        ``msg`` (str)
            The message to get indented by ``token``.

    :Return:
        ``str``
            The message ``msg``, indented by the ``token``.
    '''
    return indent(msg, token, predicate=lambda x: True)


def exclaim(err_msg):
    return "\n{}".format("(!) ").join("{}{}".format("(!) ", err_msg).splitlines())


def colorize(msg, ansi_fmt):
    return "\033[{0}{1}\033[0m".format(ansi_fmt, msg)


def _use_color(msg, ansi_fmt, output_stream):
    '''
    Based on :data:`~exhale.configs.alwaysColorize`, returns the colorized or
    non-colorized output when ``output_stream`` is not a TTY (e.g. redirecting
    to a file).

    **Parameters**
        ``msg`` (str)
            The message that is going to be printed by the caller of this method.

        ``ansi_fmt`` (str)
            The ANSI color format to use when coloring is supposed to happen.

        ``output_stream`` (file)
            Assumed to be either ``sys.stdout`` or ``sys.stderr``.

    **Return**
        ``str``
            The message ``msg`` in color, or not, depending on both
            :data:`~exhale.configs.alwaysColorize` and whether or not the
            ``output_stream`` is a TTY.
    '''
    if configs._on_rtd or (not configs.alwaysColorize and not output_stream.isatty()):
        log = msg
    else:
        log = colorize(msg, ansi_fmt)
    return log


def progress(msg, ansi_fmt=AnsiColors.BOLD_GREEN, output_stream=sys.stdout):
    return _use_color(prefix("[+] ", msg), ansi_fmt, output_stream)


def info(msg, ansi_fmt=AnsiColors.BOLD_BLUE, output_stream=sys.stdout):
    return _use_color(prefix("[~] ", msg), ansi_fmt, output_stream)


def critical(msg, ansi_fmt=AnsiColors.BOLD_RED, output_stream=sys.stderr):
    return _use_color(prefix("(!) ", msg), ansi_fmt, output_stream)


def verbose_log(msg, ansi_fmt=None):
    if configs.verboseBuild:
        if ansi_fmt:
            log = _use_color(msg, ansi_fmt, sys.stderr)
        else:
            log = msg
        sys.stderr.write("{log}\n".format(log=log))


def __fancy(text, language, fmt):
    if not configs._on_rtd and __USE_PYGMENTS:
        try:
            lang_lex = lexers.find_lexer_class_by_name(language)
            fmt      = formatters.get_formatter_by_name(fmt)
            highlighted = pygments.highlight(text, lang_lex(), fmt)
            return highlighted
        except:
            return text
    else:
        return text


def fancyErrorString(lex):
    try:
        # fancy error printing aka we want the traceback, but
        # don't want the exclaimed stuff printed again
        err = traceback.format_exc()
        # shenanigans = "During handling of the above exception, another exception occurred:"
        # err = err.split(shenanigans)[0]
        return __fancy("{0}\n".format(err), lex, "console")
    except:
        return "CRITICAL: could not extract traceback.format_exc!"


def fancyError(critical_msg=None, lex="py3tb", singleton_hook=None):
    if critical_msg:
        sys.stderr.write(critical(critical_msg))

    sys.stderr.write(fancyErrorString(lex))

    if singleton_hook:
        # Only let shutdown happen once.  Useful for when singleton_hook may also create
        # errors (e.g. why you got here in the first place).
        fancyError.__defaults__ = (None, None)
        try:
            singleton_hook()
        except Exception as e:
            sys.stderr.write(critical(
                "fancyError: `singleton_hook` caused exception: {0}".format(e)
            ))

    os._exit(1)
