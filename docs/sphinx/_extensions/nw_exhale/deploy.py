# -*- coding: utf8 -*-
########################################################################################
# This file is part of exhale.  Copyright (c) 2017-2022, Stephen McDowell.             #
# Full BSD 3-Clause license available here:                                            #
#                                                                                      #
#                https://github.com/svenevs/exhale/blob/master/LICENSE                 #
########################################################################################

'''
The deploy module is responsible for two primary actions:

1. Executing Doxygen (if requested in ``exhale_args``).
2. Launching the full API generation via the :func:`~exhale.deploy.explode` function.
'''

from __future__ import unicode_literals

from . import configs
from . import utils
from .graph import ExhaleRoot

import os
import sys
import six
import re
import codecs
import tempfile
import textwrap
from subprocess import PIPE, Popen, STDOUT


def _generate_doxygen(doxygen_input):
    '''
    This method executes doxygen based off of the specified input.  By the time this
    method is executed, it is assumed that Doxygen is intended to be run in the
    **current working directory**.  Search for ``returnPath`` in the implementation of
    :func:`~exhale.configs.apply_sphinx_configurations` for handling of this aspect.

    This method is intended to be called by :func:`~exhale.deploy.generateDoxygenXML`,
    which is in turn called by :func:`~exhale.configs.apply_sphinx_configurations`.

    Two versions of the
    doxygen command can be executed:

    1. If ``doxygen_input`` is exactly ``"Doxyfile"``, then it is assumed that a
       ``Doxyfile`` exists in the **current working directory**.  Meaning the command
       being executed is simply ``doxygen``.
    2. For all other values, ``doxygen_input`` represents the arguments as to be
       specified on ``stdin`` to the process.

    **Parameters**
        ``doxygen_input`` (str)
            Either the string ``"Doxyfile"`` to run vanilla ``doxygen``, or the
            selection of doxygen inputs (that would ordinarily be in a ``Doxyfile``)
            that will be ``communicate``d to the ``doxygen`` process on ``stdin``.

            .. note::

               If using Python **3**, the input **must** still be a ``str``.  This
               method will convert the input to ``bytes`` as follows:

               .. code-block:: py

                  if sys.version[0] == "3":
                      doxygen_input = bytes(doxygen_input, "utf-8")

    **Return**
        ``str`` or ``None``
            If an error occurs, a string describing the error is returned with the
            intention of the caller raising the exception.  If ``None`` is returned,
            then the process executed without error.  Example usage:

            .. code-block:: py

               status = _generate_doxygen("Doxygen")
               if status:
                   raise RuntimeError(status)

            Though a little awkward, this is done to enable the intended caller of this
            method to restore some state before exiting the program (namely, the working
            directory before propagating an exception to ``sphinx-build``).
    '''
    if not isinstance(doxygen_input, six.string_types):
        return "Error: the `doxygen_input` variable must be of type `str`."

    doxyfile = doxygen_input == "Doxyfile"
    try:
        # Setup the arguments to launch doxygen
        if doxyfile:
            args   = ["doxygen"]
            kwargs = {}
        else:
            args   = ["doxygen", "-"]
            kwargs = {"stdin": PIPE}

        if configs._on_rtd:
            # On RTD, any capturing of Doxygen output can cause buffer overflows for
            # even medium sized projects.  So it is disregarded entirely to ensure the
            # build will complete (otherwise, it silently fails after `cat conf.py`)
            devnull_file     = open(os.devnull, "w")
            kwargs["stdout"] = devnull_file
            kwargs["stderr"] = STDOUT
        else:
            # TL;DR: strictly enforce that (verbose) doxygen output doesn't cause the
            # `communicate` to hang due to buffer overflows.
            #
            # See excellent synopsis:
            # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
            if six.PY2:
                tempfile_kwargs = {}
            else:
                # encoding argument introduced in python 3
                tempfile_kwargs = {"encoding": "utf-8"}
            tempfile_kwargs["mode"] = "r+"
            tmp_out_file = tempfile.TemporaryFile(
                prefix="doxygen_stdout_buff", **tempfile_kwargs
            )
            tmp_err_file = tempfile.TemporaryFile(
                prefix="doxygen_stderr_buff", **tempfile_kwargs
            )

            # Write to the tempfiles over PIPE to avoid buffer overflowing
            kwargs["stdout"] = tmp_out_file
            kwargs["stderr"] = tmp_err_file

        # Note: overload of args / kwargs, Popen is expecting a list as the first
        #       parameter (aka no *args, just args)!
        doxygen_proc = Popen(args, **kwargs)

        # Communicate can only be called once, arrange whether or not stdin has value
        if not doxyfile:
            # In Py3, make sure we are communicating a bytes-like object which is no
            # longer interchangeable with strings (as was the case in Py2).
            if sys.version[0] == "3":
                doxygen_input = bytes(doxygen_input, "utf-8")
            comm_kwargs = {"input": doxygen_input}
        else:
            comm_kwargs = {}

        # Waits until doxygen has completed
        doxygen_proc.communicate(**comm_kwargs)

        # Print out what was written to the tmpfiles by doxygen
        if not configs._on_rtd and not configs.exhaleSilentDoxygen:
            # Doxygen output (some useful information, mostly just enumeration of the
            # configurations you gave it {useful for debugging...})
            if tmp_out_file.tell() > 0:
                tmp_out_file.seek(0)
                print(tmp_out_file.read())
            # Doxygen error (e.g. any warnings, or invalid input)
            if tmp_err_file.tell() > 0:
                # Making them stick out, ideally users would reduce this output to 0 ;)
                # This will print a yellow [~] before every line, but not make the
                # entire line yellow because it's definitively not helpful
                prefix = utils._use_color(
                    utils.prefix("[~]", " "), utils.AnsiColors.BOLD_YELLOW, sys.stderr
                )
                tmp_err_file.seek(0)
                sys.stderr.write(utils.prefix(prefix, tmp_err_file.read()))

        # Close the file handles opened for communication with subprocess
        if configs._on_rtd:
            devnull_file.close()
        else:
            # Delete the tmpfiles
            tmp_out_file.close()
            tmp_err_file.close()

        # Make sure we had a valid execution of doxygen
        exit_code = doxygen_proc.returncode
        if exit_code != 0:
            raise RuntimeError("Non-zero return code of [{0}] from 'doxygen'...".format(exit_code))
    except Exception as e:
        return "Unable to execute 'doxygen': {0}".format(e)

    # returning None signals _success_
    return None


def _valid_config(config, required):
    '''
    .. todo:: add documentation of this method

    ``config``: doxygen input we're looking for
    ``required``: if ``True``, must be present.  if ``False``, NOT ALLOWED to be present
    '''
    re_template = r"\s*{config}\s*=.*".format(config=config)
    found = re.search(re_template, configs.exhaleDoxygenStdin)
    if required:
        return found is not None
    else:
        return found is None


def generateDoxygenXML():
    # If this happens, we really shouldn't be here...
    if not configs.exhaleExecutesDoxygen:
        return textwrap.dedent('''
            `generateDoxygenXML` should *ONLY* be called internally.  You should
            set `exhaleExecutesDoxygen=True` in `exhale_args` in `conf.py`.
        ''')

    # Case 1: the user has their own `Doxyfile`.
    if configs.exhaleUseDoxyfile:
        return _generate_doxygen("Doxyfile")
    # Case 2: use stdin, with some defaults and potentially additional specs from user
    else:
        # There are two doxygen specs that we explicitly disallow
        #
        # 1. OUTPUT_DIRECTORY: this is *ALREADY* specified implicitly via breathe
        # 2. STRIP_FROM_PATH: this is a *REQUIRED* config (`doxygenStripFromPath`)
        #
        # There is one doxygen spec that is REQUIRED to be given:
        #
        # 1. INPUT (where doxygen should parse).
        #
        # The below is a modest attempt to validate that these were / were not given.
        if not isinstance(configs.exhaleDoxygenStdin, six.string_types):
            return "`exhaleDoxygenStdin` config must be a string!"

        if not _valid_config("OUTPUT_DIRECTORY", False):
            # If we are hitting this code, these should both exist and be configured
            # since this method is called **AFTER** the configuration verification code
            # performed in configs.apply_sphinx_configurations
            breathe_projects = configs._the_app.config.breathe_projects
            breathe_default_project = configs._the_app.config.breathe_default_project
            return textwrap.dedent('''
                `exhaleDoxygenStdin` may *NOT* specify `OUTPUT_DIRECTORY`.  Exhale does
                this internally by reading what you provided to `breathe_projects` in
                your `conf.py`.

                Based on what you had in `conf.py`, Exhale will be using

                - The `breathe_default_project`:

                      {default}

                - The output path specfied (`breathe_projects[breathe_default_project]`):

                      {path}

                  NOTE: the above path has the `xml` portion removed from what you
                        provided.  This path is what is sent to Doxygen, Breathe
                        requires you include the `xml` directory path; so Exhale simply
                        re-uses this variable and adapts the value for our needs.
            '''.format(
                default=breathe_default_project,
                path=breathe_projects[breathe_default_project].rsplit("{sep}xml".format(sep=os.sep), 1)[0]
            ))

        if not _valid_config("STRIP_FROM_PATH", False):
            return textwrap.dedent('''
                `exhaleDoxygenStdin` may *NOT* specify `STRIP_FROM_PATH`.  Exhale does
                this internally by using the value you provided to `exhale_args` in
                your `conf.py` for the key `doxygenStripFromPath`.

                Based on what you had in `conf.py`, Exhale will be using:

                    {strip}

                NOTE: the above is what you specified directly in `exhale_args`.  Exhale
                      will be using an absolute path to send to Doxygen.  It is:

                    {absolute}
            '''.format(
                strip=configs._the_app.config.exhale_args["doxygenStripFromPath"],
                absolute=configs.doxygenStripFromPath
            ))

        if not _valid_config("INPUT", True):
            return textwrap.dedent('''
                `exhaleDoxygenStdin` *MUST* specify the `INPUT` doxygen config variable.
                The INPUT variable is what tells Doxygen where to look for code to
                extract documentation from.  For example, if you had a directory layout

                    project_root/
                        docs/
                            conf.py
                            Makefile
                            ... etc ...
                        include/
                            my_header.hpp
                        src/
                            my_header.cpp

                Then you would include the line

                    INPUT = ../include

                in the string provided to `exhale_args["exhaleDoxygenStdin"]`.
            ''')

        # For these, we just want to warn them of the impact but still allow an override
        re_template = r"\s*{config}\s*=\s*(.*)"
        for cfg in ("ALIASES", "PREDEFINED"):
            found = re.search(re_template.format(config=cfg), configs.exhaleDoxygenStdin)
            if found:
                sys.stderr.write(utils.info(textwrap.dedent('''
                    You have supplied to `exhaleDoxygenStdin` a configuration of:

                        {cfg}   =   {theirs}

                    This has an important impact, as it overrides a default setting that
                    Exhale is using.

                    1. If you are intentionally overriding this configuration, simply
                       ignore this message --- what you intended will happen.

                    2. If you meant to _continue_ adding to the defaults Exhale provides,
                       you need to use a `+=` instead of a raw `=`.  So do instead

                           {cfg}   +=   {theirs}

                '''.format(cfg=cfg, theirs=found.groups()[0])), utils.AnsiColors.BOLD_YELLOW))

        # Include their custom doxygen definitions after the defaults so that they can
        # override anything they want to.  Populate the necessary output dir and strip path.
        doxy_dir = configs._doxygen_xml_output_directory.rsplit("{sep}xml".format(sep=os.sep), 1)[0]
        internal_configs = textwrap.dedent('''
            # Tell doxygen to output wherever breathe is expecting things
            OUTPUT_DIRECTORY       = "{out}"
            # Tell doxygen to strip the path names (RTD builds produce long abs paths...)
            STRIP_FROM_PATH        = "{strip}"
        '''.format(out=doxy_dir, strip=configs.doxygenStripFromPath))
        external_configs = textwrap.dedent(configs.exhaleDoxygenStdin)
        # Place external configs last so that if the _valid_config method isn't actually
        # catching what it should be, the internal configs will override theirs
        full_input = "{base}\n{external}\n{internal}\n\n".format(base=configs.DEFAULT_DOXYGEN_STDIN_BASE,
                                                                 external=external_configs,
                                                                 internal=internal_configs)

        # << verboseBuild
        if configs.verboseBuild:
            msg = "[*] The following input will be sent to Doxygen:\n"
            if not configs.alwaysColorize and not sys.stderr.isatty():
                sys.stderr.write(msg)
                sys.stderr.write(full_input)
            else:
                sys.stderr.write(utils.colorize(msg, utils.AnsiColors.BOLD_CYAN))
                sys.stderr.write(utils.__fancy(full_input, "make", "console"))

        return _generate_doxygen(full_input)


########################################################################################
#
##
###
####
##### Primary entry point.
####
###
##
#
########################################################################################
def explode():
    '''
    This method **assumes** that :func:`~exhale.configs.apply_sphinx_configurations` has
    already been applied.  It performs minimal sanity checking, and then performs in
    order

    1. Creates a :class:`~exhale.graph.ExhaleRoot` object.
    2. Executes :func:`~exhale.graph.ExhaleRoot.parse` for this object.
    3. Executes :func:`~exhale.graph.ExhaleRoot.generateFullAPI` for this object.
    4. Executes :func:`~exhale.graph.ExhaleRoot.toConsole` for this object (which will
       only produce output when :data:`~exhale.configs.verboseBuild` is ``True``).

    This results in the full API being generated, and control is subsequently passed
    back to Sphinx to now read in the source documents (many of which were just
    generated in :data:`~exhale.configs.containmentFolder`), and proceed to writing the
    final output.
    '''
    # Quick sanity check to make sure the bare minimum have been set in the configs
    err_msg = "`configs.{config}` was `None`.  Do not call `deploy.explode` directly."
    if configs.containmentFolder is None:
        raise RuntimeError(err_msg.format(config="containmentFolder"))
    if configs.rootFileName is None:
        raise RuntimeError(err_msg.format(config="rootFileName"))
    if configs.doxygenStripFromPath is None:
        raise RuntimeError(err_msg.format(config="doxygenStripFromPath"))

    # From here on, we assume that everything else has been checked / configured.
    try:
        textRoot = ExhaleRoot()
    except:
        utils.fancyError("Unable to create an `ExhaleRoot` object:")

    try:
        sys.stdout.write("{0}\n".format(utils.info("Exhale: parsing Doxygen XML.")))
        start = utils.get_time()

        textRoot.parse()

        end = utils.get_time()
        sys.stdout.write("{0}\n".format(
            utils.progress("Exhale: finished parsing Doxygen XML in {0}.".format(
                utils.time_string(start, end)
            ))
        ))
    except:
        utils.fancyError("Exception caught while parsing:")
    try:
        sys.stdout.write("{0}\n".format(
            utils.info("Exhale: generating reStructuredText documents.")
        ))
        start = utils.get_time()

        textRoot.generateFullAPI()

        end = utils.get_time()
        sys.stdout.write("{0}\n".format(
            utils.progress("Exhale: generated reStructuredText documents in {0}.".format(
                utils.time_string(start, end)
            ))
        ))
    except:
        utils.fancyError("Exception caught while generating:")

    # << verboseBuild
    #   toConsole only prints if verbose mode is enabled
    textRoot.toConsole()

    # allow access to the result after-the-fact
    configs._the_app.exhale_root = textRoot
