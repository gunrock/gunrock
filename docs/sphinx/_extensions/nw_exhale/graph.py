########################################################################################
# This file is part of exhale.  Copyright (c) 2017-2022, Stephen McDowell.             #
# Full BSD 3-Clause license available here:                                            #
#                                                                                      #
#                https://github.com/svenevs/exhale/blob/master/LICENSE                 #
########################################################################################

from __future__ import unicode_literals

from . import configs
from . import parse
from . import utils

import re
import os
import sys
import codecs
import hashlib
import itertools
from pathlib import Path
import platform
import textwrap

from bs4 import BeautifulSoup

try:
    # Python 2 StringIO
    from cStringIO import StringIO
except ImportError:
    # Python 3 StringIO
    from io import StringIO

__all__       = ["ExhaleRoot", "ExhaleNode"]


########################################################################################
#
##
###
####
##### Graph representation.
####
###
##
#
########################################################################################
class ExhaleNode(object):
    '''
    A wrapper class to track parental relationships, filenames, etc.

    **Parameters**
        ``name`` (str)
            The name of the compound.

        ``kind`` (str)
            The kind of the compound (see :data:`~exhale.utils.AVAILABLE_KINDS`).

        ``refid`` (str)
            The reference ID that Doxygen has associated with this compound.

    **Attributes**
        ``kind`` (str)
            The value of the ``kind`` parameter.

        ``name`` (str)
            The value of the ``name`` parameter.

        ``refid`` (str)
            The value of the ``refid`` parameter.

        ``children`` (list)
            A potentially empty list of ``ExhaleNode`` object references that are
            considered a child of this Node.  Please note that a child reference in any
            ``children`` list may be stored in **many** other lists.  Mutating a given
            child will mutate the object, and therefore affect other parents of this
            child.  Lastly, a node of kind ``enum`` will never have its ``enumvalue``
            children as it is impossible to rebuild that relationship without more
            Doxygen xml parsing.

        ``parent`` (:class:`~exhale.graph.ExhaleNode`)
            If an ExhaleNode is determined to be a child of another ExhaleNode, this
            node will be added to its parent's ``children`` list, and a reference to
            the parent will be in this field.  Initialized to ``None``, make sure you
            check that it is an object first.

            .. warning::

               Do not ever set the ``parent`` of a given node if the would-be parent's
               kind is ``"file"``.  Doing so will break many important relationships,
               such as nested class definitions.  Effectively, **every** node will be
               added as a child to a file node at some point.  The file node will track
               this, but the child should not.

        The following three member variables are stored internally, but managed
        externally by the :class:`~exhale.graph.ExhaleRoot` class:

        ``file_name`` (str)
            The name of the file to create.  Set to ``None`` on creation, refer to
            :func:`~exhale.graph.ExhaleRoot.initializeNodeFilenameAndLink`.

        ``link_name`` (str)
            The name of the reStructuredText link that will be at the top of the file.
            Set to ``None`` on creation, refer to
            :func:`~exhale.graph.ExhaleRoot.initializeNodeFilenameAndLink`.

        ``title`` (str)
            The title that will appear at the top of the reStructuredText file
            ``file_name``. When the reStructuredText document for this node is being
            written, the root object will set this field.

        The following two fields are used for tracking what has or has not already been
        included in the hierarchy views.  Things like classes or structs in the global
        namespace will not be found by :func:`~exhale.graph.ExhaleNode.inClassHierarchy`,
        and the ExhaleRoot object will need to track which ones were missed.

        ``in_class_hierarchy`` (bool)
            Whether or not this node has already been incorporated in the class view.

        ``in_file_hierarchy`` (bool)
            Whether or not this node has already been incorporated in the file view.

        This class wields duck typing.  If ``self.kind == "file"``, then the additional
        member variables below exist:

        ``namespaces_used`` (list)
            A list of namespace nodes that are either defined or used in this file.

        ``includes`` (list)
            A list of strings that are parsed from the Doxygen xml for this file as
            include directives.

        ``included_by`` (list)
            A list of (refid, name) string tuples that are parsed from the Doxygen xml
            for this file presenting all of the other files that include this file.
            They are stored this way so that the root class can later link to that file
            by its refid.

        ``location`` (str)
            A string parsed from the Doxygen xml for this file stating where this file
            is physically in relation to the *Doxygen* root.

        ``program_listing`` (list)
            A list of strings that is the Doxygen xml <programlisting>, without the
            opening or closing <programlisting> tags.

        ``program_file`` (list)
            Managed externally by the root similar to ``file_name`` etc, this is the
            name of the file that will be created to display the program listing if it
            exists.  Set to ``None`` on creation, refer to
            :func:`~exhale.graph.ExhaleRoot.initializeNodeFilenameAndLink`.

        ``program_link_name`` (str)
            Managed externally by the root similar to ``file_name`` etc, this is the
            reStructuredText link that will be declared at the top of the
            ``program_file``. Set to ``None`` on creation, refer to
            :func:`~exhale.graph.ExhaleRoot.initializeNodeFilenameAndLink`.
    '''
    def __init__(self, name, kind, refid):
        self.name        = os.path.normpath(name) if kind == 'dir' else name
        self.kind        = kind
        self.refid       = refid
        self.root_owner  = None  # the ExhaleRoot owner

        self.template_params = []  # only populated if found

        # for inheritance
        self.base_compounds = []
        self.derived_compounds = []

        # used for establishing a link to the file something was done in for leaf-like
        # nodes conveniently, files also have this defined as their name making
        # comparison easy :)
        self.def_in_file = None
        # la familia
        self.children    = []    # ExhaleNodes
        self.parent      = None  # if reparented, will be an ExhaleNode
        # managed externally
        self.file_name   = None
        self.link_name   = None
        self.title       = None
        # representation of hierarchies
        self.in_page_hierarchy = False
        self.in_class_hierarchy = False
        self.in_file_hierarchy = False
        # kind-specific additional information
        if self.kind == "file":
            self.namespaces_used   = []  # ExhaleNodes
            self.includes          = []  # strings
            self.included_by       = []  # (refid, name) tuples
            self.language          = ""
            self.location          = ""
            self.program_listing   = []  # strings
            self.program_file      = ""
            self.program_link_name = ""

        if self.kind == "function":
            self.return_type = None # string (void, int, etc)
            self.parameters = [] # list of strings: ["int", "int"] for foo(int x, int y)
            self.template = None # list of strings

        if self.kind == "concept":
            self.template = None # list of strings

    def __lt__(self, other):
        '''
        The ``ExhaleRoot`` class stores a bunch of lists of ``ExhaleNode`` objects.
        When these lists are sorted, this method will be called to perform the sorting.

        :Parameters:
            ``other`` (ExhaleNode)
                The node we are comparing whether ``self`` is less than or not.

        :Return (bool):
            True if ``self`` is less than ``other``, False otherwise.
        '''
        # allows alphabetical sorting within types
        if self.kind == other.kind:
            if self.kind != "page":
                return self.name.lower() < other.name.lower()
            else:
                # Arbitrarily stuff "indexpage" refid to the front.  As doxygen presents
                # things, it shows up last, but it does not matter since the sort we
                # really care about will be with lists that do *NOT* have indexpage in
                # them (for creating the page view hierarchy).
                if self.refid == "indexpage":
                    return True
                elif other.refid == "indexpage":
                    return False

                # NOTE: kind of wasteful, but ordered_refs has ALL pages
                # but realistically, there wont be *that* many pages. right? ;)
                ordered_refs = [
                    p.refid for p in self.root_owner.index_xml_page_ordering
                ]
                return ordered_refs.index(self.refid) < ordered_refs.index(other.refid)
        # treat structs and classes as the same type
        elif self.kind == "struct" or self.kind == "class":
            if other.kind != "struct" and other.kind != "class":
                return True
            else:
                if self.kind == "struct" and other.kind == "class":
                    return True
                elif self.kind == "class" and other.kind == "struct":
                    return False
                else:
                    return self.name.lower() < other.name.lower()
        # otherwise, sort based off the kind
        else:
            return self.kind < other.kind

    def set_owner(self, root):
        """Sets the :class:`~exhale.graph.ExhaleRoot` owner ``self.root_owner``."""
        # needed to be able to track the page orderings as presented in index.xml
        self.root_owner = root

    def breathe_identifier(self):
        """
        The unique identifier for breathe directives.

        .. note::

            This method is currently assumed to only be called for nodes that are
            in :data:`exhale.utils.LEAF_LIKE_KINDS` (see also
            :func:`exhale.graph.ExhaleRoot.generateSingleNodeRST` where it is used).

        **Return**

            :class:`python:str`
                Usually, this will just be ``self.name``.  However, for functions in
                particular the signature must be included to distinguish overloads.
        """
        if self.kind == "function":
            # TODO: breathe bug with templates and overloads, don't know what to do...
            return "{name}({parameters})".format(
                name=self.name,
                parameters=", ".join(self.parameters)
            )

        return self.name

    def full_signature(self):
        """
        The full signature of a ``"function"`` node.

        **Return**
            :class:`python:str`
                The full signature of the function, including template, return type,
                name, and parameter types.

        **Raises**
            :class:`python:RuntimeError`
                If ``self.kind != "function"``.
        """
        if self.kind == "function":
            return "{template}{return_type} {name}({parameters})".format(
                template="template <{0}> ".format(", ".join(self.template)) if self.template is not None else "",
                return_type=self.return_type,
                name=self.name,
                parameters=", ".join(self.parameters)
            )
        if self.kind == "concept":
            return "{template} {name}".format(
                template="template <{0}> ".format(", ".join(self.template)) if self.template is not None else "",
                name=self.name)

        raise RuntimeError(
            "full_signature may only be called for a 'function' or 'concept', but {name} is a '{kind}' node.".format(
                name=self.name, kind=self.kind
            )
        )


    def templateParametersStringAsRestList(self, nodeByRefid):
        '''
        .. todo::

           document this, create another method for creating this without the need for
           generating links, to be used in making the node titles and labels
        '''
        if not self.template_params:
            return None
        else:
            param_stream = StringIO()
            for param_t, decl_n, def_n in self.template_params:
                refid, typeid = param_t
                # Say you wanted a custom link text 'custom', and somewhere
                # else you had an internal link '.. _some_link:'.  Then you do
                #     `custom <some_link_>`_
                # LOL. RST is confusing
                if refid:
                    # Easy case: the refid is something Exhale is explicitly documenting
                    if refid in nodeByRefid:
                        link = "{0}_".format(nodeByRefid[refid].link_name)
                    else:
                        # It's going to get generated by Breathe down the line, we need
                        # to reference the page the directive will appear on.
                        parent_refid = ""
                        for key in nodeByRefid:
                            if len(key) > len(parent_refid) and key in refid:
                                parent_refid = key
                        parent = nodeByRefid[parent_refid]
                        parent_page = os.path.basename(parent.file_name.replace(".rst", ".html"))
                        link = "{page}#{refid}".format(page=parent_page, refid=refid)
                    param_stream.write(
                        "#. `{typeid} <{link}>`_".format(
                            typeid=typeid,
                            # Not necessarily an ExhaleNode link, should be a link by
                            # the time Breathe is finished?
                            link=link
                        )
                    )
                    close_please = False
                else:
                    param_stream.write("#. ``{typeid}".format(typeid=typeid))
                    close_please = True

                # The type is in there, but when parsed it may have given something like
                # `class X` for the typeid (meaning nothing else to write).  For others,
                # the decl_n is the declared name of the template parameter.  E.g. it
                # was parsed as `typeid <- class` and `decl_n <- X`.
                if decl_n:
                    param_stream.write(" ")
                    if not close_please:
                        param_stream.write("``")
                    param_stream.write("{decl_n}".format(decl_n=decl_n))
                    close_please = True

                # When templates provide a default value, `def_n` is it.  When parsed,
                # if the `decl_n` and `def_n` are the same, `def_n` is explicitly set
                # to be None.
                if def_n:
                    param_stream.write(" ")
                    if not close_please:
                        param_stream.write("``")
                    param_stream.write("= {def_n}``".format(def_n=def_n))
                    close_please = True

                if close_please:
                    param_stream.write("``")

                param_stream.write("\n")

            param_stream.write("\n")
            param_value = param_stream.getvalue()
            param_stream.close()
            return param_value

    def baseOrDerivedListString(self, lst, nodeByRefid):
        '''
        .. todo:: long time from now: intersphinx should be possible here
        '''
        # lst should either be self.base_compounds or self.derived_compounds
        if not lst:
            return None

        bod_stream = StringIO()
        for prot, refid, string in lst:
            bod_stream.write("- ")

            # Include the prototype
            if prot:
                bod_stream.write("``{0}".format(prot))
                please_close = True
            else:
                please_close = False

            # Create the link, if possible
            # TODO: how to do intersphinx links here?
            # NOTE: refid is *NOT* guaranteed to be in nodeByRefid
            #       https://github.com/svenevs/exhale/pull/103
            if refid and refid in nodeByRefid:
                # TODO: why are these links not working????????????????????????????????
                ###########flake8breaks :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/ :/
                # if please_close:
                #     bod_stream.write("`` ")  # close prototype
                # bod_stream.write("`{name} <{link}_>`_".format(
                #     # name=string.replace("<", "&gt;").replace(">", "&lt;"),
                #     name=string.replace("<", "").replace(">", ""),
                #     link=nodeByRefid[refid].link_name
                # ))
                if not please_close:
                    bod_stream.write("``")
                else:
                    bod_stream.write(" ")
                bod_stream.write("{string}`` (:ref:`{link}`)".format(
                    string=string,
                    link=nodeByRefid[refid].link_name
                ))
            else:
                if not please_close:
                    bod_stream.write("``")
                else:
                    bod_stream.write(" ")
                bod_stream.write("{0}``".format(string))
            bod_stream.write("\n")

        bod_value = bod_stream.getvalue()
        bod_stream.close()
        return bod_value

    def findNestedNamespaces(self, lst):
        '''
        Recursive helper function for finding nested namespaces.  If this node is a
        namespace node, it is appended to ``lst``.  Each node also calls each of its
        child ``findNestedNamespaces`` with the same list.

        :Parameters:
            ``lst`` (list)
                The list each namespace node is to be appended to.
        '''
        if self.kind == "namespace":
            lst.append(self)
        for c in self.children:
            c.findNestedNamespaces(lst)

    def findNestedDirectories(self, lst):
        '''
        Recursive helper function for finding nested directories.  If this node is a
        directory node, it is appended to ``lst``.  Each node also calls each of its
        child ``findNestedDirectories`` with the same list.

        :Parameters:
            ``lst`` (list)
                The list each directory node is to be appended to.
        '''
        if self.kind == "dir":
            lst.append(self)
        for c in self.children:
            c.findNestedDirectories(lst)

    def findNestedClassLike(self, lst):
        '''
        Recursive helper function for finding nested classes and structs.  If this node
        is a class or struct, it is appended to ``lst``.  Each node also calls each of
        its child ``findNestedClassLike`` with the same list.

        :Parameters:
            ``lst`` (list)
                The list each class or struct node is to be appended to.
        '''
        if self.kind == "class" or self.kind == "struct":
            lst.append(self)
        for c in self.children:
            c.findNestedClassLike(lst)

    def findNestedEnums(self, lst):
        '''
        Recursive helper function for finding nested enums.  If this node is a class or
        struct it may have had an enum added to its child list.  When this occurred, the
        enum was removed from ``self.enums`` in the :class:`~exhale.graph.ExhaleRoot`
        class and needs to be rediscovered by calling this method on all of its
        children.  If this node is an enum, it is because a parent class or struct
        called this method, in which case it is added to ``lst``.

        **Note**: this is used slightly differently than nested directories, namespaces,
        and classes will be.  Refer to
        :func:`~exhale.graph.ExhaleRoot.generateNodeDocuments`.

        :Parameters:
            ``lst`` (list)
                The list each enum is to be appended to.
        '''
        if self.kind == "enum":
            lst.append(self)
        for c in self.children:
            c.findNestedEnums(lst)

    def findNestedUnions(self, lst):
        '''
        Recursive helper function for finding nested unions.  If this node is a class or
        struct it may have had a union added to its child list.  When this occurred, the
        union was removed from ``self.unions`` in the :class:`~exhale.graph.ExhaleRoot`
        class and needs to be rediscovered by calling this method on all of its
        children.  If this node is a union, it is because a parent class or struct
        called this method, in which case it is added to ``lst``.

        **Note**: this is used slightly differently than nested directories, namespaces,
        and classes will be.  Refer to
        :func:`~exhale.graph.ExhaleRoot.generateNodeDocuments`.

        :Parameters:
            ``lst`` (list)
                The list each union is to be appended to.
        '''
        if self.kind == "union":
            lst.append(self)
        for c in self.children:
            c.findNestedUnions(lst)

    def toConsole(self, level, fmt_spec, printChildren=True):
        '''
        Debugging tool for printing hierarchies / ownership to the console.  Recursively
        calls children ``toConsole`` if this node is not a directory or a file, and
        ``printChildren == True``.

        .. todo:: fmt_spec docs needed. keys are ``kind`` and values are color spec

        :Parameters:
            ``level`` (int)
                The indentation level to be used, should be greater than or equal to 0.

            ``printChildren`` (bool)
                Whether or not the ``toConsole`` method for the children found in
                ``self.children`` should be called with ``level+1``.  Default is True,
                set to False for directories and files.
        '''
        indent = "  " * level
        utils.verbose_log("{indent}- [{kind}]: {name}".format(
            indent=indent,
            kind=utils._use_color(self.kind, fmt_spec[self.kind], sys.stderr),
            name=self.name
        ))
        # files are children of directories, the file section will print those children
        if self.kind == "dir":
            for c in self.children:
                c.toConsole(level + 1, fmt_spec, printChildren=False)
        elif printChildren:
            if self.kind == "file":
                next_indent = "  " * (level + 1)
                utils.verbose_log("{next_indent}[[[ location=\"{loc}\" ]]]".format(
                    next_indent=next_indent,
                    loc=self.location
                ))
                for incl in self.includes:
                    utils.verbose_log("{next_indent}- #include <{incl}>".format(
                        next_indent=next_indent,
                        incl=incl
                    ))
                for ref, name in self.included_by:
                    utils.verbose_log("{next_indent}- included by: [{name}]".format(
                        next_indent=next_indent,
                        name=name
                    ))
                for n in self.namespaces_used:
                    n.toConsole(level + 1, fmt_spec, printChildren=False)
                for c in self.children:
                    c.toConsole(level + 1, fmt_spec)
            elif self.kind == "class" or self.kind == "struct":
                relevant_children = []
                for c in self.children:
                    if c.kind == "class" or c.kind == "struct" or \
                       c.kind == "enum"  or c.kind == "union":
                        relevant_children.append(c)

                for rc in sorted(relevant_children):
                    rc.toConsole(level + 1, fmt_spec)
            elif self.kind != "union":
                for c in self.children:
                    c.toConsole(level + 1, fmt_spec)

    def typeSort(self):
        '''
        Sorts ``self.children`` in place, and has each child sort its own children.
        Refer to :func:`~exhale.graph.ExhaleRoot.deepSortList` for more information on
        when this is necessary.
        '''
        self.children.sort()
        for c in self.children:
            c.typeSort()

    def inPageHierarchy(self):
        '''
        Whether or not this node should be included in the page view hierarchy.  Helper
        method for :func:`~exhale.graph.ExhaleNode.toHierarchy`.  Sets the member
        variable ``self.in_page_hierarchy`` to True if appropriate.

        :Return (bool):
            True if this node should be included in the page view --- if it is a
            node of kind ``page``. Returns False otherwise.
        '''
        self.in_page_hierarchy = self.kind == "page"
        return self.in_page_hierarchy

    def inClassHierarchy(self):
        '''
        Whether or not this node should be included in the class view hierarchy.  Helper
        method for :func:`~exhale.graph.ExhaleNode.toHierarchy`.  Sets the member
        variable ``self.in_class_hierarchy`` to True if appropriate.

        :Return (bool):
            True if this node should be included in the class view --- either it is a
            node of kind ``struct``, ``class``, ``enum``, ``union``, or it is a
            ``namespace`` that one or more if its descendants was one of the previous
            four kinds.  Returns False otherwise.
        '''
        if self.kind == "namespace":
            for c in self.children:
                if c.inClassHierarchy():
                    return True
            return False
        else:
            # flag that this node is already in the class view so we can find the
            # missing top level nodes at the end
            self.in_class_hierarchy = True

            # Skip children whose names were requested to be explicitly ignored.
            for exclude in configs._compiled_listing_exclude:
                if exclude.match(self.name):
                    return False

            return self.kind in {"struct", "class", "enum", "union"}

    def inFileHierarchy(self):
        '''
        Whether or not this node should be included in the file view hierarchy.  Helper
        method for :func:`~exhale.graph.ExhaleNode.toHierarchy`.  Sets the member
        variable ``self.in_file_hierarchy`` to True if appropriate.

        :Return (bool):
            True if this node should be included in the file view --- either it is a
            node of kind ``file``, or it is a ``dir`` that one or more if its
            descendants was a ``file``.  Returns False otherwise.
        '''
        if self.kind == "file":
            # flag that this file is already in the directory view so that potential
            # missing files can be found later.
            self.in_file_hierarchy = True
            return True
        elif self.kind == "dir":
            for c in self.children:
                if c.inFileHierarchy():
                    return True
        return False

    def inHierarchy(self, hierarchyType):
        if hierarchyType == "page":
            return self.inPageHierarchy()
        elif hierarchyType == "class":
            return self.inClassHierarchy()
        elif hierarchyType == "file":
            return self.inFileHierarchy()
        else:
            raise RuntimeError("'{}' is not a valid hierarchy type".format(hierarchyType))

    def hierarchySortedDirectDescendants(self, hierarchyType):
        if hierarchyType == "page":
            if self.kind != "page":
                raise RuntimeError(
                    "Page hierarchies do not apply to '{}' nodes".format(self.kind)
                )
            return sorted(self.children)
        elif hierarchyType == "class":
            # search for nested children to display as sub-items in the tree view
            if self.kind == "class" or self.kind == "struct":
                # first find all of the relevant children
                nested_class_like = []
                nested_enums      = []
                nested_unions     = []
                # important: only scan self.children, do not use recursive findNested* methods
                for c in self.children:
                    if c.kind == "struct" or c.kind == "class":
                        nested_class_like.append(c)
                    elif c.kind == "enum":
                        nested_enums.append(c)
                    elif c.kind == "union":
                        nested_unions.append(c)

                # sort the lists we just found
                nested_class_like.sort()
                nested_enums.sort()
                nested_unions.sort()

                # return a flattened listing with everything in the order it should be
                return [
                    child for child in itertools.chain(nested_class_like, nested_enums, nested_unions)
                ]
            # namespaces include nested namespaces, and any top-level class_like, enums,
            # and unions.  include nested namespaces first
            elif self.kind == "namespace":
                # pre-process and find everything that is relevant
                nested_nspaces = []
                nested_kids    = []
                for c in self.children:
                    if c.inHierarchy(hierarchyType):
                        if c.kind == "namespace":
                            nested_nspaces.append(c)
                        else:
                            nested_kids.append(c)

                # sort the lists
                nested_nspaces.sort()
                nested_kids.sort()

                # return a flattened listing with everything in the order it should be
                return [
                    child for child in itertools.chain(nested_nspaces, nested_kids)
                ]
            else:
                # everything else is a terminal node
                return []
        elif hierarchyType == "file":
            if self.kind == "dir":
                # find the nested children of interest
                nested_dirs = []
                nested_kids = []
                for c in self.children:
                    if c.inHierarchy(hierarchyType):
                        if c.kind == "dir":
                            nested_dirs.append(c)
                        elif c.kind == "file":
                            nested_kids.append(c)

                # sort the lists
                nested_dirs.sort()
                nested_kids.sort()

                # return a flattened listing with everything in the order it should be
                return [
                    child for child in itertools.chain(nested_dirs, nested_kids)
                ]
            else:
                # files are terminal nodes in this hierarchy view
                return []
        else:
            raise RuntimeError("{} is not a valid hierarchy type".format(hierarchyType))

    def toHierarchy(self, hierarchyType, level, stream, lastChild=False):
        '''
        **Parameters**
            ``hierarchyType`` (str)
                ``"page"`` if generating the Page Hierarchy,
                ``"class"`` if generating the Class Hierarchy,
                ``"file"`` if generating the File Hierarchy.

            ``level`` (int)
                Recursion level used to determine indentation.

            ``stream`` (StringIO)
                The stream to write the contents to.

            ``lastChild`` (bool)
                When :data:`~exhale.configs.createTreeView` is ``True`` and
                :data:`~exhale.configs.treeViewIsBootstrap` is ``False``, the generated
                HTML ``li`` elements need to add a ``class="lastChild"`` to use the
                appropriate styling.

        .. todo:: add thorough documentation of this
        '''
        # NOTE: indexpage needs to be treated specially, you need to include the
        # children at the *same* level, and not actually include indexpage.
        if hierarchyType == "page" and self.refid == "indexpage":
            nested_children = self.hierarchySortedDirectDescendants(hierarchyType)
            last_child_index = len(nested_children) - 1
            child_idx        = 0
            for child in nested_children:
                child.toHierarchy(
                    hierarchyType, level, stream, child_idx == last_child_index)
                child_idx += 1
            return
        if self.inHierarchy(hierarchyType):
            # For the Tree Views, we need to know if there are nested children before
            # writing anything.  If there are, we need to open a new list
            nested_children = self.hierarchySortedDirectDescendants(hierarchyType)

            ############################################################################
            # Write out this node.                                                     #
            ############################################################################
            # Easy case: just write another bullet point
            if not configs.createTreeView:
                stream.write("{indent}- :ref:`{link}`\n".format(
                    indent='    ' * level,
                    link=self.link_name
                ))
            # Otherwise, we're generating some raw HTML and/or JavaScript depending on
            # whether we are using bootstrap or not
            else:
                # Declare the relevant links needed for the Tree Views
                indent = "  " * (level * 2)
                next_indent = "  {0}".format(indent)

                # turn double underscores into underscores, then underscores into hyphens
                html_link = self.link_name.replace("__", "_").replace("_", "-")
                href = "{file}.html#{anchor}".format(
                    file=self.file_name.rsplit(".rst", 1)[0],
                    anchor=html_link
                )

                if self.kind != "page":
                    # should always have at least two parts (templates will have more)
                    title_as_link_parts = self.title.split(" ")
                    if self.template_params:
                        # E.g. 'Template Class Foo'
                        q_start = 0
                        q_end   = 2
                    else:
                        # E.g. 'Class Foo'
                        q_start = 0
                        q_end   = 1
                    # the qualifier will not be part of the hyperlink (for clarity of
                    # navigation), the link_title will be
                    qualifier   = " ".join(title_as_link_parts[q_start:q_end])
                    link_title  = " ".join(title_as_link_parts[q_end:])
                else:
                    # E.g. 'Foo'
                    qualifier = ""
                    link_title = self.title

                link_title  = link_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # the actual text / link inside of the list item
                li_text     = '{qualifier} <a href="{href}">{link_title}</a>'.format(
                    qualifier=qualifier,
                    href=href,
                    link_title=link_title
                )

                if configs.treeViewIsBootstrap:
                    text = "text: \"<span class=\\\"{span_cls}\\\">{qualifier}</span> {link_title}\"".format(
                        span_cls=configs.treeViewBootstrapTextSpanClass,
                        qualifier=qualifier,
                        link_title=link_title
                    )
                    link = "href: \"{href}\"".format(href=href)
                    # write some json data, something like
                    #     {
                    #         text: "<span class=\\\"text-muted\\\"> some text",
                    #         href: "link to actual item",
                    #         selectable: false,
                    stream.write("{indent}{{\n{next_indent}{text},\n".format(
                        indent=indent,
                        next_indent=next_indent,
                        text=text
                    ))
                    stream.write("{next_indent}{link},\n{next_indent}selectable: false,\n".format(
                        next_indent=next_indent,
                        link=link
                    ))
                    # if requested, add the badge indicating how many children there are
                    # only add this if there are children
                    if configs.treeViewBootstrapUseBadgeTags and nested_children:
                        stream.write("{next_indent}tags: ['{num_children}'],\n".format(
                            next_indent=next_indent,
                            num_children=len(nested_children)
                        ))

                    if nested_children:
                        # If there are children then `nodes: [ ... ]` will be next
                        stream.write("\n{next_indent}nodes: [\n".format(next_indent=next_indent))
                    else:
                        # Otherwise, this element is ending.  JavaScript doesn't care
                        # about trailing commas :)
                        stream.write("{indent}}},\n".format(indent=indent))
                else:
                    if lastChild:
                        opening_li = '<li class="lastChild">'
                    else:
                        opening_li = "<li>"

                    if nested_children:
                        # write this list element and begin the next list
                        # writes something like
                        #     <li>
                        #         some text with an href
                        #         <ul>
                        #
                        # the <ul> started here gets closed below
                        stream.write("{indent}{li}\n{next_indent}{li_text}\n{next_indent}<ul>\n".format(
                            indent=indent,
                            li=opening_li,
                            next_indent=next_indent,
                            li_text=li_text
                        ))
                    else:
                        # write this list element and end it now (since no children)
                        # writes something like
                        #    <li>
                        #        some text with an href
                        #    </li>
                        stream.write("{indent}{li}{li_text}</li>\n".format(
                            indent=indent,
                            li=opening_li,
                            li_text=li_text
                        ))

            ############################################################################
            # Write out all of the children (if there are any).                        #
            ############################################################################
            last_child_index = len(nested_children) - 1
            child_idx        = 0
            for child in nested_children:
                child.toHierarchy(hierarchyType, level + 1, stream, child_idx == last_child_index)
                child_idx += 1

            ############################################################################
            # If there were children, close the lists we started above.                #
            ############################################################################
            if configs.createTreeView and nested_children:
                if configs.treeViewIsBootstrap:
                    # close the `nodes: [ ... ]` and final } for element
                    # the final comma IS necessary, and extra commas don't matter in javascript
                    stream.write("{next_indent}]\n{indent}}},\n".format(
                        next_indent=next_indent,
                        indent=indent
                    ))
                else:
                    stream.write("{next_indent}</ul>\n{indent}</li>\n".format(
                        next_indent=next_indent,
                        indent=indent
                    ))


class ExhaleRoot(object):
    '''
    The full representation of the hierarchy graphs.  In addition to containing specific
    lists of ExhaleNodes of interest, the ExhaleRoot class is responsible for comparing
    the parsed breathe hierarchy and rebuilding lost relationships using the Doxygen
    xml files.  Once the graph parsing has finished, the ExhaleRoot generates all of the
    relevant reStructuredText documents and links them together.

    The ExhaleRoot class is not designed for reuse at this time.  If you want to
    generate a new hierarchy with a different directory or something, changing all of
    the right fields may be difficult and / or unsuccessful.  Refer to the
    :func:`~exhale.deploy.explode` function for intended usage.

    .. danger::

       Zero checks are in place to enforce this usage, and if you are modifying the
       execution of this class and things are not working make sure you follow the
       ordering of those methods.

    .. todo::

       many attributes currently stored do not need to be, refactor in future release
       to just use the ``configs`` module.

    **Attributes**
        ``root_directory`` (str)
            The value of the parameter ``rootDirectory``.

        ``root_file_name`` (str)
            The value of the parameter ``rootFileName``.

        ``full_root_file_path`` (str)
            The full file path of the root file (``"root_directory/root_file_name"``).

        ``class_hierarchy_file`` (str)
            The full file path the class view hierarchy will be written to.  This is
            incorporated into ``root_file_name`` using an ``.. include:`` directive.

        ``file_hierarchy_file`` (str)
            The full file path the file view hierarchy will be written to.  This is
            incorporated into ``root_file_name`` using an ``.. include:`` directive.

        ``unabridged_api_file`` (str)
            The full file path the full API will be written to.  This is incorporated
            into ``root_file_name`` using a ``.. toctree:`` directive with a
            ``:maxdepth:`` according to the value of
            :data:`~exhale.configs.fullToctreeMaxDepth`.

        ``use_tree_view`` (bool)
            The value of the parameter ``createTreeView``.

        ``all_compounds`` (list)
            A list of all the Breathe compound objects discovered along the way.
            Populated during :func:`~exhale.graph.ExhaleRoot.discoverAllNodes`.

        ``all_nodes`` (list)
            A list of all of the ExhaleNode objects created.  Populated during
            :func:`~exhale.graph.ExhaleRoot.discoverAllNodes`.

        ``node_by_refid`` (dict)
            A dictionary with string ExhaleNode ``refid`` values, and values that are the
            ExhaleNode it came from.  Storing it this way is convenient for when the
            Doxygen xml file is being parsed.

        ``concepts`` (list)
            The full list of ExhaleNodes of kind ``concept``

        ``class_like`` (list)
            The full list of ExhaleNodes of kind ``struct`` or ``class``

        ``defines`` (list)
            The full list of ExhaleNodes of kind ``define``.

        ``enums`` (list)
            The full list of ExhaleNodes of kind ``enum``.

        ``enum_values`` (list)
            The full list of ExhaleNodes of kind ``enumvalue``.  Populated, not used.

        ``functions`` (list)
            The full list of ExhaleNodes of kind ``function``.

        ``dirs`` (list)
            The full list of ExhaleNodes of kind ``dir``.

        ``files`` (list)
            The full list of ExhaleNodes of kind ``file``.

        ``groups`` (list)
            The full list of ExhaleNodes of kind ``group``.  Pupulated, not used.

        ``namespaces`` (list)
            The full list of ExhaleNodes of kind ``namespace``.

        ``typedefs`` (list)
            The full list of ExhaleNodes of kind ``typedef``.

        ``unions`` (list)
            The full list of ExhaleNodes of kind ``union``.

        ``variables`` (list)
            The full list of ExhaleNodes of kind ``variable``.
    '''
    def __init__(self):
        # file generation location and root index data
        self.root_directory         = configs.containmentFolder
        self.root_file_name         = configs.rootFileName
        self.full_root_file_path    = os.path.join(self.root_directory, self.root_file_name)
        # The {page,class,file}_view_hierarchy files are all `.. include::`ed in the
        # root library document.  Though we are generating rst, we will want to use a
        # file extension `.rst.include` to bypass the fact that the sphinx builder will
        # process them separately if we leave them as .rst (via the source_suffix
        # configuration of the sphinx app).  If users are getting warnings about it
        # then we can actually check for `.include` in app.config.source_suffix, but
        # it is very unlikely this is going to be a problem.
        # See https://github.com/sphinx-doc/sphinx/issues/1668
        self.page_hierarchy_file    = os.path.join(self.root_directory, "page_view_hierarchy.rst.include")
        self.class_hierarchy_file   = os.path.join(self.root_directory, "class_view_hierarchy.rst.include")
        self.file_hierarchy_file    = os.path.join(self.root_directory, "file_view_hierarchy.rst.include")
        self.unabridged_api_file    = os.path.join(self.root_directory, "unabridged_api.rst.include")
        # NOTE: do *NOT* do .rst.include for the unabridged orphan kinds, the purpose of
        # that document is to have it be processed by sphinx with its corresponding
        # .. toctree:: calls to kinds that the user has asked to be excluded.  Sphinx
        # processing this document directly is desired (it is also marked :orphan: to
        # avoid a warning on the fact that it is *NOT* included in any exhale toctree).
        self.unabridged_orphan_file = os.path.join(self.root_directory, "unabridged_orphan.rst")

        # whether or not we should generate the raw html tree view
        self.use_tree_view = configs.createTreeView

        # track all compounds to build all nodes (ExhaleNodes)
        self.all_compounds = []##### update how this is used (compounds inserted are from xml parsing)
        self.all_nodes = []

        # convenience lookup: keys are string Doxygen refid's, values are ExhaleNodes
        self.node_by_refid = {}

        # breathe directive    breathe kind
        # -------------------+----------------+
        # autodoxygenfile  <-+-> IGNORE       |
        # doxygenindex     <-+-> IGNORE       |
        # autodoxygenindex <-+-> IGNORE       |
        # -------------------+----------------+
        # doxygenconcept   <-+-> "concept"    |
        self.concepts        = []           # |
        # doxygenclass     <-+-> "class"      |
        # doxygenstruct    <-+-> "struct"     |
        self.class_like      = []           # |
        # doxygendefine    <-+-> "define"     |
        self.defines         = []           # |
        # doxygenenum      <-+-> "enum"       |
        self.enums           = []           # |
        # ---> largely ignored by framework,  |
        #      but stored if desired          |
        # doxygenenumvalue <-+-> "enumvalue"  |
        self.enum_values     = []           # |
        # doxygenfunction  <-+-> "function"   |
        self.functions       = []           # |
        # no directive     <-+-> "dir"        |
        self.dirs = []                      # |
        # doxygenfile      <-+-> "file"       |
        self.files           = []           # |
        # not used, but could be supported in |
        # the future?                         |
        # doxygengroup     <-+-> "group"      |
        self.groups          = []           # |
        # doxygennamespace <-+-> "namespace"  |
        self.namespaces      = []           # |
        # doxygentypedef   <-+-> "typedef"    |
        self.typedefs        = []           # |
        # doxygenunion     <-+-> "union"      |
        self.unions          = []           # |
        # doxygenvariable  <-+-> "variable"   |
        self.variables       = []           # |
        # doxygenpage      <-+-> "page"       |
        self.pages           = []           # |
        # -------------------+----------------+
        # tracks the named ordering of pages as they show up in index.xml
        # so that the page hierarchy can be presented in the same order.
        # the only node not placed in here is "indexpage" since it is not
        # included in the page view hierarchy (indexpage is dumped right above)
        self.index_xml_page_ordering = []

    ####################################################################################
    #
    ##
    ### Parsing
    ##
    #
    ####################################################################################
    def parse(self):
        '''
        The first method that should be called after creating an ExhaleRoot object.  The
        Breathe graph is parsed first, followed by the Doxygen xml documents.  By the
        end of this method, all of the ``self.<breathe_kind>``, ``self.all_compounds``,
        and ``self.all_nodes`` lists as well as the ``self.node_by_refid`` dictionary
        will be populated.  Lastly, this method sorts all of the internal lists.  The
        order of execution is exactly

        1. :func:`~exhale.graph.ExhaleRoot.discoverAllNodes`
        2. :func:`~exhale.graph.ExhaleRoot.reparentAll`
        3. Populate ``self.node_by_refid`` using ``self.all_nodes``.
        4. :func:`~exhale.graph.ExhaleRoot.fileRefDiscovery`
        5. :func:`~exhale.graph.ExhaleRoot.filePostProcess`
        6. :func:`~exhale.graph.ExhaleRoot.parseFunctionSignatures`.
        7. :func:`~exhale.graph.ExhaleRoot.sortInternals`
        '''
        self.discoverAllNodes()
        # now reparent everything we can
        # NOTE: it's very important that this happens before `fileRefDiscovery`, since
        #       in that method we only want to consider direct descendants
        self.reparentAll()

        # now that we have all of the nodes, store them in a convenient manner for refid
        # lookup when parsing the Doxygen xml files
        for n in self.all_nodes:
            self.node_by_refid[n.refid] = n

        # find missing relationships using the Doxygen xml files
        self.fileRefDiscovery()
        self.filePostProcess()

        # gather the function signatures
        self.parseFunctionSignatures()

        # sort all of the lists we just built
        self.sortInternals()

    def discoverAllNodes(self):
        '''
        .. todo:: node discovery has changed, breathe no longer used...update docs
        '''
        doxygen_index_xml = os.path.join(
            configs._doxygen_xml_output_directory,
            "index.xml"
        )
        try:
            with codecs.open(doxygen_index_xml, "r", "utf-8") as index:
                index_contents = index.read()
        except:
            raise RuntimeError("Could not read the contents of [{0}].".format(doxygen_index_xml))

        try:
            index_soup = BeautifulSoup(index_contents, "lxml-xml")
        except:
            raise RuntimeError("Could not parse the contents of [{0}] as an xml.".format(doxygen_index_xml))

        doxygen_root = index_soup.doxygenindex
        if not doxygen_root:
            raise RuntimeError(
                "Did not find root XML node named 'doxygenindex' parsing [{0}].".format(doxygen_index_xml)
            )

        for compound in doxygen_root.find_all("compound"):
            if compound.find("name") and "kind" in compound.attrs and "refid" in compound.attrs:
                curr_name  = compound.find("name").get_text()
                curr_kind  = compound.attrs["kind"]
                curr_refid = compound.attrs["refid"]
                curr_node  = ExhaleNode(curr_name, curr_kind, curr_refid)
                self.trackNodeIfUnseen(curr_node)

                # For things like files and namespaces, a "member" list will include
                # things like defines, enums, etc.  For classes and structs, we don't
                # need to pay attention because the members are the various methods or
                # data members by the class
                if curr_kind in ["file", "namespace"]:
                    for member in compound.find_all("member"):
                        if member.find("name") and "kind" in member.attrs and "refid" in member.attrs:
                            child_name  = member.find("name").get_text()
                            child_kind  = member.attrs["kind"]
                            child_refid = member.attrs["refid"]
                            child_node  = ExhaleNode(child_name, child_kind, child_refid)
                            self.trackNodeIfUnseen(child_node)

                            if curr_kind == "namespace":
                                child_node.parent = curr_node
                            else:  # curr_kind == "file"
                                child_node.def_in_file = curr_node

                            curr_node.children.append(child_node)

        for page in self.pages:
            node_xml_contents = utils.nodeCompoundXMLContents(page)
            if node_xml_contents:
                try:
                    page.soup = BeautifulSoup(node_xml_contents, "lxml-xml")
                except:
                    utils.fancyError("Unable to parse file xml [{0}]:".format(page.name))

                try:
                    cdef = page.soup.doxygen.compounddef

                    title = cdef.find("title")
                    if title and title.string:
                        page.title = title.string

                    err_non = "[CRITICAL] did not find refid [{0}] in `self.node_by_refid`."
                    err_dup = "Conflicting page definition: [{0}] appears to be defined in both [{1}] and [{2}]."  # noqa
                    # process subpages
                    inner_pages = cdef.find_all("innerpage", recursive=False)

                    utils.verbose_log(
                        "*** [{0}] had [{1}] innerpages found".format(page.name, len(inner_pages)),
                        utils.AnsiColors.BOLD_MAGENTA
                    )

                    for subpage in inner_pages:
                        if "refid" in subpage.attrs:
                            refid = subpage.attrs["refid"]
                            if refid in self.node_by_refid:
                                node = self.node_by_refid[refid]

                                # << verboseBuild
                                utils.verbose_log(
                                    "    - [{0}]".format(node.name),
                                    utils.AnsiColors.BOLD_MAGENTA
                                )

                                if node.parent:
                                    utils.verbose_log(
                                        err_dup.format(node.name, node.parent.name, page.name),
                                        utils.AnsiColors.BOLD_YELLOW
                                    )

                                if node not in page.children:
                                    page.children.append(node)
                                    node.parent = page
                            else:
                                # << verboseBuild
                                utils.verbose_log(err_non.format(refid), utils.AnsiColors.BOLD_RED)

                    # the location of the page as determined by doxygen
                    location = cdef.find("location")
                    if location and "file" in location.attrs:
                        location_str = os.path.normpath(location.attrs["file"])
                        # some older versions of doxygen don't reliably strip from path
                        # so make sure to remove it
                        abs_strip_path = os.path.normpath(os.path.abspath(
                            configs.doxygenStripFromPath
                        ))
                        if location_str.startswith(abs_strip_path):
                            location_str = os.path.relpath(location_str, abs_strip_path)
                        page.location = os.path.normpath(location_str)

                except:
                    utils.fancyError(
                        "Could not process Doxygen xml for file [{0}]".format(f.name)
                    )
        self.pages = [page for page in self.pages if not page.parent]

        # Now that we have discovered everything, we need to explicitly parse the file
        # xml documents to determine where leaf-like nodes have been declared.
        #
        # TODO: change formatting of namespace to provide a listing of all files using it
        for f in self.files:
            node_xml_contents = utils.nodeCompoundXMLContents(f)
            if node_xml_contents:
                try:
                    f.soup = BeautifulSoup(node_xml_contents, "lxml-xml")
                except:
                    utils.fancyError("Unable to parse file xml [{0}]:".format(f.name))

                try:
                    cdef = f.soup.doxygen.compounddef

                    if "language" in cdef.attrs:
                        f.language = cdef.attrs["language"]

                    err_non = "[CRITICAL] did not find refid [{0}] in `self.node_by_refid`."
                    err_dup = "Conflicting file definition: [{0}] appears to be defined in both [{1}] and [{2}]."  # noqa
                    # process classes
                    inner_classes = cdef.find_all("innerclass", recursive=False)

                    # << verboseBuild
                    utils.verbose_log(
                        "*** [{0}] had [{1}] innerclasses found".format(f.name, len(inner_classes)),
                        utils.AnsiColors.BOLD_MAGENTA
                    )

                    for class_like in inner_classes:
                        if "refid" in class_like.attrs:
                            refid = class_like.attrs["refid"]
                            if refid in self.node_by_refid:
                                node = self.node_by_refid[refid]

                                # << verboseBuild
                                utils.verbose_log(
                                    "    - [{0}]".format(node.name),
                                    utils.AnsiColors.BOLD_MAGENTA
                                )

                                if not node.def_in_file:
                                    node.def_in_file = f
                                elif node.def_in_file != f:
                                    # << verboseBuild
                                    utils.verbose_log(
                                        err_dup.format(node.name, node.def_in_file.name, f.name),
                                        utils.AnsiColors.BOLD_YELLOW
                                    )
                            else:
                                # << verboseBuild
                                utils.verbose_log(err_non.format(refid), utils.AnsiColors.BOLD_RED)
                        else:
                            # TODO: can this ever happen?
                            # << verboseBuild
                            catastrophe  = "CATASTROPHIC: doxygen xml for `{0}` found `innerclass` [{1}] that"
                            catastrophe += " does *NOT* have a `refid` attribute!"
                            catastrophe  = catastrophe.format(f, str(class_like))
                            utils.verbose_log(
                                utils.prefix("(!) ", catastrophe),
                                utils.AnsiColors.BOLD_RED
                            )

                    # try and find anything else
                    memberdefs = cdef.find_all("memberdef", recursive=False)

                    # << verboseBuild
                    utils.verbose_log(
                        "*** [{0}] had [{1}] memberdef".format(f.name, len(memberdefs)),
                        utils.AnsiColors.BOLD_MAGENTA
                    )

                    for member in cdef.find_all("memberdef", recursive=False):
                        if "id" in member.attrs:
                            refid = member.attrs["id"]
                            if refid in self.node_by_refid:
                                node = self.node_by_refid[refid]

                                # << verboseBuild
                                utils.verbose_log(
                                    "    - [{0}]".format(node.name),
                                    utils.AnsiColors.BOLD_MAGENTA
                                )

                                if not node.def_in_file:
                                    node.def_in_file = f

                    # the location of the file as determined by doxygen
                    location = cdef.find("location")
                    if location and "file" in location.attrs:
                        location_str = os.path.normpath(location.attrs["file"])
                        # some older versions of doxygen don't reliably strip from path
                        # so make sure to remove it
                        abs_strip_path = os.path.normpath(os.path.abspath(
                            configs.doxygenStripFromPath
                        ))
                        if location_str.startswith(abs_strip_path):
                            location_str = os.path.relpath(location_str, abs_strip_path)
                        f.location = os.path.normpath(location_str)

                except:
                    utils.fancyError(
                        "Could not process Doxygen xml for file [{0}]".format(f.name)
                    )

        ###### TODO: explain how the parsing works // move it to exhale.parse
        # last chance: we will still miss some, but need to pause and establish namespace relationships
        for nspace in self.namespaces:
            node_xml_contents = utils.nodeCompoundXMLContents(nspace)
            if node_xml_contents:
                try:
                    name_soup = BeautifulSoup(node_xml_contents, "lxml-xml")
                except:
                    continue

                cdef = name_soup.doxygen.compounddef
                for class_like in cdef.find_all("innerclass", recursive=False):
                    if "refid" in class_like.attrs:
                        refid = class_like.attrs["refid"]
                        if refid in self.node_by_refid:
                            node = self.node_by_refid[refid]
                            if node not in nspace.children:
                                nspace.children.append(node)
                                node.parent = nspace

                for nested_nspace in cdef.find_all("innernamespace", recursive=False):
                    if "refid" in nested_nspace.attrs:
                        refid = nested_nspace.attrs["refid"]
                        if refid in self.node_by_refid:
                            node = self.node_by_refid[refid]
                            if node not in nspace.children:
                                nspace.children.append(node)
                                node.parent = nspace

                # This is where things get interesting
                for sectiondef in cdef.find_all("sectiondef", recursive=False):
                    for memberdef in sectiondef.find_all("memberdef", recursive=False):
                        if "id" in memberdef.attrs:
                            refid = memberdef.attrs["id"]
                            if refid in self.node_by_refid:
                                node = self.node_by_refid[refid]
                                location = memberdef.find("location")
                                if location and "file" in location.attrs:
                                    filedef = os.path.normpath(location.attrs["file"])
                                    for f in self.files:
                                        if filedef == f.location:
                                            node.def_in_file = f
                                            if node not in f.children:
                                                f.children.append(node)
                                            break

        # Find the nodes that did not have their file location definition assigned
        missing_file_def            = {} # keys: refid, values: ExhaleNode
        missing_file_def_candidates = {} # keys: refid, values: ExhaleNode (file kind only!)
        for refid in self.node_by_refid:
            node = self.node_by_refid[refid]
            if node.def_in_file is None and node.kind not in ("file", "dir", "group", "namespace", "enumvalue"):
                missing_file_def[refid] = node
                missing_file_def_candidates[refid] = []

        # Some compounds like class / struct have their own XML file and if documented
        # correctly will have a <location> tag.  For example, one may need to add the
        #
        #     \class namespace::ClassName file_basename.hpp full/file/path/file_basename.hpp
        #
        # in order for the <location> tag to be generated.  And in the case of forward
        # declarations (e.g., for PIMPL patterns), in order for the class XML to be
        # generated at all it seems this must be used.
        #
        #     <?xml version='1.0' encoding='UTF-8' standalone='no'?>
        #     <doxygen xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="compound.xsd" version="1.8.13">
        #       <compounddef id="classpimpl_1_1EarthImpl" kind="class" language="C++" prot="public">
        #         <compoundname>pimpl::EarthImpl</compoundname>
        #         <includes refid="earth_8hpp" local="no">include/pimpl/earth.hpp</includes>
        #         <briefdescription>
        #     <para>The <ref refid="classpimpl_1_1Earth" kindref="compound">Earth</ref> PIMPL. </para>    </briefdescription>
        #         <detaileddescription>
        #         </detaileddescription>
        #         <location file="include/pimpl/earth.hpp" line="30" column="1"/>
        #         <listofallmembers>
        #         </listofallmembers>
        #       </compounddef>
        #     </doxygen>
        #
        # So we're taking advantage of the fact that
        #
        #    namespace pimpl {
        #        /**
        #         * \class pimpl::EarthImpl earth.hpp include/pimpl/earth.hpp
        #         * \brief The Earth PIMPL.
        #         */
        #         class EarthImpl;
        #     }
        #
        # Has a <location file="include/pimpl/earth.hpp" line="30" column="1"/>
        #
        # TODO: clarify this in the docs?  You don't understand the full cause though.
        refid_removals = []
        for refid in missing_file_def:
            node = missing_file_def[refid]
            node_xml_contents = utils.nodeCompoundXMLContents(node)
            # None is returned when no {refid}.xml exists (e.g., for enum or union).
            if not node_xml_contents:
                pass

            try:
                node_soup = BeautifulSoup(node_xml_contents, "lxml-xml")
                cdef = node_soup.doxygen.compounddef
                location = cdef.find("location", recursive=False)
                if location and "file" in location.attrs:
                    file_path = os.path.normpath(location["file"])
                    for f in self.files:
                        if f.location == file_path:
                            node.def_in_file = f
                            f.children.append(node)
                            refid_removals.append(refid)
            except:
                pass

        # We found the def_in_file, don't parse the programlisting for these nodes.
        for refid in refid_removals:
            del missing_file_def[refid]

        # Go through every file and see if the refid associated with a node missing a
        # file definition location is present in the <programlisting>
        for f in self.files:
            cdef = f.soup.doxygen.compounddef
            # try and find things in the programlisting as a last resort
            programlisting = cdef.find("programlisting")
            if programlisting:
                for ref in programlisting.find_all("ref"):
                    if "refid" in ref.attrs:
                        refid = ref.attrs["refid"]
                        # be careful not to just consider any refid found, e.g. don't
                        # use the `compound` kindref's because those are just stating
                        # it was used in this file, not that it was declared here
                        if "kindref" in ref.attrs and ref.attrs["kindref"] == "member":
                            if refid in missing_file_def and f not in missing_file_def_candidates[refid]:
                                missing_file_def_candidates[refid].append(f)

        # For every refid missing a file definition location, see if we found it only
        # once in a file node's <programlisting>.  If so, assign that as the file the
        # node was defined in
        for refid in missing_file_def:
            node = missing_file_def[refid]
            candidates = missing_file_def_candidates[refid]
            # If only one found, life is good!
            if len(candidates) == 1:
                node.def_in_file = candidates[0]
                # << verboseBuild
                utils.verbose_log(utils.info(
                    "Manually setting file definition of {0} {1} to [{2}]".format(
                        node.kind, node.name, node.def_in_file.location
                    ),
                    utils.AnsiColors.BOLD_CYAN
                ))
            # More than one found, don't know what to do...
            elif len(candidates) > 1:
                # << verboseBuild
                err_msg = StringIO()
                err_msg.write(textwrap.dedent('''
                    While attempting to discover the file that Doxygen refid `{0}` was
                    defined in, more than one candidate was found.  The candidates were:
                '''.format(refid)))
                # NOTE: candidates should only ever contain File nodes (thus c.location
                #       should exist, and already be populated).
                for c in candidates:
                    err_msg.write("  - path=[{0}], refid={1}\n".format(c.location, c.refid))
                err_msg.write("\n")
                utils.verbose_log(utils.critical(err_msg.getvalue()))
            # NOTE: no 'else' clause here, a warning about no file link generated is
            #       produced when the rst file is written

        # now that all nodes have been discovered, process template parameters, and
        # coordinate any base / derived inheritance relationships
        for node in self.class_like:
            node_xml_contents = utils.nodeCompoundXMLContents(node)
            if node_xml_contents:
                try:
                    name_soup = BeautifulSoup(node_xml_contents, "lxml-xml")
                except:
                    utils.fancyError("Could not process [{0}]".format(
                        os.path.join(configs._doxygen_xml_output_directory, "{0}".format(node.refid))
                    ))

                try:
                    cdef = name_soup.doxygen.compounddef
                    tparams = cdef.find("templateparamlist", recursive=False)
                    #
                    # DANGER DANGER DANGER
                    # No, you may not build links directly right now.  Cuz they aren't initialized
                    #

                    # first, find template parameters
                    if tparams:
                        for param in tparams.find_all("param", recursive=False):
                            # Doxygen seems to produce unreliable results.  For example,
                            # sometimes you will get `param.type <- class X` with empty
                            # decloname and defname, and sometimes you will get
                            # `param.type <- class` and declname `X`.  Similar behavior
                            # is observed with `typename X`.  These are generally just
                            # ignored (falling in the broader category of a typename)
                            #
                            # Sometimes you will get a refid in the type, so deal with
                            # that as they come too (yay)!
                            param_t = param.type
                            decl_n  = param.declname
                            def_n   = param.defname

                            # TODO: this doesn't seem to happen, should probably investigate more
                            # do something with `param.defval` ?

                            # By the end:
                            # param_t <- (None | str, str) tuple
                            #             ^^^^^^^^^^
                            #             only a refid, or None
                            # decl_n  <- str; declared name
                            # def_n   <- None | str; defined name
                            #
                            # When decl_n and def_n are the same, this means no explicit
                            # default template parameter is given.  This will ultimately
                            # mean that def_n is set to None for consistency.
                            if param_t.ref:
                                if "refid" in param_t.ref.attrs:
                                    refid = param_t.ref.attrs["refid"]
                                else:
                                    # I hope this never happens.
                                    refid = None
                                param_t = (refid, param_t.ref.string)
                            else:
                                param_t = (None, param_t.string)

                            # Right now these are the soup tags, get the strings
                            if decl_n:
                                decl_n = decl_n.string
                            if def_n:
                                def_n  = def_n.string

                            # Unset def_n if same as decl_n
                            if decl_n and def_n and decl_n == def_n:
                                def_n = None

                            node.template_params.append((param_t, decl_n, def_n))

                    def prot_ref_str(soup_node):
                        if "prot" in soup_node.attrs:
                            prot = soup_node.attrs["prot"]
                        else:
                            prot = None
                        if "refid" in soup_node.attrs:
                            refid = soup_node.attrs["refid"]
                        else:
                            refid = None
                        return (prot, refid, soup_node.string)

                    # Now see if there is a reference to any base classes
                    for base in cdef.find_all("basecompoundref", recursive=False):
                        node.base_compounds.append(prot_ref_str(base))

                    # Now see if there is a reference to any derived classes
                    for derived in cdef.find_all("derivedcompoundref", recursive=False):
                        node.derived_compounds.append(prot_ref_str(derived))
                except:
                    utils.fancyError("Error processing Doxygen XML for [{0}]".format(node.name), "txt")

    def trackNodeIfUnseen(self, node):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.discoverAllNodes`.  If the node
        is not in self.all_nodes yet, add it to both self.all_nodes as well as the
        corresponding ``self.<breathe_kind>`` list.

        :Parameters:
            ``node`` (ExhaleNode)
                The node to begin tracking if not already present.
        '''
        if node not in self.all_nodes:
            node.set_owner(self)
            self.all_nodes.append(node)
            self.node_by_refid[node.refid] = node
            if node.kind == "concept":
                self.concepts.append(node)
            if node.kind == "class" or node.kind == "struct":
                self.class_like.append(node)
            elif node.kind == "namespace":
                self.namespaces.append(node)
            elif node.kind == "enum":
                self.enums.append(node)
            elif node.kind == "enumvalue":
                self.enum_values.append(node)
            elif node.kind == "define":
                self.defines.append(node)
            elif node.kind == "file":
                self.files.append(node)
            elif node.kind == "dir":
                self.dirs.append(node)
            elif node.kind == "function":
                self.functions.append(node)
            elif node.kind == "variable":
                self.variables.append(node)
            elif node.kind == "group":
                self.groups.append(node)
            elif node.kind == "typedef":
                self.typedefs.append(node)
            elif node.kind == "union":
                self.unions.append(node)
            elif node.kind == "page":
                self.pages.append(node)
                if node.refid != "indexpage":
                    self.index_xml_page_ordering.append(node)

    def reparentAll(self):
        '''
        Fixes some of the parental relationships lost in parsing the Breathe graph.
        File relationships are recovered in
        :func:`~exhale.graph.ExhaleRoot.fileRefDiscovery`.  This method simply calls in
        this order:

        1. :func:`~exhale.graph.ExhaleRoot.reparentUnions`
        2. :func:`~exhale.graph.ExhaleRoot.reparentClassLike`
        3. :func:`~exhale.graph.ExhaleRoot.reparentDirectories`
        4. :func:`~exhale.graph.ExhaleRoot.renameToNamespaceScopes`
        5. :func:`~exhale.graph.ExhaleRoot.reparentNamespaces`
        '''
        self.reparentUnions()
        self.reparentClassLike()
        self.reparentDirectories()
        self.renameToNamespaceScopes()

        # NOTE: must be last in current setup, reparenting of unions and class_like
        # relies on self.namespaces having all namespaces in self.namespaces, after this
        # nested namespaces are not in self.namespaces.
        self.reparentNamespaces()

        # make sure all children lists are unique (no duplicate children)
        for node in self.all_nodes:
            node.children = list(set(node.children))

    def reparentUnions(self):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.reparentAll`.  Namespaces and
        classes should have the unions defined in them to be in the child list of itself
        rather than floating around.  Union nodes that are reparented (e.g. a union
        defined in a class) will be removed from the list ``self.unions`` since the
        Breathe directive for its parent (e.g. the class) will include the documentation
        for the union.  The consequence of this is that a union defined in a class will
        **not** appear in the full api listing of Unions.
        '''
        # unions declared in a class will not link to the individual union page, so
        # we will instead elect to remove these from the list of unions
        removals = []
        for u in self.unions:
            parts = u.name.split("::")
            if len(parts) >= 2:
                # TODO: nested unions are not supported right now...
                parent_name = "::".join(p for p in parts[:-1])
                reparented  = False
                # see if the name matches any potential parents
                for node in itertools.chain(self.class_like, self.namespaces):
                    if node.name == parent_name:
                        node.children.append(u)
                        u.parent = node
                        reparented = True
                        break
                # if not reparented, try the namespaces
                if reparented:
                    removals.append(u)
                else:
                    # << verboseBuild
                    utils.verbose_log(
                        "The union {0} has '::' in its name, but no parent was found!".format(u.name),
                        utils.AnsiColors.BOLD_RED
                    )

        # remove the unions from self.unions that were declared in class_like objects
        for rm in removals:
            self.unions.remove(rm)

    def reparentClassLike(self):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.reparentAll`. Iterates over the
        ``self.class_like`` list and adds each object as a child to a namespace if the
        class, or struct is a member of that namespace.  Many classes / structs will be
        reparented to a namespace node, these will remain in ``self.class_like``.
        However, if a class or struct is reparented to a different class or struct (it
        is a nested class / struct), it *will* be removed from so that the class view
        hierarchy is generated correctly.
        '''
        removals = []
        for cl in self.class_like:
            parts = cl.name.split("::")
            if len(parts) > 1:
                parent_name = "::".join(parts[:-1])

                # Try and reparent to class_like first.  If it is a nested class then
                # we remove from the top level self.class_like.
                for parent_cl in self.class_like:
                    if parent_cl.name == parent_name:
                        parent_cl.children.append(cl)
                        cl.parent = parent_cl
                        removals.append(cl)
                        break

                # Next, reparent to namespaces.  Do not delete from self.class_like.
                for parent_nspace in self.namespaces:
                    if parent_nspace.name == parent_name:
                        parent_nspace.children.append(cl)
                        cl.parent = parent_nspace
                        break

        for rm in removals:
            if rm in self.class_like:
                self.class_like.remove(rm)

    def reparentDirectories(self):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.reparentAll`.  Adds
        subdirectories as children to the relevant directory ExhaleNode.  If a node in
        ``self.dirs`` is added as a child to a different directory node, it is removed
        from the ``self.dirs`` list.
        '''
        dir_parts = []
        dir_ranks = []
        for d in self.dirs:
            parts = d.name.split(os.sep)
            for p in parts:
                if p not in dir_parts:
                    dir_parts.append(p)
            dir_ranks.append((len(parts), d))

        traversal = sorted(dir_ranks)
        removals = []
        for rank, directory in reversed(traversal):
            # rank one means top level directory
            if rank < 2:
                break
            # otherwise, this is nested
            for p_rank, p_directory in reversed(traversal):
                if p_rank == rank - 1:
                    if p_directory.name == os.path.dirname(directory.name):
                        p_directory.children.append(directory)
                        directory.parent = p_directory
                        if directory not in removals:
                            removals.append(directory)
                        break

        for rm in removals:
            self.dirs.remove(rm)

    def renameToNamespaceScopes(self):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.reparentAll`. Some compounds in
        Breathe such as functions and variables do not have the namespace name they are
        declared in before the name of the actual compound.  This method prepends the
        appropriate (nested) namespace name before the name of any child that does not
        already have it.

        For example, the variable ``MAX_DEPTH`` declared in namespace ``external`` would
        have its ExhaleNode's ``name`` attribute changed from ``MAX_DEPTH`` to
        ``external::MAX_DEPTH``.
        '''
        for n in self.namespaces:
            namespace_name = "{0}::".format(n.name)
            for child in n.children:
                if namespace_name not in child.name:
                    child.name = "{0}{1}".format(namespace_name, child.name)

    def reparentNamespaces(self):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.reparentAll`.  Adds nested
        namespaces as children to the relevant namespace ExhaleNode.  If a node in
        ``self.namespaces`` is added as a child to a different namespace node, it is
        removed from the ``self.namespaces`` list.  Because these are removed from
        ``self.namespaces``, it is important that
        :func:`~exhale.graph.ExhaleRoot.renameToNamespaceScopes` is called before this
        method.
        '''
        namespace_parts = []
        namespace_ranks = []
        for n in self.namespaces:
            parts = n.name.split("::")
            for p in parts:
                if p not in namespace_parts:
                    namespace_parts.append(p)
            namespace_ranks.append((len(parts), n))

        traversal = sorted(namespace_ranks)
        removals = []
        for rank, namespace in reversed(traversal):
            # rank one means top level namespace
            if rank < 2:
                continue
            # otherwise, this is nested
            for p_rank, p_namespace in reversed(traversal):
                if p_rank == rank - 1:
                    if p_namespace.name == "::".join(namespace.name.split("::")[:-1]):
                        p_namespace.children.append(namespace)
                        namespace.parent = p_namespace
                        if namespace not in removals:
                            removals.append(namespace)
                        continue

        removals = []
        for nspace in self.namespaces:
            if nspace.parent and nspace.parent.kind == "namespace" and nspace not in removals:
                removals.append(nspace)

        for rm in removals:
            self.namespaces.remove(rm)

    def fileRefDiscovery(self):
        '''
        Finds the missing components for file nodes by parsing the Doxygen xml (which is
        just the ``doxygen_output_dir/node.refid``).  Additional items parsed include
        adding items whose ``refid`` tag are used in this file, the <programlisting> for
        the file, what it includes and what includes it, as well as the location of the
        file (with respsect to the *Doxygen* root).

        Care must be taken to only include a refid found with specific tags.  The
        parsing of the xml file was done by just looking at some example outputs.  It
        seems to be working correctly, but there may be some subtle use cases that break
        it.

        .. warning::
            Some enums, classes, variables, etc declared in the file will not have their
            associated refid in the declaration of the file, but will be present in the
            <programlisting>.  These are added to the files' list of children when they
            are found, but this parental relationship cannot be formed if you set
            ``XML_PROGRAMLISTING = NO`` with Doxygen.  An example of such an enum would
            be an enum declared inside of a namespace within this file.
        '''
        if not os.path.isdir(configs._doxygen_xml_output_directory):
            utils.fancyError("The doxygen xml output directory [{0}] is not valid!".format(
                configs._doxygen_xml_output_directory
            ))

        # parse the doxygen xml file and extract all refid's put in it
        # keys: file object, values: list of refid's
        doxygen_xml_file_ownerships = {}
        # innerclass, innernamespace, etc
        ref_regex    = re.compile(r'.*<inner.*refid="(\w+)".*')
        # what files this file includes
        inc_regex    = re.compile(r'.*<includes.*>(.+)</includes>')
        # what files include this file
        inc_by_regex = re.compile(r'.*<includedby refid="(\w+)".*>(.*)</includedby>')
        # the actual location of the file
        loc_regex    = re.compile(r'.*<location file="(.*)"/>')

        for f in self.files:
            doxygen_xml_file_ownerships[f] = []
            try:
                doxy_xml_path = os.path.join(configs._doxygen_xml_output_directory, "{0}.xml".format(f.refid))
                with codecs.open(doxy_xml_path, "r", "utf-8") as doxy_file:
                    processing_code_listing = False  # shows up at bottom of xml
                    for line in doxy_file:
                        # see if this line represents the location tag
                        match = loc_regex.match(line)
                        if match is not None:
                            f.location = os.path.normpath(match.groups()[0])
                            continue

                        if not processing_code_listing:
                            # gather included by references
                            match = inc_by_regex.match(line)
                            if match is not None:
                                ref, name = match.groups()
                                f.included_by.append((ref, name))
                                continue
                            # gather includes lines
                            match = inc_regex.match(line)
                            if match is not None:
                                inc = match.groups()[0]
                                f.includes.append(inc)
                                continue
                            # gather any classes, namespaces, etc declared in the file
                            match = ref_regex.match(line)
                            if match is not None:
                                match_refid = match.groups()[0]
                                if match_refid in self.node_by_refid:
                                    doxygen_xml_file_ownerships[f].append(match_refid)
                                continue
                            # lastly, see if we are starting the code listing
                            if "<programlisting>" in line:
                                processing_code_listing = True
                        elif processing_code_listing:
                            if "</programlisting>" in line:
                                processing_code_listing = False
                            else:
                                f.program_listing.append(line)
            except:
                utils.fancyError(
                    "Unable to process doxygen xml for file [{0}].\n".format(f.name)
                )

        #
        # IMPORTANT: do not set the parent field of anything being added as a child to the file
        #

        # hack to make things work right on RTD
        # TODO: do this at construction rather than as a post process!
        if configs.doxygenStripFromPath is not None:
            for node in itertools.chain(self.files, self.dirs):
                if node.kind == "file":
                    manip = node.location
                else:  # node.kind == "dir"
                    manip = node.name

                abs_strip_path = os.path.normpath(os.path.abspath(
                    configs.doxygenStripFromPath
                ))
                if manip.startswith(abs_strip_path):
                    manip = os.path.relpath(manip, abs_strip_path)

                manip = os.path.normpath(manip)
                if node.kind == "file":
                    node.location = manip
                else:  # node.kind == "dir"
                    node.name = manip

        # now that we have parsed all the listed refid's in the doxygen xml, reparent
        # the nodes that we care about
        allowable_child_kinds = ["struct", "class", "function", "typedef", "define", "enum", "union"]
        for f in self.files:
            for match_refid in doxygen_xml_file_ownerships[f]:
                child = self.node_by_refid[match_refid]
                if child.kind in allowable_child_kinds:
                    if child not in f.children:
                        f.children.append(child)
                elif child.kind == "namespace":
                    if child not in f.namespaces_used:
                        f.namespaces_used.append(child)

        # last but not least, some different kinds declared in the file that are scoped
        # in a namespace they will show up in the programlisting, but not at the toplevel.
        for f in self.files:
            potential_orphans = []
            for n in f.namespaces_used:
                for child in n.children:
                    if child.kind == "enum"     or child.kind == "variable" or \
                       child.kind == "function" or child.kind == "typedef"  or \
                       child.kind == "union":
                        potential_orphans.append(child)

            # now that we have a list of potential orphans, see if this doxygen xml had
            # the refid of a given child present.
            for orphan in potential_orphans:
                unresolved_name = orphan.name.split("::")[-1]
                if f.refid in orphan.refid and any(unresolved_name in line for line in f.program_listing):
                    if orphan not in f.children:
                        f.children.append(orphan)

        # Last but not least, make sure all children know where they were defined.
        for f in self.files:
            for child in f.children:
                if child.def_in_file is None:
                    child.def_in_file = f
                elif child.def_in_file != f:
                    # << verboseBuild
                    utils.verbose_log(
                        "Conflicting file definition for [{0}]: both [{1}] and [{2}] found.".format(
                            child.name, child.def_in_file.name, f.name
                        ),
                        utils.AnsiColors.BOLD_RED
                    )

    def filePostProcess(self):
        '''
        The real name of this method should be ``reparentFiles``, but to avoid confusion
        with what stage this must happen at it is called this instead.  After the
        :func:`~exhale.graph.ExhaleRoot.fileRefDiscovery` method has been called, each
        file will have its location parsed.  This method reparents files to directories
        accordingly, so the file view hierarchy can be complete.
        '''
        # directories are already reparented, traverse the children and get a flattened
        # list of all directories. previously, all directories should have had their
        # names adjusted to remove a potentially leading path separator
        nodes_remaining = [d for d in self.dirs]
        all_directories = []
        while len(nodes_remaining) > 0:
            d = nodes_remaining.pop()
            all_directories.append(d)
            for child in d.children:
                if child.kind == "dir":
                    nodes_remaining.append(child)

        all_directories.sort()

        for f in self.files:
            if not f.location:
                sys.stderr.write(utils.critical(
                    "Cannot reparent file [{0}] because it's location was not discovered.\n".format(
                        f.name
                    )
                ))
                continue
            elif os.sep not in f.location:
                # top-level file, cannot parent do a directory
                utils.verbose_log(
                    "### File [{0}] with location [{1}] was identified as being at the top level".format(
                        f.name, f.location
                    ),
                    utils.AnsiColors.BOLD_YELLOW
                )
                continue

            dirname = os.path.dirname(f.location)
            found = False
            for d in all_directories:
                if dirname == d.name:
                    d.children.append(f)
                    f.parent = d
                    found = True
                    break

            if not found:
                sys.stderr.write(utils.critical(
                    "Could not find directory parent of file [{0}] with location [{1}].\n".format(
                        f.name, f.location
                    )
                ))

    def parseFunctionSignatures(self):
        """Search file and namespace node XML contents for function signatures."""
        # Keys: string refid of either namespace or file nodes
        # Values: list of function objects that should be defined there
        parent_to_func = {}
        for func in self.functions:
            # Case 1: it is a function inside a namespace, the function information
            # is in the namespace's XML file.
            if func.parent:
                parent_refid = None
                if func.parent.kind == "namespace":
                    parent_refid = func.parent.refid
                else:
                    raise RuntimeError(textwrap.dedent('''
                        Function [{0}] with refid=[{1}] had a parent of kind '{2}':
                        Parent name=[{3}], refid=[{4}].

                        Functions may only have namespace parents.  Please report this
                        issue online, Exhale has a parsing error.
                    '''.format(func.name, func.refid, func.parent.name, func.parent.refid)))
            # Case 2: top-level function, it's information is in the file node's XML.
            elif func.def_in_file:
                parent_refid = func.def_in_file.refid
            else:
                utils.verbose_log(utils.critical(
                    "Cannot parse function [{0}] signature, refid=[{2}], no parent/def_in_file found!".format(
                        func.name, func.refid
                    )
                ))

            # If we found a suitable parent refid, gather in parent_to_func.
            if parent_refid:
                if parent_refid not in parent_to_func:
                    parent_to_func[parent_refid] = []
                parent_to_func[parent_refid].append(func)

        # Now we have a mapping of all defining elements to where the function
        # signatures _should_ live.
        # TODO: setwise comparison / report when children vs parent_to_func[refid] differ?
        for refid in parent_to_func:
            parent = self.node_by_refid[refid]
            parent_contents = utils.nodeCompoundXMLContents(parent)
            if not parent_contents:
                continue  ############flake8efphase: TODO: error, log?

            try:
                parent_soup = BeautifulSoup(parent_contents, "lxml-xml")
            except:
                continue

            cdef = parent_soup.doxygen.compounddef
            func_section = None
            for section in cdef.find_all("sectiondef", recursive=False):
                if "kind" in section.attrs and section.attrs["kind"] == "func":
                    func_section = section
                    break

            if not func_section:
                continue############flake8efphase: TODO: error, log?

            functions = parent_to_func[refid]
            for memberdef in func_section.find_all("memberdef", recursive=False):
                if "kind" not in memberdef.attrs or memberdef.attrs["kind"] != "function":
                    continue

                func_refid = memberdef.attrs["id"]
                func = None
                for candidate in functions:
                    if candidate.refid == func_refid:
                        func = candidate
                        break

                if not func:
                    continue ############flake8efphase: TODO: error, log?
                functions.remove(func)

                # At last, we can actually parse the function signature
                # 1. The function return type.
                func.return_type = utils.sanitize(
                    memberdef.find("type", recursive=False).text
                )
                # 2. The function parameter list.
                parameters = []
                for param in memberdef.find_all("param", recursive=False):
                    parameters.append(param.type.text)
                func.parameters = utils.sanitize_all(parameters)
                # 3. The template parameter list.
                templateparamlist = memberdef.templateparamlist
                if templateparamlist:
                    template = []
                    for param in templateparamlist.find_all("param", recursive=False):
                        template.append(param.type.text)
                    func.template = utils.sanitize_all(template)


    def sortInternals(self):
        '''
        Sort all internal lists (``class_like``, ``namespaces``, ``variables``, etc)
        mostly how doxygen would, alphabetical but also hierarchical (e.g. structs
        appear before classes in listings).  Some internal lists are just sorted, and
        some are deep sorted (:func:`~exhale.graph.ExhaleRoot.deepSortList`).
        '''
        # some of the lists only need to be sorted, some of them need to be sorted and
        # have each node sort its children
        # leaf-like lists: no child sort
        self.defines.sort()
        self.concepts.sort()
        self.enums.sort()
        self.enum_values.sort()
        self.functions.sort()
        self.groups.sort()
        self.typedefs.sort()
        self.variables.sort()
        self.pages.sort()

        # hierarchical lists: sort children
        self.deepSortList(self.class_like)
        self.deepSortList(self.namespaces)
        self.deepSortList(self.unions)
        self.deepSortList(self.files)
        self.deepSortList(self.dirs)
        self.deepSortList(self.pages)

    def deepSortList(self, lst):
        '''
        For hierarchical internal lists such as ``namespaces``, we want to sort both the
        list as well as have each child sort its children by calling
        :func:`~exhale.graph.ExhaleNode.typeSort`.

        :Parameters:
            ``lst`` (list)
                The list of ExhaleNode objects to be deep sorted.
        '''
        lst.sort()
        for l in lst:
            l.typeSort()

    ####################################################################################
    #
    ##
    ### Library generation.
    ##
    #
    ####################################################################################
    def generateFullAPI(self):
        '''
        Since we are not going to use some of the breathe directives (e.g. namespace or
        file), when representing the different views of the generated API we will need:

        1. Generate a single file restructured text document for all of the nodes that
           have either no children, or children that are leaf nodes.
        2. When building the view hierarchies (page, class, and file view and), provide
           a link to the appropriate files generated previously.

        If adding onto the framework to say add another view (from future import groups)
        you would link from a restructured text document to one of the individually
        generated files using the value of ``link_name`` for a given ExhaleNode object.

        This method calls in this order:

        1. :func:`~exhale.graph.ExhaleRoot.generateAPIRootHeader`
        2. :func:`~exhale.graph.ExhaleRoot.generateNodeDocuments`
        3. :func:`~exhale.graph.ExhaleRoot.generateAPIRootBody`
        '''
        self.generateAPIRootHeader()
        self.generateNodeDocuments()
        self.generateAPIRootBody()

    def generateAPIRootHeader(self):
        '''
        This method creates the root library api file that will include all of the
        different hierarchy views and full api listing.  If ``self.root_directory`` is
        not a current directory, it is created first.  Afterward, the root API file is
        created and its title is written, as well as the value of
        ``configs.afterTitleDescription``.
        '''
        try:
            # TODO: update to pathlib everywhere...
            root_directory_path = Path(self.root_directory)
            root_directory_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            utils.fancyError(
                "Cannot create the directory {0} {1}".format(self.root_directory, e)
            )
        try:
            with codecs.open(self.full_root_file_path, "w", "utf-8") as generated_index:
                # Add the metadata if they requested it
                if configs.pageLevelConfigMeta:
                    generated_index.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                if configs.rootFileTitle:
                    generated_index.write(textwrap.dedent('''\
                        {heading_mark}
                        {heading}
                        {heading_mark}

                    '''.format(
                        heading=configs.rootFileTitle,
                        heading_mark=utils.heading_mark(
                            configs.rootFileTitle,
                            configs.SECTION_HEADING_CHAR
                        )
                    )))

                if configs.afterTitleDescription:
                    generated_index.write("\n{0}\n\n".format(configs.afterTitleDescription))
        except:
            utils.fancyError(
                "Unable to create the root api file / header: {0}".format(self.full_root_file_path)
            )

    def generateNodeDocuments(self):
        '''
        Creates all of the reStructuredText documents related to types parsed by
        Doxygen.  This includes all leaf-like documents (``class``, ``struct``,
        ``enum``, ``typedef``, ``union``, ``variable``, and ``define``), as well as
        namespace, file, and directory pages.

        During the reparenting phase of the parsing process, nested items were added as
        a child to their actual parent.  For classes, structs, enums, and unions, if
        it was reparented to a ``namespace`` it will *remain* in its respective
        ``self.<breathe_kind>`` list.  However, if it was an internally declared child
        of a class or struct (nested classes, structs, enums, and unions), this node
        will be removed from its ``self.<breathe_kind>`` list to avoid duplication in
        the class hierarchy generation.

        When generating the full API, though, we will want to include all of these and
        therefore must call :func:`~exhale.graph.ExhaleRoot.generateSingleNodeRST` with
        all of the nested items.  For nested classes and structs, this is done by just
        calling ``node.findNestedClassLike`` for every node in ``self.class_like``.  The
        resulting list then has all of ``self.class_like``, as well as any nested
        classes and structs found.  With ``enum`` and ``union``, these would have been
        reparented to a **class** or **struct** if it was removed from the relevant
        ``self.<breathe_kind>`` list.  Meaning we must make sure that we genererate the
        single node RST documents for everything by finding the nested enums and unions
        from ``self.class_like``, as well as everything in ``self.enums`` and
        ``self.unions``.
        '''
        # initialize all of the nodes first
        for node in self.all_nodes:
            self.initializeNodeFilenameAndLink(node)

        self.adjustFunctionTitles()

        # now that all potential ``node.link_name`` members are initialized, generate
        # the leaf-like documents
        for node in self.all_nodes:
            if node.kind in utils.LEAF_LIKE_KINDS:
                self.generateSingleNodeRST(node)

        self.generatePageDocuments()

        # generate the remaining parent-like documents
        self.generateNamespaceNodeDocuments()
        self.generateFileNodeDocuments()
        self.generateDirectoryNodeDocuments()

    def initializeNodeFilenameAndLink(self, node):
        '''
        Sets the ``file_name`` and ``link_name`` for the specified node.  If the kind
        of this node is "file", then this method will also set the ``program_file``
        as well as the ``program_link_name`` fields.

        Since we are operating inside of a ``containmentFolder``, this method **will**
        include ``self.root_directory`` in this path so that you can just use::

            with codecs.open(node.file_name, "w", "utf-8") as gen_file:
                # ... write the file ...

        Having the ``containmentFolder`` is important for when we want to generate the
        file, but when we want to use it with ``include`` or ``toctree`` this will
        need to change.  Refer to
        :func:`~exhale.graph.ExhaleRoot.gerrymanderNodeFilenames`.

        This method also sets the value of ``node.title``, which will be used in both
        the reStructuredText document of the node as well as the links generated in the
        class view hierarchy (<a href="..."> for the ``createTreeView = True`` option).

        :type:  exhale.graph.ExhaleNode
        :param: node
            The node that we are setting the above information for.
        '''
        # Flag for title special treatment at end.
        template_special = False

        # Special cases: directories and files do not have an equivalent C++ domain
        # construct in Sphinx, as well as Exhale does not use the corresponding Breathe
        # directive for these compounds.  Similarly, Exhale does not use the Breathe
        # namespace directive.  As such, where possible, the filename should be as
        # human-friendly has possible so that users can conveniently link to the
        # internal Exhal ref's using e.g. :ref:`file_dir_subdir_filename.h`.
        SPECIAL_CASES = ["dir", "file", "namespace", "page"]
        if node.kind in SPECIAL_CASES:
            if node.kind == "file":
                unique_id = node.location
            else:
                unique_id = node.name

            unique_id = unique_id.replace(":", "_").replace(os.sep, "_").replace(" ", "_")
            if node.kind == "namespace":
                title = node.name.split("::")[-1]
            else:
                # NOTE: for files, node.name := basename(node.location) aka don't matter
                title = os.path.basename(node.name)
        else:
            unique_id = node.refid

            # special treatment for templates
            first_lt = node.name.find("<")
            last_gt  = node.name.rfind(">")
            # dealing with a template when this is true
            if first_lt > -1 and last_gt > -1:
                # NOTE: this has to happen for partial / full template specializations
                #       When specializations occur, the "<param1, param2>" etc show up
                #       in `node.name`.
                template_special = True
                #flake8failhere
                # TODO: when specializations occur, can you find a way to link to them
                # in the title?  Issue: nested templates prevent splitting on ','
                title = "{cls}{templates}".format(
                    cls=node.name[:first_lt].split("::")[-1],  # remove namespaces
                    templates=node.name[first_lt:last_gt + 1]  # template params
                )
            else:
                title = node.name.split("::")[-1]

            # additionally, I feel that nested classes should have their fully qualified
            # name without namespaces for clarity
            prepend_parent = False
            if node.kind in ["class", "struct", "enum", "union"]:
                if node.parent is not None and node.parent.kind in ["class", "struct"]:
                    prepend_parent = True
            if prepend_parent:
                title = "{parent}::{child}".format(
                    parent=node.parent.name.split("::")[-1],
                    child=title
                )

        # `unique_id` and `title` should be set approriately for all nodes by this point
        if node.kind in SPECIAL_CASES:
            node.link_name = "{kind}_{id}".format(kind=node.kind, id=unique_id)
            node.file_name = "{link_name}.rst".format(link_name=node.link_name)
            # Like the tree view documents, we want to .. include:: the indexpage on
            # the root library document without having sphinx generate html for the page
            # that is being included (otherwise there are duplicate label warnings).
            #
            # Unless a user has `.include` in their source_suffix, this skips this.
            if node.refid == "indexpage":
                node.file_name += ".include"
        else:
            # The node.link_name is the internal reference for exhale to link to in the
            # library API listing.  We cannot use unique_id in "non-special-cases"
            # because that will be the Doxygen refid, which Breathe will use as the
            # actual documentation ref.  This link_name is an anchor point to the top
            # of the page, but it cannot be a duplicate.
            #
            # Lastly, the Doxygen refid _may_ have the kind in it (e.g., a class or
            # struct), but also may _not_ (e.g., a function is a hash appended to the
            # file that defined it).  So a little bit of trickery is used to make sure
            # that the generated filename is at least _somewhat_ understandable for a
            # human to know what it is documenting (or at least its kind...).
            node.link_name = "exhale_{kind}_{id}".format(kind=node.kind, id=unique_id)
            if unique_id.startswith(node.kind):
                node.file_name = "{id}.rst".format(id=unique_id)
            else:
                node.file_name = "{kind}_{id}.rst".format(kind=node.kind, id=unique_id)

        # Make sure this file can always be generated.  We do not need to change the
        # node.link_name, just make sure the file being written to is OK.
        if len(node.file_name) >= configs.MAXIMUM_FILENAME_LENGTH:
            # hashlib.sha1 will produce a length 40 string.
            node.file_name = "{kind}_{sha1}.rst".format(
                kind=node.kind, sha1=hashlib.sha1(node.link_name.encode()).hexdigest()
            )

        # Create the program listing internal link and filename.
        if node.kind == "file":
            node.program_link_name = "program_listing_{link_name}".format(link_name=node.link_name)
            node.program_file = "{pl_link_name}.rst".format(pl_link_name=node.program_link_name)

            # Adding a 'program_listing_' prefix may have made this too long.  If so,
            # change both node.file_name and node.program_file for consistency.
            if len(node.program_file) >= configs.MAXIMUM_FILENAME_LENGTH:
                sha1 = hashlib.sha1(node.link_name.encode()).hexdigest()
                node.file_name = "{kind}_{sha1}.rst".format(kind=node.kind, sha1=sha1)
                node.program_file = "program_listing_{file_name}".format(file_name=node.file_name)

        # Now force everything in the containment folder
        for attr in ["file_name", "program_file"]:
            if hasattr(node, attr):
                full_path = os.path.join(self.root_directory, getattr(node, attr))
                if platform.system() == "Windows" and len(full_path) >= configs.MAXIMUM_WINDOWS_PATH_LENGTH:
                    # NOTE: self.root_directory is *ALREADY* an absolute path, this
                    #       prefix requires absolute paths!  See documentation for
                    #       configs.MAXIMUM_WINDOWS_PATH_LENGTH.
                    full_path = "{magic}{full_path}".format(
                        magic="{slash}{slash}?{slash}".format(slash="\\"),  # \\?\ I HATE YOU WINDOWS
                        full_path=full_path
                    )
                setattr(node, attr, full_path)

        #flake8failhereplease: add a test with decltype!
        # account for decltype(&T::var) etc, could be in name or template params
        # html_safe_name = html_safe_name.replace("&", "_AMP_").replace("*", "_STAR_")
        # html_safe_name = html_safe_name.replace("(", "_LPAREN_").replace(")", "_RPAREN_")
        # html_safe_name = html_safe_name.replace("<", "_LT_").replace(">", "_GT_").replace(",", "_COMMA_")

        # breathe does not prepend the namespace for variables and typedefs, so
        # I choose to leave the fully qualified name in the title for added clarity
        if node.kind in ["variable", "typedef"]:
            title = node.name

        # Last but not least, set the title for the page to be generated.
        if node.kind != "page":
            node.title = "{kind} {title}".format(
                kind=utils.qualifyKind(node.kind),
                title=title
            )
        if node.template_params or template_special:
            node.title = "Template {title}".format(
                title=node.title.replace('*', r'\*'))

    def adjustFunctionTitles(self):
        # keys: string (func.name)
        # values: list of nodes (length 2 or larger indicates overload)
        overloads = {}
        for func in self.functions:
            if func.name not in overloads:
                overloads[func.name] = [func]
            else:
                overloads[func.name].append(func)

        # Now that we know what is / is not overloaded, only include the parameters
        # when actually needed in the title.
        # TODO: should this be exclusive to functions?  What about classes etc?
        # TODO: include full signature instead of just parameters????
        # TODO: this is making me so sad
        for name in overloads:
            functions = overloads[name]
            needs_parameters = len(functions) > 1

            # Problems with Breathe and template overloads, best I can do right now is warn.
            # Keys: strings, ", " joined with parameter list of current function
            # Values: list of function objects, len > 1 indicates problem to print to console.
            parameter_warning_map = {}

            for func in functions:
                # TODO: make this more like classes and include the templates?
                #       problem: SFINAE -> death of readability
                #
                # SOLUTION? only include when overloads found?
                if func.template is not None:
                    if len(func.template) == 0:
                        prefix = "Specialized Template Function"
                    else:
                        prefix = "Template Function"
                else:
                    prefix = "Function"

                if needs_parameters:
                    # Must escape asterisks in heading else they get treated as refs:
                    # http://docutils.sourceforge.net/docs/user/rst/quickstart.html#text-styles
                    suffix = func.breathe_identifier().replace("*", r"\*")
                else:
                    suffix = func.name

                func.title = "{prefix} {suffix}".format(prefix=prefix, suffix=suffix)

                # Build the warning set in a way that can recover things in the outer loop.
                parameters_str = ", ".join(func.parameters)
                if parameters_str in parameter_warning_map:
                    parameter_warning_map[parameters_str].append(func)
                else:
                    parameter_warning_map[parameters_str] = [func]

            # Inform user when specified breathe directive will create problems
            for parameters_str in parameter_warning_map:
                warn_functions = parameter_warning_map[parameters_str]
                if len(warn_functions) > 1:
                    sys.stderr.write(utils.critical(
                        textwrap.dedent('''
                            Current limitations in .. doxygenfunction:: directive affect your code!

                            Right now there are {num} functions that will all be generating the
                            *SAME* directive on different pages:

                                .. doxygenfunction:: {breathe_identifier}

                            This will result in all {num} pages documenting the same function, however
                            which function is not known (possibly dependent upon order of Doxygen's
                            index.xml?).  We hope to resolve this issue soon, and appreciate your
                            understanding.

                            The full function signatures as parsed by Exhale that will point to the
                            same function:
                        '''.format(
                            num=len(warn_functions), breathe_identifier=warn_functions[0].breathe_identifier()
                        ))                                                                        + \
                        "".join(["\n- {0}".format(wf.full_signature()) for wf in warn_functions]) + \
                        textwrap.dedent('''

                            Unfortunately, there are no known workarounds at this time.  Your only options

                            1. Ignore it, hopefully this will be resolved sooner rather than later.
                            2. Only let Doxygen document *ONE* of these functions, e.g., by doing

                                   #if !defined(DOXYGEN_SHOULD_SKIP_THIS)
                                       // function declaration and/or implementation
                                   #endif // DOXYGEN_SHOULD_SKIP_THIS

                               Making sure that your Doxygen configuration has

                                   PREDEFINED += DOXYGEN_SHOULD_SKIP_THIS

                               (added by default when using "exhaleDoxygenStdin").

                            Sorry :(

                        ''')
                    ))

    def generateSingleNodeRST(self, node):
        '''
        Creates the reStructuredText document for the leaf like node object.

        It is **assumed** that the specified ``node.kind`` is in
        :data:`~exhale.utils.LEAF_LIKE_KINDS`.  File, directory, and namespace nodes are
        treated separately.

        :Parameters:
            ``node`` (ExhaleNode)
                The leaf like node being generated by this method.
        '''
        try:
            # if (node.kind == 'concept'):
                # print("**** found concept")


            with codecs.open(node.file_name, "w", "utf-8") as gen_file:
                ########################################################################
                # Page header / linking.                                               #
                ########################################################################
                # generate a link label for every generated file
                link_declaration = ".. _{0}:".format(node.link_name)

                # acquire the file this was originally defined in
                if node.def_in_file:
                    defined_in = "- Defined in :ref:`{where}`".format(where=node.def_in_file.link_name)
                else:
                    defined_in = ".. did not find file this was defined in"
                    sys.stderr.write(utils.critical(
                        "Did not locate file that defined {0} [{1}]; no link generated.\n".format(node.kind,
                                                                                                  node.name)
                    ))

                # Add the metadata if they requested it
                if configs.pageLevelConfigMeta:
                    gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                gen_file.write(textwrap.dedent('''\
                    {link}

                    {heading}
                    {heading_mark}

                    {defined_in}

                '''.format(
                    link=link_declaration,
                    heading=node.title,
                    heading_mark=utils.heading_mark(
                        node.title,
                        configs.SECTION_HEADING_CHAR
                    ),
                    defined_in=defined_in
                )))

                contents = utils.contentsDirectiveOrNone(node.kind)
                if contents:
                    gen_file.write(contents)

                ########################################################################
                # Nested relationships.                                                #
                ########################################################################
                # link to outer types if this node is a nested type
                if node.parent and (node.parent.kind == "struct" or node.parent.kind == "class"):
                    nested_type_of = "This {kind} is a nested type of :ref:`{parent}`.".format(
                        kind=node.kind,
                        parent=node.parent.link_name
                    )
                else:
                    nested_type_of = None

                # if this has nested types, link to them
                nested_defs = None
                if node.kind == "class" or node.kind == "struct":
                    nested_children = []
                    for c in node.children:
                        c.findNestedEnums(nested_children)
                        c.findNestedUnions(nested_children)
                        c.findNestedClassLike(nested_children)

                    if nested_children:
                        # build up a list of links, custom sort function will force
                        # double nested and beyond to appear after their parent by
                        # sorting on their name
                        nested_children.sort(key=lambda x: x.name)
                        nested_child_stream = StringIO()
                        for nc in nested_children:
                            nested_child_stream.write("- :ref:`{0}`\n".format(nc.link_name))

                        # extract the list of links and add them as a subsection in the header
                        nested_child_string = nested_child_stream.getvalue()
                        nested_child_stream.close()
                        heading = "Nested Types"
                        nested_defs = textwrap.dedent('''
                            {heading}
                            {heading_mark}

                        '''.format(
                            heading=heading,
                            heading_mark=utils.heading_mark(
                                heading,
                                configs.SUB_SUB_SECTION_HEADING_CHAR
                            )
                        ))
                        nested_defs = "{0}{1}\n".format(nested_defs, nested_child_string)

                if nested_type_of or nested_defs:
                    heading = "Nested Relationships"
                    gen_file.write(textwrap.dedent('''
                        {heading}
                        {heading_mark}

                    '''.format(
                        heading=heading,
                        heading_mark=utils.heading_mark(
                            heading,
                            configs.SUB_SECTION_HEADING_CHAR
                        )
                    )))
                    if nested_type_of:
                        gen_file.write("{0}\n\n".format(nested_type_of))
                    if nested_defs:
                        gen_file.write(nested_defs)

                ########################################################################
                # Inheritance relationships.                                           #
                ########################################################################
                ##### remove this duplicated nonsense someday
                if node.base_compounds or node.derived_compounds:
                    heading = "Inheritance Relationships"
                    gen_file.write(textwrap.dedent('''
                        {heading}
                        {heading_mark}
                    '''.format(
                        heading=heading,
                        heading_mark=utils.heading_mark(
                            heading,
                            configs.SUB_SECTION_HEADING_CHAR
                        )
                    )))
                    if node.base_compounds:
                        if len(node.base_compounds) == 1:
                            title = "Base Type"
                        else:
                            title = "Base Types"

                        gen_file.write(textwrap.dedent('''
                            {heading}
                            {heading_mark}

                        '''.format(
                            heading=title,
                            heading_mark=utils.heading_mark(
                                title,
                                configs.SUB_SUB_SECTION_HEADING_CHAR
                            )
                        )))
                        gen_file.write("{0}\n".format(node.baseOrDerivedListString(
                            node.base_compounds, self.node_by_refid
                        )))
                    if node.derived_compounds:
                        if len(node.derived_compounds) == 1:
                            title = "Derived Type"
                        else:
                            title = "Derived Types"
                        gen_file.write(textwrap.dedent('''
                            {heading}
                            {heading_mark}

                        '''.format(
                            heading=title,
                            heading_mark=utils.heading_mark(
                                title,
                                configs.SUB_SUB_SECTION_HEADING_CHAR
                            )
                        )))
                        gen_file.write("{0}\n".format(node.baseOrDerivedListString(
                            node.derived_compounds, self.node_by_refid
                        )))

                ########################################################################
                # Template parameter listing.                                          #
                ########################################################################
                if configs.includeTemplateParamOrderList:
                    template = node.templateParametersStringAsRestList(self.node_by_refid)
                    if template:
                        heading = "Template Parameter Order"
                        gen_file.write(textwrap.dedent('''
                            {heading}
                            {heading_mark}

                        '''.format(
                            heading=heading,
                            heading_mark=utils.heading_mark(
                                heading,
                                configs.SUB_SECTION_HEADING_CHAR
                            )
                        )))

                        gen_file.write("{template_params}\n\n".format(template_params=template))

                        # << verboseBuild
                        utils.verbose_log(
                            "+++ {kind} {name} has usable template parameters:\n{params}".format(
                                kind=node.kind,
                                name=node.name,
                                params=utils.prefix("    ", template)
                            ),
                            utils.AnsiColors.BOLD_CYAN
                        )

                ########################################################################
                # The Breathe directive!!!                                             #
                ########################################################################
                heading = "{kind} Documentation".format(kind=utils.qualifyKind(node.kind))
                gen_file.write(textwrap.dedent('''
                    {heading}
                    {heading_mark}

                '''.format(
                    heading=heading,
                    heading_mark=utils.heading_mark(
                        heading,
                        configs.SUB_SECTION_HEADING_CHAR
                    )
                )))
                # inject the appropriate doxygen directive and name of this node
                directive = ".. {directive}:: {breathe_identifier}".format(
                    directive=utils.kindAsBreatheDirective(node.kind),
                    breathe_identifier=node.breathe_identifier()
                )
                gen_file.write("\n{directive}\n".format(directive=directive))
                # include any specific directives for this doxygen directive
                specifications = utils.prefix(
                    "   ",
                    "\n".join(spec for spec in utils.specificationsForKind(node.kind))
                )
                gen_file.write(specifications)
        except:
            utils.fancyError(
                "Critical error while generating the file for [{0}].".format(node.file_name)
            )

    def generatePageDocuments(self):
        '''
        Generates the reStructuredText document for every page.
        '''
        all_pages = [p for p in self.pages]
        while len(all_pages) > 0:
            page = all_pages.pop()
            self.generateSinglePageDocument(page)
            for subpage in page.children:
                all_pages.append(subpage)

    def generateSinglePageDocument(self, node):
        '''
        Creates the reStructuredText document for a page.

        :Parameters:
            ``node`` (ExhaleNode)
                The "page" node being generated by this method.
        '''
        try:
            with codecs.open(node.file_name, "w", "utf-8") as gen_file:
                ########################################################################
                # Page header / linking.                                               #
                ########################################################################
                # generate a link label for every generated file
                link_declaration = ".. _{0}:".format(node.link_name)

                # Add the metadata if they requested it
                if configs.pageLevelConfigMeta:
                    gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                gen_file.write(textwrap.dedent('''\
                    {link}

                    {heading}
                    {heading_mark}

                '''.format(
                    link=link_declaration,
                    heading=node.title,
                    heading_mark=utils.heading_mark(
                        node.title, configs.SECTION_HEADING_CHAR
                    )
                )))

                contents = utils.contentsDirectiveOrNone(node.kind)
                if contents:
                    gen_file.write(contents)

                # inject the appropriate doxygen directive and name of this node
                directive = ".. {directive}:: {breathe_identifier}".format(
                    directive=utils.kindAsBreatheDirective(node.kind),
                    breathe_identifier=node.breathe_identifier()
                )
                gen_file.write("{directive}\n".format(directive=directive))
                # include any specific directives for this doxygen directive
                specifications = utils.prefix(
                    "   ",
                    "\n".join(spec for spec in utils.specificationsForKind(node.kind))
                )
                gen_file.write(specifications)
        except:
            utils.fancyError(
                "Critical error while generating the file for [{0}].".format(node.file_name)
            )

    def generateNamespaceNodeDocuments(self):
        '''
        Generates the reStructuredText document for every namespace, including nested
        namespaces that were removed from ``self.namespaces`` (but added as children
        to one of the namespaces in ``self.namespaces``).

        The documents generated do not use the Breathe namespace directive, but instead
        link to the relevant documents associated with this namespace.
        '''
        # go through all of the top level namespaces
        for n in self.namespaces:
            # find any nested namespaces
            nested_namespaces = []
            for child in n.children:
                child.findNestedNamespaces(nested_namespaces)
            # generate the children first
            for nested in reversed(sorted(nested_namespaces)):
                self.generateSingleNamespace(nested)
            # generate this top level namespace
            self.generateSingleNamespace(n)

    def generateSingleNamespace(self, nspace):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.generateNamespaceNodeDocuments`.
        Writes the reStructuredText file for the given namespace.

        :Parameters:
            ``nspace`` (ExhaleNode)
                The namespace node to create the reStructuredText document for.
        '''
        try:
            with codecs.open(nspace.file_name, "w", "utf-8") as gen_file:
                # Add the metadata if they requested it
                if configs.pageLevelConfigMeta:
                    gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                nspace.title = "{0} {1}".format(utils.qualifyKind(nspace.kind), nspace.name)

                # generate a link label for every generated file
                gen_file.write(textwrap.dedent('''
                    .. _{link}:

                    {heading}
                    {heading_mark}

                '''.format(
                    link=nspace.link_name,
                    heading=nspace.title,
                    heading_mark=utils.heading_mark(nspace.title, configs.SECTION_HEADING_CHAR)
                )))

                brief, detailed = parse.getBriefAndDetailedRST(self, nspace)
                if brief:
                    gen_file.write("{0}\n\n".format(brief))

                # include the contents directive if requested
                contents = utils.contentsDirectiveOrNone(nspace.kind)
                if contents:
                    gen_file.write("{0}\n\n".format(contents))

                if detailed:
                    gen_file.write("{0}\n\n".format(detailed))

                # generate the headings and links for the children
                children_string = self.generateNamespaceChildrenString(nspace)
                gen_file.write(children_string)
        except:
            utils.fancyError(
                "Critical error while generating the file for [{0}]".format(nspace.file_name)
            )

    def generateNamespaceChildrenString(self, nspace):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.generateSingleNamespace`, and
        :func:`~exhale.graph.ExhaleRoot.generateFileNodeDocuments`.  Builds the
        body text for the namespace node document that links to all of the child
        namespaces, structs, classes, functions, typedefs, unions, and variables
        associated with this namespace.

        :Parameters:
            ``nspace`` (ExhaleNode)
                The namespace node we are generating the body text for.

        :Return (str):
            The string to be written to the namespace node's reStructuredText document.
        '''
        # sort the children
        nsp_namespaces        = []
        nsp_nested_class_like = []
        nsp_concepts          = []
        nsp_enums             = []
        nsp_functions         = []
        nsp_typedefs          = []
        nsp_unions            = []
        nsp_variables         = []
        for child in nspace.children:
            # Skip children whose names were requested to be explicitly ignored.
            should_exclude = False
            for exclude in configs._compiled_listing_exclude:
                if exclude.match(child.name):
                    should_exclude = True
            if should_exclude:
                continue

            if child.kind == "namespace":
                nsp_namespaces.append(child)
            elif child.kind == "struct" or child.kind == "class":
                child.findNestedClassLike(nsp_nested_class_like)
                child.findNestedEnums(nsp_enums)
                child.findNestedUnions(nsp_unions)
            elif child.kind == "concept":
                nsp_concepts.append(child)
            elif child.kind == "enum":
                nsp_enums.append(child)
            elif child.kind == "function":
                nsp_functions.append(child)
            elif child.kind == "typedef":
                nsp_typedefs.append(child)
            elif child.kind == "union":
                nsp_unions.append(child)
            elif child.kind == "variable":
                nsp_variables.append(child)

        # generate their headings if they exist (no Defines...that's not a C++ thing...)
        children_stream = StringIO()
        self.generateSortedChildListString(children_stream, "Namespaces", nsp_namespaces)
        self.generateSortedChildListString(children_stream, "Concepts", nsp_concepts)
        self.generateSortedChildListString(children_stream, "Classes", nsp_nested_class_like)
        self.generateSortedChildListString(children_stream, "Enums", nsp_enums)
        self.generateSortedChildListString(children_stream, "Functions", nsp_functions)
        self.generateSortedChildListString(children_stream, "Typedefs", nsp_typedefs)
        self.generateSortedChildListString(children_stream, "Unions", nsp_unions)
        self.generateSortedChildListString(children_stream, "Variables", nsp_variables)
        # read out the buffer contents, close it and return the desired string
        children_string = children_stream.getvalue()
        children_stream.close()
        return children_string

    def generateSortedChildListString(self, stream, sectionTitle, lst):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.generateNamespaceChildrenString`.
        Used to build up a continuous string with all of the children separated out into
        titled sections.

        This generates a new titled section with ``sectionTitle`` and puts a link to
        every node found in ``lst`` in this section.  The newly created section is
        appended to the existing ``stream`` buffer.

        :Parameters:
            ``stream`` (StringIO)
                The already-open StringIO to write the result to.

            ``sectionTitle`` (str)
                The title of the section for this list of children.

            ``lst`` (list)
                A list of ExhaleNode objects that are to be linked to from this section.
                This method sorts ``lst`` in place.
        '''
        if lst:
            lst.sort()
            stream.write(textwrap.dedent('''

                {heading}
                {heading_mark}

            '''.format(
                heading=sectionTitle,
                heading_mark=utils.heading_mark(
                    sectionTitle,
                    configs.SUB_SECTION_HEADING_CHAR
                )
            )))
            for l in lst:
                stream.write(textwrap.dedent('''
                    - :ref:`{link}`
                '''.format(link=l.link_name)))

    def generateFileNodeDocuments(self):
        '''
        Generates the reStructuredText documents for files as well as the file's
        program listing reStructuredText document if applicable.  Refer to
        :ref:`usage_customizing_file_pages` for changing the output of this method.
        The remainder of the file lists all nodes that have been discovered to be
        defined (e.g. classes) or referred to (e.g. included files or files that include
        this file).

        .. todo::

           writing the actual file should be set in one method so that things for files,
           namespaces, and leaflike nodes don't keep getting out of sync
        '''
        for f in self.files:
            # if the programlisting was included, length will be at least 1 line
            if len(f.program_listing) > 0:
                include_program_listing = True
                lexer = utils.doxygenLanguageToPygmentsLexer(f.location, f.language)
                full_program_listing = '.. code-block:: {0}\n\n'.format(lexer)

                # need to reformat each line to remove xml tags / put <>& back in
                for pgf_line in f.program_listing:
                    fixed_whitespace = re.sub(r'<sp/>', ' ', pgf_line)
                    # for our purposes, this is good enough:
                    #     http://stackoverflow.com/a/4869782/3814202
                    no_xml_tags  = re.sub(r'<[^<]+?>', '', fixed_whitespace)
                    revive_lt    = re.sub(r'&lt;', '<', no_xml_tags)
                    revive_gt    = re.sub(r'&gt;', '>', revive_lt)
                    revive_quote = re.sub(r'&quot;', '"', revive_gt)
                    revive_apos  = re.sub(r'&apos;', "'", revive_quote)
                    revive_amp   = re.sub(r'&amp;', '&', revive_apos)
                    full_program_listing = "{}   {}".format(full_program_listing, revive_amp)

                # create the programlisting file
                try:
                    with codecs.open(f.program_file, "w", "utf-8") as gen_file:
                        # Add the metadata if they requested it
                        if configs.pageLevelConfigMeta:
                            gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                        # generate a link label for every generated file
                        link_declaration = ".. _{}:".format(f.program_link_name)
                        # every generated file must have a header for sphinx to be happy
                        prog_title = "Program Listing for {} {}".format(utils.qualifyKind(f.kind), f.name)
                        gen_file.write(textwrap.dedent('''
                            {link}

                            {heading}
                            {heading_mark}

                            |exhale_lsh| :ref:`Return to documentation for file <{file}>` (``{location}``)

                            .. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

                        '''.format(
                            link=link_declaration,
                            heading=prog_title,
                            heading_mark=utils.heading_mark(
                                prog_title,
                                configs.SECTION_HEADING_CHAR
                            ),
                            file=f.link_name,
                            location=f.location
                        )))
                        gen_file.write(full_program_listing)
                except:
                    utils.fancyError(
                        "Critical error while generating the file for [{0}]".format(f.file_name)
                    )
            else:
                include_program_listing = False

        for f in self.files:
            if len(f.location) > 0:
                heading = "Definition (``{where}``)".format(where=f.location)
                file_definition = textwrap.dedent('''
                    {heading}
                    {heading_mark}

                '''.format(
                    heading=heading,
                    heading_mark=utils.heading_mark(
                        heading,
                        configs.SUB_SECTION_HEADING_CHAR
                    )
                ))
            else:
                file_definition = ""

            if include_program_listing and file_definition != "":
                prog_file_definition = textwrap.dedent('''
                    .. toctree::
                       :maxdepth: 1

                       {prog_link}
                '''.format(prog_link=os.path.basename(f.program_file)))
                file_definition = "{}{}".format(file_definition, prog_file_definition)

            if len(f.includes) > 0:
                file_includes_stream = StringIO()
                heading = "Includes"
                file_includes_stream.write(textwrap.dedent('''
                    {heading}
                    {heading_mark}

                '''.format(
                    heading=heading,
                    heading_mark=utils.heading_mark(
                        heading,
                        configs.SUB_SECTION_HEADING_CHAR
                    )
                )))
                for incl in sorted(f.includes):
                    local_file = None
                    for incl_file in self.files:
                        if incl in incl_file.location:
                            local_file = incl_file
                            break
                    if local_file is not None:
                        file_includes_stream.write(textwrap.dedent('''
                            - ``{include}`` (:ref:`{link}`)
                        '''.format(include=incl, link=local_file.link_name)))
                    else:
                        file_includes_stream.write(textwrap.dedent('''
                            - ``{include}``
                        '''.format(include=incl)))

                file_includes = file_includes_stream.getvalue()
                file_includes_stream.close()
            else:
                file_includes = ""

            if len(f.included_by) > 0:
                file_included_by_stream = StringIO()
                heading = "Included By"
                file_included_by_stream.write(textwrap.dedent('''
                    {heading}
                    {heading_mark}

                '''.format(
                    heading=heading,
                    heading_mark=utils.heading_mark(
                        heading,
                        configs.SUB_SECTION_HEADING_CHAR
                    )
                )))
                for incl_ref, incl_name in f.included_by:
                    for incl_file in self.files:
                        if incl_ref == incl_file.refid:
                            file_included_by_stream.write(textwrap.dedent('''
                                - :ref:`{link}`
                            '''.format(link=incl_file.link_name)))
                            break
                file_included_by = file_included_by_stream.getvalue()
                file_included_by_stream.close()
            else:
                file_included_by = ""

            # generate their headings if they exist --- DO NOT USE findNested*, these are included recursively
            file_structs    = []
            file_concepts   = []
            file_classes    = []
            file_enums      = []
            file_functions  = []
            file_typedefs   = []
            file_unions     = []
            file_variables  = []
            file_defines    = []
            for child in f.children:
                if child.kind == "struct":
                    file_structs.append(child)
                elif child.kind == "concept":
                    file_concepts.append(child)
                elif child.kind == "class":
                    file_classes.append(child)
                elif child.kind == "enum":
                    file_enums.append(child)
                elif child.kind == "function":
                    file_functions.append(child)
                elif child.kind == "typedef":
                    file_typedefs.append(child)
                elif child.kind == "union":
                    file_unions.append(child)
                elif child.kind == "variable":
                    file_variables.append(child)
                elif child.kind == "define":
                    file_defines.append(child)

            # generate the listing of children referenced to from this file
            children_stream = StringIO()
            self.generateSortedChildListString(children_stream, "Namespaces", f.namespaces_used)
            self.generateSortedChildListString(children_stream, "Concepts", file_concepts)
            self.generateSortedChildListString(children_stream, "Classes", file_structs + file_classes)
            self.generateSortedChildListString(children_stream, "Enums", file_enums)
            self.generateSortedChildListString(children_stream, "Functions", file_functions)
            self.generateSortedChildListString(children_stream, "Defines", file_defines)
            self.generateSortedChildListString(children_stream, "Typedefs", file_typedefs)
            self.generateSortedChildListString(children_stream, "Unions", file_unions)
            self.generateSortedChildListString(children_stream, "Variables", file_variables)

            children_string = children_stream.getvalue()
            children_stream.close()

            try:
                with codecs.open(f.file_name, "w", "utf-8") as gen_file:
                    # Add the metadata if they requested it
                    if configs.pageLevelConfigMeta:
                        gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                    # generate a link label for every generated file
                    link_declaration = ".. _{0}:".format(f.link_name)
                    # every generated file must have a header for sphinx to be happy
                    f.title = "{0} {1}".format(utils.qualifyKind(f.kind), f.name)
                    gen_file.write(textwrap.dedent('''
                        {link}

                        {heading}
                        {heading_mark}
                    '''.format(
                        link=link_declaration,
                        heading=f.title,
                        heading_mark=utils.heading_mark(
                            f.title,
                            configs.SECTION_HEADING_CHAR
                        )
                    )))

                    if f.parent and f.parent.kind == "dir":
                        gen_file.write(textwrap.dedent('''
                            |exhale_lsh| :ref:`Parent directory <{parent_link}>` (``{parent_name}``)

                            .. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS
                        '''.format(
                            parent_link=f.parent.link_name, parent_name=f.parent.name
                        )))

                    brief, detailed = parse.getBriefAndDetailedRST(self, f)
                    if brief:
                        gen_file.write("\n{brief}\n".format(brief=brief))

                    # include the contents directive if requested
                    contents = utils.contentsDirectiveOrNone(f.kind)
                    if contents:
                        gen_file.write(contents)

                    gen_file.write(textwrap.dedent('''
                        {definition}

                        {detailed}

                        {includes}

                        {includeby}

                        {children}
                    '''.format(
                        definition=file_definition,
                        detailed=detailed,
                        includes=file_includes,
                        includeby=file_included_by,
                        children=children_string
                    )).lstrip())
            except:
                utils.fancyError(
                    "Critical error while generating the file for [{0}]".format(f.file_name)
                )

            if configs.generateBreatheFileDirectives:
                try:
                    with codecs.open(f.file_name, "a", "utf-8") as gen_file:
                        heading        = "Full File Listing"
                        heading_mark   = utils.heading_mark(
                            heading, configs.SUB_SECTION_HEADING_CHAR
                        )
                        directive      = utils.kindAsBreatheDirective(f.kind)
                        node           = f.location
                        specifications = "\n   ".join(
                            spec for spec in utils.specificationsForKind(f.kind)
                        )

                        gen_file.write(textwrap.dedent('''
                            {heading}
                            {heading_mark}

                            .. {directive}:: {node}
                               {specifications}
                        '''.format(
                            heading=heading,
                            heading_mark=heading_mark,
                            directive=directive,
                            node=node,
                            specifications=specifications
                        )))
                except:
                    utils.fancyError(
                        "Critical error while generating the breathe directive for [{0}]".format(f.file_name)
                    )

    def generateDirectoryNodeDocuments(self):
        '''
        Generates all of the directory reStructuredText documents.
        '''
        all_dirs = []
        for d in self.dirs:
            d.findNestedDirectories(all_dirs)

        for d in all_dirs:
            self.generateDirectoryNodeRST(d)

    def generateDirectoryNodeRST(self, node):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.generateDirectoryNodeDocuments`.
        Generates the reStructuredText documents for the given directory node.
        Directory nodes will only link to files and subdirectories within it.

        :Parameters:
            ``node`` (ExhaleNode)
                The directory node to generate the reStructuredText document for.
        '''
        # find the relevant children: directories and files only
        child_dirs  = []
        child_files = []
        for c in node.children:
            if c.kind == "dir":
                child_dirs.append(c)
            elif c.kind == "file":
                child_files.append(c)

        # generate the subdirectory section
        if len(child_dirs) > 0:
            heading = "Subdirectories"
            child_dirs_string = textwrap.dedent('''
                {heading}
                {heading_mark}

            '''.format(
                heading=heading,
                heading_mark=utils.heading_mark(
                    heading,
                    configs.SUB_SECTION_HEADING_CHAR
                )
            ))
            for child_dir in sorted(child_dirs):
                child_dirs_string = "{}- :ref:`{}`\n".format(child_dirs_string, child_dir.link_name)
        else:
            child_dirs_string = ""

        # generate the files section
        if len(child_files) > 0:
            heading = "Files"
            child_files_string = textwrap.dedent('''
                {heading}
                {heading_mark}

            '''.format(
                heading=heading,
                heading_mark=utils.heading_mark(
                    heading,
                    configs.SUB_SECTION_HEADING_CHAR
                )
            ))
            for child_file in sorted(child_files):
                child_files_string = "{}- :ref:`{}`\n".format(child_files_string, child_file.link_name)
        else:
            child_files_string = ""

        if node.parent and node.parent.kind == "dir":
            parent_directory = textwrap.dedent('''
                |exhale_lsh| :ref:`Parent directory <{parent_link}>` (``{parent_name}``)

                .. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS
            '''.format(
                parent_link=node.parent.link_name, parent_name=node.parent.name
            ))
        else:
            parent_directory = ""

        # generate the file for this directory
        try:
            #flake8fail get rid of {} in this method
            with codecs.open(node.file_name, "w", "utf-8") as gen_file:
                # Add the metadata if they requested it
                if configs.pageLevelConfigMeta:
                    gen_file.write("{0}\n\n".format(configs.pageLevelConfigMeta))

                # generate a link label for every generated file
                link_declaration = ".. _{0}:\n\n".format(node.link_name)
                header = textwrap.dedent('''
                    {heading}
                    {heading_mark}

                '''.format(
                    heading=node.title,
                    heading_mark=utils.heading_mark(
                        node.title,
                        configs.SECTION_HEADING_CHAR
                    )
                ))
                path = "\n*Directory path:* ``{path}``\n".format(path=node.name)
                # write it all out
                gen_file.write("{0}{1}{2}{3}{4}\n{5}\n\n".format(
                    link_declaration, header, parent_directory, path, child_dirs_string, child_files_string)
                )
        except:
            utils.fancyError(
                "Critical error while generating the file for [{0}]".format(node.file_name)
            )

    def generateAPIRootBody(self):
        '''
        Generates the root library api file's body text.  The method calls
        :func:`~exhale.graph.ExhaleRoot.gerrymanderNodeFilenames` first to enable proper
        internal linkage between reStructuredText documents.  Afterward, it calls
        :func:`~exhale.graph.ExhaleRoot.generateViewHierarchies` followed by
        :func:`~exhale.graph.ExhaleRoot.generateUnabridgedAPI` to generate both
        hierarchies as well as the full API listing.  As a result, three files will now
        be ready:

        1. ``self.page_hierarchy_file``
        2. ``self.class_hierarchy_file``
        3. ``self.file_hierarchy_file``
        4. ``self.unabridged_api_file``

        These three files are then *included* into the root library file.  The
        consequence of using an ``include`` directive is that Sphinx will complain about
        these three files never being included in any ``toctree`` directive.  These
        warnings are expected, and preferred to using a ``toctree`` because otherwise
        the user would have to click on the class view link from the ``toctree`` in
        order to see it.  This behavior has been acceptable for me so far, but if it
        is causing you problems please raise an issue on GitHub and I may be able to
        conditionally use a ``toctree`` if you really need it.
        '''
        try:
            self.gerrymanderNodeFilenames()
            self.generateViewHierarchies()
            self.generateUnabridgedAPI()
            with codecs.open(self.full_root_file_path, "a", "utf-8") as generated_index:
                # Include index page, if present
                for page in self.pages:
                    if page.refid == "indexpage":
                        generated_index.write(".. include:: {0}\n\n".format(
                            os.path.basename(page.file_name)
                        ))
                        break
                # Include the page, class, and file hierarchies
                if os.path.exists(self.page_hierarchy_file):
                    generated_index.write(".. include:: {0}\n\n".format(
                        os.path.basename(self.page_hierarchy_file)
                    ))
                if os.path.exists(self.class_hierarchy_file):
                    generated_index.write(".. include:: {0}\n\n".format(
                        os.path.basename(self.class_hierarchy_file)
                    ))
                if os.path.exists(self.file_hierarchy_file):
                    generated_index.write(".. include:: {0}\n\n".format(
                        os.path.basename(self.file_hierarchy_file)
                    ))

                # Add the afterHierarchyDescription if provided
                if configs.afterHierarchyDescription:
                    generated_index.write(
                        "\n{0}\n\n".format(configs.afterHierarchyDescription)
                    )

                # Include the unabridged API
                generated_index.write(".. include:: {0}\n\n".format(
                    os.path.basename(self.unabridged_api_file)
                ))

                # Add the afterBodySummary if provided
                if configs.afterBodySummary:
                    generated_index.write(
                        "\n{0}\n\n".format(configs.afterBodySummary)
                    )

                # The following should only be applied to the page library root page
                # Applying it to other pages will result in an error
                if self.use_tree_view and configs.treeViewIsBootstrap:
                    generated_index.write(textwrap.dedent('''

                        .. raw:: html

                           <script type="text/javascript">
                               /* NOTE: if you are reading this, Exhale generated this directly. */
                               $(document).ready(function() {{
                                   /* Inspired by very informative answer to get color of links:
                                      https://stackoverflow.com/a/2707837/3814202 */
                                   var $fake_link = $('<a href="#"></a>').hide().appendTo("body");
                                   var linkColor = $fake_link.css("color");
                                   $fake_link.remove();

                                   var $fake_p = $('<p class="{icon_mimic}"></p>').hide().appendTo("body");
                                   var iconColor = $fake_p.css("color");
                                   $fake_p.remove();

                                   /* After much deliberation, using JavaScript directly to enforce that the
                                    * link and glyphicon receive different colors is fruitless, because the
                                    * bootstrap treeview library will overwrite the style every time.  Instead,
                                    * leaning on the library code itself to append some styling to the head,
                                    * I choose to mix a couple of things:
                                    *
                                    * 1. Set the `color` property of bootstrap treeview globally, this would
                                    *    normally affect the color of both the link text and the icon.
                                    * 2. Apply custom forced styling of the glyphicon itself in order to make
                                    *    it a little more clear to the user (via different colors) that the
                                    *    act of clicking the icon and the act of clicking the link text perform
                                    *    different actions.  The icon expands, the text navigates to the page.
                                    */
                                    // Part 1: use linkColor as a parameter to bootstrap treeview

                                    // apply the page view hierarchy
                                    $("#{page_idx}").treeview({{
                                        data: {page_func_name}(),
                                        enableLinks: true,
                                        color: linkColor,
                                        showTags: {show_tags},
                                        collapseIcon: "{collapse_icon}",
                                        expandIcon: "{expand_icon}",
                                        levels: {levels},
                                        onhoverColor: "{onhover_color}"
                                    }});

                                    // apply the class view hierarchy
                                    $("#{class_idx}").treeview({{
                                        data: {class_func_name}(),
                                        enableLinks: true,
                                        color: linkColor,
                                        showTags: {show_tags},
                                        collapseIcon: "{collapse_icon}",
                                        expandIcon: "{expand_icon}",
                                        levels: {levels},
                                        onhoverColor: "{onhover_color}"
                                    }});

                                    // apply the file view hierarchy
                                    $("#{file_idx}").treeview({{
                                        data: {file_func_name}(),
                                        enableLinks: true,
                                        color: linkColor,
                                        showTags: {show_tags},
                                        collapseIcon: "{collapse_icon}",
                                        expandIcon: "{expand_icon}",
                                        levels: {levels},
                                        onhoverColor: "{onhover_color}"
                                    }});

                                    // Part 2: override the style of the glyphicons by injecting some CSS
                                    $('<style type="text/css" id="exhaleTreeviewOverride">' +
                                      '    .treeview span[class~=icon] {{ '                 +
                                      '        color: ' + iconColor + ' ! important;'       +
                                      '    }}'                                              +
                                      '</style>').appendTo('head');
                               }});
                           </script>
                    '''.format(
                        icon_mimic=configs.treeViewBootstrapIconMimicColor,
                        page_idx=configs._page_hierarchy_id,
                        page_func_name=configs._bstrap_page_hierarchy_fn_data_name,
                        class_idx=configs._class_hierarchy_id,
                        class_func_name=configs._bstrap_class_hierarchy_fn_data_name,
                        file_idx=configs._file_hierarchy_id,
                        file_func_name=configs._bstrap_file_hierarchy_fn_data_name,
                        show_tags="true" if configs.treeViewBootstrapUseBadgeTags else "false",
                        collapse_icon=configs.treeViewBootstrapCollapseIcon,
                        expand_icon=configs.treeViewBootstrapExpandIcon,
                        levels=configs.treeViewBootstrapLevels,
                        onhover_color=configs.treeViewBootstrapOnhoverColor
                    )))
        except:
            utils.fancyError(
                "Unable to create the root api body: [{0}]".format(self.full_root_file_path)
            )

    def gerrymanderNodeFilenames(self):
        '''
        When creating nodes, the filename needs to be relative to ``conf.py``, so it
        will include ``self.root_directory``.  However, when generating the API, the
        file we are writing to is in the same directory as the generated node files so
        we need to remove the directory path from a given ExhaleNode's ``file_name``
        before we can ``include`` it or use it in a ``toctree``.
        '''
        for node in self.all_nodes:
            node.file_name = os.path.basename(node.file_name)
            if node.kind == "file":
                node.program_file = os.path.basename(node.program_file)

    def generateViewHierarchies(self):
        '''
        Wrapper method to create the view hierarchies.  Currently it just calls
        :func:`~exhale.graph.ExhaleRoot.generatePageView`,
        :func:`~exhale.graph.ExhaleRoot.generateClassView`, and
        :func:`~exhale.graph.ExhaleRoot.generateDirectoryView` --- if you want to implement
        additional hierarchies, implement the additionaly hierarchy method and call it
        from here.  Then make sure to ``include`` it in
        :func:`~exhale.graph.ExhaleRoot.generateAPIRootBody`.
        '''
        # gather the page hierarchy data and write it out
        page_view_data = self.generatePageView()
        self.writeOutHierarchy({
            "idx": configs._page_hierarchy_id,
            "bstrap_data_func_name": configs._bstrap_page_hierarchy_fn_data_name,
            "file_name": self.page_hierarchy_file,
            "file_title": configs.pageHierarchySubSectionTitle,
            "type": "page"
        }, page_view_data)
        # gather the class hierarchy data and write it out
        class_view_data = self.generateClassView()
        self.writeOutHierarchy({
            "idx": configs._class_hierarchy_id,
            "bstrap_data_func_name": configs._bstrap_class_hierarchy_fn_data_name,
            "file_name": self.class_hierarchy_file,
            "file_title": "Class Hierarchy",
            "type": "class"
        }, class_view_data)
        # gather the file hierarchy data and write it out
        file_view_data = self.generateDirectoryView()
        self.writeOutHierarchy({
            "idx": configs._file_hierarchy_id,
            "bstrap_data_func_name": configs._bstrap_file_hierarchy_fn_data_name,
            "file_name": self.file_hierarchy_file,
            "file_title": "File Hierarchy",
            "type": "file"
        }, file_view_data)

    def writeOutHierarchy(self, hierarchy_config, data):
        # inject the raw html for the treeView unordered lists
        if configs.createTreeView:
            # Cheap minification.  The `data` string is either
            #
            # 1. The interior of an HTML <ul> ... </ul> (collapsible lists)
            # 2. A json array for returning from a javascript function (bootstrap)
            #
            # In either case, the data is currently well-formatted, no "suprise"
            # newlines should appear, etc.  So we can just split the lines and strip
            # the leading indentation.
            if configs.minifyTreeView:
                data = "".join([line.strip() for line in data.splitlines()])
                # For the bootstrap version we can also further elminate some extra
                # spaces between colons and their mapped value, and delete some
                # erroneous commas that don't hurt but don't help ;)
                if configs.treeViewIsBootstrap:
                    data = data.replace(': ', ':').replace(",}", "}").replace(",,", ",").replace(",]", "]")

            if data:
                # conveniently, both get indented to the same level.  a happy accident
                indent = " " * 9  # indent by 6 + 3 for being under .. raw:: html
                indented_data = re.sub(r'(.+)', r'{indent}\1'.format(indent=indent), data)
                idx = hierarchy_config["idx"]

                final_data_stream = StringIO()
                if configs.treeViewIsBootstrap:
                    func_name = hierarchy_config["bstrap_data_func_name"]
                    # developer note: when using string formatting with {curly_braces}, if
                    # you want a literal curly brace you escape it with curly braces.  so
                    # the left curly brace is `{{` rather than `{` so that the formatting
                    # knows you want a literal `{` in the end.
                    final_data_stream.write(textwrap.dedent('''
                        .. raw:: html

                           <div id="{idx}"></div>
                           <script type="text/javascript">
                             function {func_name}() {{
                                return [
                    '''.format(idx=idx, func_name=func_name)))
                    final_data_stream.write(indented_data)
                    # NOTE: the final .. end raw html line "tricks" textwrap.dedent into
                    #       only stripping out until there. DO NOT REMOVE EVER!
                    final_data_stream.write(textwrap.dedent('''
                                ]
                             }}
                           </script><!-- end {func_name}() function -->

                        .. end raw html for treeView
                    '''.format(idx=idx, func_name=func_name)))
                else:
                    final_data_stream.write(textwrap.dedent('''
                        .. raw:: html

                           <ul class="treeView" id="{idx}">
                             <li>
                               <ul class="collapsibleList">
                    '''.format(idx=idx)))
                    final_data_stream.write(indented_data)
                    # NOTE: the final .. end raw html line "tricks" textwrap.dedent into
                    #       only stripping out until there. DO NOT REMOVE EVER!
                    final_data_stream.write(textwrap.dedent('''
                               </ul>
                             </li><!-- only tree view element -->
                           </ul><!-- /treeView {idx} -->

                        .. end raw html for treeView
                    '''.format(idx=idx)))

                # the appropriate raw html has been created, grab the final value
                final_data_string = final_data_stream.getvalue()
                final_data_stream.close()
            else:
                final_data_string = data
        else:
            # non-treeView is already done formatting, just a bulleted list
            final_data_string = data

        # Last but not least, we need the file to write to
        file_name = hierarchy_config["file_name"]

        # write everything to file to be incorporated with `.. include::` later
        try:
            if final_data_string:
                with codecs.open(file_name, "w", "utf-8") as hierarchy_file:
                    file_title = hierarchy_config["file_title"]
                    hierarchy_file.write(textwrap.dedent('''
                        {heading}
                        {heading_mark}

                    ''').format(
                        heading=file_title,
                        heading_mark=utils.heading_mark(
                            file_title,
                            configs.SUB_SECTION_HEADING_CHAR
                        )
                    ))
                    hierarchy_file.write(final_data_string)
                    hierarchy_file.write("\n\n")  # just in case, extra whitespace causes no harm
        except:
            h_type = hierarchy_config["type"]
            utils.fancyError("Error writing the {h_type} hierarchy.".format(h_type=h_type))

    def generatePageView(self):
        '''
        Generates the pages view hierarchy, writing it to ``self.page_hierarchy_file``.
        '''
        page_view_stream = StringIO()

        for p in self.pages:
            p.toHierarchy("page", 0, page_view_stream)

        # extract the value from the stream and close it down
        page_view_string = page_view_stream.getvalue()
        page_view_stream.close()

        return page_view_string

    def generateClassView(self):
        '''
        Generates the class view hierarchy, writing it to ``self.class_hierarchy_file``.
        '''
        class_view_stream = StringIO()

        for n in self.namespaces:
            n.toHierarchy("class", 0, class_view_stream)

        # Add everything that was not nested in a namespace.
        missing = []
        # class-like objects (structs and classes)
        for cl in sorted(self.class_like):
            if not cl.in_class_hierarchy:
                missing.append(cl)
        # enums
        for e in sorted(self.enums):
            if not e.in_class_hierarchy:
                missing.append(e)
        # unions
        for u in sorted(self.unions):
            if not u.in_class_hierarchy:
                missing.append(u)

        if len(missing) > 0:
            idx = 0
            last_missing_child = len(missing) - 1
            for m in missing:
                m.toHierarchy("class", 0, class_view_stream, idx == last_missing_child)
                idx += 1
        elif configs.createTreeView:
            # need to restart since there were no missing children found, otherwise the
            # last namespace will not correctly have a lastChild
            class_view_stream.close()
            class_view_stream = StringIO()

            last_nspace_index = len(self.namespaces) - 1
            for idx in range(last_nspace_index + 1):
                nspace = self.namespaces[idx]
                nspace.toHierarchy("class", 0, class_view_stream, idx == last_nspace_index)

        # extract the value from the stream and close it down
        class_view_string = class_view_stream.getvalue()
        class_view_stream.close()
        return class_view_string

    def generateDirectoryView(self):
        '''
        Generates the file view hierarchy, writing it to ``self.file_hierarchy_file``.
        '''
        file_view_stream = StringIO()

        for d in self.dirs:
            d.toHierarchy("file", 0, file_view_stream)

        # add potential missing files (not sure if this is possible though)
        missing = []
        for f in sorted(self.files):
            if not f.in_file_hierarchy:
                missing.append(f)

        found_missing = len(missing) > 0
        if found_missing:
            idx = 0
            last_missing_child = len(missing) - 1
            for m in missing:
                m.toHierarchy("file", 0, file_view_stream, idx == last_missing_child)
                idx += 1
        elif configs.createTreeView:
            # need to restart since there were no missing children found, otherwise the
            # last directory will not correctly have a lastChild
            file_view_stream.close()
            file_view_stream = StringIO()

            last_dir_index = len(self.dirs) - 1
            for idx in range(last_dir_index + 1):
                curr_d = self.dirs[idx]
                curr_d.toHierarchy("file", 0, file_view_stream, idx == last_dir_index)

        # extract the value from the stream and close it down
        file_view_string = file_view_stream.getvalue()
        file_view_stream.close()
        return file_view_string

    def generateUnabridgedAPI(self):
        '''
        Generates the unabridged (full) API listing into ``self.unabridged_api_file``.
        This is necessary as some items may not show up in either hierarchy view,
        depending on:

        1. The item.  For example, if a namespace has only one member which is a
           variable, then neither the namespace nor the variable will be declared in the
           class view hierarchy.  It will be present in the file page it was declared in
           but not on the main library page.

        2. The configurations of Doxygen.  For example, see the warning in
           :func:`~exhale.graph.ExhaleRoot.fileRefDiscovery`.  Items whose parents cannot
           be rediscovered withouth the programlisting will still be documented, their
           link appearing in the unabridged API listing.

        Currently, the API is generated in the following (somewhat arbitrary) order:

        - Namespaces
        - Concepts
        - Classes and Structs
        - Enums
        - Unions
        - Functions
        - Variables
        - Defines
        - Typedefs
        - Directories
        - Files
        '''
        try:
            from collections.abc import MutableMapping
        except ImportError:
            # TODO: remove when dropping python 2.7
            from collections import MutableMapping
        class UnabridgedDict(MutableMapping):
            def __init__(self):
                self.items = {}
                for kind in utils.AVAILABLE_KINDS:
                    self.__setitem__(kind, [])

            def _key(self, k):
                # Just need to fold class and struct to same bucket.
                if k == "struct":
                    return "class"
                return k

            def __getitem__(self, key):
                k = self._key(key)
                if k not in self.items:
                    sys.stderr.write(utils.critical(
                        "Unabridged API: unexpected kind '{}' (IGNORED)\n".format(key)
                    ))
                    self.items[k] = []
                return self.items[k]

            def __setitem__(self, key, value):
                self.items[self._key(key)] = value

            def __delitem__(self, key):
                del self.items[self._key(key)]

            def __iter__(self):
                return iter(self.items)

            def __len__(self):
                return len(self.items)

        try:
            # Gather all nodes in an easy to index dictionary mapping node.kind to the
            # node itself.  "class" and "struct" are stored together.
            unabridged_specs = UnabridgedDict()
            for node in self.all_nodes:
                if node.kind == "page" and node.refid == "indexpage":
                    continue
                unabridged_specs[node.kind].append(node)

            # Create the buffers to write to and dump the page headings.
            unabridged_api = StringIO()
            orphan_api = StringIO()
            for page, is_orphan in [(unabridged_api, False), (orphan_api, True)]:
                if is_orphan:
                    page.write(":orphan:\n\n")
                page.write(textwrap.dedent('''
                    {heading}
                    {heading_mark}
                '''.format(
                    heading=configs.fullApiSubSectionTitle,
                    heading_mark=utils.heading_mark(
                        configs.fullApiSubSectionTitle,
                        configs.SECTION_HEADING_CHAR if is_orphan
                        else configs.SUB_SECTION_HEADING_CHAR
                    )
                )))

            dump_order = [
                ("Namespaces", "namespace"),
                ("Concepts", "concept"),
                ("Classes and Structs", "class"),  # NOTE: class/struct stored together!
                ("Enums", "enum"),
                ("Unions", "union"),
                ("Functions", "function"),
                ("Variables", "variable"),
                ("Defines", "define"),
                ("Typedefs", "typedef"),
                ("Directories", "dir"),
                ("Files", "file"),
                ("Pages", "page")
            ]
            for title, kind in dump_order:
                node_list = unabridged_specs[kind]
                # Write to orphan_api if this kind is to be ignored, or the kind is
                # "class" and "struct" was ignored (stored together).
                if kind in configs.unabridgedOrphanKinds or \
                        (kind == "class" and "struct" in configs.unabridgedOrphanKinds) or \
                        (kind == "struct" and "class" in configs.unabridgedOrphanKinds):
                    dest = orphan_api
                else:
                    dest = unabridged_api
                self.enumerateAll(title, node_list, dest)

            # Write out the unabridged api file (gets included to root).
            with codecs.open(self.unabridged_api_file, "w", "utf-8") as full_api_file:
                full_api_file.write(unabridged_api.getvalue())

            # If the orphan file has any .. toctree:: in there, then we want to make
            # sure to write it.  For example, if files and directories are dumped here,
            # we want Sphinx to be convinced that they show up in a toctree somewhere.
            orphan_api_value = orphan_api.getvalue()
            if "toctree" in orphan_api_value:
                with codecs.open(self.unabridged_orphan_file, "w", "utf-8") as orphan_file:
                    orphan_file.write(orphan_api_value)
        except:
            utils.fancyError("Error writing the unabridged API.")

    def enumerateAll(self, subsectionTitle, lst, openFile):
        '''
        Helper function for :func:`~exhale.graph.ExhaleRoot.generateUnabridgedAPI`.
        Simply writes a subsection to ``openFile`` (a ``toctree`` to the ``file_name``)
        of each ExhaleNode in ``sorted(lst)`` if ``len(lst) > 0``.  Otherwise, nothing
        is written to the file.

        :Parameters:
            ``subsectionTitle`` (str)
                The title of this subsection, e.g. ``"Namespaces"`` or ``"Files"``.

            ``lst`` (list)
                The list of ExhaleNodes to be enumerated in this subsection.

            ``openFile`` (File)
                The **already open** file object to write to directly.  No safety checks
                are performed, make sure this is a real file object that has not been
                closed already.
        '''
        if len(lst) > 0:
            openFile.write(textwrap.dedent('''
                {heading}
                {heading_mark}

            '''.format(
                heading=subsectionTitle,
                heading_mark=utils.heading_mark(
                    subsectionTitle,
                    configs.SUB_SUB_SECTION_HEADING_CHAR
                )
            )))
            for l in sorted(lst):
                openFile.write(textwrap.dedent('''
                    .. toctree::
                       :maxdepth: {depth}

                       {file}
                '''.format(
                    depth=configs.fullToctreeMaxDepth,
                    file=l.file_name
                )))

    ####################################################################################
    #
    ##
    ### Miscellaneous utility functions.
    ##
    #
    ####################################################################################
    def toConsole(self):
        '''
        Convenience function for printing out the entire API being generated to the
        console.  Unused in the release, but is helpful for debugging ;)
        '''
        fmt_spec = {
            "concept":   utils.AnsiColors.BOLD_MAGENTA,
            "class":     utils.AnsiColors.BOLD_MAGENTA,
            "struct":    utils.AnsiColors.BOLD_CYAN,
            "define":    utils.AnsiColors.BOLD_YELLOW,
            "enum":      utils.AnsiColors.BOLD_MAGENTA,
            "enumvalue": utils.AnsiColors.BOLD_RED,     # red means unused in framework
            "function":  utils.AnsiColors.BOLD_CYAN,
            "file":      utils.AnsiColors.BOLD_YELLOW,
            "dir":       utils.AnsiColors.BOLD_MAGENTA,
            "group":     utils.AnsiColors.BOLD_RED,     # red means unused in framework
            "namespace": utils.AnsiColors.BOLD_CYAN,
            "typedef":   utils.AnsiColors.BOLD_YELLOW,
            "union":     utils.AnsiColors.BOLD_MAGENTA,
            "variable":  utils.AnsiColors.BOLD_CYAN,
            "page":      utils.AnsiColors.BOLD_YELLOW
        }

        self.consoleFormat(
            "{0} and {1}".format(
                utils._use_color("Concepts", fmt_spec["concept"],  sys.stderr),
                utils._use_color("Classes", fmt_spec["class"],  sys.stderr),
                utils._use_color("Structs", fmt_spec["struct"], sys.stderr),
            ),
            self.class_like,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Defines", fmt_spec["define"], sys.stderr),
            self.defines,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Enums", fmt_spec["enum"], sys.stderr),
            self.enums,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Enum Values (unused)", fmt_spec["enumvalue"], sys.stderr),
            self.enum_values,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Functions", fmt_spec["function"], sys.stderr),
            self.functions,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Files", fmt_spec["file"], sys.stderr),
            self.files,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Directories", fmt_spec["dir"], sys.stderr),
            self.dirs,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Groups (unused)", fmt_spec["group"], sys.stderr),
            self.groups,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Namespaces", fmt_spec["namespace"], sys.stderr),
            self.namespaces,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Typedefs", fmt_spec["typedef"], sys.stderr),
            self.typedefs,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Unions", fmt_spec["union"], sys.stderr),
            self.unions,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Variables", fmt_spec["variable"], sys.stderr),
            self.variables,
            fmt_spec
        )
        self.consoleFormat(
            utils._use_color("Pages", fmt_spec["page"], sys.stderr),
            self.pages,
            fmt_spec
        )

    def consoleFormat(self, sectionTitle, lst, fmt_spec):
        '''
        Helper method for :func:`~exhale.graph.ExhaleRoot.toConsole`.  Prints the given
        ``sectionTitle`` and calls :func:`~exhale.graph.ExhaleNode.toConsole` with ``0``
        as the level for every ExhaleNode in ``lst``.

        **Parameters**
            ``sectionTitle`` (str)
                The title that will be printed with some visual separators around it.

            ``lst`` (list)
                The list of ExhaleNodes to print to the console.
        '''
        if not configs.verboseBuild:
            return

        utils.verbose_log(textwrap.dedent('''
            ###########################################################
            ## {0}
            ###########################################################'''.format(sectionTitle)))
        for l in lst:
            l.toConsole(0, fmt_spec)
