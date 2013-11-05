Gunrock documentation               {#mainpage}
=====================

Introduction
============

Gunrock is a CUDA library for graph primitives that refactors,
integrates, and generalizes best-of-class GPU implementations
of breadth-first search, connected components, and betweenness
centrality into a unified code base useful for future
development of high-performance GPU graph primitives.

Home Page
---------

Homepage for Gunrock: <http://gunrock.github.io/>

Getting Started with Gunrock
----------------------------

For information on building Gunrock, see [Building Gunrock](@ref
building_gunrock).

The "tests" subdirectory included with Gunrock has a comprehensive test
application for all the functionality of Gunrock.

We have also provided a code walkthrough of a [simple example](@ref
simple_example).

Reporting Problems
==================

To report Gunrock bugs or request features, please file an issue
directly using [Github](https://github.com/gunrock/gunrock/issues).

<!-- TODO: Algorithm Input Size Limitations -->

Operating System Support and Requirements
=========================================

This release (0.1) has only been tested on Linux Mint 15 (64-bit) with
CUDA 5.5 installed. We expect Gunrock to build and run correctly on
other 64-bit and 32-bit Linux distributions. The current release (0.1)
does not support any other platforms.

Requirements
------------

Gunrock has not been tested with any CUDA version < 5.5.

The CPU validity code for connected component and betweenness
centrality uses Boost Graph Library v1.53.0.

CUDA
====

Gunrock is implemented in [CUDA C/C++](http://developer.nvidia.com/cuda).
It requires the CUDA Toolkit. Please see the NVIDIA
[CUDA](http://developer.nvidia.com/cuda) homepage to download CUDA as well
as the CUDA Programming Guide and CUDA SDK, which includes many CUDA code
examples.

Design Goals
============

Gunrock aims to provide a core set of vertex-centric or edge-centric
operators for solving graph related problems and use these
parallel-friendly abstractions to improve programmer productivity
while maintaining high performance.

Road Map
========

 - Framework: The structure of the operator code in Gunrock may change
   significantly during near-term future development. Generally we
   want to find the right set of operators that can abstract most
   graph primitives while delivering high performance.

 - Primitives: Our near-term goal is to implement direction-optimal
   BFS (as described in *Direction-Optimizing Breadth-First Search*
   ([DOI](http://dx.doi.org/10.1109/SC.2012.50)) by Scott Beamer,
   Krste Asanovic and David Patterson) using the backward edge-mapping
   operator in Gunrock. The long-term goal includes other basic graph
   primitives such as single-source shortest path and minimal spanning
   tree.

Credits
=======

Gunrock Developers
------------------

- [Yangzihao Wang](http://www.idav.ucdavis.edu/~yzhwang/), University of
  California, Davis

- [John Owens](http://www.ece.ucdavis.edu/~jowens/), University of California,
  Davis [general help]

Acknowledgements
----------------

Thanks to the following developers who contributed code: The
connected-component implementation was derived from code written by
Jyothish Soman, Kothapalli Kishore, and P. J. Narayanan and described
in their IPDPSW '10 paper *A Fast GPU Algorithm for Graph
Connectivity* ([DOI](http://dx.doi.org/10.1109/IPDPSW.2010.5470817)).
The breadth-first search implementation and many of the utility
functions in Gunrock are derived from the
[b40c](http://code.google.com/p/back40computing/) library of
[Duane Merrill](https://sites.google.com/site/duanemerrill/). The
algorithm is described in his PPoPP '12 paper *Scalable GPU Graph
Traversal* ([DOI](http://dx.doi.org/10.1145/2370036.2145832)). Thanks
to Erich Elsen and Vishal Vaidyanathan from
[Royal Caliber](http://www.royal-caliber.com/) for their discussion on
library development and the dataset auto-generating code.

This work was funded by the DARPA XDATA program under AFRL Contract
FA8750-13-C-0002 and by NSF awards CCF-1017399 and OCI-1032859. Our
XDATA principal investigator is Eric Whyne of
[Data Tactics Corporation](http://www.data-tactics.com/) and our DARPA
program manager is
[Dr. Christopher White](http://www.darpa.mil/Our_Work/I2O/Personnel/Dr_Christopher_White.aspx).

Gunrock Copyright and Software License
======================================

Gunrock is copyright The Regents of the University of
California, 2013. The library, examples, and all source code are
released under
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
