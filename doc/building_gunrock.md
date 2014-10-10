Building Gunrock              {#building_gunrock}
==============

This release (0.2) has currently been tested on Linux and has only been tuned
for architectures with CUDA compute capability equal to or larger than 3.0. It
is most likely that we will concentrate our future effort on 3.0+-capability
devices going forward.

Boost Dependency           {#build_boost}
=================

Gunrock uses the [Boost Graph Library](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/index.html) for
the implementation of connected component, betweenness centrality, PageRank,
and Single-source shortest path CPU reference implementation. You will need to
download the boost source distribution, install it, and modify the `BOOST_INC`
variable in `tests/primitive_name/Makefile` to build test applications for bc,
cc, pr, and sssp. To build the simple example, you will also need to modify the
`BOOST_INC` variable in `simple_example/Makefile`.

Moderngpu Dependency           {#build_mgpu}
=================

Gunrock uses APIs from [moderngpu](https://github.com/NVlabs/moderngpu). You
will need to get it and place it at the same directory as Gunrock project. If
you place it somewhere else, please change the `MGPU_INC` in Makefiles
accordingly.

Generating Datasets           {#generating_datasets}
===================

All dataset-related code is under the `dataset` subdirectory. The current
version of Gunrock only supports Matrix-market coordinate-formatted graph file.
The datasets are divided into two categories according to their sizes. Under
the `dataset/small` subdirectory, there are trivial graph datasets for testing
the correctness of the graph primitives. All of them are ready to use.  Under
the `dataset/large` subdirectory, there are large graph datasets for doing
performance regression tests. To download them to your local machine, just type
`make` in the dataset subdirectory. You can also choose to only download one
specific dataset to your local machine by stepping into the subdirectory of
that dataset and typing `make` inside that subdirectory.

Running the Tests           {#running_tests}
=================

Before running the tests, make sure you have installed boost and have
successfully generated the datasets.

To build the tests, go into `tests/primitive_name` and simply type `make`.

To run the tests for each graph primitive, go into `tests/primitive_name` and
simply type `sh run.sh`.

The current release (v0.2) has only been tuned for architectures with CUDA
compute capability equal to or larger than 3.0. It is most likely that we will
concentrate our future effort on 3.0+-capability devices going forward.

Building Shared Library
======================
If you wish to build a shared library and load Gunrock`s primitives in your project
which uses a language has a C-friendly interface, just create a directory to put
all the build files and then type `cmake -i [directory of Gunrock]`. When asks about
whether to show advanced options, answer `Yes`. You should be able to specify the
directory of Boost then. You can then type `make` to build all test files
and the shared library. The examples of how to call Gunrock APIs are located at
shared_lib_tests.
