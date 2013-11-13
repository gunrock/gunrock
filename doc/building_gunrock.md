Building Gunrock              {#building_gunrock}
==============

The release(0.1) has currently been tested on Linux.

Boost Dependency           {#build_boost}
=================
Gunrock uses the [Boost Graph Library](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/index.html)
for the implementation of connected component and betweenness centrality
CPU reference implementation. You will need to download the boost source
distribution, install it and modify the BOOST_INC variable in tests/cc/
Makefile and tests/bc/Makefile to build test applications for CC and BC.
To build simple example, you will also need to modify the BOOST_INC variable
in simple_example/Makefile.

Generating Datasets           {#generating_datasets}
===================
All the dataset-related code are under dataset subdirectory. The current
version of Gunrock only supports Matrix-market coordinate-formatted graph
file. The datasets are divided into two categories according to their
sizes. Under dataset/small subdirectory, there are trivial graph datasets
for testing the correctness of the graph primitives. All of them are ready
to use. Under dataset/large subdirectory, there are large graph datasets
for doing performance regression tests. To download them to your local
machine. Just type "make" in the dataset subdirectory. You can also choose
to only download one specific dataset to your local machine by step into
the subdirectory of that dataset, and type "make" inside that subdirectory.

Running the Tests           {#running_tests}
=================
Before running the tests, make sure you have installed boost and have
successfully generated datasets.

To build the tests, go into tests/bfs, tests/bc or tests/cc and simply
type "make".

To run the tests for each graph primitive, go into tests/bfs, tests/bc
or tests/cc and simply type "sh run.sh".

The current release (v0.1) has only been tuned for architectures with the
compute capability equal or larger than 3.0.
