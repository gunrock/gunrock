Building Gunrock              {#building_gunrock}
==============

This default branch has currently been tested on Linux and has only been tuned
for architectures with CUDA
[compute capability](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)
equal to or larger than 3.0. It is most likely that we will concentrate our
future effort on 3.0+-capability devices going forward.

Boost Dependency           {#build_boost}
=================

Gunrock uses the
[Boost Graph Library (BGL)](http://www.boost.org/doc/libs/1_58_0/libs/graph/doc/index.html)
for the CPU reference implementations of Connected Component, Betweenness
Centrality, PageRank, Single-Source Shortest Path, and Minimum Spanning Tree.
You will need to
[install Boost](http://www.boost.org/doc/libs/1_58_0/doc/html/bbv2/installation.html)
to build test applications.

METIS Dependency {#build_metis}
=================
Gunrock uses the
[METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) as one
possible partitioner. You will need to
[install METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)
to build test applications.

ModernGPU & CUB Dependency           {#build_mgpu}
=================

Gunrock uses APIs from [Modern GPU](https://github.com/NVlabs/moderngpu)
and [CUB](http://nvlabs.github.io/cub/). You
will need to download or clone them and place them to `gunrock/externals`.
Alternatively, you can clone gunrock recursively with the git command:

    git clone --recursive https://github.com/gunrock/gunrock

or if you already cloned gunrock, under `gunrock/`:

    git submodule init
    git submodule update

Generating Datasets           {#generating_datasets}
===================

All dataset-related code is under the `gunrock/dataset/` subdirectory. The
current version of Gunrock only supports
[Matrix-market coordinate-formatted graph](http://math.nist.gov/MatrixMarket/formats.html)
format. The datasets are divided into two categories according to their scale.
Under the `dataset/small/` subdirectory, there are trivial graph datasets for
testing the correctness of the graph primitives. All of them are ready to use.
Under the `dataset/large/` subdirectory, there are large graph datasets for
doing performance regression tests. To download them to your local machine,
just type `make` in the `dataset/large/` subdirectory. You can also choose to
only download one specific dataset to your local machine by stepping into the
subdirectory of that dataset and typing `make` inside that subdirectory.

Running the Tests           {#running_tests}
=================
Before running the tests, make sure you have installed boost and have
successfully generated the datasets.

- Clone gunrock: `git clone --recursive https://github.com/gunrock/gunrock`
- Create and enter build directory: `mkdir build && cd build`
- Make: `cmake [gunrock directory]`, Then type: `make`
Binary test files are in directory: `build/bin`

It will also build a shared library with a C-friendly interface; the example
test files of calling Gunrock APIs are located at `gunrock/shared_lib_tests`.

Alternatively, you can build gunrock using individual `Makefiles` under
`gunrock/test/primitive_name` and simply type `make`.  You can either run the
test for all primitives by typing `make test` or `ctest` in the build
directory, or do your own testings manually.
