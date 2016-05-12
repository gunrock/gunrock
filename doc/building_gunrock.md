Building Gunrock              {#building_gunrock}
==============

This default branch has been tested on Linux and has only been tuned
for architectures with CUDA
[compute capability](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)
equal to or larger than 3.0. It is most likely that we will concentrate our
future effort on 3.0+-capability devices going forward.

CUDA Version {#cuda_version}
============

Gunrock uses C++11 internally and thus requires a CUDA compiler that supports C++11 (CUDA 7.0 or greater). We identified and reported a compiler bug in CUDA 7.0 that was fixed in recent CUDA 7.5 builds, so we recommend CUDA 7.5 or higher. CUDA 7.0 may allow successful building of some primitives but not all. CUDA 8 release candidates appear to successfully compile and run Gunrock.

Boost Dependency           {#build_boost}
=================

Gunrock uses the
[Boost Graph Library (BGL)](http://www.boost.org/doc/libs/1_58_0/libs/graph/doc/index.html)
for the CPU reference implementations of Connected Component, Betweenness
Centrality, PageRank, Single-Source Shortest Path, and Minimum Spanning Tree.
You will need to
[install Boost](http://www.boost.org/doc/libs/1_58_0/doc/html/bbv2/installation.html)
to build test applications.

One external user has reported that a user-installed Boost in a local directory did not work but Boost installed as root does work, so we recommend the latter.

METIS Dependency {#build_metis}
=================
Gunrock uses the
[METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) as one
possible partitioner. You will need to
[install METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)
to build test applications.

One external user has reported that a user-installed METIS in a local directory did not work but METIS installed as root does work, so we recommend the latter. This is particularly important for finding METIS header files.

If the build cannot find your METIS library, please set the `METIS_DLL` environment variable to the full path of the library.

ModernGPU & CUB Dependency           {#build_mgpu}
=================

Gunrock uses APIs from [Modern GPU](https://github.com/NVlabs/moderngpu)
and [CUB](http://nvlabs.github.io/cub/). You
will need to download or clone them and place them into `gunrock/externals`.
Alternatively, you can clone gunrock recursively with the git command:

    git clone --recursive https://github.com/gunrock/gunrock

or if you already cloned gunrock, under `gunrock/`:

    git submodule init
    git submodule update
    
Even if users have these two packages elsewhere on their systems, please
install them into gunrock/externals for proper compilation.

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
