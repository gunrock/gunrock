<<<<<<< HEAD
Building Gunrock {#building_gunrock}
=====================
Gunrock current release has been tested on Linux Mint 15 (64-bit), Ubuntu 12.04, 14.04 and 15.10 with CUDA 7.5 installed, compute architecture 3.0 and g++ 4.8. We expect Gunrock to build and run correctly on other 64-bit and 32-bit Linux distributions, Mac OS, and Windows.

Installation {#installation}
----------------------------
**Overview:**
* [Prerequisites](#prerequisites)
* [Compilation](#compilation)
* [Hardware](#hardware)


Prerequisites {#prerequisites}
----------------------------
**Required Dependencies:**
* [CUDA](https://developer.nvidia.com/cuda-zone) (7.5 or higher) is used to implement Gunrock.
  * Refer to NVIDIA's [CUDA](https://developer.nvidia.com/cuda-downloads) homepage to download and install CUDA.
  * Refer to NVIDIA [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) for detailed information and examples on programming CUDA.

* [Boost](http://www.boost.org/users/history/version_1_58_0.html) (1.58 or higher) is used for for the CPU reference implementations of Connected Component, Betweenness Centrality, PageRank, Single-Source Shortest Path, and Minimum Spanning Tree.
  * Refer to Boost [Getting Started Guide](http://www.boost.org/doc/libs/1_58_0/more/getting_started/unix-variants.html) to install the required Boost libraries.
  * Alternatively, you can also install Boost by running `/gunrock/dep/install_boost.sh` script (recommended installation method).
  * Ideal location for Boost installation is `/usr/local/`. If the build cannot find your Boost library, make sure a symbolic link for boost installation exists somewhere in `/usr/` directory.

* [ModernGPU](https://github.com/moderngpu/moderngpu) and [CUB](http://nvlabs.github.io/cub/) used as external submodules for Gunrock's APIs.
  * You will need to download or clone ModernGPU and CUB, and place them to `gunrock/externals`.
  * Alternatively, you can clone gunrock recursively with `git clone --recursive https://github.com/gunrock/gunrock`
  * or if you already cloned gunrock, under `gunrock/` directory:

```
git submodule init
git submodule update
```

**Optional Dependencies:**
* [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) is used as one possible partitioner to partition graphs for multi-gpu primitives implementations.
  * Refer to METIS [Installation Guide](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)
  * Alternatively, you can also install METIS by running `/gunrock/dep/install_metis.sh` script.
  * If the build cannot find your METIS library, please set the `METIS_DLL` environment variable to the full path of the library.

Compilation {#compilation}
----------------------------
**Simple Gunrock Compilation:**
* Downloading gunrock:
```
# Using git (recursively) download gunrock
git clone --recursive https://github.com/gunrock/gunrock
# Using wget to download gunrock
wget --no-check-certificate https://github.com/gunrock/gunrock/archive/master.zip
```
* Compiling gunrock:
```
cd gunrock
mkdir build && cd build
cmake ..
make
```
* Binary test files are available in `build/bin` directory.
* You can either run the test for all primitives by typing `make check` or `ctest` in the build directory, or do your own testings manually.
* Detailed test log from `ctest` can be found in `/build/Testing/Temporary/LastTest.log`, alternatively you can run tests with verbose option enabled `ctest -v`.

**Advance Gunrock Compilation:**

You can also compile gunrock with more specific/advanced settings using `cmake -D[OPTION]=ON/OFF`. Following options are available:

* **GUNROCK_BUILD_LIB** (default: ON) - Builds main gunrock library.
* **GUNROCK_BUILD_SHARED_LIBS** (default: ON) - Turn off to build for static libraries.
* **GUNROCK_BUILD_APPLICATIONS** (default: ON) - Set off to only build one of the following primitive (GUNROCK\_APP\_* must be set on to build if this option is turned off.)
  * **GUNROCK_APP_BC** (default: OFF)
  * **GUNROCK_APP_BFS** (default: OFF)
  * **GUNROCK_APP_CC** (default: OFF)
  * **GUNROCK_APP_PR** (default: OFF)
  * **GUNROCK_APP_SSSP** (default: OFF)
  * **GUNROCK_APP_DOBFS** (default: OFF)
  * **GUNROCK_APP_HITS** (default: OFF)
  * **GUNROCK_APP_SALSA** (default: OFF)
  * **GUNROCK_APP_MST** (default: OFF)
  * **GUNROCK_APP_WTF** (default: OFF)
  * **GUNROCK_APP_TOPK** (default: OFF)

* **GUNROCK_MGPU_TESTS** (default: OFF) - If on, tests multiple GPU primitives with `ctest`.
* **GUNROCK_GENCODE_SM<>** (default: GUNROCK_GENCODE_SM30=ON) change to generate code for different compute capability.
* **CUDA_VERBOSE_PTXAS** (default: OFF) - ON to enable verbose output from the PTXAS assembler.

Example for compiling gunrock with only *Breadth First Search (BFS)* primitive:
```
mkdir build && cd build
cmake -DGUNROCK_BUILD_APPLICATIONS=OFF -DGUNROCK_APP_BFS=ON ..
make
```

Generating Datasets {#generating_datasets}
----------------------------
All dataset-related code is under the `gunrock/dataset/` subdirectory. The current version of Gunrock only supports [Matrix-market coordinate-formatted graph](http://math.nist.gov/MatrixMarket/formats.html) format. The datasets are divided into two categories according to their scale. Under the `dataset/small/` subdirectory, there are trivial graph datasets for testing the correctness of the graph primitives. All of them are ready to use. Under the `dataset/large/` subdirectory, there are large graph datasets for doing performance regression tests.
* To download them to your local machine, just type `make` in the `dataset/large/` subdirectory.
* You can also choose to only download one specific dataset to your local machine by stepping into the subdirectory of that dataset and typing make inside that subdirectory.



##Hardware {#hardware}
----------------------------
**Laboratory Tested Hardware:** Gunrock with GeForce GTX 970, Tesla K40s. We have not encountered any trouble in-house with devices with CUDA capability >= 3.0.

=======
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
>>>>>>> c0678c645c5d5175562ce329b8e6d40a4d977ebb
