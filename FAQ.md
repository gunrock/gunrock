Gunrock FAQ     {#faq}
===========

What does it do?
----------------

Gunrock is a fast and efficient graph processing library on the GPU that
provides a set of graph algorithms used in big data analytics and visualization
with high performance.  It also provides a set of operators which abstract the
general operations in graph processing for other developers to build
high-performance graph algorithm prototypes with minimum programming effort.

How does it do it?
------------------

Gunrock takes advantage of the immense computational power available in
commodity-level, off-the-shelf Graphics Processing Units (GPUs), originally
designed to handle the parallel computational tasks in computer graphics, to
perform graph traversal and computation in parallel on thousands of GPU's
computing cores.

Who should want this?
---------------------

Gunrock is built with two kinds of users in mind: The first kind of users are
programmers who build big graph analytics and visualization projects and need to
use existing graph primitives provided by Gunrock.  The second kind of users
are programmers who want to use Gunrock's high-level, programmable abstraction
to express, develop, and refine their own (and often more complicated) graph
primitives.

What is the skill set users need to use it?
-------------------------------------------

For the first kind of users, C/C++ background is sufficient. We are also
building Gunrock as a shared library with C interfaces that can be loaded by
other languages such as Python and Julia.  For the second kind of users, they
need to have the C/C++ background and also an understanding of parallel
programming, especially BSP (Bulk-Synchronous Programming) model used by Gunrock.

What platforms/languages do people need to know in order to modify or integrate it with other tools?
----------------------------------------------------------------------------------------------------

Using the exposed interface, the users do not need to know CUDA or OpenCL to
modify or integrate Gunrock to their own tools. However, an essential
understanding of parallel programming and BSP model is necessary if one wants
to add/modify graph primitives in Gunrock.

Why would someone want this?
----------------------------

The study of social networks, webgraphs, biological networks, and unstructured
meshes in scientific simulation has raised a significant demand for efficient
parallel frameworks for processing and analytics on large-scale graphs. Initial
research efforts in using GPUs for graph processing and analytics are promising.

How is it better than the current state of the art?
---------------------------------------------------

Most existing CPU large graph processing libraries perform worse on large
graphs with billions of edges. Supercomputer or expensive clusters can achieve
close to real-time feedback with high cost on hardware infrastructure. With
GPUs, we can achieve the same real-time feedback with much lower cost on
hardware. Gunrock has the best performance among the limited research efforts
toward GPU graph processing. Our peak Edge Traversed Per Second (ETPS) can
reach 3.5G.  And all the primitives in Gunrock have 10x to 25x speedup over the
equivalent single-node CPU implementations. With a set of general graph
processing operators exposed to users, Gunrock is also more flexible than other
GPU/CPU graph library in terms of programmability.

How would someone get it?
-------------------------

Gunrock is an open-source library. The code, documentation, and quick start
guide are all on its [GitHub page](gunrock.github.io).

Is a user account required?
---------------------------

No. One can use either git clone or download directly to get the source code
and documentation of Gunrock.

Are all of its components/dependencies easy to find?
----------------------------------------------------

Gunrock has three dependencies. Two of them are also GPU primitive libraries which
also reside on GitHub. The third one is Boost (Gunrock uses Boost Graph Library
to implement CPU reference testing algorithms). All dependencies do not require
installation. To use, one only needs to download or git clone them and put them
in the according directories. More details in the installation section of this
documentation.

How would someone install it?
-----------------------------

For C/C++ programmer, integrating Gunrock into your projects is easy. Since it
is a template based library, just add the include files in your code. The
simple example and all the testrigs will provide detailed information on how to
do this.

For programmers who use Python, Julia, or other language and want to call
Gunrock APIs, we are building a shared library with binary compatible
C interfaces. It will be included in the soon-to-arrive next release of
Gunrock.

Can anyone install it? Do they need IT help?
--------------------------------------------

Gunrock is targeted at developers who are familiar with basic software
engineering. For non-technical people, IT help might be needed.

Does this process actually work? All the time? On all systems specified?
------------------------------------------------------------------------
Currently, Gunrock has been tested on two Linux distributions: Linux Mint and
Ubuntu. But we expect it to run correctly on other Linux distributions too.
We are currently building a CMake solution to port Gunrock to Mac and Windows.
The feature will be included in the soon-to-arrive next release of Gunrock.

How would someone test that it's working with provided sample data?
-------------------------------------------------------------------

Testrigs are provided as well as a small simple example for users to test the
correctness and performance of every graph primitive.

Is the "using" of sample data clear?
------------------------------------

On Linux, one only needs to go to the dataset directory and run "make"; the
script will automatically download all the needed datasets. One can also choose
to download a single dataset in its separate directory.

How would someone use it with their own data?
---------------------------------------------

Gunrock supports Matrix Market (.mtx) file format; users need to pre-process
the graph data into this format before running Gunrock.
