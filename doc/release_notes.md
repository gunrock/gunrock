Gunrock v0.3 Release Notes {#release_notes}
==========================

Release 0.3
8th August 2015

Gunrock release 0.3 is a feature release that adds two new graph primitives: 
Stochastic Approach for Link-Structure Analysis (SALSA) and Minimal Spanning 
Tree (MST), and improves several existing primitives. The new release uses a
unified framework for both single-GPU and single-node multi-GPUs. Five graph
primitives (BFS, CC, PR, BC, and SSSP) can be launched on multi-GPUs now by
adding `--device=GPU_index_1,GPU_index_2,...,GPU_index_n`. A simple pure C
interface allow users to easily integrate Gunrock into their own work. A 
stats logging and performance chart generating pipeline is developed and
integrated in this new release.

v0.3 ChangeLog
==============
 - Uses a unified framework for both single-GPU and single-node multi-GPUs.
 - Added a stats logging and performance chart generating pipeline.
 - Fixed bugs in BC, SALSA, and MST.
 - Fixed bugs in E2V Advance traversal mode.
 - Added C interfaces and Python sample code for five graph primitives (BFS, CC, PR, BC, and SSSP).
 - Can use both CMake system and make under each primitive directory.

v0.3 Known Issues
=================
 - Direction-Optimizing BFS does not work with multi-GPU.
