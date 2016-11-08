Gunrock v0.4 Release Notes {#release_notes}
==========================

Release 0.4
8th November 2016

Gunrock release 0.4 is a feature release that adds

 - New optimizations to both advance and filter operators
 - Multi-iteration support for BFS, SSSP, BC, CC and PR
 - Better error handling
 - Updates on several interfaces
 - Overall performance improvement for both single and multi-GPU execution

v0.4 ChangeLog
==============
 - Integrated direction-optimizing BFS with normal BFS. Now for BFS
   there is only one executable, named bfs. The direction-optimizing
   switch is enabled by the command-line option
   `--direction-optimized`.
 - Added three new strategies for advance (triggered by setting
   `ADVANCE_MODE` accordingly):
    - `ALL_EDGES`, optimized for advance on all edges with all vertices
      of the graph. With `ALL_EDGES`, there is no need to use sorted
      search for load balancing, just binary search over the whole row
      offsets array; used in CC.
    - `LB_CULL`, fused LB advance with a subsequent CULL filter; used in
      BFS, SSSP and BC.
    - `LB_LIGHT_CULL`, fused `LB_LIGHT` advance with a subsequent CULL
      filter; used in BFS, SSSP and BC.
 - Added three new strategies of filter (triggered by setting
   `FILTER_MODE` accordingly):
    - `COMPACTED_CULL`, optimized on several culling heuristics
    - `SIMPLIFIED`, another implementation of the CULL filter, without
      some optimizations
    - `BY_PASS`, optimized for a filter with no elements
      to remove from the input frontier; used in CC and PR.
 - Added multi-iteration support for BFS, SSSP, BC, CC and PR. Users
   can set the number of iterations to run and specify the source node
   for each run (if necessary) via `InitSetup()` defined in gunrock.h.

v0.4 Known Issues
=================
 - HITS and SALSA do not have CPU reference yet
 - HITS, SALSA, and who-to-Follow do not have multi-GPU support yet
 - An out-of-memory error (for graphs that approach the memory limit
   of GPUs) will cause result validation to fail
