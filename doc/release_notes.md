Gunrock v0.4 Release Notes {#release_notes}
==========================

Release 0.4
?th October 2015

Gunrock release 0.4 is a feature release that adds 
 - New optimizations to both advance and filter operators,
 - Multi-iteration supports for BFS, SSSP, BC, CC and PR,
 - Better error handling,
 - Updates on several interfaces,
 - Overall performance improvement for both single and multi-GPU execution.

v0.4 ChangeLog
==============
 - Integrated direction-optimizing BFS with normal BFS.
 - Added three new strategies of advance:
    - ALL_EDGES, optimized for advance on all edges with all 
      vertices of the graph, no need to use sorted search, 
      just binary search over the whole row offsets array;
    - LB_CULL, fuzed LB advance with a subsequent CULL filter.
    - LB_LIGHT_CULL, fuzed LB_LIGHT advance with a subsequent CULL filter.
 - Added three new strategies of filter: 
    - COMPACTED_CULL, optimized on several culling heuristics;
    - SIMPLIFIED, an other implementation of the CULL filter, without 
      some optimizations;
    - BY_PASS, optimized for filter with no elements 
      to remove from the input frontier.
 - Added multi-iteration support for BFS, SSSP, BC, CC and PR.

v0.4 Known Issues
=================
 - HITS, SALSA, Who-to-Follow do not have multi-GPU support yet.
 - Out of memory error will cause result validation to fail.
