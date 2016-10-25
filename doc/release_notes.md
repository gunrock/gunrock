Gunrock v0.4 Release Notes {#release_notes}
==========================

Release 0.4
?th October 2015

Gunrock release 0.4 is a feature release that includes new optimizations to
both advance and filter operators, adds multi-iteration support for both BFS
and SSSP, adds better error handling, and updates several interfaces. The new
release improves the overall performance for both single and multi-GPU
execution.

v0.4 ChangeLog
==============
 - Integrated direction-optimizing BFS with normal BFS
 - Added two new strategies for advance:
     - `all_edges_advance` (optimized for advance on all edges with all vertices in the input frontier, no need to use sorted search, just binary search over the whole row offsets array),
     - `fused_advance_filter` (optimized for advance followed with filter step)
 - Added three new strategies for filter:
     - `compacted_cull_filter` (optimized on several culling heuristics)
     - `bypass_filter` and `simplified_filter` (optimized for filters with no elements to remove from the input frontier)
 - Added multi-iteration support for BFS and SSSP

v0.4 Known Issues
=================
 - HITS, SALSA, Who-to-Follow do not have multi-GPU support yet.
 - Out of memory error will cause result validation to fail.
