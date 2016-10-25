Gunrock v0.4 Release Notes {#release_notes}
==========================

Release 0.4
?th October 2015

Gunrock release 0.4 is a feature release that includes new optimizations to
both advance and filter operators, adds multi-iteration supports for both BFS
and SSSP, adds better error handling, and updates several interfaces. The new
release improves the overall performance for both single and multi-GPU
execution.

v0.4 ChangeLog
==============
 - Integrate direction-optimizing BFS with normal BFS. Now for BFS there is only one executable, named bfs. The direction-optimizing switch can be turned on by this command line option: '--direction-optimized'.
 - Added two new strategies of advance: all_edges_advance (optimized for
   advance on all edges with all vertices in the input frontier, no need to use
   sorted search, just binary search over the whole row offsets array),
   fused_advance_filter (optimized for advance followed with filter step).
   all_edges_advance is triggered by setting advance mode to ALL_EDGES, it can
   be used for any case that the users know the input frontier contains all
   vertices and the advance step will visit all edges. Such cases include PR,
   Who-to-Follow, MIS, label propagation, and more. fused_advance_filter is
   automatically turned on for two advance modes: LB_CULL and LB_LIGHT_CULL.
   The fused advance and filter happen in BFS, SSSP, BC, and a lot other
   traversal-based graph primitives. A filter followed with
   fused_advance_filter is harmless but would be a waste of performance.
 - Added three new strategies of filter: compacted_cull_filter (optimized on
   several culling heuristics), bypass_filter and simplified_filter (optimized
   for filter with no elements to remove from the input frontier).
   compacted_cull_filter could be triggered by setting the filter mode to
   COMPACTED_CULL, in general it removes more redundant elements and could
   improve the performance from CC to traversal-based graph primitives like BFS
   and SSSP. It is also the default filter in fused_advance_and_filter
   operator. by_pass_filter and simplified_filter are both used for per-node
   -edge computation without filtering, such as in CC and PR.  They could be
   triggered with filter mode: BY_PASS and SIMPLIFIED respectively.
 - Added multi-iteration support for BFS and SSSP.

v0.4 Known Issues
=================
 - HITS, SALSA, Who-to-Follow do not have multi-GPU support yet.
 - Out of memory error will cause result validation to fail.
