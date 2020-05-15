/**
 * @file rake.cuh
 *
 * @brief Serial Reduce-Then-Scan. Description of a (typically) conflict-free
 serial-reduce-then-scan (raking) shared-memory grid.
 *
 * A "lane" for reduction/scan consists of one value (i.e., "partial") per
 * active thread.  A grid consists of one or more scan lanes. The lane(s) can be
 * sequentially "raked" by the specified number of raking threads (e.g., for
 * upsweep reduction or downsweep scanning), where each raking thread progresses
 * serially through a segment that is its share of the total grid.
 *
 * Depending on how the raking threads are further reduced/scanned, the lanes
 * can be independent (i.e., only reducing the results from every SEGS_PER_LANE
 * raking threads), or fully dependent (i.e., reducing the results from every
 * raking thread)
 *
 * Must have at least as many raking threads as lanes (i.e., at least one raking
 * thread for each lane).
 *
 * If (there are prefix dependences between lanes) AND (more than one warp of
 * raking threads is specified), a secondary raking grid will be typed-out in
 * order to facilitate communication between warps of raking threads.
 *
 * @note Typically two-level grids are a losing performance proposition.
 *
 * @todo Explore "Single-pass Parallel Prefix Scan with Decoupled Look-back"
 * instead. Authors, Duane Merrill and Michael Garland note: > Contemporary GPU
 * scan parallelization strategies such as reduce-then-scan are typically
 * memory-bound, but impose ~3n global data movement. Furthermore, they perform
 * two full passes over the input, which precludes them from serving as in-situ
 * global allocation mechanisms within computations that oversubscribe the
 * processor. Finally, these scan algorithms cannot be modified for in-place
 * compaction behavior (selection, run-length-encoding, duplicate removal, etc.)
 * because the execution order of thread blocks within the output pass is
 * unconstrained. Separate storage is required for the compacted output to
 * prevent race conditions where inputs might otherwise be overwritten before
 * they can be read. Alternatively, the chained-scan GPU parallelization
 * operates in a single pass, but is hindered by serial prefix dependences
 * between adjacent processors that prevent memory I/O from fully saturating. In
 * comparison, our decoupledlookback algorithm elides these serial dependences
 * at the expense of bounded redundant computation. As a result, our prefix scan
 * computations (as well as adaptations for in-place compaction behavior and
 * in-situ allocation) are typically capable of saturating memory bandwidth in a
 * single pass.
 *
 */