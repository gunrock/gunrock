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

#pragma once
#include <gunrock/util/limits.hxx>
#include <gunrock/util/meta.hxx>
#include <gunrock/util/numeric_traits.hxx>

namespace gunrock {
namespace algo {

/**
 * @namespace raking
 */
namespace raking {

namespace device {

/**
 * @brief
 * @todo should this really be called grid?
 */
namespace grid {

template<architecture_t GUNROCK_CUDA_ARCH,
         unsigned _LOG_ACTIVE_THREADS, // Number of threads placing a lane
                                       // partial (i.e., the number of partials
                                       // per lane)
         unsigned _LOG_SCAN_LANES,     // Number of scan lanes
         unsigned _LOG_RAKING_THREADS, // Number of threads used for raking
                                       // (typically 1 warp)
         bool _DEPENDENT_LANES, // If there are prefix dependences between lanes
                                // (i.e., downsweeping will incorporate
                                // aggregates from previous lanes)
         typename raking_value_t // Type of items we will be reducing/scanning
         >
struct rake
{
  // Type of items we will be reducing/scanning
  typedef raking_value_t value_t;

  // Warpscan type (using volatile storage for built-in types allows us to omit
  // thread-fence operations during warp-synchronous code)
  typedef
    typename util::meta::_if<(util::numeric_traits<value_t>::REPRESENTATION ==
                              util::representation_t::not_a_number),
                             value_t,
                             volatile value_t>::type_t warp_scan_t;

  /**
   * @note We use an enum type here b/c of a NVCC-win compiler bug where
   * the compiler can't handle ternary expressions in static-const fields having
   * both evaluation targets as local const expressions
   * @todo verify if the bug still exists, if not, remove this enum
   * representation.
   */

  enum
  {
    // Number of scan lanes
    log_scan_lanes = _LOG_SCAN_LANES,
    scan_lanes = 1 << log_scan_lanes,

    // Number of partials per lange
    log_partials_per_lane = _LOG_ACTIVE_THREADS,
    partials_per_lane = 1 << log_partials_per_lane,

    // Number of raking threads
    log_raking_threads = _LOG_RAKING_THREADS,
    raking_threads = 1 << log_raking_threads,

    // Number of raking threads per lane
    log_raking_threads_per_lane =
      log_raking_threads - log_scan_lanes, // must be positive!
    raking_threads_per_lane = 1 << log_raking_threads_per_lane,

    // Partials to be raked per raking thread
    log_partials_per_segment =
      log_partials_per_lane - log_raking_threads_per_lane,
    partials_per_segment = 1 << log_partials_per_segment,

    // Number of partials that we can put in one stripe across the shared memory
    // banks
    log_partials_per_bank_array =
      util::math::log2(util::properties::shared_memory_banks()) +
      util::math::log2(util::properties::shared_memory_bank_stride()) -
      util::math::log2(sizeof(value_t)),
    partials_per_bank_array = 1 << log_partials_per_bank_array,

    log_segments_per_bank_array =
      util::limits::max(0,
                        log_partials_per_bank_array - log_partials_per_segment),
    segments_per_bank_array = 1 << log_segments_per_bank_array,

    // Whether or not one warp of raking threads can rake entirely in one stripe
    // across the shared memory banks
    no_padding =
      (log_segments_per_bank_array >=
       util::math::log2(util::properties::maximum_threads_per_warp())),

    // Number of raking segments we can have without padding (i.e., a "row")
    log_segments_per_row =
      (no_padding) ? log_raking_threads : // All raking threads (segments)
        util::limits::min(
          log_raking_threads_per_lane,
          log_segments_per_bank_array), // Up to as many segments per lane (all
                                        // lanes must have same amount of
                                        // padding to have constant lane stride)

    segments_per_row = 1 << log_segments_per_row,

    // Number of partials per row
    log_partials_per_row = log_segments_per_row + log_partials_per_segment,
    partials_per_row = 1 << log_partials_per_row,

    // Number of partials that we must use to "pad out" one memory bank
    log_bank_padding_partials = util::limits::max(
      0,
      util::math::log2(util::properties::shared_memory_bank_stride()) -
        util::math::log2(sizeof(value_t))),
    bank_padding_partials = 1 << log_bank_padding_partials,

    // Number of partials that we must use to "pad out" a lane to one memory
    // bank
    lane_padding_partials =
      util::limits::max(0, partials_per_bank_array - partials_per_lane),

    // Number of partials (including padding) per "row"
    padded_partials_per_row =
      (no_padding)
        ? partials_per_row
        : partials_per_row + lane_padding_partials + bank_padding_partials,

    // Number of rows in a grid
    log_rows = log_raking_threads - log_segments_per_row,
    rows = 1 << log_rows,

    // Number of rows per lane (always at least one)
    log_rows_per_lane =
      util::limits::max(0, log_raking_threads_per_lane - log_segments_per_row),
    rows_per_lane = 1 << log_rows_per_lane,

    // Padded stride between lanes (in partials)
    lane_stride = (no_padding) ? partials_per_lane
                               : rows_per_lane * padded_partials_per_row,

    // Number of elements needed to back this level of the raking grid
    raking_elements = rows * padded_partials_per_row,
  }; // enum
};

} // namespace grid

} // namespace device
} // namespace raking

} // namespace algo
} // namespace gunrock