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
#include <gunrock/util/device_properties.hxx>
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

/**
 * @brief
 *
 * @tparam GUNROCK_CUDA_ARCH
 * @tparam _LOG_ACTIVE_THREADS Number of threads placing a lane partial (i.e.,
 * the number of partials per lane)
 * @tparam _LOG_SCAN_LANES Number of scan lanes
 * @tparam _LOG_RAKING_THREADS Number of threads used for raking (typically 1
 * warp)
 * @tparam _DEPENDENT_LANES If there are prefix dependences between lanes (i.e.,
 * downsweeping will incorporate aggregates from previous lanes)
 * @tparam reducing/scanning Type of items we will be reducing/scanning
 */
template<
  // architecture_t GUNROCK_CUDA_ARCH,
  unsigned _LOG_ACTIVE_THREADS,
  unsigned _LOG_SCAN_LANES,
  unsigned _LOG_RAKING_THREADS,
  bool _DEPENDENT_LANES,
  typename raking_value_t>
struct raking
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
   * @note (06/02/2020) removed enum
   */

  // Number of scan lanes
  auto log_scan_lanes = _LOG_SCAN_LANES;
  auto scan_lanes = 1 << log_scan_lanes;

  // Number of partials per lange
  auto log_partials_per_lane = _LOG_ACTIVE_THREADS;
  auto partials_per_lane = 1 << log_partials_per_lane;

  // Number of raking threads
  auto log_raking_threads = _LOG_RAKING_THREADS;
  auto raking_threads = 1 << log_raking_threads;

  // Number of raking threads per lane
  auto log_raking_threads_per_lane =
    log_raking_threads - log_scan_lanes; // must be positive!
  auto raking_threads_per_lane = 1 << log_raking_threads_per_lane;

  // Partials to be raked per raking thread
  auto log_partials_per_segment =
    log_partials_per_lane - log_raking_threads_per_lane;
  auto partials_per_segment = 1 << log_partials_per_segment;

  // Number of partials that we can put in one stripe across the shared memory
  // banks
  auto log_partials_per_bank_array =
    util::math::log2(util::properties::shared_memory_banks()) +
    util::math::log2(util::properties::shared_memory_bank_stride()) -
    util::math::log2(sizeof(value_t));
  auto partials_per_bank_array = 1 << log_partials_per_bank_array;

  auto log_segments_per_bank_array =
    util::limits::max(0,
                      log_partials_per_bank_array - log_partials_per_segment);
  auto segments_per_bank_array = 1 << log_segments_per_bank_array;

  // Whether or not one warp of raking threads can rake entirely in one stripe
  // across the shared memory banks
  auto no_padding =
    (log_segments_per_bank_array >=
     util::math::log2(util::properties::maximum_threads_per_warp()));

  // Number of raking segments we can have without padding (i.e., a "row")
  auto log_segments_per_row =
    (no_padding) ? log_raking_threads : // All raking threads (segments)
      util::limits::min(
        log_raking_threads_per_lane,
        log_segments_per_bank_array); // Up to as many segments per lane (all
                                      // lanes must have same amount of
                                      // padding to have constant lane stride)

  auto segments_per_row = 1 << log_segments_per_row;

  // Number of partials per row
  auto log_partials_per_row = log_segments_per_row + log_partials_per_segment;
  auto partials_per_row = 1 << log_partials_per_row;

  // Number of partials that we must use to "pad out" one memory bank
  auto log_bank_padding_partials = util::limits::max(
    0,
    util::math::log2(util::properties::shared_memory_bank_stride()) -
      util::math::log2(sizeof(value_t)));
  auto bank_padding_partials = 1 << log_bank_padding_partials;

  // Number of partials that we must use to "pad out" a lane to one memory
  // bank
  auto lane_padding_partials =
    util::limits::max(0, partials_per_bank_array - partials_per_lane);

  // Number of partials (including padding) per "row"
  auto padded_partials_per_row =
    (no_padding)
      ? partials_per_row
      : partials_per_row + lane_padding_partials + bank_padding_partials;

  // Number of rows in a grid
  auto log_rows = log_raking_threads - log_segments_per_row;
  auto rows = 1 << log_rows;

  // Number of rows per lane (always at least one)
  auto log_rows_per_lane =
    util::limits::max(0, log_raking_threads_per_lane - log_segments_per_row);
  auto rows_per_lane = 1 << log_rows_per_lane;

  // Padded stride between lanes (in partials)
  auto lane_stride =
    (no_padding) ? partials_per_lane : rows_per_lane * padded_partials_per_row;

  // Number of elements needed to back this level of the raking grid
  auto raking_elements = rows * padded_partials_per_row;

  // If there are prefix dependences between lanes, a secondary raking grid
  // type will be needed in the event we have more than one warp of raking
  // threads

  typedef
    typename util::meta::_if<(util::numeric_traits<value_t>::REPRESENTATION ==
                              util::representation_t::not_a_number),
                             value_t,
                             volatile value_t>::type_t warp_scan_t;

  typedef typename util::meta::_if<
    _DEPENDENT_LANES &&
      (log_raking_threads >
       util::math::log2(util::properties::maximum_threads_per_warp())),
    raking<               // Secondary grid
                          // GUNROCK_CUDA_ARCH,
      log_raking_threads, // Depositing threads (the primary raking threads)
      0, // 1 lane (the primary raking threads only make one deposit)
      util::math::log2(
        util::properties::maximum_threads_per_warp()), // Raking threads (1
                                                       // warp)
      false,    // There is only one lane, so there are no inter-lane prefix
                // dependences
      value_t>, // Partial type
    util::meta::null_t> // No secondary grid
    ::type_t secondary_grid_t;

  /**
   * Utility class for totaling the SMEM elements needed for an raking grid
   * hierarchy
   */
  template<typename raking, int __dummy = 0>
  struct total_raking_elements
  {
    // Recurse
    auto VALUE = raking::raking_elements +
                 total_raking_elements<typename raking::secondary_grid_t>;
  };

  template<int __dummy>
  struct total_raking_elements<util::meta::null_t, __dummy>
  {
    // Terminate
    auto VALUE = 0
  };

  // Total number of smem raking elements needed back this hierarchy
  // of raking grids (may be reused for other purposes)
  auto total_raking_elements = total_raking_elements<raking>::VALUE;

  /**
   * Type of pointer for inserting partials into lanes, e.g.,
   * lane_partial[LANE][0] = ...
   */
  typedef value_t (*lane_partial_t)[lane_stride];

  /**
   * Type of pointer for raking across lane segments
   */
  typedef value_t* raking_segment_t;

  // Returns the location in the smem grid where the calling thread can
  // insert/extract its partial for raking reduction/scan into the first lane.
  // Positions in subsequent lanes can be obtained via increments of
  // lane_stride.
  static GUNROCK_HOST_DEVICE lane_partial_t my_lane_partial(value_t* smem)
  {
    int row = threadIdx.x >> log_partials_per_row;
    int col = threadIdx.x & (partials_per_row - 1);

    return (lane_partial_t)(smem + (row * padded_partials_per_row) + col);
  }

  // Returns the location in the smem grid where the calling thread can begin
  // serial raking/scanning
  static GUNROCK_HOST_DEVICE raking_segment_t my_raking_segment(value_t* smem)
  {
    int row = threadIdx.x >> log_segments_per_row;
    int col = (threadIdx.x & (segments_per_row - 1))
              << log_partials_per_segment;
    return (raking_segment_t)(smem + (row * padded_partials_per_row) + col);
  }

  /**
   * Displays configuration to standard out
   */
  static GUNROCK_HOST_DEVICE void print()
  {
    printf("SCAN_LANES: %d\n"
           "PARTIALS_PER_LANE: %d\n"
           "RAKING_THREADS: %d\n"
           "RAKING_THREADS_PER_LANE: %d\n"
           "PARTIALS_PER_SEG: %d\n"
           "PARTIALS_PER_BANK_ARRAY: %d\n"
           "SEGS_PER_BANK_ARRAY: %d\n"
           "NO_PADDING: %d\n"
           "SEGS_PER_ROW: %d\n"
           "PARTIALS_PER_ROW: %d\n"
           "BANK_PADDING_PARTIALS: %d\n"
           "LANE_PADDING_PARTIALS: %d\n"
           "PADDED_PARTIALS_PER_ROW: %d\n"
           "ROWS: %d\n"
           "ROWS_PER_LANE: %d\n"
           "LANE_STRIDE: %d\n"
           "RAKING_ELEMENTS: %d\n",
           scan_lanes,
           partials_per_lane,
           raking_threads,
           raking_threads_per_lane,
           partials_per_segment,
           partials_per_bank_array,
           segments_per_bank_array,
           no_padding,
           segments_per_row,
           partials_per_row,
           bank_padding_partials,
           lane_padding_partials,
           padded_partials_per_row,
           rows,
           rows_per_lane,
           lane_stride,
           raking_elements);
  }

}; // struct raking

} // namespace grid

namespace details {

/**
 * Operational details for threads working in an raking grid
 */
template<typename grid::raking,
         typename secondary_raking_t = typename grid::raking::secondary_grid_t>
struct raking;

/**
 * Operational details for threads working in an raking grid (specialized for
 * one-level raking grid)
 */
template<typename grid::raking>
struct raking<grid::raking, util::meta::null_t> : grid::raking
{
  auto queue_rsvn_thread = 0;
  auto cumulative_thread = grid::raking::raking_threads - 1;
  auto warp_threads = util::properties::maximum_threads_per_warp();

  typedef typename grid::raking::value_t value_t; // Partial type
  typedef typename grid::raking::warpscan_t (
    *warpscan_storage_t)[warp_threads]; // Warpscan storage type
  typedef typename util::meta::null_t
    secondary_raking_t; // Type of next-level grid raking details

  /**
   * Smem pool backing raking grid lanes
   */
  value_t* smem_pool;

  /**
   * Warpscan storage
   */
  warpscan_storage_t warpscan;

  /**
   * The location in the smem grid where the calling thread can insert/extract
   * its partial for raking reduction/scan into the first lane.
   */
  typename grid::raking::lane_partial_t lane_partial;

  /**
   * Returns the location in the smem grid where the calling thread can begin
   * serial raking/scanning
   */
  typename grid::raking::raking_segment_t raking_segment;

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool)
    : smem_pool(smem_pool)
    , lane_partial(
        grid::raking::my_lane_partial(smem_pool)) // set lane partial pointer
  {
    if (threadIdx.x < grid::raking::raking_threads) {
      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool, warpscan_storage_t warpscan)
    : smem_pool(smem_pool)
    , warpscan(warpscan)
    , lane_partial(
        grid::raking::my_lane_partial(smem_pool)) // set lane partial pointer
  {
    if (threadIdx.x < grid::raking::raking_threads) {
      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool,
                          warpscan_storage_t warpscan,
                          value_t warpscan_identity)
    : smem_pool(smem_pool)
    , warpscan(warpscan)
    , lane_partial(
        grid::raking::my_lane_partial(smem_pool)) // set lane partial pointer
  {
    if (threadIdx.x < grid::raking::RAKING_THREADS) {
      // Initialize first half of warpscan storage to identity
      warpscan[0][threadIdx.x] = warpscan_identity;

      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  GUNROCK_F_DEVICE value_t cumulative_partial() const
  {
    return warpscan[1][cumulative_thread];
  }

  // Return the queue reservation in the first warpscan cell
  GUNROCK_F_DEVICE value_t queue_reservation() const
  {
    return warpscan[1][queue_rsvn_thread];
  }

  GUNROCK_F_DEVICE value_t* get_smem_pool() { return smem_pool; }

}; // struct raking

/**
 * Operational details for threads working in a hierarchical raking grid
 */
template<typename grid::raking, typename secondary_raking_t>
struct raking : grid::raking
{

  auto cumulative_thread = grid::raking::raking_threads - 1;
  auto warp_threads = util::properties::maximum_threads_per_warp();

  typedef typename grid::raking::value_t value_t; // Partial type
  typedef typename grid::raking::warpscan_t (
    *warpscan_storage_t)[warp_threads]; // Warpscan storage type
  typedef raking<secondary_raking_t>
    secondary_raking_details_t; // Type of next-level grid raking details

  /**
   * The location in the smem grid where the calling thread can insert/extract
   * its partial for raking reduction/scan into the first lane.
   */
  typename grid::raking::lane_partial_t lane_partial;

  /**
   * Returns the location in the smem grid where the calling thread can begin
   * serial raking/scanning
   */
  typename grid::raking::raking_segment_t raking_segment;

  /**
   * Secondary-level grid details
   */
  secondary_raking_details_t secondary_raking;

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool)
    : lane_partial(grid::raking::my_lane_partial(smem_pool))
    , // set lane partial pointer
    secondary_raking(smem_pool + grid::raking::raking_elements)
  {
    if (threadIdx.x < grid::raking::raking_threads) {
      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool,
                          warpscan_storage_t warpscan)
    : lane_partial(grid::raking::my_lane_partial(smem_pool))
    , // set lane partial pointer
    secondary_raking(smem_pool + grid::raking::raking_elements, warpscan)
  {
    if (threadIdx.x < grid::raking::raking_threads) {
      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Constructor
  GUNROCK_F_DEVICE raking(value_t* smem_pool,
                          warpscan_storage_t warpscan,
                          value_t warpscan_identity)
    : lane_partial(grid::raking::my_lane_partial(smem_pool))
    , // set lane partial pointer
    secondary_raking(smem_pool + grid::raking::raking_elements,
                     warpscan,
                     warpscan_identity)
  {
    if (threadIdx.x < grid::raking::raking_threads) {
      // Set raking segment pointer
      raking_segment = grid::raking::my_raking_segment(smem_pool);
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  GUNROCK_F_DEVICE value_t cumulative_partial() const
  {
    return secondary_raking.cumulative_partial();
  }

  // Return the queue reservation in the first warpscan cell
  GUNROCK_F_DEVICE value_t queue_reservation() const
  {
    return secondary_raking.queue_reservation();
  }

  /**
   *
   */
  GUNROCK_F_DEVICE value_t* get_smem_pool()
  {
    return secondary_raking.get_smem_pool();
  }
};

} // namespace details
} // namespace device
} // namespace raking

} // namespace algo
} // namespace gunrock