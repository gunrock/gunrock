// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cooperative_soa_scan.cuh
 *
 * @brief Cooperative tile SOA (structure-of-arrays) scan within CTAs
 */

#pragma once

#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/reduction/soa/cooperative_soa_reduction.cuh>
#include <gunrock/util/scan/soa/serial_soa_scan.cuh>
#include <gunrock/util/scan/soa/warp_soa_scan.cuh>

namespace gunrock {
namespace util {
namespace scan {
namespace soa {

/**
 * Cooperative SOA reduction in raking smem grid hierarchies
 */
template <typename RakingSoaDetails,
          typename SecondaryRakingSoaDetails =
              typename RakingSoaDetails::SecondaryRakingSoaDetails>
struct CooperativeSoaGridScan;

/**
 * Cooperative SOA tile scan
 */
template <int VEC_SIZE,  // Length of vector-loads (e.g, vec-1, vec-2, vec-4)
          bool EXCLUSIVE = true>  // Whether or not this is an exclusive scan
struct CooperativeSoaTileScan {
  //---------------------------------------------------------------------
  // Iteration structures for extracting partials from raking lanes and
  // using them to seed scans of tile vectors
  //---------------------------------------------------------------------

  // Next lane/load
  template <int LANE, int TOTAL_LANES>
  struct ScanLane {
    template <typename RakingSoaDetails, typename TileSoa, typename ReductionOp>
    static __device__ __forceinline__ void Invoke(
        RakingSoaDetails raking_soa_details, TileSoa tile_soa,
        ReductionOp scan_op) {
      // Retrieve partial reduction from raking grid
      typename RakingSoaDetails::TileTuple exclusive_partial;
      raking_soa_details.lane_partials.Get(exclusive_partial, LANE, 0);

      // Scan the partials in this lane/load
      SerialSoaScan<VEC_SIZE, EXCLUSIVE>::Scan(tile_soa, exclusive_partial,
                                               LANE, scan_op);

      // Next load
      ScanLane<LANE + 1, TOTAL_LANES>::Invoke(raking_soa_details, tile_soa,
                                              scan_op);
    }
  };

  // Terminate
  template <int TOTAL_LANES>
  struct ScanLane<TOTAL_LANES, TOTAL_LANES> {
    template <typename RakingSoaDetails, typename TileSoa, typename ReductionOp>
    static __device__ __forceinline__ void Invoke(
        RakingSoaDetails raking_soa_details, TileSoa tile_soa,
        ReductionOp scan_op) {}
  };

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /*
   * Scan a single tile where carry is assigned (or updated if REDUCE_INTO_CARRY
   * is set) with the total aggregate only in raking threads.
   *
   * No post-synchronization needed before grid reuse.
   */
  template <bool REDUCE_INTO_CARRY, typename RakingSoaDetails, typename TileSoa,
            typename TileTuple, typename ReductionOp>
  static __device__ __forceinline__ void ScanTileWithCarry(
      RakingSoaDetails raking_soa_details, TileSoa tile_soa, TileTuple &carry,
      ReductionOp scan_op) {
    // Reduce vectors in tile, placing resulting partial into corresponding
    // raking grid lanes
    reduction::soa::CooperativeSoaTileReduction<VEC_SIZE>::template ReduceLane<
        0, RakingSoaDetails::SCAN_LANES>::Invoke(raking_soa_details, tile_soa,
                                                 scan_op);

    __syncthreads();

    CooperativeSoaGridScan<RakingSoaDetails>::template ScanTileWithCarry<
        REDUCE_INTO_CARRY>(raking_soa_details, carry, scan_op);

    __syncthreads();

    // Scan partials in tile, retrieving resulting partial from raking grid lane
    // partial
    ScanLane<0, RakingSoaDetails::SCAN_LANES>::Invoke(raking_soa_details,
                                                      tile_soa, scan_op);
  }

  /*
   * Scan a single tile.  Total aggregate is computed and returned in all
   * threads.
   *
   * No post-synchronization needed before grid reuse.
   */
  template <typename RakingSoaDetails, typename TileSoa, typename TileTuple,
            typename ReductionOp>
  static __device__ __forceinline__ void ScanTile(
      TileTuple &retval, RakingSoaDetails raking_soa_details, TileSoa tile_soa,
      ReductionOp scan_op) {
    // Reduce vectors in tile, placing resulting partial into corresponding
    // raking grid lanes
    reduction::soa::CooperativeSoaTileReduction<VEC_SIZE>::template ReduceLane<
        0, RakingSoaDetails::SCAN_LANES>::Invoke(raking_soa_details, tile_soa,
                                                 scan_op);

    __syncthreads();

    CooperativeSoaGridScan<RakingSoaDetails>::ScanTile(raking_soa_details,
                                                       scan_op);

    __syncthreads();

    // Scan partials in tile, retrieving resulting partial from raking grid lane
    // partial
    ScanLane<0, RakingSoaDetails::SCAN_LANES>::Invoke(raking_soa_details,
                                                      tile_soa, scan_op);

    // Return last thread's inclusive partial
    retval = raking_soa_details.CumulativePartial();
  }
};

/******************************************************************************
 * CooperativeSoaGridScan
 ******************************************************************************/

/**
 * Cooperative SOA raking grid reduction (specialized for last-level of raking
 * grid)
 */
template <typename RakingSoaDetails>
struct CooperativeSoaGridScan<RakingSoaDetails, NullType> {
  typedef typename RakingSoaDetails::TileTuple TileTuple;

  /*
   * Scan in last-level raking grid.
   */
  template <typename ReductionOp>
  static __device__ __forceinline__ void ScanTile(
      RakingSoaDetails raking_soa_details, ReductionOp scan_op) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Raking reduction
      TileTuple inclusive_partial;
      reduction::soa::SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::
          Reduce(inclusive_partial, raking_soa_details.raking_segments,
                 scan_op);

      // Exclusive warp scan
      TileTuple exclusive_partial =
          WarpSoaScan<RakingSoaDetails::LOG_RAKING_THREADS>::Scan(
              inclusive_partial, raking_soa_details.warpscan_partials, scan_op);

      // Exclusive raking scan
      SerialSoaScan<RakingSoaDetails::PARTIALS_PER_SEG>::Scan(
          raking_soa_details.raking_segments, exclusive_partial, scan_op);
    }
  }

  /*
   * Scan in last-level raking grid.  Carry-in/out is updated only in raking
   * threads (homogeneously)
   */
  template <bool REDUCE_INTO_CARRY, typename ReductionOp>
  static __device__ __forceinline__ void ScanTileWithCarry(
      RakingSoaDetails raking_soa_details, TileTuple &carry,
      ReductionOp scan_op) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Raking reduction
      TileTuple inclusive_partial;
      reduction::soa::SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::
          Reduce(inclusive_partial, raking_soa_details.raking_segments,
                 scan_op);

      // Exclusive warp scan, get total
      TileTuple warpscan_total;
      TileTuple exclusive_partial =
          WarpSoaScan<RakingSoaDetails::LOG_RAKING_THREADS>::Scan(
              inclusive_partial, warpscan_total,
              raking_soa_details.warpscan_partials, scan_op);

      // Seed exclusive partial with carry-in
      if (REDUCE_INTO_CARRY) {
        if (!ReductionOp::IDENTITY_STRIDES && (threadIdx.x == 0)) {
          // Thread-zero can't use the exclusive partial from the warpscan
          // because it contains garbage
          exclusive_partial = carry;

        } else {
          // Seed exclusive partial with the carry partial
          exclusive_partial = scan_op(carry, exclusive_partial);
        }

        // Update carry
        carry = scan_op(carry, warpscan_total);

      } else {
        // Set carry
        carry = warpscan_total;
      }

      // Exclusive raking scan
      SerialSoaScan<RakingSoaDetails::PARTIALS_PER_SEG>::Scan(
          raking_soa_details.raking_segments, exclusive_partial, scan_op);
    }
  }
};

/**
 * Cooperative SOA raking grid reduction (specialized for last-level of raking
 * grid)
 */
template <typename RakingSoaDetails, typename SecondaryRakingSoaDetails>
struct CooperativeSoaGridScan {
  typedef typename RakingSoaDetails::TileTuple TileTuple;

  /*
   * Scan in last-level raking grid.
   */
  template <typename ReductionOp>
  static __device__ __forceinline__ void ScanTile(
      RakingSoaDetails raking_soa_details, ReductionOp scan_op) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Raking reduction
      TileTuple inclusive_partial;
      reduction::soa::SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::
          Reduce(inclusive_partial, raking_soa_details.raking_segments,
                 scan_op);

      // Store partial reduction into next raking grid
      raking_soa_details.secondary_details.lane_partials.Set(inclusive_partial,
                                                             0, 0);
    }

    __syncthreads();

    // Collectively scan in next grid
    CooperativeSoaGridScan<SecondaryRakingSoaDetails>::ScanTile(
        raking_soa_details.secondary_details, scan_op);

    __syncthreads();

    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Retrieve partial reduction from next raking grid
      TileTuple exclusive_partial;
      raking_soa_details.secondary_details.lane_partials.Get(exclusive_partial,
                                                             0, 0);

      // Exclusive raking scan
      SerialSoaScan<RakingSoaDetails::PARTIALS_PER_SEG>::Scan(
          raking_soa_details.raking_segments, exclusive_partial, scan_op);
    }
  }

  /*
   * Scan in last-level raking grid.  Carry-in/out is updated only in raking
   * threads (homogeneously)
   */
  template <bool REDUCE_INTO_CARRY, typename ReductionOp>
  static __device__ __forceinline__ void ScanTileWithCarry(
      RakingSoaDetails raking_soa_details, TileTuple &carry,
      ReductionOp scan_op) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Raking reduction
      TileTuple inclusive_partial;
      reduction::soa::SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::
          Reduce(inclusive_partial, raking_soa_details.raking_segments,
                 scan_op);

      // Store partial reduction into next raking grid
      raking_soa_details.secondary_details.lane_partials.Set(inclusive_partial,
                                                             0, 0);
    }

    __syncthreads();

    // Collectively scan in next grid
    CooperativeSoaGridScan<SecondaryRakingSoaDetails>::
        template ScanTileWithCarry<REDUCE_INTO_CARRY>(
            raking_soa_details.secondary_details, carry, scan_op);

    __syncthreads();

    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Retrieve partial reduction from next raking grid
      TileTuple exclusive_partial;
      raking_soa_details.secondary_details.lane_partials.Get(exclusive_partial,
                                                             0, 0);

      // Exclusive raking scan
      SerialSoaScan<RakingSoaDetails::PARTIALS_PER_SEG>::Scan(
          raking_soa_details.raking_segments, exclusive_partial, scan_op);
    }
  }
};

}  // namespace soa
}  // namespace scan
}  // namespace util
}  // namespace gunrock
