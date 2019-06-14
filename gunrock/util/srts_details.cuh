// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * srts_details.cuh
 *
 * @brief Operational details for threads working in an raking grid
 */

#pragma once

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

/**
 * Operational details for threads working in an raking grid
 */
template <typename RakingGrid,
          typename SecondaryRakingGrid = typename RakingGrid::SecondaryGrid>
struct RakingDetails;

/**
 * Operational details for threads working in an raking grid (specialized for
 * one-level raking grid)
 */
template <typename RakingGrid>
struct RakingDetails<RakingGrid, NullType> : RakingGrid {
  enum {
    QUEUE_RSVN_THREAD = 0,
    CUMULATIVE_THREAD = RakingGrid::RAKING_THREADS - 1,
    WARP_THREADS = GR_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
  };

  typedef typename RakingGrid::T T;  // Partial type
  typedef typename RakingGrid::WarpscanT (
      *WarpscanStorage)[WARP_THREADS];  // Warpscan storage type
  typedef NullType
      SecondaryRakingDetails;  // Type of next-level grid raking details

  /**
   * Smem pool backing raking grid lanes
   */
  T *smem_pool;

  /**
   * Warpscan storage
   */
  WarpscanStorage warpscan;

  /**
   * The location in the smem grid where the calling thread can insert/extract
   * its partial for raking reduction/scan into the first lane.
   */
  typename RakingGrid::LanePartial lane_partial;

  /**
   * Returns the location in the smem grid where the calling thread can begin
   * serial raking/scanning
   */
  typename RakingGrid::RakingSegment raking_segment;

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool)
      : smem_pool(smem_pool),
        lane_partial(
            RakingGrid::MyLanePartial(smem_pool))  // set lane partial pointer
  {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool,
                                           WarpscanStorage warpscan)
      : smem_pool(smem_pool),
        warpscan(warpscan),
        lane_partial(
            RakingGrid::MyLanePartial(smem_pool))  // set lane partial pointer
  {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool,
                                           WarpscanStorage warpscan,
                                           T warpscan_identity)
      : smem_pool(smem_pool),
        warpscan(warpscan),
        lane_partial(
            RakingGrid::MyLanePartial(smem_pool))  // set lane partial pointer
  {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Initialize first half of warpscan storage to identity
      warpscan[0][threadIdx.x] = warpscan_identity;

      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  __device__ __forceinline__ T CumulativePartial() const {
    return warpscan[1][CUMULATIVE_THREAD];
  }

  // Return the queue reservation in the first warpscan cell
  __device__ __forceinline__ T QueueReservation() const {
    return warpscan[1][QUEUE_RSVN_THREAD];
  }

  __device__ __forceinline__ T *SmemPool() { return smem_pool; }
};

/**
 * Operational details for threads working in a hierarchical raking grid
 */
template <typename RakingGrid, typename SecondaryRakingGrid>
struct RakingDetails : RakingGrid {
  enum {
    CUMULATIVE_THREAD = RakingGrid::RAKING_THREADS - 1,
    WARP_THREADS = GR_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
  };

  typedef typename RakingGrid::T T;  // Partial type
  typedef typename RakingGrid::WarpscanT (
      *WarpscanStorage)[WARP_THREADS];  // Warpscan storage type
  typedef RakingDetails<SecondaryRakingGrid>
      SecondaryRakingDetails;  // Type of next-level grid raking details

  /**
   * The location in the smem grid where the calling thread can insert/extract
   * its partial for raking reduction/scan into the first lane.
   */
  typename RakingGrid::LanePartial lane_partial;

  /**
   * Returns the location in the smem grid where the calling thread can begin
   * serial raking/scanning
   */
  typename RakingGrid::RakingSegment raking_segment;

  /**
   * Secondary-level grid details
   */
  SecondaryRakingDetails secondary_details;

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool)
      : lane_partial(
            RakingGrid::MyLanePartial(smem_pool)),  // set lane partial pointer
        secondary_details(smem_pool + RakingGrid::RAKING_ELEMENTS) {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool,
                                           WarpscanStorage warpscan)
      : lane_partial(
            RakingGrid::MyLanePartial(smem_pool)),  // set lane partial pointer
        secondary_details(smem_pool + RakingGrid::RAKING_ELEMENTS, warpscan) {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Constructor
  __device__ __forceinline__ RakingDetails(T *smem_pool,
                                           WarpscanStorage warpscan,
                                           T warpscan_identity)
      : lane_partial(
            RakingGrid::MyLanePartial(smem_pool)),  // set lane partial pointer
        secondary_details(smem_pool + RakingGrid::RAKING_ELEMENTS, warpscan,
                          warpscan_identity) {
    if (threadIdx.x < RakingGrid::RAKING_THREADS) {
      // Set raking segment pointer
      raking_segment = RakingGrid::MyRakingSegment(smem_pool);
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  __device__ __forceinline__ T CumulativePartial() const {
    return secondary_details.CumulativePartial();
  }

  // Return the queue reservation in the first warpscan cell
  __device__ __forceinline__ T QueueReservation() const {
    return secondary_details.QueueReservation();
  }

  /**
   *
   */
  __device__ __forceinline__ T *SmemPool() {
    return secondary_details.SmemPool();
  }
};

}  // namespace util
}  // namespace gunrock
