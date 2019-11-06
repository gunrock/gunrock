// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * srts_soa_details.cuh
 *
 * @brief Operational Details for Threads Working in an SOA (structure of
 * arrays) Raking Grid
 */

#pragma once

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

/**
 * Operational details for threads working in an raking grid
 */
template <
    typename TileTuple, typename RakingGridTuple,
    int Grids = RakingGridTuple::NUM_FIELDS,
    typename SecondaryRakingGridTuple = typename If<
        Equals<NullType, typename RakingGridTuple::T0::SecondaryGrid>::VALUE,
        NullType,
        Tuple<typename RakingGridTuple::T0::SecondaryGrid,
              typename RakingGridTuple::T1::SecondaryGrid> >::Type>
struct RakingSoaDetails;

/**
 * Two-field raking details
 */
template <typename _TileTuple, typename RakingGridTuple>
struct RakingSoaDetails<_TileTuple, RakingGridTuple, 2, NullType>
    : RakingGridTuple::T0 {
  enum {
    CUMULATIVE_THREAD = RakingSoaDetails::RAKING_THREADS - 1,
    WARP_THREADS = GR_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
  };

  // Simple SOA tuple "slice" type
  typedef _TileTuple TileTuple;

  // SOA type of raking lanes
  typedef Tuple<typename TileTuple::T0*, typename TileTuple::T1*>
      GridStorageSoa;

  // SOA type of warpscan storage
  typedef Tuple<typename RakingGridTuple::T0::WarpscanT (*)[WARP_THREADS],
                typename RakingGridTuple::T1::WarpscanT (*)[WARP_THREADS]>
      WarpscanSoa;

  // SOA type of partial-insertion pointers
  typedef Tuple<typename RakingGridTuple::T0::LanePartial,
                typename RakingGridTuple::T1::LanePartial>
      LaneSoa;

  // SOA type of raking segments
  typedef Tuple<typename RakingGridTuple::T0::RakingSegment,
                typename RakingGridTuple::T1::RakingSegment>
      RakingSoa;

  typedef NullType SecondaryRakingSoaDetails;

  /**
   * Warpscan storages
   */
  WarpscanSoa warpscan_partials;

  /**
   * Lane insertion/extraction pointers.
   */
  LaneSoa lane_partials;

  /**
   * Raking pointers
   */
  RakingSoa raking_segments;

  // Constructor
  __host__ __device__ __forceinline__
  RakingSoaDetails(GridStorageSoa smem_pools, WarpscanSoa warpscan_partials)
      :

        warpscan_partials(warpscan_partials),
        lane_partials(  // set lane partial pointer
            RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
            RakingGridTuple::T1::MyLanePartial(smem_pools.t1)) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Set raking segment pointers
      raking_segments =
          RakingSoa(RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
                    RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
    }
  }

  // Constructor
  __host__ __device__ __forceinline__
  RakingSoaDetails(GridStorageSoa smem_pools, WarpscanSoa warpscan_partials,
                   TileTuple soa_tuple_identity)
      :

        warpscan_partials(warpscan_partials),
        lane_partials(  // set lane partial pointer
            RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
            RakingGridTuple::T1::MyLanePartial(smem_pools.t1)) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Set raking segment pointers
      raking_segments =
          RakingSoa(RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
                    RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));

      // Initialize first half of warpscan storages to identity
      warpscan_partials.Set(soa_tuple_identity, 0, threadIdx.x);
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  __device__ __forceinline__ TileTuple CumulativePartial() {
    TileTuple retval;
    warpscan_partials.Get(retval, 1, CUMULATIVE_THREAD);
    return retval;
  }
};

/**
 * Two-field raking details
 */
template <typename _TileTuple, typename RakingGridTuple,
          typename SecondaryRakingGridTuple>
struct RakingSoaDetails<_TileTuple, RakingGridTuple, 2,
                        SecondaryRakingGridTuple> : RakingGridTuple::T0 {
  enum {
    CUMULATIVE_THREAD = RakingSoaDetails::RAKING_THREADS - 1,
    WARP_THREADS = GR_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
  };

  // Simple SOA tuple "slice" type
  typedef _TileTuple TileTuple;

  // SOA type of raking lanes
  typedef Tuple<typename TileTuple::T0*, typename TileTuple::T1*>
      GridStorageSoa;

  // SOA type of warpscan storage
  typedef Tuple<typename RakingGridTuple::T0::WarpscanT (*)[WARP_THREADS],
                typename RakingGridTuple::T1::WarpscanT (*)[WARP_THREADS]>
      WarpscanSoa;

  // SOA type of partial-insertion pointers
  typedef Tuple<typename RakingGridTuple::T0::LanePartial,
                typename RakingGridTuple::T1::LanePartial>
      LaneSoa;

  // SOA type of raking segments
  typedef Tuple<typename RakingGridTuple::T0::RakingSegment,
                typename RakingGridTuple::T1::RakingSegment>
      RakingSoa;

  // SOA type of secondary details
  typedef RakingSoaDetails<TileTuple, SecondaryRakingGridTuple>
      SecondaryRakingSoaDetails;

  /**
   * Lane insertion/extraction pointers.
   */
  LaneSoa lane_partials;

  /**
   * Raking pointers
   */
  RakingSoa raking_segments;

  /**
   * Secondary-level grid details
   */
  SecondaryRakingSoaDetails secondary_details;

  // Constructor
  __host__ __device__ __forceinline__
  RakingSoaDetails(GridStorageSoa smem_pools,
                   WarpscanSoa warpscan_partials)
      :

        lane_partials(  // set lane partial pointer
            RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
            RakingGridTuple::T1::MyLanePartial(smem_pools.t1)),
        secondary_details(
            GridStorageSoa(
                smem_pools.t0 + RakingGridTuple::T0::RAKING_ELEMENTS,
                smem_pools.t1 + RakingGridTuple::T1::RAKING_ELEMENTS),
            warpscan_partials) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Set raking segment pointers
      raking_segments =
          RakingSoa(RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
                    RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
    }
  }

  // Constructor
  __host__ __device__ __forceinline__
  RakingSoaDetails(GridStorageSoa smem_pools, WarpscanSoa warpscan_partials,
                   TileTuple soa_tuple_identity)
      :

        lane_partials(  // set lane partial pointer
            RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
            RakingGridTuple::T1::MyLanePartial(smem_pools.t1)),
        secondary_details(
            GridStorageSoa(
                smem_pools.t0 + RakingGridTuple::T0::RAKING_ELEMENTS,
                smem_pools.t1 + RakingGridTuple::T1::RAKING_ELEMENTS),
            warpscan_partials) {
    if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {
      // Set raking segment pointers
      raking_segments =
          RakingSoa(RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
                    RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
    }
  }

  // Return the cumulative partial left in the final warpscan cell
  __device__ __forceinline__ TileTuple CumulativePartial() {
    return secondary_details.CumulativePartial();
  }
};

}  // namespace util
}  // namespace gunrock
