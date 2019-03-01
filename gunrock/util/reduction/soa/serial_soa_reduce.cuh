// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * serial_soa_reduce.cuh
 *
 * @brief Serial tuple reduction over structure-of-array types.
 */

#pragma once

#include <gunrock/util/soa_tuple.cuh>

namespace gunrock {
namespace util {
namespace reduction {
namespace soa {

/**
 * Have each thread perform a serial reduction over its specified SOA segment
 */
template <int NUM_ELEMENTS>  // Length of SOA array segment to reduce
struct SerialSoaReduce {
  //---------------------------------------------------------------------
  // Iteration Structures
  //---------------------------------------------------------------------

  // Next SOA tuple
  template <int COUNT, int TOTAL>
  struct Iterate {
    // Reduce
    template <typename Tuple, typename RakingSoa, typename ReductionOp>
    static __host__ __device__ __forceinline__ Tuple
    Reduce(RakingSoa raking_partials, Tuple exclusive_partial,
           ReductionOp reduction_op) {
      // Load current partial
      Tuple current_partial;
      raking_partials.Get(current_partial, COUNT);

      // Compute inclusive partial from exclusive and current partials
      Tuple inclusive_partial =
          reduction_op(exclusive_partial, current_partial);

      // Recurse
      return Iterate<COUNT + 1, TOTAL>::Reduce(raking_partials,
                                               inclusive_partial, reduction_op);
    }

    // Reduce 2D
    template <typename Tuple, typename RakingSoa, typename ReductionOp>
    static __host__ __device__ __forceinline__ Tuple
    Reduce(RakingSoa raking_partials, Tuple exclusive_partial, int row,
           ReductionOp reduction_op) {
      // Load current partial
      Tuple current_partial;
      raking_partials.Get(current_partial, row, COUNT);

      // Compute inclusive partial from exclusive and current partials
      Tuple inclusive_partial =
          reduction_op(exclusive_partial, current_partial);

      // Recurse
      return Iterate<COUNT + 1, TOTAL>::Reduce(
          raking_partials, inclusive_partial, row, reduction_op);
    }
  };

  // Terminate
  template <int TOTAL>
  struct Iterate<TOTAL, TOTAL> {
    // Reduce
    template <typename Tuple, typename RakingSoa, typename ReductionOp>
    static __host__ __device__ __forceinline__ Tuple
    Reduce(RakingSoa raking_partials, Tuple exclusive_partial,
           ReductionOp reduction_op) {
      return exclusive_partial;
    }

    // Reduce 2D
    template <typename Tuple, typename RakingSoa, typename ReductionOp>
    static __host__ __device__ __forceinline__ Tuple
    Reduce(RakingSoa raking_partials, Tuple exclusive_partial, int row,
           ReductionOp reduction_op) {
      return exclusive_partial;
    }
  };

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /*
   * Reduce a structure-of-array RakingSoa into a single Tuple "slice"
   */
  template <typename Tuple, typename RakingSoa, typename ReductionOp>
  static __host__ __device__ __forceinline__ void Reduce(
      Tuple &retval, RakingSoa raking_partials, ReductionOp reduction_op) {
    // Get first partial
    Tuple current_partial;
    raking_partials.Get(current_partial, 0);

    retval = Iterate<1, NUM_ELEMENTS>::Reduce(raking_partials, current_partial,
                                              reduction_op);
  }

  /*
   * Reduce a structure-of-array RakingSoa into a single Tuple "slice", seeded
   * with exclusive_partial
   */
  template <typename Tuple, typename RakingSoa, typename ReductionOp>
  static __host__ __device__ __forceinline__ Tuple
  SeedReduce(RakingSoa raking_partials, Tuple exclusive_partial,
             ReductionOp reduction_op) {
    return Iterate<0, NUM_ELEMENTS>::Reduce(raking_partials, exclusive_partial,
                                            reduction_op);
  }

  /*
   * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple
   * "slice"
   */
  template <typename Tuple, typename RakingSoa, typename ReductionOp>
  static __host__ __device__ __forceinline__ void Reduce(
      Tuple &retval, RakingSoa raking_partials, int row,
      ReductionOp reduction_op) {
    // Get first partial
    Tuple current_partial;
    raking_partials.Get(current_partial, row, 0);

    retval = Iterate<1, NUM_ELEMENTS>::Reduce(raking_partials, current_partial,
                                              row, reduction_op);
  }

  /*
   * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple
   * "slice", seeded with exclusive_partial
   */
  template <typename Tuple, typename RakingSoa, typename ReductionOp>
  static __host__ __device__ __forceinline__ Tuple
  SeedReduce(RakingSoa raking_partials, Tuple exclusive_partial, int row,
             ReductionOp reduction_op) {
    return Iterate<0, NUM_ELEMENTS>::Reduce(raking_partials, exclusive_partial,
                                            row, reduction_op);
  }
};

}  // namespace soa
}  // namespace reduction
}  // namespace util
}  // namespace gunrock
