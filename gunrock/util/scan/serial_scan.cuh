// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * serial_scan.cuh
 *
 * @brief Serial Scan Over Array Types
 */

#pragma once

#include <gunrock/util/operators.cuh>

namespace gunrock {
namespace util {
namespace scan {

/**
 * Have each thread perform a serial scan over its specified segment.
 */
template <int NUM_ELEMENTS,       // Length of array segment to scan
          bool EXCLUSIVE = true>  // Whether or not this is an exclusive scan
struct SerialScan {
  //---------------------------------------------------------------------
  // Iteration Structures
  //---------------------------------------------------------------------

  // Iterate
  template <int COUNT, int TOTAL>
  struct Iterate {
    template <typename T, typename ReductionOp>
    static __device__ __forceinline__ T Invoke(T partials[], T results[],
                                               T exclusive_partial,
                                               ReductionOp scan_op) {
      T inclusive_partial = scan_op(partials[COUNT], exclusive_partial);
      results[COUNT] = (EXCLUSIVE) ? exclusive_partial : inclusive_partial;
      return Iterate<COUNT + 1, TOTAL>::Invoke(partials, results,
                                               inclusive_partial, scan_op);
    }
  };

  // Terminate
  template <int TOTAL>
  struct Iterate<TOTAL, TOTAL> {
    template <typename T, typename ReductionOp>
    static __device__ __forceinline__ T Invoke(T partials[], T results[],
                                               T exclusive_partial,
                                               ReductionOp scan_op) {
      return exclusive_partial;
    }
  };

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /*
   * Serial scan with the specified operator
   */
  template <typename T, typename ReductionOp>
  static __device__ __forceinline__ T
  Invoke(T partials[],
         T exclusive_partial,  // Exclusive partial to seed with
         ReductionOp scan_op) {
    return Iterate<0, NUM_ELEMENTS>::Invoke(partials, partials,
                                            exclusive_partial, scan_op);
  }

  /*
   * Serial scan with the addition operator
   */
  template <typename T>
  static __device__ __forceinline__ T
  Invoke(T partials[],
         T exclusive_partial)  // Exclusive partial to seed with
  {
    Sum<T> reduction_op;
    return Invoke(partials, exclusive_partial, reduction_op);
  }

  /*
   * Serial scan with the specified operator
   */
  template <typename T, typename ReductionOp>
  static __device__ __forceinline__ T
  Invoke(T partials[], T results[],
         T exclusive_partial,  // Exclusive partial to seed with
         ReductionOp scan_op) {
    return Iterate<0, NUM_ELEMENTS>::Invoke(partials, results,
                                            exclusive_partial, scan_op);
  }

  /*
   * Serial scan with the addition operator
   */
  template <typename T>
  static __device__ __forceinline__ T
  Invoke(T partials[], T results[],
         T exclusive_partial)  // Exclusive partial to seed with
  {
    Sum<T> reduction_op;
    return Invoke(partials, results, exclusive_partial, reduction_op);
  }
};

}  // namespace scan
}  // namespace util
}  // namespace gunrock
