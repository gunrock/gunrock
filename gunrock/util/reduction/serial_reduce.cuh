// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * serial_reduce.cuh
 *
 * @brief Serial reduction over array types
 */

#pragma once

#include <gunrock/util/operators.cuh>

namespace gunrock {
namespace util {
namespace reduction {

/**
 * Have each thread perform a serial reduction over its specified segment
 */
template <int NUM_ELEMENTS>
struct SerialReduce {
  //---------------------------------------------------------------------
  // Iteration Structures
  //---------------------------------------------------------------------

  // Iterate
  template <int COUNT, int TOTAL>
  struct Iterate {
    template <typename T, typename ReductionOp>
    static __device__ __forceinline__ T Invoke(T *partials,
                                               ReductionOp reduction_op) {
      T a = Iterate<COUNT - 2, TOTAL>::Invoke(partials, reduction_op);
      T b = partials[TOTAL - COUNT];
      T c = partials[TOTAL - (COUNT - 1)];

      // TODO: consider specializing with a video 3-op instructions on SM2.0+,
      // e.g., asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(a) : "r"(a),
      // "r"(b), "r"(c));
      return reduction_op(a, reduction_op(b, c));
    }
  };

  // Terminate
  template <int TOTAL>
  struct Iterate<2, TOTAL> {
    template <typename T, typename ReductionOp>
    static __device__ __forceinline__ T Invoke(T *partials,
                                               ReductionOp reduction_op) {
      return reduction_op(partials[TOTAL - 2], partials[TOTAL - 1]);
    }
  };

  // Terminate
  template <int TOTAL>
  struct Iterate<1, TOTAL> {
    template <typename T, typename ReductionOp>
    static __device__ __forceinline__ T Invoke(T *partials,
                                               ReductionOp reduction_op) {
      return partials[TOTAL - 1];
    }
  };

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /*
   * Serial reduction with the specified operator
   */
  template <typename T, typename ReductionOp>
  static __device__ __forceinline__ T Invoke(T *partials,
                                             ReductionOp reduction_op) {
    return Iterate<NUM_ELEMENTS, NUM_ELEMENTS>::Invoke(partials, reduction_op);
  }

  /*
   * Serial reduction with the addition operator
   */
  template <typename T>
  static __device__ __forceinline__ T Invoke(T *partials) {
    Sum<T> reduction_op;
    return Invoke(partials, reduction_op);
  }

  /*
   * Serial reduction with the specified operator, seeded with the
   * given exclusive partial
   */
  template <typename T, typename ReductionOp>
  static __device__ __forceinline__ T Invoke(T *partials, T exclusive_partial,
                                             ReductionOp reduction_op) {
    return reduction_op(exclusive_partial, Invoke(partials, reduction_op));
  }

  /*
   * Serial reduction with the addition operator, seeded with the
   * given exclusive partial
   */
  template <typename T, typename ReductionOp>
  static __device__ __forceinline__ T Invoke(T *partials, T exclusive_partial) {
    Sum<T> reduction_op;
    return Invoke(partials, exclusive_partial, reduction_op);
  }
};

}  // namespace reduction
}  // namespace util
}  // namespace gunrock
