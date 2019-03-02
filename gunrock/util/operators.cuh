// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * opearators.cuh
 *
 * @brief Simple Reduction Operators
 */

#pragma once

namespace gunrock {
namespace util {

/**
 * Static operator wrapping structure.
 *
 * (N.B. due to an NVCC/cudafe 4.0 regression, we can't specify static templated
 * functions inside other types...)
 */
template <typename T, typename R = T>
struct Operators {
  // Empty default transform function
  static __device__ __forceinline__ void NopTransform(T &val) {}
};

/**
 * Default equality functor
 */
template <typename T>
struct Equality {
  __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) {
    return a == b;
  }
};

/**
 * Default sum functor
 */
template <typename T>
struct Sum {
  // Binary reduction
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) {
    return a + b;
  }

  // Identity
  __host__ __device__ __forceinline__ T operator()() { return (T)0; }
};

}  // namespace util
}  // namespace gunrock
