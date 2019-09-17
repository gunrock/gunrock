// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * binary search.cuh
 *
 * @brief Binary search routine
 */

#pragma once

#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace util {

template <typename T1, typename ArrayT, typename SizeT, typename CompareLambda>
__host__ __device__ SizeT BinarySearch(const T1 &element_to_find,
                                       const ArrayT &elements, SizeT left_index,
                                       SizeT right_index,
                                       CompareLambda comp)  // strictly less
{
  SizeT center_index = 0;
  do {
    if (right_index - left_index <= 1) {
      if (comp(elements[right_index], element_to_find)) return right_index + 1;
      if (comp(elements[left_index], element_to_find)) return right_index;
      return left_index;
    } else {
      center_index = ((long long)left_index + (long long)right_index) >> 1;
      if (comp(elements[center_index], element_to_find)) {
        left_index = center_index + 1;
      } else {
        right_index = center_index - 1;
      }
#ifndef __CUDA_ARCH__
      /*PrintMsg("Find " + std::to_string(element_to_find) +
          " Pos " + std::to_string(center_index) +
          " -> [" + std::to_string(left_index) +
          ", " + std::to_string(right_index) + "]");*/
#endif
    }
  } while (true);
  // return 0;
}

template <typename T1, typename ArrayT, typename SizeT>
__host__ __device__ __forceinline__ SizeT
BinarySearch(const T1 &element_to_find, const ArrayT &elements,
             SizeT left_index, SizeT right_index) {
  return BinarySearch(element_to_find, elements, left_index, right_index,
                      [](const T1 &a, const T1 &b) { return a < b; });
}

template <typename T1, typename ArrayT, typename SizeT, typename LessOp,
          typename EqualOp>
__host__ __device__ SizeT
BinarySearch_LeftMost(const T1 &element_to_find,
                      const ArrayT &elements,  // the array
                      SizeT left_index,        // left index of range, inclusive
                      SizeT right_index,  // right index of range, inclusive
                      LessOp less,        // strictly less
                      EqualOp equal, bool check_boundaries = true) {
  SizeT org_right_index = right_index;
  SizeT center_index = 0;

  if (check_boundaries) {
    if ((!(less(elements[left_index], element_to_find))) &&
        (!(equal(elements[left_index], element_to_find))))
      return left_index - 1;
    if (less(elements[right_index], element_to_find)) return right_index;
  }

  while (right_index - left_index > 1) {
    center_index = ((long long)left_index + (long long)right_index) >> 1;
    if (less(elements[center_index], element_to_find))
      left_index = center_index;
    else
      right_index = center_index;
  }

  if (center_index < org_right_index &&
      less(elements[center_index], element_to_find) &&
      equal(elements[center_index + 1], element_to_find)) {
    center_index++;
  } else if (center_index > 0 &&
             !less(elements[center_index], element_to_find) &&
             !equal(elements[center_index], element_to_find)) {
    center_index--;
  }

  while (center_index > 0 && equal(elements[center_index - 1], element_to_find))
    center_index--;
  return center_index;
}

template <typename T1, typename ArrayT, typename SizeT, typename LessOp>
__host__ __device__ __forceinline__ SizeT BinarySearch_LeftMost(
    const T1 &element_to_find, const ArrayT &elements, SizeT left_index,
    SizeT right_index, LessOp less, bool check_boundaries = true) {
  return BinarySearch_LeftMost(
      element_to_find, elements, left_index, right_index, less,
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

template <typename T1, typename ArrayT, typename SizeT>
__host__ __device__ __forceinline__ SizeT BinarySearch_LeftMost(
    const T1 &element_to_find, const ArrayT &elements, SizeT left_index,
    SizeT right_index, bool check_boundaries = true) {
  return BinarySearch_LeftMost(
      element_to_find, elements, left_index, right_index,
      [](const T1 &a, const T1 &b) { return (a < b); },
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

template <typename T1, typename ArrayT, typename SizeT, typename LessOp,
          typename EqualOp>
__host__ __device__ SizeT
BinarySearch_RightMost(const T1 &element_to_find,
                       const ArrayT &elements,  // the array
                       SizeT left_index,   // left index of range, inclusive
                       SizeT right_index,  // right index of range, inclusive
                       LessOp less,        // strictly less
                       EqualOp equal, bool check_boundaries = true) {
  SizeT org_right_index = right_index;
  SizeT center_index = 0;

  if (check_boundaries) {
    if (!less(elements[left_index], element_to_find) &&
        !equal(elements[left_index], element_to_find))
      return left_index - 1;
    if (less(elements[right_index], element_to_find) ||
        equal(elements[right_index], element_to_find))
      return right_index;
  }

  while (right_index - left_index > 1) {
    center_index = ((long long)left_index + (long long)right_index) >> 1;
    if (less(elements[center_index], element_to_find))
      left_index = center_index;
    else
      right_index = center_index;
  }

  /*if (center_index < org_right_index &&
       less (elements[center_index], element_to_find) &&
       equal(elements[center_index + 1], element_to_find))
  {
      center_index ++;
  } else*/
  if (center_index > 0 && !less(elements[center_index], element_to_find) &&
      !equal(elements[center_index], element_to_find)) {
    center_index--;
  }

  while (center_index < org_right_index - 1 &&
         equal(elements[center_index + 1], element_to_find))
    center_index++;
  return center_index;
}

template <typename T1, typename ArrayT, typename SizeT, typename LessOp>
__host__ __device__ __forceinline__ SizeT BinarySearch_RightMost(
    const T1 &element_to_find, const ArrayT &elements, SizeT left_index,
    SizeT right_index, LessOp less, bool check_boundaries = true) {
  return BinarySearch_RightMost(
      element_to_find, elements, left_index, right_index, less,
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

template <typename T1, typename ArrayT, typename SizeT>
__host__ __device__ __forceinline__ SizeT BinarySearch_RightMost(
    const T1 &element_to_find, const ArrayT &elements, SizeT left_index,
    SizeT right_index, bool check_boundaries = true) {
  return BinarySearch_RightMost(
      element_to_find, elements, left_index, right_index,
      [](const T1 &a, const T1 &b) { return (a < b); },
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
