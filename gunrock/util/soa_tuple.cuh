// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * soa_tuple.cuh
 *
 * @brief Simple tuple types for assisting AOS <-> SOA work
 */

#pragma once

#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

template <typename T0 = NullType, typename T1 = NullType,
          typename T2 = NullType, typename T3 = NullType>
struct Tuple;

/**
 * 1 element tuple
 */
template <typename _T0>
struct Tuple<_T0, NullType, NullType, NullType> {
  // Typedefs
  typedef _T0 T0;

  enum {
    NUM_FIELDS = 1,
    VOLATILE = util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE,
  };

  // Fields
  T0 t0;

  // Constructors
  __host__ __device__ __forceinline__ Tuple() {}
  __host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}

  // Manipulators

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int col) {
    t0[col] = tuple.t0;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int row, const int col) {
    t0[row][col] = tuple.t0;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int col) const {
    retval = TupleSlice(t0[col]);
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int row,
                                               const int col) const {
    retval = TupleSlice(t0[row][col]);
  }
};

/**
 * 2 element tuple
 */
template <typename _T0, typename _T1>
struct Tuple<_T0, _T1, NullType, NullType> {
  // Typedefs
  typedef _T0 T0;
  typedef _T1 T1;

  enum {
    NUM_FIELDS = 2,
    VOLATILE =
        (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE),
  };

  // Fields
  T0 t0;
  T1 t1;

  // Constructors
  __host__ __device__ __forceinline__ Tuple() {}
  __host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}
  __host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1) : t0(t0), t1(t1) {}

  // Manipulators

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int col) {
    t0[col] = tuple.t0;
    t1[col] = tuple.t1;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int row, const int col) {
    t0[row][col] = tuple.t0;
    t1[row][col] = tuple.t1;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int col) const {
    retval = TupleSlice(t0[col], t1[col]);
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int row,
                                               const int col) const {
    retval = TupleSlice(t0[row][col], t1[row][col]);
  }
};

/**
 * 3 element tuple
 */
template <typename _T0, typename _T1, typename _T2>
struct Tuple<_T0, _T1, _T2, NullType> {
  // Typedefs
  typedef _T0 T0;
  typedef _T1 T1;
  typedef _T2 T2;

  enum {
    NUM_FIELDS = 3,
    VOLATILE =
        (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T2>::Type>::VALUE),
  };

  // Fields
  T0 t0;
  T1 t1;
  T2 t2;

  // Constructor
  __host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2)
      : t0(t0), t1(t1), t2(t2) {}

  // Manipulators

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int col) {
    t0[col] = tuple.t0;
    t1[col] = tuple.t1;
    t2[col] = tuple.t2;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int row, const int col) {
    t0[row][col] = tuple.t0;
    t1[row][col] = tuple.t1;
    t2[row][col] = tuple.t2;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int col) const {
    retval = TupleSlice(t0[col], t1[col], t2[col]);
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int row,
                                               const int col) const {
    retval = TupleSlice(t0[row][col], t1[row][col], t2[row][col]);
  }
};

/**
 * 4 element tuple
 */
template <typename _T0, typename _T1, typename _T2, typename _T3>
struct Tuple {
  // Typedefs
  typedef _T0 T0;
  typedef _T1 T1;
  typedef _T2 T2;
  typedef _T3 T3;

  enum {
    NUM_FIELDS = 3,
    VOLATILE =
        (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T2>::Type>::VALUE &&
         util::IsVolatile<typename util::RemovePointers<T3>::Type>::VALUE),
  };

  // Fields
  T0 t0;
  T1 t1;
  T2 t2;
  T3 t3;

  // Constructor
  __host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2, T3 t3)
      : t0(t0), t1(t1), t2(t2), t3(t3) {}

  // Manipulators

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int col) {
    t0[col] = tuple.t0;
    t1[col] = tuple.t1;
    t2[col] = tuple.t2;
    t3[col] = tuple.t3;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Set(const TupleSlice &tuple,
                                               const int row, const int col) {
    t0[row][col] = tuple.t0;
    t1[row][col] = tuple.t1;
    t2[row][col] = tuple.t2;
    t3[row][col] = tuple.t3;
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int col) const {
    retval = TupleSlice(t0[col], t1[col], t2[col], t3[col]);
  }

  template <typename TupleSlice>
  __host__ __device__ __forceinline__ void Get(TupleSlice &retval,
                                               const int row,
                                               const int col) const {
    retval = TupleSlice(t0[row][col], t1[row][col], t2[row][col], t3[row][col]);
  }
};

}  // namespace util
}  // namespace gunrock
