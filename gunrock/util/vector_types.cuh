// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * vector_types.cuh
 *
 * @brief Utility code for working with vector types of arbitrary typenames
 */

#pragma once

namespace gunrock {
namespace util {

template <typename T, int SIZE>
struct VectorType {
  /*typedef UnknownType Type;*/
};
template <>
struct VectorType<char, 1> {
  typedef char1 Type;
};
template <>
struct VectorType<char, 2> {
  typedef char2 Type;
};
template <>
struct VectorType<char, 3> {
  typedef char3 Type;
};
template <>
struct VectorType<char, 4> {
  typedef char4 Type;
};
template <>
struct VectorType<unsigned char, 1> {
  typedef uchar1 Type;
};
template <>
struct VectorType<unsigned char, 2> {
  typedef uchar2 Type;
};
template <>
struct VectorType<unsigned char, 3> {
  typedef uchar3 Type;
};
template <>
struct VectorType<unsigned char, 4> {
  typedef uchar4 Type;
};

template <>
struct VectorType<short, 1> {
  typedef short1 Type;
};
template <>
struct VectorType<short, 2> {
  typedef short2 Type;
};
template <>
struct VectorType<short, 3> {
  typedef short3 Type;
};
template <>
struct VectorType<short, 4> {
  typedef short4 Type;
};
template <>
struct VectorType<unsigned short, 1> {
  typedef ushort1 Type;
};
template <>
struct VectorType<unsigned short, 2> {
  typedef ushort2 Type;
};
template <>
struct VectorType<unsigned short, 3> {
  typedef ushort3 Type;
};
template <>
struct VectorType<unsigned short, 4> {
  typedef ushort4 Type;
};

template <>
struct VectorType<int, 1> {
  typedef int1 Type;
};
template <>
struct VectorType<int, 2> {
  typedef int2 Type;
};
template <>
struct VectorType<int, 3> {
  typedef int3 Type;
};
template <>
struct VectorType<int, 4> {
  typedef int4 Type;
};
template <>
struct VectorType<unsigned int, 1> {
  typedef uint1 Type;
};
template <>
struct VectorType<unsigned int, 2> {
  typedef uint2 Type;
};
template <>
struct VectorType<unsigned int, 3> {
  typedef uint3 Type;
};
template <>
struct VectorType<unsigned int, 4> {
  typedef uint4 Type;
};

template <>
struct VectorType<long, 1> {
  typedef long1 Type;
};
template <>
struct VectorType<long, 2> {
  typedef long2 Type;
};
template <>
struct VectorType<long, 3> {
  typedef long3 Type;
};
template <>
struct VectorType<long, 4> {
  typedef long4 Type;
};
template <>
struct VectorType<unsigned long, 1> {
  typedef ulong1 Type;
};
template <>
struct VectorType<unsigned long, 2> {
  typedef ulong2 Type;
};
template <>
struct VectorType<unsigned long, 3> {
  typedef ulong3 Type;
};
template <>
struct VectorType<unsigned long, 4> {
  typedef ulong4 Type;
};

template <>
struct VectorType<long long, 1> {
  typedef longlong1 Type;
};
template <>
struct VectorType<long long, 2> {
  typedef longlong2 Type;
};
template <>
struct VectorType<long long, 3> {
  typedef longlong3 Type;
};
template <>
struct VectorType<long long, 4> {
  typedef longlong4 Type;
};
template <>
struct VectorType<unsigned long long, 1> {
  typedef ulonglong1 Type;
};
template <>
struct VectorType<unsigned long long, 2> {
  typedef ulonglong2 Type;
};
template <>
struct VectorType<unsigned long long, 3> {
  typedef ulonglong3 Type;
};
template <>
struct VectorType<unsigned long long, 4> {
  typedef ulonglong4 Type;
};

template <>
struct VectorType<float, 1> {
  typedef float1 Type;
};
template <>
struct VectorType<float, 2> {
  typedef float2 Type;
};
template <>
struct VectorType<float, 3> {
  typedef float3 Type;
};
template <>
struct VectorType<float, 4> {
  typedef float4 Type;
};
template <>
struct VectorType<double, 1> {
  typedef double1 Type;
};
template <>
struct VectorType<double, 2> {
  typedef double2 Type;
};
template <>
struct VectorType<double, 3> {
  typedef double3 Type;
};
template <>
struct VectorType<double, 4> {
  typedef double4 Type;
};

template <typename T, int SIZE>
struct VecType {
  typedef typename VectorType<T, SIZE>::Type Type;
};

// DO NOT USE following definations
// cuda buildin vector types are defined in <CUDA_DIR>/include/vector_types.h

/*
 * Specializations of this vector-type template can be used to indicate the
 * proper vector type for a given typename and vector size. We can use the
 * ::Type typedef to declare and work with the appropriate vectorized type for a
 * given typename T.
 *
 * For example, consider the following copy kernel that uses vec-2 loads
 * and stores:
 *
 *      template <typename T>
 *      __global__ void CopyKernel(T *d_in, T *d_out)
 *      {
 *          typedef typename VecType<T, 2>::Type Vector;
 *
 *          Vector datum;
 *
 *          Vector *d_in_v2 = (Vector *) d_in;
 *          Vector *d_out_v2 = (Vector *) d_out;
 *
 *          datum = d_in_v2[threadIdx.x];
 *          d_out_v2[threadIdx.x] = datum;
 *      }
 *
 */
// template <typename T, int vec_elements> struct VecType;

/**
 * Partially-specialized generic vec1 type
 */
// template <typename T>
// struct VecType<T, 1> {
//    T x;
//    typedef VecType<T, 1> Type;
//};

/**
 * Partially-specialized generic vec2 type
 */
// template <typename T>
// struct VecType<T, 2> {
//    T x;
//    T y;
//    typedef VecType<T, 2> Type;
//};

/**
 * Partially-specialized generic vec4 type
 */
// template <typename T>
// struct VecType<T, 4> {
//    T x;
//    T y;
//    T z;
//    T w;
//    typedef VecType<T, 4> Type;
//};

/**
 * Macro for expanding partially-specialized built-in vector types
 */
/*#define GR_DEFINE_VECTOR_TYPE(base_type,short_type) \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; }; \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; }; \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };

GR_DEFINE_VECTOR_TYPE(char, char)
GR_DEFINE_VECTOR_TYPE(signed char, char)
GR_DEFINE_VECTOR_TYPE(short, short)
GR_DEFINE_VECTOR_TYPE(int, int)
GR_DEFINE_VECTOR_TYPE(long, long)
GR_DEFINE_VECTOR_TYPE(long long, longlong)
GR_DEFINE_VECTOR_TYPE(unsigned char, uchar)
GR_DEFINE_VECTOR_TYPE(unsigned short, ushort)
GR_DEFINE_VECTOR_TYPE(unsigned int, uint)
GR_DEFINE_VECTOR_TYPE(unsigned long, ulong)
GR_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
GR_DEFINE_VECTOR_TYPE(float, float)
GR_DEFINE_VECTOR_TYPE(double, double)

#undef GR_DEFINE_VECTOR_TYPE
*/

}  // namespace util
}  // namespace gunrock
