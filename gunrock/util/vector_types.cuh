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


/*
 * Specializations of this vector-type template can be used to indicate the 
 * proper vector type for a given typename and vector size. We can use the ::Type
 * typedef to declare and work with the appropriate vectorized type for a given 
 * typename T.
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
template <typename T, int vec_elements> struct VecType;

/**
 * Partially-specialized generic vec1 type 
 */
template <typename T> 
struct VecType<T, 1> {
    T x;
    typedef VecType<T, 1> Type;
};

/**
 * Partially-specialized generic vec2 type 
 */
template <typename T> 
struct VecType<T, 2> {
    T x;
    T y;
    typedef VecType<T, 2> Type;
};

/**
 * Partially-specialized generic vec4 type 
 */
template <typename T> 
struct VecType<T, 4> {
    T x;
    T y;
    T z;
    T w;
    typedef VecType<T, 4> Type;
};


/**
 * Macro for expanding partially-specialized built-in vector types
 */
#define GR_DEFINE_VECTOR_TYPE(base_type,short_type)                           \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; };      \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; };      \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };     

GR_DEFINE_VECTOR_TYPE(char,               char)
GR_DEFINE_VECTOR_TYPE(signed char,        char)
GR_DEFINE_VECTOR_TYPE(short,              short)
GR_DEFINE_VECTOR_TYPE(int,                int)
GR_DEFINE_VECTOR_TYPE(long,               long)
GR_DEFINE_VECTOR_TYPE(long long,          longlong)
GR_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
GR_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
GR_DEFINE_VECTOR_TYPE(unsigned int,       uint)
GR_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
GR_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
GR_DEFINE_VECTOR_TYPE(float,              float)
GR_DEFINE_VECTOR_TYPE(double,             double)

#undef GR_DEFINE_VECTOR_TYPE


} // namespace util
} // namespace gunrock

