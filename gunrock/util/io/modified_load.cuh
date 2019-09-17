// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * modified_load.cuh
 *
 * @brief Kernel utilities for loading types through global memory
 * with cache modifiers
 */

#pragma once

#include <cuda.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/vector_types.cuh>

namespace gunrock {
namespace util {
namespace io {

/**
 * Enumeration of data movement cache modifiers.
 */
namespace ld {

enum CacheModifier {
  NONE,  // default (currently ca)
  cg,    // cache global
  ca,    // cache all
  cs,    // cache streaming

  LIMIT
};

}  // namespace ld

#define CacheModifierToString(modifier)                                    \
  (((int)modifier == b40c::util::io::ld::NONE)                             \
       ? "util::io::ld::NONE"                                              \
       : ((int)modifier == b40c::util::io::ld::cg)                         \
             ? "util::io::ld::cg"                                          \
             : ((int)modifier == b40c::util::io::ld::ca)                   \
                   ? "util::io::ld::ca"                                    \
                   : ((int)modifier == b40c::util::io::ld::cs)             \
                         ? "util::io::ld::cs"                              \
                         : ((int)modifier == b40c::util::io::st::NONE)     \
                               ? "util::io::st::NONE"                      \
                               : ((int)modifier == b40c::util::io::st::cg) \
                                     ? "util::io::st::cg"                  \
                                     : ((int)modifier ==                   \
                                        b40c::util::io::st::wb)            \
                                           ? "util::io::st::wb"            \
                                           : ((int)modifier ==             \
                                              b40c::util::io::st::cs)      \
                                                 ? "util::io::st::cs"      \
                                                 : "<ERROR>")

/**
 * Basic utility for performing modified loads through cache.
 */
template <ld::CacheModifier CACHE_MODIFIER>
struct ModifiedLoad {
  /*
   * Load operation we will provide specializations for
   */
  template <typename T>
  __device__ __forceinline__ static void Ld(T &val, T *ptr);

  /*
   * Vec-4 loads for 64-bit types are implemented as two vec-2 loads
   */
  __device__ __forceinline__ static void Ld(double4 &val, double4 *ptr) {
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<double2 *>(&val.x),
                                     reinterpret_cast<double2 *>(ptr));
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<double2 *>(&val.z),
                                     reinterpret_cast<double2 *>(ptr) + 1);
  }

  __device__ __forceinline__ static void Ld(ulonglong4 &val, ulonglong4 *ptr) {
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2 *>(&val.x),
                                     reinterpret_cast<ulonglong2 *>(ptr));
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2 *>(&val.z),
                                     reinterpret_cast<ulonglong2 *>(ptr) + 1);
  }

  __device__ __forceinline__ static void Ld(longlong4 &val, longlong4 *ptr) {
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2 *>(&val.x),
                                     reinterpret_cast<longlong2 *>(ptr));
    ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2 *>(&val.z),
                                     reinterpret_cast<longlong2 *>(ptr) + 1);
  }
};

#if __CUDA_ARCH__ >= 200

/**
 * Specialization for NONE modifier
 */
template <>
template <typename T>
__device__ __forceinline__ void ModifiedLoad<ld::NONE>::Ld(T &val, T *ptr) {
  val = *ptr;
}

/**
 * Singleton load op
 */
#define GR_LOAD(base_type, ptx_type, reg_mod, cast_type, modifier)      \
  template <>                                                           \
  template <>                                                           \
  void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type *ptr) { \
    asm volatile("ld.global." #modifier "." #ptx_type " %0, [%1];"      \
                 : "=" #reg_mod(reinterpret_cast<cast_type &>(val))     \
                 : _GR_ASM_PTR_(ptr));                                  \
  }

/**
 * Vector load ops
 */
#define GR_LOAD_VEC1(base_type, ptx_type, reg_mod, cast_type, modifier) \
  template <>                                                           \
  template <>                                                           \
  void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type *ptr) { \
    asm volatile("ld.global." #modifier "." #ptx_type " %0, [%1];"      \
                 : "=" #reg_mod(reinterpret_cast<cast_type &>(val.x))   \
                 : _GR_ASM_PTR_(ptr));                                  \
  }

#define GR_LOAD_VEC2(base_type, ptx_type, reg_mod, cast_type, modifier)     \
  template <>                                                               \
  template <>                                                               \
  void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type *ptr) {     \
    asm volatile("ld.global." #modifier ".v2." #ptx_type " {%0, %1}, [%2];" \
                 : "=" #reg_mod(reinterpret_cast<cast_type &>(val.x)),      \
                   "=" #reg_mod(reinterpret_cast<cast_type &>(val.y))       \
                 : _GR_ASM_PTR_(ptr));                                      \
  }

#define GR_LOAD_VEC4(base_type, ptx_type, reg_mod, cast_type, modifier) \
  template <>                                                           \
  template <>                                                           \
  void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type *ptr) { \
    asm volatile("ld.global." #modifier ".v4." #ptx_type                \
                 " {%0, %1, %2, %3}, [%4];"                             \
                 : "=" #reg_mod(reinterpret_cast<cast_type &>(val.x)),  \
                   "=" #reg_mod(reinterpret_cast<cast_type &>(val.y)),  \
                   "=" #reg_mod(reinterpret_cast<cast_type &>(val.z)),  \
                   "=" #reg_mod(reinterpret_cast<cast_type &>(val.w))   \
                 : _GR_ASM_PTR_(ptr));                                  \
  }

/**
 * Defines specialized load ops for only the base type
 */
#define GR_LOAD_BASE(base_type, ptx_type, reg_mod, cast_type) \
  GR_LOAD(base_type, ptx_type, reg_mod, cast_type, cg)        \
  GR_LOAD(base_type, ptx_type, reg_mod, cast_type, ca)        \
  GR_LOAD(base_type, ptx_type, reg_mod, cast_type, cs)

/**
 * Defines specialized load ops for the base type and for its derivative vec1
 * and vec2 types
 */
#define GR_LOAD_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, \
                             reg_mod, cast_type)                         \
  GR_LOAD_BASE(base_type, ptx_type, reg_mod, cast_type)                  \
                                                                         \
  GR_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, cg)          \
  GR_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, ca)          \
  GR_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, cs)          \
                                                                         \
  GR_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, cg)          \
  GR_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, ca)          \
  GR_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, cs)

/**
 * Defines specialized load ops for the base type and for its derivative vec1,
 * vec2, and vec4 types
 */
#define GR_LOAD_BASE_ONE_TWO_FOUR(base_type, dest_type, short_type, ptx_type, \
                                  reg_mod, cast_type)                         \
  GR_LOAD_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, reg_mod,   \
                       cast_type)                                             \
  GR_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, cg)               \
  GR_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, ca)               \
  GR_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, cs)

#if CUDA_VERSION >= 4000
#define GR_REG8 h
#define GR_REG16 h
#define GR_CAST8 short
#else
#define GR_REG8 r
#define GR_REG16 r
#define GR_CAST8 char
#endif

/**
 * Define cache-modified loads for all 4-byte (and smaller) structures
 */
GR_LOAD_BASE_ONE_TWO_FOUR(char, char, char, s8, GR_REG8, GR_CAST8)
GR_LOAD_BASE_ONE_TWO_FOUR(short, short, short, s16, GR_REG16, short)
GR_LOAD_BASE_ONE_TWO_FOUR(int, int, int, s32, r, int)
GR_LOAD_BASE_ONE_TWO_FOUR(unsigned char, unsigned char, uchar, u8, GR_REG8,
                          unsigned GR_CAST8)
GR_LOAD_BASE_ONE_TWO_FOUR(unsigned short, unsigned short, ushort, u16, GR_REG16,
                          unsigned short)
GR_LOAD_BASE_ONE_TWO_FOUR(unsigned int, unsigned int, uint, u32, r,
                          unsigned int)
GR_LOAD_BASE_ONE_TWO_FOUR(float, float, float, f32, f, float)

#if !defined(__LP64__) || (__LP64__ == 0)
// longs are 64-bit on non-Windows 64-bit compilers
GR_LOAD_BASE_ONE_TWO_FOUR(long, long, long, s32, r, long)
GR_LOAD_BASE_ONE_TWO_FOUR(unsigned long, unsigned long, ulong, u32, r,
                          unsigned long)
#endif

GR_LOAD_BASE(signed char, s8, r,
             unsigned int)  // Only need to define base: char2,char4, etc
                            // already defined from char

/**
 * Define cache-modified loads for all 8-byte structures
 */
GR_LOAD_BASE_ONE_TWO(unsigned long long, unsigned long long, ulonglong, u64, l,
                     unsigned long long)
GR_LOAD_BASE_ONE_TWO(long long, long long, longlong, s64, l, long long)
GR_LOAD_BASE_ONE_TWO(
    double, double, double, s64, l,
    long long)  // Cast to 64-bit long long a workaround for the fact that
                // the 3.x assembler has no register constraint for doubles

#if defined(__LP64__)
// longs are 64-bit on non-Windows 64-bit compilers
GR_LOAD_BASE_ONE_TWO(long, long, long, s64, l, long)
GR_LOAD_BASE_ONE_TWO(unsigned long, unsigned long, ulong, u64, l, unsigned long)
#endif

/**
 * Undefine macros
 */
#undef GR_LOAD_VEC1
#undef GR_LOAD_VEC2
#undef GR_LOAD_VEC4
#undef GR_LOAD_BASE
#undef GR_LOAD_BASE_ONE_TWO
#undef GR_LOAD_BASE_ONE_TWO_FOUR
#undef GR_CAST8
#undef GR_REG8
#undef GR_REG16

#else  //__CUDA_ARCH__

template <ld::CacheModifier READ_MODIFIER>
template <typename T>
__device__ __forceinline__ void ModifiedLoad<READ_MODIFIER>::Ld(T &val,
                                                                T *ptr) {
  val = *ptr;
}

#endif  //__CUDA_ARCH__

}  // namespace io
}  // namespace util
}  // namespace gunrock
