// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * oprtr_base.cuh
 *
 * @brief Base defination for operators
 */

#pragma once

#include <gunrock/util/io/cub_io.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>

namespace gunrock {
namespace oprtr {

/**
 * Load instruction cache-modifier const defines.
 */

// Load instruction cache-modifier for reading incoming frontier vertex-ids.
// Valid on SM2.0 or newer
static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER = util::io::ld::cg;

// Load instruction cache-modifier for reading CSR column-indices.
static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER =
    util::io::ld::NONE;

// Load instruction cache-modifier for reading edge values.
static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER =
    util::io::ld::NONE;

// Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER =
    util::io::ld::cg;

// Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER =
    util::io::ld::NONE;

// Store instruction cache-modifier for writing outgoing frontier vertex-ids.
// Valid on SM2.0 or newer
static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER =
    util::io::st::cg;

#ifndef CUDA_ARCH
static const int CUDA_ARCH = 300;  // CUDA_ARCH compiled for
#endif

/**
 * @brief Operator Modes
 */
using OprtrMode = uint32_t;
enum : OprtrMode {
  OprtrMode_AdvanceMask = 0x0F0000,
  OptrtMode_None = 0x000000,
  OprtrMode_TWC = 0x010000,
  OprtrMode_TWC_BACKWARD = 0x020000,
  OprtrMode_LB_BACKWARD = 0x030000,
  OprtrMode_LB = 0x040000,
  OprtrMode_LB_LIGHT = 0x050000,
  OprtrMode_LB_CULL = 0x060000,
  OprtrMode_LB_LIGHT_CULL = 0x070000,
  OprtrMode_ALL_EDGES = 0x080000,

  OprtrMode_FilterMask = 0xF00000,
  OprtrMode_CULL = 0x100000,
  OprtrMode_SIMPLIFIED = 0x200000,
  OprtrMode_SIMPLIFIED2 = 0x300000,
  OprtrMode_COMPACTED_CULL = 0x400000,
  OprtrMode_BY_PASS = 0x500000,

  OprtrMode_ReduceMask = 0xF000000,
  OprtrMode_REDUCE_TO_INPUT_POS = 0x1000000,
  OprtrMode_REDUCE_TO_SRC = 0x2000000,
  OprtrMode_REDUCE_TO_DEST = 0x3000000,
};

using OprtrFlag = uint32_t;
enum : OprtrFlag {
  OprtrFlag_None = 0x00,
};

/**
 * @brief Four types of advance operator
 */
using OprtrType = uint32_t;
enum : OprtrType {
  OprtrType_Mask = 0x0F,
  OprtrType_V2V = 0x01,
  OprtrType_V2E = 0x02,
  OprtrType_E2V = 0x04,
  OprtrType_E2E = 0x08,
};

/**
 * @brief opeartion to use for mgpu primitives
 */
using ReduceOp = uint32_t;
enum : ReduceOp {
  ReduceOp_Mask = 0xF0,
  ReduceOp_None = 0x00,
  ReduceOp_Plus = 0x10,
  ReduceOp_Minus = 0x20,
  ReduceOp_Multiply = 0x30,
  ReduceOp_Divide = 0x40,
  ReduceOp_Mod = 0x50,
  ReduceOp_Bitor = 0x60,
  ReduceOp_Bitand = 0x70,
  ReduceOp_Xor = 0x80,
  ReduceOp_Max = 0x90,
  ReduceOp_Min = 0xA0,
};

using ReduceType = uint32_t;
enum : ReduceType {
  ReduceType_Mask = 0xF00,
  ReduceType_None = 0x000,
  ReduceType_Vertex = 0x100,
  ReduceType_Edge = 0x200,
};

using OprtrOption = uint32_t;
enum : OprtrOption {
  OprtrOption_Mask = 0xF000,
  OprtrOption_None = 0x0000,
  OprtrOption_Idempotence = 0x1000,
  OprtrOption_Mark_Predecessors = 0x2000,
};

template <OprtrFlag FLAG>
bool isFused() {
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_CULL) return true;
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT_CULL)
    return true;
  else
    return false;
}

template <OprtrFlag FLAG>
bool hasPreScan() {
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB) return true;
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT) return true;
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_CULL) return true;
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT_CULL) return true;
  return false;
}

template <OprtrFlag FLAG>
bool isBackward() {
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_TWC_BACKWARD) return true;
  if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_BACKWARD) return true;
  return false;
}

template <typename T, ReduceOp R_OP>
struct Reduce {
  static const T Identity = util::PreDefinedValues<T>::InvalidValue;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return util::PreDefinedValues<T>::InvalidValue;
  }
};

// template <typename T>
// struct Identity<T, ReduceOp_None>
//{
//    static const T Identity = 0;
//};

template <typename T>
struct Reduce<T, ReduceOp_Plus> {
  static const T Identity = 0;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a + b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Minus> {
  static const T Identity = 0;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a - b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Multiply> {
  static const T Identity = 1;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a * b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Divide> {
  static const T Identity = 1;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a / b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Mod> {
  static const T Identity = util::PreDefinedValues<T>::InvalidValue;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a % b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Bitor> {
  static const T Identity = util::PreDefinedValues<T>::AllZeros;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a | b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Bitand> {
  static const T Identity = util::PreDefinedValues<T>::AllOnes;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a & b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Xor> {
  static const T Identity = util::PreDefinedValues<T>::AllZeros;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return a ^ b;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Max> {
  static const T Identity = util::PreDefinedValues<T>::MinValue;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return (a < b) ? b : a;
  }
};

template <typename T>
struct Reduce<T, ReduceOp_Min> {
  static const T Identity = util::PreDefinedValues<T>::MaxValue;

  __device__ __host__ __forceinline__ static T op(const T &a, const T &b) {
    return (a < b) ? a : b;
  }
};

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
