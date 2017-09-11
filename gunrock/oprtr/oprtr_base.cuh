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

namespace gunrock {
namespace oprtr {

/**
 * Load instruction cache-modifier const defines.
 */

// Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER
    = util::io::ld::cg;

// Load instruction cache-modifier for reading CSR column-indices.
static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER
    = util::io::ld::NONE;

// Load instruction cache-modifier for reading edge values.
static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER
    = util::io::ld::NONE;

// Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER
    = util::io::ld::cg;

// Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER
    = util::io::ld::NONE;

// Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer
static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER
    = util::io::st::cg;

#ifndef CUDA_ARCH
    static const int CUDA_ARCH = 300; // CUDA_ARCH compiled for
#endif

/**
 * @brief Operator Modes
 */
using OprtrMode = uint32_t;
enum : OprtrMode {
    OprtrMode_AdvanceMask    = 0x0F0000,
    OptrtMode_None           = 0x000000,
    OprtrMode_TWC            = 0x010000,
    OprtrMode_TWC_BACKWARD   = 0x020000,
    OprtrMode_LB_BACKWARD    = 0x030000,
    OprtrMode_LB             = 0x040000,
    OprtrMode_LB_LIGHT       = 0x050000,
    OprtrMode_LB_CULL        = 0x060000,
    OprtrMode_LB_LIGHT_CULL  = 0x070000,
    OprtrMode_ALL_EDGES      = 0x080000,

    OprtrMode_FilterMask     = 0xF00000,
    OprtrMode_CULL           = 0x100000,
    OprtrMode_SIMPLIFIED     = 0x200000,
    OprtrMode_SIMPLIFIED2    = 0x300000,
    OprtrMode_COMPACTED_CULL = 0x400000,
    OprtrMode_BY_PASS        = 0x500000,
};

using OprtrFlag = uint32_t;
enum : OprtrFlag
{
    OprtrFlag_None = 0x00,
};

/**
 * @brief Four types of advance operator
 */
using OprtrType = uint32_t;
enum : OprtrType
{
    OprtrType_Mask = 0x0F,
    OprtrType_V2V  = 0x01,
    OprtrType_V2E  = 0x02,
    OprtrType_E2V  = 0x04,
    OprtrType_E2E  = 0x08,
};

/**
 * @brief opeartion to use for mgpu primitives
 */
using ReduceOp = uint32_t;
enum : ReduceOp
{
    ReduceOp_Mask       = 0xF0,
    ReduceOp_None       = 0x00,
    ReduceOp_Plus       = 0x10,
    ReduceOp_Minus      = 0x20,
    ReduceOp_Multiples  = 0x30,
    ReduceOp_Modulus    = 0x40,
    ReduceOp_Bit_Or     = 0x50,
    ReduceOp_Bit_And    = 0x60,
    ReduceOp_Bit_Xor    = 0x70,
    ReduceOp_Maximum    = 0x80,
    ReduceOp_Minimum    = 0x90,
};

using ReduceType = uint32_t;
enum : ReduceType
{
    ReduceType_Mask   = 0xF00,
    ReduceType_None   = 0x000,
    ReduceType_Vertex = 0x100,
    ReduceType_Edge   = 0x200,
};

using OprtrOption = uint32_t;
enum : OprtrOption
{
    OprtrOption_Mask              = 0xF000,
    OprtrOption_None              = 0x0000,
    OprtrOption_Idempotence       = 0x1000,
    OprtrOption_Mark_Predecessors = 0x2000,
};

template <OprtrFlag FLAG>
bool isFused()
{
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_CULL      ) return true;
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT_CULL) return true;
    else return false;
}

template <OprtrFlag FLAG>
bool hasPreScan()
{
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB           ) return true;
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT     ) return true;
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_CULL      ) return true;
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_LIGHT_CULL) return true;
    return false;
}

template <OprtrFlag FLAG>
bool isBackward()
{
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_TWC_BACKWARD) return true;
    if ((FLAG & OprtrMode_AdvanceMask) == OprtrMode_LB_BACKWARD ) return true;
    return false;
}

template <typename T, ReduceOp R_OP>
struct Identity
{
    static const T Val = 0;
    //__device__ __host__ __forceinline__ T operator()()
    //{
    //    extern __device__ __host__ void Error_UnsupportedOperation();
    //    Error_UnsupportedOperation();
    //    return 0;
    //}
};

template <typename T>
struct Identity<T, ReduceOp_None>
{
    static const T Val = 0;
};

template <typename T>
struct Identity<T, ReduceOp_Plus>
{
    static const T Val = 0;
};

template <typename T>
struct Identity<T, ReduceOp_Multiples>
{
    static const T Val = 1;
};

template <typename T>
struct Identity<T, ReduceOp_Bit_Or>
{
    static const T Val = util::PreDefinedValues<T>::AllZeros;
};

template <typename T>
struct Identity<T, ReduceOp_Bit_And>
{
    static const T Val = util::PreDefinedValues<T>::AllOnes;
};

template <typename T>
struct Identity<T, ReduceOp_Bit_Xor>
{
    static const T Val = util::PreDefinedValues<T>::AllZeros;
};

template <typename T>
struct Identity<T, ReduceOp_Maximum>
{
    static const T Val = util::PreDefinedValues<T>::MinValue;
};

template <typename T>
struct Identity<T, ReduceOp_Minimum>
{
    static const T Val = util::PreDefinedValues<T>::MaxValue;
};

} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
