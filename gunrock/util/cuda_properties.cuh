// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cuda_properties.cuh
 *
 * @brief CUDA Properties
 */

#pragma once

namespace gunrock {
namespace util {

/*****************************************************************
 * Macros for guiding compilation paths
 *****************************************************************/

/**
 * CUDA architecture of the current compilation path
 */
#ifndef __CUDA_ARCH__
//#define __GR_CUDA_ARCH__ 0                      // Host path
#else
#define __GR_CUDA_ARCH__ __CUDA_ARCH__  // Device path
#endif

/*****************************************************************
 * Device properties by SM architectural version
 *****************************************************************/

// Invalid CUDA device ordinal
#define GR_INVALID_DEVICE (-1)

// Threads per warp.
#define GR_LOG_WARP_THREADS(arch) (5)  // 32 threads in a warp
#define GR_WARP_THREADS(arch) (1 << GR_LOG_WARP_THREADS(arch))

// SM memory bank stride (in bytes)
#define GR_LOG_BANK_STRIDE_BYTES(arch) (2)  // 4 byte words
#define GR_BANK_STRIDE_BYTES(arch) (1 << GR_LOG_BANK_STRIDE_BYTES)

// Memory banks per SM
#define GR_SM20_LOG_MEM_BANKS() (5)  // 32 banks on SM2.0+
#define GR_SM10_LOG_MEM_BANKS() (4)  // 16 banks on SM1.0-SM1.3
#define GR_LOG_MEM_BANKS(arch) \
  ((arch >= 200) ? GR_SM20_LOG_MEM_BANKS() : GR_SM10_LOG_MEM_BANKS())

// Physical shared memory per SM (bytes)
#define GR_SM75_SMEM_BYTES() (65536)   // 64KB on SM7.5
#define GR_SM61_SMEM_BYTES() (98304)   // 96KB on SM6.1+
#define GR_SM60_SMEM_BYTES() (65536)   // 64KB on SM6.0+
#define GR_SM52_SMEM_BYTES() (98304)   // 64KB on SM5.2
#define GR_SM50_SMEM_BYTES() (65536)   // 64KB on SM5.0+
#define GR_SM37_SMEM_BYTES() (114688)  // 48KB + 64KB on SM3.7
#define GR_SM20_SMEM_BYTES() (49152)   // 48KB on SM2.0+
#define GR_SM10_SMEM_BYTES() (16384)   // 32KB on SM1.0-SM1.3
#define GR_SMEM_BYTES(arch)                                                 \
  ((arch == 750)                                                            \
       ? GR_SM75_SMEM_BYTES()                                               \
       : (arch >= 610)                                                      \
             ? GR_SM61_SMEM_BYTES()                                         \
             : (arch >= 600)                                                \
                   ? GR_SM60_SMEM_BYTES()                                   \
                   : (arch == 520)                                          \
                         ? GR_SM52_SMEM_BYTES()                             \
                         : (arch >= 500)                                    \
                               ? GR_SM50_SMEM_BYTES()                       \
                               : (arch == 370)                              \
                                     ? GR_SM37_SMEM_BYTES()                 \
                                     : (arch >= 200) ? GR_SM20_SMEM_BYTES() \
                                                     : GR_SM10_SMEM_BYTES())

// Physical threads per SM
#define GR_SM30_SM_THREADS() (2048)  // 2048 threads on SM3.0+
#define GR_SM20_SM_THREADS() (1536)  // 1536 threads on SM2.0+
#define GR_SM12_SM_THREADS() (1024)  // 1024 threads on SM1.2-SM1.3
#define GR_SM10_SM_THREADS() (768)   // 768 threads on SM1.0-SM1.1
#define GR_SM_THREADS(arch)                                             \
  ((arch >= 300) ? GR_SM30_SM_THREADS()                                 \
                 : (arch >= 200) ? GR_SM20_SM_THREADS()                 \
                                 : (arch >= 130) ? GR_SM12_SM_THREADS() \
                                                 : GR_SM10_SM_THREADS())

// Physical threads per CTA
#define GR_SM20_LOG_CTA_THREADS() (10)  // 1024 threads on SM2.0+
#define GR_SM10_LOG_CTA_THREADS() (9)   // 512 threads on SM1.0-SM1.3
#define GR_LOG_CTA_THREADS(arch) \
  ((arch >= 200) ? GR_SM20_LOG_CTA_THREADS() : GR_SM10_LOG_CTA_THREADS())

// Max CTAs per SM
#define GR_SM50_SM_CTAS() (32)  // 32 CTAs on SM5.0+
#define GR_SM30_SM_CTAS() (16)  // 16 CTAs on SM3.0+
#define GR_SM20_SM_CTAS() (8)   // 8 CTAs on SM2.0+
#define GR_SM12_SM_CTAS() (8)   // 8 CTAs on SM1.2-SM1.3
#define GR_SM10_SM_CTAS() (8)   // 8 CTAs on SM1.0-SM1.1
#define GR_SM_CTAS(arch)                                                   \
  ((arch >= 500)                                                           \
       ? GR_SM50_SM_CTAS()                                                 \
       : (arch >= 300) ? GR_SM30_SM_CTAS()                                 \
                       : (arch >= 200) ? GR_SM20_SM_CTAS()                 \
                                       : (arch >= 130) ? GR_SM12_SM_CTAS() \
                                                       : GR_SM10_SM_CTAS())

// Max registers per SM
#define GR_SM30_SM_REGISTERS() (65536)  // 65536 registers on SM3.0+
#define GR_SM20_SM_REGISTERS() (32768)  // 32768 registers on SM2.0+
#define GR_SM12_SM_REGISTERS() (16384)  // 16384 registers on SM1.2-SM1.3
#define GR_SM10_SM_REGISTERS() (8192)   // 8192 registers on SM1.0-SM1.1
#define GR_SM_REGISTERS(arch)                                             \
  ((arch >= 300) ? GR_SM30_SM_REGISTERS()                                 \
                 : (arch >= 200) ? GR_SM20_SM_REGISTERS()                 \
                                 : (arch >= 130) ? GR_SM12_SM_REGISTERS() \
                                                 : GR_SM10_SM_REGISTERS())

/*****************************************************************
 * Inlined PTX helper macros
 *****************************************************************/

// Register modifier for pointer-types (for inlining PTX assembly)
#if defined(_WIN64) || defined(__LP64__)
#define __GR_LP64__ 1
// 64-bit register modifier for inlined asm
#define _GR_ASM_PTR_ "l"
#else
#define __GR_LP64__ 0
// 32-bit register modifier for inlined asm
#define _GR_ASM_PTR_ "r"
#endif

/*****************************************************************
 * CUDA/GPU inspection utilities
 *****************************************************************/

/**
 * Empty Kernel
 */
template <typename T>
__global__ void FlushKernel(void) {}

/**
 * Class encapsulating device properties for dynamic host-side inspection
 */
class CudaProperties {
 public:
  // Information about our target device
  cudaDeviceProp device_props;
  int device_sm_version;

  // Information about our kernel assembly
  int kernel_ptx_version;

 public:
  CudaProperties() {}
  CudaProperties(int gpu) { Setup(gpu); }
  void Setup() {
    int current_device;
    cudaGetDevice(&current_device);
    Setup(current_device);
  }

  // Constructor
  void Setup(int gpu) {
    // Get current device properties
    cudaGetDeviceProperties(&device_props, gpu);
    device_sm_version = device_props.major * 100 + device_props.minor * 10;

    // Get SM version of compiled kernel assemblies
    cudaFuncAttributes flush_kernel_attrs;
    cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
    kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
  }
};

}  // namespace util
}  // namespace gunrock
