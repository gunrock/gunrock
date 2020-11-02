/**
 * @file device_properties.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once
#include <iostream>
#include <gunrock/cuda/device.hxx>
#include <gunrock/error.hxx>

namespace gunrock {
namespace cuda {

typedef cudaDeviceProp device_properties_t;
typedef struct {
  const int cuda_arch_;
  const int major = (int)(cuda_arch_ / 100);
  const int minor = (cuda_arch_ / 10) % 10;
  constexpr bool operator>=(const int &i) const { return cuda_arch_ >= i; }
} architecture_t;

/**
 * @namespace properties
 * C++ based CUDA device properties.
 */
namespace properties {

enum : size_t {
  KiB = 1024,
  K   = 1024
};

/**
 * Device properties retrieved from:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
 */

/**
 * Maximum number of threads per block
 */
inline constexpr unsigned cta_max_threads() {
  return 1 << 10;  // 1024 threads per CTA
}

/**
 * Warp size
 */
inline constexpr unsigned warp_max_threads() {
  return 1 << 5;  // 32 threads per warp
}

/**
 * Maximum number of resident blocks per SM
 */
inline constexpr unsigned sm_max_ctas(architecture_t arch) {
  return
    (arch >= 860) ? 16 :  // SM86+
    (arch >= 800) ? 32 :  // SM80
    (arch >= 750) ? 16 :  // SM75
    (arch >= 500) ? 32 :  // SM50-SM72
                    16 ;  // SM30-SM37
}

/**
 * Maximum number of resident threads per SM
 */
inline constexpr unsigned sm_max_threads(architecture_t arch) {
  return
    (arch >= 860) ? 1536 :  // SM86+
    (arch >= 800) ? 2048 :  // SM80
    (arch >= 750) ? 1024 :  // SM75
                    2048 ;  // SM30-SM72
}

/**
 * Number of 32-bit registers per SM
 */
inline constexpr unsigned sm_registers(architecture_t arch) {
  return
    (arch >= 500) ?  64 * K :  // SM50+
    (arch >= 370) ? 128 * K :  // SM37
                     64 * K ;  // SM30-SM35
}

/**
 * Maximum amount of shared memory per SM
 */
inline constexpr unsigned sm_max_smem_bytes(architecture_t arch) {
  return
    (arch >= 860) ? 100 * KiB :  // SM86+
    (arch >= 800) ? 164 * KiB :  // SM80
    (arch >= 750) ?  64 * KiB :  // SM75
    (arch >= 700) ?  96 * KiB :  // SM70-SM72
    (arch >= 620) ?  64 * KiB :  // SM62
    (arch >= 610) ?  96 * KiB :  // SM61
    (arch >= 530) ?  64 * KiB :  // SM53
    (arch >= 520) ?  96 * KiB :  // SM52
    (arch >= 500) ?  64 * KiB :  // SM50
    (arch >= 370) ? 112 * KiB :  // SM37
                     48 * KiB ;  // SM30-SM35
}

/**
 * Number of shared memory banks
 */
inline constexpr unsigned shared_memory_banks() {
  return 1 << 5;  // 32 memory banks per SM
}

inline constexpr unsigned shared_memory_bank_stride() {
  return 1 << 2;  // 4 byte words
}

void print(device_properties_t& prop) {
  device_id_t ordinal;
  cudaGetDevice(&ordinal);

  size_t freeMem, totalMem;
  error::error_t status = cudaMemGetInfo(&freeMem, &totalMem);
  error::throw_if_exception(status);

  double memBandwidth =
      (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8 * 2) / 1.0e9;

  std::cout << prop.name << " : " << prop.clockRate / 1000.0 << " Mhz "
            << "(Ordinal " << ordinal << ")" << std::endl;
  std::cout << "FreeMem: " << (int)(freeMem / (1 << 20)) << " MB "
            << "TotalMem: " << (int)(totalMem / (1 << 20)) << " MB "
            << ((int)8 * sizeof(int*)) << "-bit pointers." << std::endl;
  std::cout << prop.multiProcessorCount
            << " SMs enabled, Compute Capability sm_" << prop.major
            << prop.minor << std::endl;
  std::cout << "Mem Clock: " << prop.memoryClockRate / 1000.0 << " Mhz x "
            << prop.memoryBusWidth << " bits (" << memBandwidth << " GB/s)"
            << std::endl;
  std::cout << "ECC " << (prop.ECCEnabled ? "Enabled" : "Disabled")
            << std::endl;
}

}  // namespace properties

}  // namespace cuda
}  // namespace gunrock
