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
  unsigned major;
  unsigned minor;
  constexpr unsigned as_combined_number() const {
    return major * 10 + minor;
  }
  constexpr bool operator==(int i) { return (int)as_combined_number() == i; }
  constexpr bool operator!=(int i) { return (int)as_combined_number() != i; }
  constexpr bool operator>(int i) { return (int)as_combined_number() > i; }
  constexpr bool operator<(int i) { return (int)as_combined_number() < i; }
  constexpr bool operator>=(int i) { return (int)as_combined_number() >= i; }
  constexpr bool operator<=(int i) { return (int)as_combined_number() <= i; }
} compute_capability_t;

/**
 * @brief Get compute capability from major and minor versions.
 * @param major Compute capability major version
 * @param minor Compute capability minor version
 * \return compute_capability_t
 */
constexpr compute_capability_t make_compute_capability(unsigned major,
                                                       unsigned minor) {
  return compute_capability_t{major, minor};
}

/**
 * @brief Get compute capability from combined major and minor version.
 * @param combined Combined major and minor value, e.g. 86 for 8.6
 * \return compute_capability_t
 */
constexpr compute_capability_t make_compute_capability(unsigned combined) {
  return compute_capability_t{combined / 10, combined % 10};
}
/**
 * @namespace properties
 * C++ based CUDA device properties.
 */
namespace properties {

/**
 * @brief Enums for units used by device property values.
 */
enum : size_t {
  KiB = 1024,
  K   = 1024
};

/**
 * @brief Architecture name based on compute capability.
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
 * @param capability Compute capability from which to get the result
 * \return const char* architecture name or nullptr if capability is invalid
 */
inline constexpr const char* arch_name(compute_capability_t capability) {
  return
    (capability.major == 8) ?  "Ampere" :
    (capability.major == 7 && capability.minor == 5) ? "Turing" :
    (capability.major == 7) ?   "Volta" :
    (capability.major == 6) ?  "Pascal" :
    (capability.major == 5) ? "Maxwell" :
    (capability.major == 3) ?  "Kepler" :
                                nullptr ;
}

// Device properties retrieved from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications

/**
 * @brief Maximum number of threads per block.
 * \return unsigned
 */
inline constexpr unsigned cta_max_threads() {
  return 1 << 10;  // 1024 threads per CTA
}

/**
 * @brief Warp size (has always been 32, but is subject to change).
 * \return unsigned
 */
inline constexpr unsigned warp_max_threads() {
  return 1 << 5;
}

/**
 * @brief Maximum number of resident blocks per SM.
 * @param capability Compute capability from which to get the result
 * \return unsigned
 */
inline constexpr unsigned sm_max_ctas(compute_capability_t capability) {
  return
    (capability >= 86) ? 16 :  // SM86+
    (capability >= 80) ? 32 :  // SM80
    (capability >= 75) ? 16 :  // SM75
    (capability >= 50) ? 32 :  // SM50-SM72
                         16 ;  // SM30-SM37
}

/**
 * @brief Maximum number of resident threads per SM.
 * @param capability Compute capability from which to get the result
 * \return unsigned
 */
inline constexpr unsigned sm_max_threads(compute_capability_t capability) {
  return
    (capability >= 86) ? 1536 :  // SM86+
    (capability >= 80) ? 2048 :  // SM80
    (capability >= 75) ? 1024 :  // SM75
                         2048 ;  // SM30-SM72
}

/**
 * @brief Number of 32-bit registers per SM.
 * @param capability Compute capability from which to get the result
 * \return unsigned
 */
inline constexpr unsigned sm_registers(compute_capability_t capability) {
  return
    (capability >= 50) ?  64 * K :  // SM50+
    (capability >= 37) ? 128 * K :  // SM37
                          64 * K ;  // SM30-SM35
}

/**
 * @brief Maximum amount of shared memory per SM.
 * @tparam sm3XCacheConfig cudaFuncCache enum representing the shared data
 *                         cache configuration used when called on compute
 *                         capability 3.x
 * @param capability       Compute capability from which to get the result
 * \return unsigned
 * @todo Test if this function can be resolved at compile time
 */
template<enum cudaFuncCache sm3XCacheConfig = cudaFuncCachePreferNone>
inline constexpr unsigned sm_max_shared_memory_bytes(
  compute_capability_t capability
) {
  unsigned sm3XConfiguredSmem =
    (sm3XCacheConfig == cudaFuncCachePreferNone)   ? 48 * KiB :
    (sm3XCacheConfig == cudaFuncCachePreferShared) ? 48 * KiB :
    (sm3XCacheConfig == cudaFuncCachePreferL1)     ? 16 * KiB :
    (sm3XCacheConfig == cudaFuncCachePreferEqual)  ? 32 * KiB :
                                                     48 * KiB ;

  return
    (capability >= 86) ? 100 * KiB :  // SM86+
    (capability >= 80) ? 164 * KiB :  // SM80
    (capability >= 75) ?  64 * KiB :  // SM75
    (capability >= 70) ?  96 * KiB :  // SM70-SM72
    (capability >= 62) ?  64 * KiB :  // SM62
    (capability >= 61) ?  96 * KiB :  // SM61
    (capability >= 53) ?  64 * KiB :  // SM53
    (capability >= 52) ?  96 * KiB :  // SM52
    (capability >= 50) ?  64 * KiB :  // SM50
    (capability >= 37) ?  64 * KiB + sm3XConfiguredSmem :  // SM37
                sm3XConfiguredSmem ;  // SM30-SM35
}

/**
 * @brief Number of shared memory banks.
 * \return unsigned
 */
inline constexpr unsigned shared_memory_banks() {
  return 1 << 5;  // 32 memory banks per SM
}

/**
 * @brief Stride length (number of bytes per word) of shared memory in bytes.
 * @tparam sm3XSmemConfig cudaSharedMemConfig enum representing the shared
 *                        memory bank size (stride) used when called on compute
 *                        capability 3.x
 * \return unsigned
 */
template<enum cudaSharedMemConfig sm3XSmemConfig = cudaSharedMemBankSizeDefault>
inline constexpr unsigned shared_memory_bank_stride() {
  // The default config on 3.x is the same constant value for later archs
  // Only let 3.x be configurable if stride later becomes dependent on arch
  return
    (sm3XSmemConfig == cudaSharedMemBankSizeDefault)   ? 1 << 2 :
    (sm3XSmemConfig == cudaSharedMemBankSizeFourByte)  ? 1 << 2 :
    (sm3XSmemConfig == cudaSharedMemBankSizeEightByte) ? 1 << 3 :
                                                         1 << 2 ;
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
