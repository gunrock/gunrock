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
 */
constexpr compute_capability_t make_compute_capability(unsigned major,
                                                       unsigned minor) {
  return compute_capability_t{major, minor};
}

/**
 * @brief Get compute capability from combined major and minor version.
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

// Device properties retrieved from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications

/**
 * @brief Maximum number of threads per block.
 */
inline constexpr unsigned cta_max_threads() {
  return 1 << 10;  // 1024 threads per CTA
}

/**
 * @brief Warp size (has always been 32, but is subject to change).
 */
inline constexpr unsigned warp_max_threads() {
  return 1 << 5;
}

/**
 * @brief Maximum number of resident blocks per SM.
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
 */
inline constexpr unsigned sm_registers(compute_capability_t capability) {
  return
    (capability >= 50) ?  64 * K :  // SM50+
    (capability >= 37) ? 128 * K :  // SM37
                          64 * K ;  // SM30-SM35
}

/**
 * @brief Maximum amount of shared memory per SM.
 */
inline constexpr unsigned sm_max_smem_bytes(compute_capability_t capability) {
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
    (capability >= 37) ? 112 * KiB :  // SM37
                          48 * KiB ;  // SM30-SM35
}

/**
 * @brief Number of shared memory banks.
 */
inline constexpr unsigned shared_memory_banks() {
  return 1 << 5;  // 32 memory banks per SM
}

/**
 * @brief Stride length of shared memory in bytes.
 */
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
