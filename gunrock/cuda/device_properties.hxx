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

namespace gunrock {
namespace cuda {

typedef cudaDeviceProp device_properties_t;
typedef int architecture_t;

/**
 * @namespace properties
 * C++ based CUDA device properties.
 */
namespace properties {

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

inline constexpr unsigned shared_memory_banks() {
  return 1 << 5;  // 32 memory banks per SM
}

inline constexpr unsigned shared_memory_bank_stride() {
  return 1 << 2;  // 4 byte words
}

inline constexpr unsigned maximum_threads_per_warp() {
  return 1 << 5;  // 32 threads per warp
}

}  // namespace properties

}  // namespace cuda
}  // namespace gunrock