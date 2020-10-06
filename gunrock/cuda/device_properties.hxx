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

  // XXX: I don't like printfs, use std::cout here
  printf("%s : %8.3lf Mhz   (Ordinal %d)\n", prop.name, prop.clockRate / 1000.0,
         ordinal);
  printf("%d SMs enabled. Compute Capability sm_%d%d\n",
         prop.multiProcessorCount, prop.major, prop.minor);
  printf("FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n",
         (int)(freeMem / (1 << 20)), (int)(totalMem / (1 << 20)),
         (int)8 * sizeof(int*));
  printf("Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n",
         prop.memoryClockRate / 1000.0, prop.memoryBusWidth, memBandwidth);
  printf("ECC %s\n\n", prop.ECCEnabled ? "Enabled" : "Disabled");
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