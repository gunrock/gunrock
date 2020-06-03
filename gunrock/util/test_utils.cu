// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.cu
 *
 * @brief Utility Routines for Tests
 */
#include <gunrock/util/test_utils.h>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace util {

bool IsDevicePointer(const void *ptr)
{
    cudaPointerAttributes attributes;
    auto err = cudaPointerGetAttributes(&attributes, ptr);

    // An error here indicates the memory was LIKELY
    // allocated on the host or the pointer is gibberish.
    if(err != cudaSuccess)
    {
        // Clear out the last cuda error. We expected this error
        // because it implies we have a host side pointer.
        cudaGetLastError();
        return false;
    }

    if(attributes.devicePointer != nullptr)
    {
        return true;
    }

    return false;
}

/******************************************************************************
 * Device initialization
 ******************************************************************************/

void DeviceInit(CommandLineArgs &args) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "No devices supporting CUDA.\n");
    exit(1);
  }
  std::vector<int> devs;
  args.GetCmdLineArguments("device", devs);
  if (devs.size() == 0)
    for (int i = 0; i < deviceCount; i++) devs.push_back(i);
  else if (devs.size() == 1) {
    if (devs[0] < 0) {
      devs[0] = 0;
    }
    if (devs[0] > deviceCount - 1) {
      devs[0] = deviceCount - 1;
    }
  }
  for (unsigned long i = 0; i < devs.size(); i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devs[i]);
    if (deviceProp.major < 1) {
      fprintf(stderr, "Device does not support CUDA.\n");
      exit(1);
    }
    if (!args.CheckCmdLineFlag("quiet")) {
      printf("Using device %d: %s\n", devs[i], deviceProp.name);
    }
  }
  cudaSetDevice(devs[0]);
}

cudaError_t SetDevice(int dev) {
  return util::GRError(cudaSetDevice(dev), "cudaSetDevice failed.", __FILE__,
                       __LINE__);
}

cudaError_t GetDevice(int* dev) {
  return util::GRError(cudaGetDevice(dev), "cudaGetDevice failed.", __FILE__,
                      __LINE__);
}

}  // namespace util
}  // namespace gunrock
