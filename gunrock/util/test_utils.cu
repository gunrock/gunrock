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

namespace gunrock {
namespace util {

/******************************************************************************
 * Device initialization
 ******************************************************************************/

void DeviceInit(CommandLineArgs &args)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No devices supporting CUDA.\n");
        exit(1);
    }
    int dev = 0;
    args.GetCmdLineArgument("device", dev);
    if (dev < 0) {
        dev = 0;
    }
    if (dev > deviceCount - 1) {
        dev = deviceCount - 1;
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major < 1) {
        fprintf(stderr, "Device does not support CUDA.\n");
        exit(1);
    }
    if (!args.CheckCmdLineFlag("quiet")) {
        printf("Using device %d: %s\n", dev, deviceProp.name);
    }

    cudaSetDevice(dev);
}

} //util
} //gunrock
