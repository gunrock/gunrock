// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.cuh
 *
 * @brief Utility Routines for Tests
 */

#pragma once

#include <gunrock/util/test_utils.h>
#include <gunrock/util/error_utils.cuh>

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


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceResults(
    T *h_reference,
    T *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (int i = 0; i < num_elements; i++) {
            PrintValue(h_reference[i]);
            printf(", ");
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_elements; i++) {
            PrintValue(h_data[i]);
            printf(", ");
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_elements, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}

int CompareDeviceResults(
    util::NullType *h_reference,
    util::NullType *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    return 0;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_elements * sizeof(T));
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_reference, d_reference, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (int i = 0; i < num_elements; i++) {
            PrintValue(h_reference[i]);
            printf(", ");
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_elements; i++) {
            PrintValue(h_data[i]);
            printf(", ");
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_elements, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
void DisplayDeviceResults(
    T *d_data,
    size_t num_elements)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    printf("\n\nData:\n");
    for (int i = 0; i < num_elements; i++) {
        PrintValue(h_data[i]);
        printf(", ");
    }
    printf("\n\n");

    // Cleanup
    if (h_data) free(h_data);
}

/******************************************************************************
 * Timing
 ******************************************************************************/

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Check available device memory
bool EnoughDeviceMemory(unsigned int mem_needed)
{
    size_t free_mem, total_mem;
    if (util::GRError(cudaMemGetInfo(&free_mem, &total_mem),
                "cudaMemGetInfo failed", __FILE__, __LINE__)) return false;
    return (mem_needed <= free_mem);
}


}// namespace util
}// namespace gunrock
