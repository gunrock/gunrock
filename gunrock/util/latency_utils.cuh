// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_loop.cuh
 *
 * @brief Base Iteration Loop
 */

#pragma once

namespace gunrock {
namespace util {
namespace latency {

#define NUM_BLOCKS 120
#define BLOCK_SIZE 1024

__global__ void Load_Kernel(
    int num_repeats,
    int *d_data)
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int x = d_data[pos];
    const int a = 19;
    const int b = 717;

    for (int i=0; i<num_repeats; i++)
    {
        x = x * a + b;
    }
    d_data[pos] = x;
}

cudaError_t Get_BaseLine(
    //int num_blocks,
    //int block_size,
    int num_repeats,
    cudaStream_t stream,
    float &elapsed_ms,
    int *d_data)
{
    cudaError_t retval = cudaSuccess;
    cudaEvent_t start, stop;
    if (retval = cudaEventCreate(&start)) return retval;
    if (retval = cudaEventCreate(&stop )) return retval;

    if (retval = cudaEventRecord(start, stream)) return retval;
    for (int i=0; i<10; i++)
        Load_Kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>
            (num_repeats, d_data);
    if (retval = cudaEventRecord(stop , stream)) return retval;
    if (retval = cudaEventSynchronize(stop)) return retval;

    if (retval = cudaEventElapsedTime(&elapsed_ms, start, stop)) return retval;
    elapsed_ms /= 10;
    if (retval = cudaEventDestroy(start)) return retval;
    if (retval = cudaEventDestroy(stop )) return retval;
    return retval;
}

template <typename Array>
cudaError_t Test_BaseLine(
    const char* name,
    int num_repeats,
    cudaStream_t stream,
    Array &data)
{
    cudaError_t retval = cudaSuccess;
    float elapsed_time = 0;
    if (num_repeats == 0) return retval;
    if (retval = util::latency::Get_BaseLine(num_repeats, stream, 
        elapsed_time, data.GetPointer(util::DEVICE)))
        return retval;
    printf("%s\t = %d\t = %f us\n",
        name, num_repeats, elapsed_time * 1000);
    return retval;
}

cudaError_t Insert_Latency(
    //int num_blocks,
    //int block_size,
    int num_repeats,
    cudaStream_t stream,
    int *d_data)
{
    Load_Kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>
        (num_repeats, d_data);
    return cudaSuccess;
}

} // namespace latency
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

