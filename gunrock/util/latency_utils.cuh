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

template <typename SizeT>
__global__ void Load_Kernel(SizeT num_repeats, SizeT num_elements,
                            int *d_data) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  const int a = 19;
  const int b = 717;

  long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long STRIDE = (long long)blockDim.x * gridDim.x;
  while (x < num_elements) {
    int val = d_data[pos];
    for (int i = 0; i < num_repeats; i++) {
      val = val * a + b;
    }
    d_data[pos] = val;

    x += STRIDE;
  }
}

template <typename SizeT>
cudaError_t Get_BaseLine(
    // int num_blocks,
    // int block_size,
    SizeT num_repeats, SizeT num_elements, cudaStream_t stream,
    float &elapsed_ms, int *d_data) {
  cudaError_t retval = cudaSuccess;
  cudaEvent_t start, stop;
  if (retval = cudaEventCreate(&start)) return retval;
  if (retval = cudaEventCreate(&stop)) return retval;

  if (retval = cudaEventRecord(start, stream)) return retval;
  for (int i = 0; i < 10; i++)
    Load_Kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(num_repeats,
                                                       num_elements, d_data);
  if (retval = cudaEventRecord(stop, stream)) return retval;
  if (retval = cudaEventSynchronize(stop)) return retval;

  if (retval = cudaEventElapsedTime(&elapsed_ms, start, stop)) return retval;
  elapsed_ms /= 10;
  if (retval = cudaEventDestroy(start)) return retval;
  if (retval = cudaEventDestroy(stop)) return retval;
  return retval;
}

template <typename Array>
cudaError_t Test_BaseLine(const char *name, long long num_repeats,
                          long long num_elements, cudaStream_t stream,
                          Array &data) {
  cudaError_t retval = cudaSuccess;
  float elapsed_time = 0;
  if (num_repeats == 0) return retval;
  if (retval = util::latency::Get_BaseLine(num_repeats, num_elements, stream,
                                           elapsed_time,
                                           data.GetPointer(util::DEVICE)))
    return retval;
  printf("%s\t = (%lld, %lld)\t = %f us\n", name, num_repeats, num_elements,
         elapsed_time * 1000);
  return retval;
}

template <typename SizeT>
cudaError_t Insert_Latency(
    // int num_blocks,
    // int block_size,
    SizeT num_repeats, SizeT num_elements, cudaStream_t stream, int *d_data) {
  Load_Kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(num_repeats, num_elements,
                                                     d_data);
  return cudaSuccess;
}

template <typename Array>
cudaError_t Test(long long num_elements, cudaStream_t stream, Array &data,
                 long long communicate_latency, float communicate_multipy,
                 long long expand_latency, long long subqueue_latency,
                 long long fullqueue_latency, long long makeout_latency) {
  cudaError_t retval = cudaSuccess;

  if (retval = util::latency::Test_BaseLine("communicate_latency",
                                            communicate_latency, num_elements,
                                            stream, data))
    return retval;

  if (retval = util::latency::Test_BaseLine("expand_latency  ", expand_latency,
                                            num_elements, stream, data))
    return retval;

  if (retval = util::latency::Test_BaseLine(
          "subqueue_latency", subqueue_latency, num_elements, stream, data))
    return retval;

  if (retval = util::latency::Test_BaseLine(
          "fullqueue_latency", fullqueue_latency, num_elements, stream, data))
    return retval;

  if (retval = util::latency::Test_BaseLine(
          "makeout_latency  ", makeout_latency, num_elements, stream, data))
    return retval;
  return retval;
}

template <typename Array>
cudaError_t Test(cudaStream_t stream, Array &data,
                 long long communicate_latency, float communicate_multipy,
                 long long expand_latency, long long subqueue_latency,
                 long long fullqueue_latency, long long makeout_latency) {
  cudaError_t retval = cudaSuccess;
  const long long elements[] = {1,     10,     100,     1000,
                                10000, 100000, 1000000, 10000000};

  if (communicate_multipy > 0)
    printf("communicate_multipy\t = %.2fx\n", communicate_multipy);

  for (int i = 0; i < 8; i++) {
    if (retval = Test(elements[i], stream, data, communicate_latency,
                      communicate_multipy, expand_latency, subqueue_latency,
                      fullqueue_latency, makeout_latency))
      return retval;
  }
  return retval;
}

}  // namespace latency
}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
