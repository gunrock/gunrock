// XXX: dummy template for unit testing

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <gunrock/container/array.cuh>
typedef cudaError_t test_error_t;

template<std::size_t N, typename T>
__global__ void kernel(T a) 
{
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx > N) return;

  a[idx] = (float)idx;
  printf("a[%i] = %f\n", idx, a[idx]);
}

test_error_t
test_array()
{
  using namespace gunrock;
  using namespace container::dense;

  test_error_t status         = cudaSuccess;
  constexpr std::size_t N     = 10;

  array<float, N>               a;

  float* pointer              = a.data();
  const float* const_pointer  = a.data();

  std::size_t size            = a.size();
  std::size_t max_size        = a.max_size();
  bool is_empty               = a.empty();

  cudaDeviceSynchronize();
  kernel<N><<<1, N>>>(a);
  cudaDeviceSynchronize();

  // Segmentation fault; no host support
  // XXX: this is trivial to add using
  // a thrust::host_vector, but we have
  // to handle move symantics ourselves,
  // and that is when things get really
  // complicated. I will consider this if
  // find it useful.
  // a[0] = 0;

  return status;
}

int
main(int argc, char** argv)
{
  return test_array();
}