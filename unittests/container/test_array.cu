// XXX: dummy template for unit testing

#include <gunrock/container/array.cuh>
typedef cudaError_t test_error_t;

template<typename T>
__global__ void kernel(T data) {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= data.size()) return;
  data[idx] = idx;
  printf("data[%i] = %i\n", idx, data[idx]);
}

test_error_t
test_array()
{
  using namespace gunrock;
  using namespace container::dense;
  typedef array<int, size_t> array_t;

  test_error_t status = cudaSuccess;
  size_t N = 128;
  
  array_t a(N);

  dim3 dimBlock(N);
  dim3 dimGrid(1);

  kernel<<<dimGrid, dimBlock>>>(a);
  cudaDeviceSynchronize();

  return status;
}

int
main(int argc, char** argv)
{
  return test_array();
}