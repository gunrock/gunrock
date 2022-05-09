#include <cstdlib>  // EXIT_SUCCESS

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#define ERROR_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <typename T>
__global__ void kernel(T t) {
  t.what_is_inside();
}

template <typename op_t>
__global__ void lambda_kernel(op_t op) {
  int idx = threadIdx.x;
  auto discard = op(idx);
}

struct thrust_container {
  // Constructor
  thrust_container() {
    m = std::make_shared<thrust::device_vector<int>>(
        thrust::device_vector<int>(1, 1));
    raw_ptr = nullptr;
  }

  // Copy Constructor
  thrust_container(thrust_container const& rhs) {
    m = rhs.m;
    raw_ptr = rhs.m.get()->data().get();
  }

  __host__ __device__ void what_is_inside() const { printf("%i\n", *raw_ptr); }

 private:
  std::shared_ptr<thrust::device_vector<int>> m;
  int* raw_ptr;
};

void test_copy_ctor(thrust_container& t) {
  // works.
  kernel<<<1, 1>>>(t);
  ERROR_CHECK(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  auto lambda_op = [=] __device__(const int& idx) {
    t.what_is_inside();
    return 0;
  };

  // works.
  lambda_kernel<<<1, 1>>>(lambda_op);
  ERROR_CHECK(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  // fails when using thrust transform.
  // fix: use copy constructor with __device__ __host__ attributes.
  thrust::transform(
      thrust::device,                          // execution policy
      thrust::make_counting_iterator<int>(0),  // input iterator: first
      thrust::make_counting_iterator<int>(1),  // input iterator: last
      thrust::make_discard_iterator(),         // output iterator: ignore
      lambda_op                                // unary operation
  );
  ERROR_CHECK(cudaPeekAtLastError());
  cudaDeviceSynchronize();
}

int main() {
  thrust_container t;
  test_copy_ctor(t);
  return EXIT_SUCCESS;
}