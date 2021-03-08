#include <stdio.h>

struct der1 {
  der1() {}
  int size1;
  __host__ __device__ int get_size() const { return size1; }
};

struct der2 {
  der2() {}
  int size2;
  __host__ __device__ int get_size() const { return size2; }
};

struct join : public der1, public der2 {
  join() : der1(), der2() {}

  template <typename view_t = der1>
  __host__ __device__ int get_size() const {
    return view_t::get_size();
  }
};

template <typename join_t>
__global__ void kernel(join_t container) {
  int size = container.template get_size<der2>();  // this doesnt compile.
  printf("size = %i\n", size);
}

void test_templated() {
  using index_t = int;
  using value_t = float;

  // Host stuff.
  join host_container;

  host_container.size1 = 10;
  host_container.size2 = 20;

  int size = host_container.template get_size<der1>();  // this compiles.
  printf("size = %i\n", size);

  // Device stuff.
  join dev_container;

  dev_container.size1 = 10;
  dev_container.size2 = 20;

  kernel<<<1, 1>>>(dev_container);
  cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
  test_templated();
}