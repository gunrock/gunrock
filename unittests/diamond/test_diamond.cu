#include <cstdlib>  // EXIT_SUCCESS
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gunrock/error.hxx>  // error checking

using namespace gunrock;

template <typename join_t>
__global__ void kernel(join_t monster) {
  float val = monster.get_value_at(1);
  int der1_size = monster.get_total_size();

  printf("value[1] = %f\n", val);
  printf("size = %i\n", der1_size);
}

struct base {
  base() {}
  __host__ __device__ virtual int get_total_size() const = 0;

  void set_base_size(int const& s) { base_size = s; }

 protected:
  int base_size;
};

// no error
struct der1 : public base {
  der1() : base() {}

  float* ptr1;
  int size1;

  __host__ __device__ float get_value_at(int const& i) const { return ptr1[i]; }

  __host__ __device__ int get_size() const { return size1; }

  __host__ __device__ int get_total_size() const override {
    return base::base_size + get_size();
  }
};

// error when using virtual
struct der2 : public virtual base {
  der2() : base() {}

  float* ptr2;
  int size2;

  __host__ __device__ float get_value_at(int const& i) const { return ptr2[i]; }

  __host__ __device__ int get_size() const { return size2; }

  __host__ __device__ int get_total_size() const override {
    return base::base_size + get_size();
  }
};

struct join : public der1 /* , public der2 */ {
  join() : der1() /* , der2() */ {}

  __host__ __device__ int get_total_size() const override {
    return der1::get_total_size();
  }
};

template <typename vector_struct_t>
auto set_diamond(vector_struct_t& v, vector_struct_t& v2) {
  join my_container;
  int base_size = 10;

  my_container.ptr1 = thrust::raw_pointer_cast(v.data());
  // my_container.ptr2 = thrust::raw_pointer_cast(v2.data());

  my_container.set_base_size(base_size);
  my_container.size1 = v.size();
  // my_container.size2 = v2.size();

  return my_container;
}

void test_diamond() {
  using index_t = int;
  using value_t = float;

  error::error_t status = cudaSuccess;

  // let's use thrust vector<type_t> for initial arrays
  thrust::host_vector<value_t> h_vector(10);
  for (index_t i = 0; i < 10; ++i)
    h_vector[i] = i;

  thrust::device_vector<value_t> d_vector = h_vector;
  thrust::device_vector<value_t> d_vector2 = h_vector;

  auto my_container = set_diamond(d_vector, d_vector2);

  // Device Output
  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);
  std::cout << "Entering Kernel (host)" << std::endl;

  kernel<<<1, 1>>>(my_container);

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  std::cout << "Kernel Exited (host)" << std::endl;
}

int main(int argc, char** argv) {
  test_diamond();
  return EXIT_SUCCESS;
}