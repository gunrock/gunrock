#include <cstdlib>            // EXIT_SUCCESS
#include <gunrock/error.hxx>  // error checking

#include <gunrock/container/array.hxx>  // array support for the loop
#include <gunrock/cuda/context.hxx>     // context to run on

#include <gunrock/framework/operators/for/for.hxx>       // for operator
#include <gunrock/framework/operators/for/for_each.hxx>  // for_each operator

void test_for() {
  using namespace gunrock;

  constexpr std::size_t N = 10;

  gunrock::array<float, N> host;

  float* pointer = host.data();
  const float* const_pointer = host.data();

  std::size_t size = host.size();
  std::size_t max_size = host.max_size();
  bool is_empty = host.empty();

  std::cout << "Array.size() = " << size << std::endl;
  std::cout << "Array.max_size() = " << max_size << std::endl;
  std::cout << "Is Array Empty? " << std::boolalpha << is_empty << std::endl;

  for (std::size_t i = 0; i < N; ++i) {
    host[i] = i;
    std::cout << "host[" << i << "] = " << host[i] << std::endl;
  }

  // Initialize the context.
  cuda::device_id_t device = 0;
  cuda::standard_context_t context(device);

  // Run parallel_for, create an array inside for and set/print the values.
  operators::parallel_for::execute(0, N,
                                   [N] __host__ __device__(std::size_t & i) {
                                     gunrock::array<float, N> device;
                                     device[i] = i;
                                     printf("a[%i] = %f\n", (int)i, device[i]);
                                   },
                                   context);

  // Without synchronize, it won't print.
  context.synchronize();

  thrust::device_vector<float> v(N);
  auto data = v.data().get();

  // Run parallel_for_each, create an array inside for and set/print the values.
  operators::parallel_for_each::execute(
      data, 0, N,
      [] __host__ __device__(float* reference) {
        *reference = 1;
        printf("%f\n", *reference);
      },
      context);

  context.synchronize();
}

int main(int argc, char** argv) {
  test_for();
  return EXIT_SUCCESS;
}