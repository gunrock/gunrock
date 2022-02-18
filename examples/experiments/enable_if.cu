/**
 * @file enable_if.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Enable if bug?
 * @version 0.1
 * @date 2022-01-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

enum my_enum_t {
  X,  /// for each X
  Y   /// for each Y
};

template <my_enum_t t,
          typename vector_t,
          typename operator_t,
          // error: due to overloading
          std::enable_if_t<t == my_enum_t::X, int> = 0>
void execute(vector_t& v, operator_t op) {
  auto data = v.data().get();
  thrust::for_each(             // for each
      thrust::cuda::par.on(0),  // stream
      v.begin(),                // begin
      v.end(),                  // end
      // Notice how the two lambdas differ (e.g. of how enum could
      // lead to different paths we may want to take.)
      [=] __device__(int x) { op(x); }  // lambda
  );
}

template <my_enum_t t,
          typename vector_t,
          typename operator_t,
          // error: due to overloading
          std::enable_if_t<t == my_enum_t::Y, int> = 0>
void execute(vector_t& v, operator_t op) {
  auto data = v.data().get();
  thrust::for_each(                              // for each
      thrust::cuda::par.on(0),                   // stream
      thrust::counting_iterator<int>(0),         // begin
      thrust::counting_iterator<int>(v.size()),  // end
      // Notice how the two lambdas differ (e.g. of how enum could
      // lead to different paths we may want to take.)
      [=] __device__(int i) { op(data[i] + (i * i)); }  // lambda
  );
}

int main(int argc, char** argv) {
  thrust::device_vector<int> v(10, 1);
  execute<my_enum_t::X>(v, [] __device__(int i) { printf("%d\n", i); });
  execute<my_enum_t::Y>(v, [] __device__(int i) { printf("%d\n", i); });
}