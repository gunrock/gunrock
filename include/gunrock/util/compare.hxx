/**
 * @file compare.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-31
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <cstddef>
#include <iostream>

namespace gunrock {
namespace util {

namespace detail {
auto default_comparator = [](auto& a, auto& b) -> bool { return a != b; };
}

/**
 * @brief Compares values between a device pointer and a host pointer of same
 * length, and returns number of errors/mismatches found.
 *
 * @tparam type_t Type of the values to be compared.
 * @tparam comp_t Type of the error comparator function.
 * @param d_ptr device pointer
 * @param h_ptr host pointer
 * @param n number of elements to compare
 * @param error_op lambda function to compare two value that result in a
 * mismatch count to increment (default a != b).
 * @param verbose if true, prints out the mismatches
 * @return std::size_t number of mismatches found
 */
template <typename type_t,
          typename comp_t = decltype(detail::default_comparator)>
std::size_t compare(const type_t* d_ptr,
                    const type_t* h_ptr,
                    const std::size_t n,
                    comp_t error_op = detail::default_comparator,
                    const bool verbose = false) {
  thrust::host_vector<type_t> d_vec(n);
  cudaMemcpy(d_vec.data(), d_ptr, n * sizeof(type_t), cudaMemcpyDeviceToHost);

  std::size_t error_count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (error_op(d_vec[i], h_ptr[i])) {
      if (verbose)
        std::cout << "Error: " << d_vec[i] << " != " << h_ptr[i] << std::endl;
      ++error_count;
    }
  }
  return error_count;
}

}  // namespace util
}  // namespace gunrock