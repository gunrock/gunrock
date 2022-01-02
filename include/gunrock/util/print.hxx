/**
 * @file print.hxx
 *
 * @brief
 */

#pragma once

#include <string>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace gunrock {

/**
 * @namespace print
 * Print utilities.
 */
namespace print {

/**
 * @brief Print the first k elements of a vector.
 *
 * @tparam vector_t
 * @param x vector to print.
 * @param k number of elements to print.
 */
template <typename vector_t>
void head(vector_t& x, int k, std::string name = "") {
  using type_t = typename vector_t::value_type;
  if (x.size() < k)
    k = x.size();

  if (name.size() > 0)
    std::cout << name << "[:" << k << "] = ";

  thrust::copy(x.begin(), x.begin() + k,
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;
}

/**
 * @brief Print the first k elements of an array (device or host). Requires
 * expensive copies, not intended to be performant.
 *
 * @tparam type_t type of the array.
 * @param x pointer to be printed.
 * @param k number of elements to be printed.
 * @param n number of elements in the array.
 */
template <typename type_t>
void head(type_t* x, int k, int n, std::string name = "") {
  if (n < k)
    k = n;

  if (name.size() > 0)
    std::cout << name << "[:" << k << "] = ";

  thrust::device_vector<type_t> d_tmp(x, x + k);
  thrust::host_vector<type_t> h_tmp = d_tmp;
  thrust::copy(h_tmp.begin(), h_tmp.end(),
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;
}

}  // namespace print
}  // namespace gunrock