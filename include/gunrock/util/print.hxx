/**
 * @file print.hxx
 *
 * @brief
 */

#pragma once

namespace gunrock {

/**
 * @namespace print
 * Print utilities.
 */
namespace print {

// Print the first k elements of a `thrust` vector
template <typename val_t, typename vec_t>
void head(vec_t& x, int k) {
  thrust::copy(
    x.begin(),
    (x.size() < k) ? x.begin() + x.size() : x.begin() + k,
    std::ostream_iterator<val_t>(std::cout, " ")
  );
  std::cout << std::endl;
}

// Print the first k elements from a device pointer
template <typename val_t>
void head(val_t* x, int k) {
  thrust::device_vector<val_t> d_tmp(x, x + k);
  thrust::host_vector<val_t> h_tmp = d_tmp;
  thrust::copy(h_tmp.begin(), h_tmp.end(), std::ostream_iterator<val_t>(std::cout, " "));
  std::cout << std::endl;
}

}  // namespace print
}  // namespace gunrock