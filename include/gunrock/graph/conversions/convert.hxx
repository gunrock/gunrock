/**
 * @file convert.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-12-04
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/cuda/cuda.hxx>

namespace gunrock {
namespace graph {
namespace convert {

template <memory_space_t space, typename index_t, typename offset_t>
void offsets_to_indices(const index_t* offsets,
                        offset_t const& size_of_offsets,
                        offset_t* indices,
                        index_t const& size_of_indices) {
  using execution_policy_t = std::conditional_t<space == memory_space_t::device,
                                                decltype(thrust::cuda::par.on(
                                                    0)),  // XXX: does this
                                                          // work on stream 0?
                                                decltype(thrust::host)>;
  execution_policy_t exec;
  // convert compressed offsets into uncompressed indices
  thrust::fill(exec, indices + 0, indices + size_of_indices, offset_t(0));

  thrust::scatter_if(
      exec,                                    // execution policy
      thrust::counting_iterator<offset_t>(0),  // begin iterator
      thrust::counting_iterator<offset_t>(size_of_offsets - 1),  // end iterator
      offsets + 0,  // where to scatter
      thrust::make_transform_iterator(
          thrust::make_zip_iterator(
              thrust::make_tuple(offsets + 0, offsets + 1)),
          [=] __host__ __device__(const thrust::tuple<offset_t, offset_t>& t) {
            thrust::not_equal_to<offset_t> comp;
            return comp(thrust::get<0>(t), thrust::get<1>(t));
          }),
      indices + 0);

  thrust::inclusive_scan(exec, indices + 0, indices + size_of_indices,
                         indices + 0, thrust::maximum<offset_t>());
}

template <memory_space_t space, typename index_t, typename offset_t>
void indices_to_offsets(const index_t* indices,
                        index_t const& size_of_indices,
                        offset_t* offsets,
                        offset_t const& size_of_offsets) {
  using execution_policy_t =
      std::conditional_t<space == memory_space_t::device,
                         decltype(thrust::device), decltype(thrust::host)>;
  execution_policy_t exec;
  // convert uncompressed indices into compressed offsets
  thrust::lower_bound(exec, indices, indices + size_of_indices,
                      thrust::counting_iterator<offset_t>(0),
                      thrust::counting_iterator<offset_t>(size_of_offsets),
                      offsets + 0);
}

}  // namespace convert
}  // namespace graph
}  // namespace gunrock
