/**
 * @file helpers.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-01-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/cuda/context.hxx>
#include <thrust/transform_scan.h>

namespace gunrock {
namespace operators {
namespace advance {

template <typename graph_t, typename frontier_t, typename work_tiles_t>
std::size_t compute_output_length(graph_t& G,
                                  frontier_t* input,
                                  work_tiles_t& segments,
                                  cuda::standard_context_t& context,
                                  bool graph_as_frontier = false) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  auto input_data = input->data();
  auto total_elems = graph_as_frontier ? G.get_number_of_vertices()
                                       : input->get_number_of_elements();

  // XXX: todo, maybe use capacity instead?
  if (segments.size() < total_elems + 1)
    segments.resize(total_elems + 1);

  auto segment_sizes = [=] __host__ __device__(std::size_t const& i) {
    if (i == total_elems)  // XXX: this is a weird exc. scan.
      return edge_t(0);

    auto v = graph_as_frontier ? vertex_t(i) : input_data[i];
    // if item is invalid, segment size is 0.
    if (!gunrock::util::limits::is_valid(v))
      return edge_t(0);
    else
      return G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_exclusive_scan(
      thrust::cuda::par.on(context.stream()),          // execution policy
      thrust::make_counting_iterator<std::size_t>(0),  // input iterator: first
      thrust::make_counting_iterator<std::size_t>(total_elems +
                                                  1),  // input iterator: last
      segments.begin(),                                // output iterator
      segment_sizes,                                   // unary operation
      edge_t(0),                                       // initial value
      thrust::plus<edge_t>()                           // binary operation
  );

  // The last item contains the total scanned items, so in a simple
  // example, where the input = {1, 0, 2, 2, 1, 3} resulted in the
  // inclusive scan output = {1, 1, 3, 5, 6, 9}, then output.size() - 1
  // will contain the element 9, which is the number of total items to process.
  // We can use this to allocate the size of the output frontier.
  auto location_of_total_scanned_items =
      thrust::distance(segments.begin(), new_length) - 1;

  // Move the last element of the scanned work-domain to host.
  // Last Element = size of active buffer - 1;
  // If the active buffer is greater than number of vertices,
  // we should TODO: resize the scanned work domain, this happens
  // when we allow duplicates to be in the active buffer.
  thrust::host_vector<edge_t> size_of_output(
      segments.data() + location_of_total_scanned_items,
      segments.data() + location_of_total_scanned_items + 1);

  // DEBUG::
  // std::cout << "Scanned Segments (THRUST) = ";
  // thrust::copy(segments.begin(), segments.end(),
  //              std::ostream_iterator<edge_t>(std::cout, " "));
  // std::cout << std::endl;
  // std::cout << "Size of Output (THRUST) = " << size_of_output[0] <<
  // std::endl;

  return size_of_output[0];
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock