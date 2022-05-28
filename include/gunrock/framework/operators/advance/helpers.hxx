/**
 * @file helpers.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Helper functions for Advance operators.
 * @todo These can be potentially moved under frontier's API.
 * @version 0.1
 * @date 2021-01-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/cuda/context.hxx>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>

namespace gunrock {
namespace operators {
namespace advance {

/**
 * @brief Given a frontier and a graph, find the offsets segments for the output
 * frontier of an advance operation (@see advance.hxx).
 *
 * @tparam graph_t Graph type
 * @tparam frontier_t Frontier type
 * @tparam work_tiles_t Segments type
 * @param G Graph
 * @param input Input frontier
 * @param segments Segments array
 * @param context CUDA context
 * @param graph_as_frontier if true, entire graph is used instead of the
 * frontier.
 * @return std::size_t number of total elements in the output frontier.
 */
template <typename graph_t, typename frontier_t, typename work_tiles_t>
std::size_t compute_output_offsets(graph_t& G,
                                   frontier_t* input,
                                   work_tiles_t& segments,
                                   gcuda::standard_context_t& context,
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

  return size_of_output[0];
}

/**
 * @brief Cheaper than compute_output_offsets, only calculates the number of
 * total elements in the output frontier. Maybe used to allocate the output
 * frontier.
 *
 * @tparam graph_t Graph type
 * @tparam frontier_t Frontier type
 * @param G Graph object
 * @param input Input frontier
 * @param context CUDA context
 * @param graph_as_frontier if true, entire graph is used instead of the
 * frontier.
 * @return std::size_t number of total elements in the output frontier
 */
template <typename graph_t, typename frontier_t>
std::size_t compute_output_length(graph_t& G,
                                  frontier_t& input,
                                  gcuda::standard_context_t& context,
                                  bool graph_as_frontier = false) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  if (graph_as_frontier)
    return G.get_number_of_edges();

  auto input_data = input.data();
  auto total_elems = input.get_number_of_elements();

  auto segment_sizes = [=] __host__ __device__(std::size_t const& i) {
    auto v = input_data[i];
    // if item is invalid, segment size is 0.
    if (!gunrock::util::limits::is_valid(v))
      return edge_t(0);
    else
      return G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_reduce(
      thrust::cuda::par.on(context.stream()),          // execution policy
      thrust::make_counting_iterator<std::size_t>(0),  // input iterator: first
      thrust::make_counting_iterator<std::size_t>(
          total_elems),       // input iterator: last
      segment_sizes,          // unary operation
      edge_t(0),              // initial value
      thrust::plus<edge_t>()  // binary operation
  );

  return new_length;
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock