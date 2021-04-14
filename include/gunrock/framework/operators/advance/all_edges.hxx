/**
 * @file all_edges.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-01-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/transform_scan.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace all_edges {

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  using edge_t = typename graph_t::edge_type;

  // Prepare output for all edges advance.

  // If no edges found in the graph,
  // output an empty frontier.
  if (G.get_number_of_edges() <= 0) {
    output->set_number_of_elements(0);
    return;
  }

  /*!
   * @todo Resize the output (inactive) buffer to the new size. Can be hidden
   * within the frontier struct.
   */
  if (output->get_capacity() < G.get_number_of_edges())
    output->reserve(G.get_number_of_edges());
  output->set_number_of_elements(G.get_number_of_edges());

  auto all_edges_kernel = [=] __device__(edge_t const& e) {
    auto pair = G.get_source_and_destination_vertices(e);
    auto w = G.get_edge_weight(e);
    return op(pair.source, pair.destination, e, w)
               ? pair.destination
               : gunrock::numeric_limits<decltype(pair.source)>::invalid();
  };

  thrust::transform(thrust::cuda::par.on(context.stream()),
                    thrust::make_counting_iterator<edge_t>(0),  // Begin: 0
                    thrust::make_counting_iterator<edge_t>(
                        G.get_number_of_edges()),  // End: # of Edges
                    output->begin(),               // Output frontier
                    all_edges_kernel               // Unary Operator
  );
}
}  // namespace all_edges
}  // namespace advance
}  // namespace operators
}  // namespace gunrock