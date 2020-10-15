/**
 * @file color.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/applications/color/color_implementation.hxx>

#pragma once

namespace gunrock {
namespace color {

using namespace memory;

template <memory_space_t space, typename graph_type, typename host_graph_type>
float color(graph_type* G,
            host_graph_type* g,
            typename graph_type::vertex_pointer_t colors) {
  using color_problem_type = color_problem_t<graph_type, host_graph_type>;
  using color_enactor_type = color_enactor_t<color_problem_type>;

  // Create contexts for all the devices
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  color_problem_type color_problem(G,              // input graph (GPU)
                                   g,              // input graph (CPU)
                                   multi_context,  // input context
                                   colors          // output color/vertex
  );

  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());

  color_enactor_type color_enactor(
      &color_problem,  // pass in a problem (contains data in/out)
      multi_context);

  float elapsed = color_enactor.enact();
  return elapsed;
}

template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename vertex_vector_t,
          typename edge_vector_t,
          typename weight_vector_t>
float execute(vertex_t const& number_of_rows,
              vertex_t const& number_of_columns,
              edge_t const& number_of_nonzeros,
              edge_vector_t& row_offsets,
              vertex_vector_t& column_indices,
              weight_vector_t& edge_values,
              vertex_vector_t& colors) {
  // Build graph structure for color
  auto G =
      graph::build::from_csr_t<space>(number_of_rows,      // number of rows
                                      number_of_columns,   // number of columns
                                      number_of_nonzeros,  // number of edges
                                      row_offsets,         // row offsets
                                      column_indices,      // column indices
                                      edge_values);        // nonzero values

  auto g = graph::build::from_csr_t<memory_space_t::host>(
      number_of_rows,      // number of rows
      number_of_columns,   // number of columns
      number_of_nonzeros,  // number of edges
      row_offsets,         // XXX: illegal device memory
      column_indices,      // XXX: illegal device memory
      edge_values);        // XXX: illegal device memory

  return color<space>(G.data().get(), g.data(), colors.data().get());
}

}  // namespace color
}  // namespace gunrock