/**
 * @file sssp.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/applications/sssp/sssp_implementation.hxx>

#pragma once

namespace gunrock {
namespace sssp {

template <memory::memory_space_t space, typename graph_type>
float sssp(graph_type* G,
           typename graph_type::vertex_type source,
           typename graph_type::weight_pointer_t distances) {
  using sssp_problem_type = sssp_problem_t<graph_type>;
  using sssp_enactor_type = sssp_enactor_t<sssp_problem_type>;

  sssp_problem_type sssp_problem(G,          // input graph
                                 source,     // input source
                                 distances,  // output distances
                                 nullptr     // output predecessors
  );

  // Create contexts for all the devices
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());

  // sssp_enactor_type sssp_enactor(
  //     &sssp_problem,  // pass in a problem (contains data in/out)
  //     multi_context);

  // float elapsed = sssp_enactor.enact();
  // return elapsed;
}

template <memory::memory_space_t space,
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
              vertex_t const& source,
              weight_vector_t& distances) {
  // Build graph structure for SSSP
  auto G =
      graph::build::from_csr_t<space>(number_of_rows,      // number of rows
                                      number_of_columns,   // number of columns
                                      number_of_nonzeros,  // number of edges
                                      row_offsets,         // row offsets
                                      column_indices,      // column indices
                                      edge_values);        // nonzero values

  return sssp<space>(G.data().get(), source, distances.data().get());
}

}  // namespace sssp
}  // namespace gunrock