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

using namespace memory;

template <typename graph_vector_t,
          typename host_graph_vector_t,
          typename graph_type = typename graph_vector_t::value_type>
float sssp(graph_vector_t& G,
           host_graph_vector_t& g,
           typename graph_type::vertex_type source,
           typename graph_type::weight_pointer_t distances) {
  using host_graph_type = typename host_graph_vector_t::value_type;

  using sssp_problem_type = sssp_problem_t<graph_type, host_graph_type>;
  using sssp_enactor_type = sssp_enactor_t<sssp_problem_type>;
  using weight_t = typename graph_type::weight_type;

  // Create contexts for all the devices
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  std::shared_ptr<sssp_problem_type> sssp_problem(
      std::make_shared<sssp_problem_type>(G.data().get(),  // input graph (GPU)
                                          g.data(),        // input graph (CPU)
                                          multi_context,   // input context
                                          source,          // input source
                                          distances,       // output distances
                                          nullptr  // output predecessors
                                          ));

  std::shared_ptr<sssp_enactor_type> sssp_enactor(
      std::make_shared<sssp_enactor_type>(
          sssp_problem.get(),  // pass in a problem (contains data in/out)
          multi_context));

  float elapsed = sssp_enactor->enact();
  return elapsed;
}

template <typename csr_device_t, typename vertex_t, typename weight_vector_t>
float execute(csr_device_t& csr,
              vertex_t const& source,
              weight_vector_t& distances) {
  // Build graph structure for SSSP
  auto G = graph::build::from_csr_t<memory_space_t::device>(
      csr.number_of_rows,      // number of rows
      csr.number_of_columns,   // number of columns
      csr.number_of_nonzeros,  // number of edges
      csr.row_offsets,         // row offsets
      csr.column_indices,      // column indices
      csr.nonzero_values);     // nonzero values

  // XXX: Rework, there should be a way to hide this:
  auto g = graph::build::meta_graph(csr.number_of_rows,     // number of rows
                                    csr.number_of_columns,  // number of columns
                                    csr.number_of_nonzeros  // number of edges
  );

  return sssp(G, g, source, distances.data().get());
}

}  // namespace sssp
}  // namespace gunrock