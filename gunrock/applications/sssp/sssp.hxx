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

#include <gunrock/framework/framework.hxx>

#include <gunrock/util/timer.hxx>
#include <gunrock/memory.hxx>

#include <gunrock/applications/sssp/sssp_problem.hxx>
#include <gunrock/applications/sssp/sssp_enactor.hxx>

#pragma once

namespace gunrock {
namespace sssp {

template <memory::memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t>
double sssp(vertex_t number_of_vertices,
            edge_t number_of_edges,
            edge_t* row_offsets,
            vertex_t* column_indices,
            weight_t* edge_values,
            vertex_t source,
            weight_t* distances) {
  // Build graph structure for SSSP
  auto G =
      graph::build::from_csr_t<space>(number_of_vertices,  // number of rows
                                      number_of_vertices,  // number of columns
                                      number_of_edges,     // number of edges
                                      row_offsets,         // row offsets
                                      column_indices,      // column indices
                                      edge_values);        // nonzero values

  std::shared_ptr<sssp_problem_t<decltype(G)>> sssp_problem(
      std::make_shared<sssp_problem_t<decltype(G)>>(G, source));

  std::shared_ptr<sssp_enactor_t<decltype(*sssp_problem)>> sssp_enactor(
      std::make_shared<sssp_enactor_t<decltype(*sssp_problem)>>(G,
                                                                queue_sizing));

  timer_t timer;
  timer.start();
  sssp_enactor->enact(sssp_problem);
  return timer.end();
}

}  // namespace sssp
}  // namespace gunrock