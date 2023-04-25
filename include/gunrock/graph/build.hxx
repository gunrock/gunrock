/**
 * @file build.hxx
 * @author Annie Robison (amrobison@ucdavis.edu)
 * @brief
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/graph/detail/build.hxx>

namespace gunrock {
namespace graph {

/**
 * @brief Builds a graph using CSR object.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param csr csr_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, csr);
}

/**
 * @brief Builds a graph using COO object.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param coo coo_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  return detail::builder<space>(properties, coo);
}

/**
 * @brief Builds a graph using CSC object.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param csc csc_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(properties, csc);
}

/**
 * @brief Builds a graph that supports COO and CSR.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param coo coo_t format with graph's data.
 * @param csr csr_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, coo, csr);
}

/**
 * @brief Builds a graph that supports CSC and CSR.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param csc csc_t format with graph's data.
 * @param csr csr_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, csc, csr);
}

/**
 * @brief Builds a graph that supports COO and CSC.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param coo coo_t format with graph's data.
 * @param csc csc_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(properties, coo, csc);
}

/**
 * @brief Builds a graph that supports COO, CSC and CSR.
 *
 * @tparam space memory space for the graph (host or device).
 * @tparam edge_t Edge type of the graph.
 * @tparam vertex_t Vertex type of the graph.
 * @tparam weight_t Weight type of the graph.
 * @param properties Graph properties.
 * @param coo coo_t format with graph's data.
 * @param csc csc_t format with graph's data.
 * @param csr csr_t format with graph's data.
 * @return graph_t the graph itself.
 */
template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, coo, csc, csr);
}

}  // namespace graph
}  // namespace gunrock