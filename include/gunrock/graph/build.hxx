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

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, csr);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  return detail::builder<space>(properties, coo);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(properties, csc);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, coo, csr);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc,
           format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(properties, csc, csr);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(graph::graph_properties_t properties,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(properties, coo, csc);
}

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