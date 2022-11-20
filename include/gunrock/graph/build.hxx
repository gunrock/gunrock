/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
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
namespace build {

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::builder<space>(csr);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  return detail::builder<space>(coo);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(csc);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  return detail::builder<space>(csr, coo);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(csr, csc);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(coo, csc);
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto build(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
           format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
           format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  return detail::builder<space>(csr, coo, csc);
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock