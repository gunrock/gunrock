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
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return detail::from_csr<space, build_views>(csr);
}

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr, format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  return detail::from_csr<space, build_views>(csr, coo);
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock