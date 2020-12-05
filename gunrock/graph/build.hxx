/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
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
auto from_csr(vertex_t const& r,
              vertex_t const& c,
              edge_t const& nnz,
              edge_t* Ap,
              vertex_t* J,
              weight_t* X) {
  auto I_deleter = [&](vertex_t* ptr) { memory::free(ptr, space); };
  std::shared_ptr<vertex_t> I_ptr(
      memory::allocate<vertex_t>(nnz * sizeof(vertex_t), space), I_deleter);
  return detail::from_csr<space, build_views>(r, c, nnz, Ap, J, X, I_ptr.get());
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock