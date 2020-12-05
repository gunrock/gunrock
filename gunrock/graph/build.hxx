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
  // Build missing data.
  vertex_t* I = nullptr;
  edge_t* Aj = nullptr;

  // Host.
  edge_t* h_Ap = Ap;
  edge_t* h_Aj = Aj;

  vertex_t* h_I = I;
  vertex_t* h_J = J;
  weight_t* h_X = X;

  // Device.
  edge_t* d_Ap = Ap;
  edge_t* d_Aj = Aj;

  vertex_t* d_I = I;
  vertex_t* d_J = J;
  weight_t* d_X = X;

  // nullify space that's not needed.
  if (space == memory_space_t::device) {
    h_Ap = (edge_t*)nullptr;
    h_J = (vertex_t*)nullptr;
    h_X = (weight_t*)nullptr;
  } else if (space == memory_space_t::host) {
    d_Ap = (edge_t*)nullptr;
    d_J = (vertex_t*)nullptr;
    d_X = (weight_t*)nullptr;
  } else {
    error::throw_if_exception(cudaErrorUnknown);
  }

  auto D = detail::builder<memory_space_t::device,  // build for device
                           build_views              // supported views
                           >(r, c, nnz, d_I, d_J, d_Ap, d_Aj, d_X);

  auto H = detail::builder<memory_space_t::host,  // build for host
                           build_views            // supported views
                           >(r, c, nnz, h_I, h_J, h_Ap, h_Aj, h_X);
  graph_container_t G(D, H);
  return G;
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock