/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-12-04
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/graph/conversions/convert.hxx>

namespace gunrock {
namespace graph {
namespace build {
namespace detail {

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(vertex_t const& r,
             vertex_t const& c,
             edge_t const& nnz,
             vertex_t* I,
             vertex_t* J,
             edge_t* Ap,
             edge_t* Aj,
             weight_t* X) {
  // Enable the types based on the different views required.
  // Enable CSR.
  using csr_v_t =
      std::conditional_t<has(build_views, view_t::csr),
                         graph::graph_csr_t<vertex_t, edge_t, weight_t>,
                         empty_csr_t>;

  // Enable CSC.
  using csc_v_t =
      std::conditional_t<has(build_views, view_t::csc),
                         graph::graph_csc_t<vertex_t, edge_t, weight_t>,
                         empty_csc_t>;

  // Enable COO.
  using coo_v_t =
      std::conditional_t<has(build_views, view_t::coo),
                         graph::graph_coo_t<vertex_t, edge_t, weight_t>,
                         empty_coo_t>;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t,
                                    csc_v_t, coo_v_t>;

  graph_type G;

  if constexpr (has(build_views, view_t::csr)) {
    G.template set<csr_v_t>(r, nnz, Ap, J, X);
  }

  if constexpr (has(build_views, view_t::csc)) {
    G.template set<csc_v_t>(r, nnz, Aj, I, X);
  }

  if constexpr (has(build_views, view_t::coo)) {
    G.template set<coo_v_t>(r, nnz, I, J, X);
  }

  return G;
}

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
              weight_t* X,
              vertex_t* I = nullptr,
              edge_t* Aj = nullptr) {
  if constexpr (has(build_views, view_t::csc) ||
                has(build_views, view_t::coo)) {
    const edge_t size_of_offsets = r + 1;
    convert::offsets_to_indices<space>(Ap, size_of_offsets, I, nnz);
  }

  if constexpr (has(build_views, view_t::csc)) {
    const edge_t size_of_offsets = r + 1;
    convert::indices_to_offsets<space>(J, nnz, Aj, size_of_offsets);
  }

  return builder<space,       // build for host
                 build_views  // supported views
                 >(r, c, nnz, I, J, Ap, Aj, X);
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock