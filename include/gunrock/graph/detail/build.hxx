/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-12-04
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/graph/conversions/convert.hxx>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/formats/formats.hxx>

namespace gunrock {
namespace graph {
namespace build {
namespace detail {

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  // Enable the types based on the different views required.
  // Enable CSR.
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using coo_v_t = empty_coo_t;
  using csc_v_t = empty_csc_t;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t,
                                    csc_v_t, coo_v_t>;
  graph_type G;

  G.template set<csr_v_t>(
      csr.number_of_rows, csr.number_of_nonzeros, csr.row_offsets.data().get(),
      csr.column_indices.data().get(), csr.nonzero_values.data().get());

  return G;
}

template <
    memory_space_t space,
    view_t build_views,
    typename edge_t,
    typename vertex_t,
    typename weight_t>
auto builder(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
             format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  // Enable the types based on the different views required.
  // Enable CSR.
  using csr_v_t =
      std::conditional_t<has(build_views, view_t::csr),
                         graph::graph_csr_t<vertex_t, edge_t, weight_t>,
                         empty_csr_t>;

  //// Enable COO.
  using coo_v_t =
      std::conditional_t<has(build_views, view_t::coo),
                         graph::graph_coo_t<vertex_t, edge_t, weight_t>,
                         empty_coo_t>;

  using csc_v_t = empty_csc_t;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t,
                                    csc_v_t, coo_v_t>;

  graph_type G;

  if constexpr (has(build_views, view_t::csr)) {
    G.template set<csr_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                            csr.row_offsets.data().get(),
                            csr.column_indices.data().get(),
                            csr.nonzero_values.data().get());
  }

  if constexpr (has(build_views, view_t::coo)) {
    G.template set<coo_v_t, space, vertex_t, edge_t, weight_t, format::coo_t<space, vertex_t, edge_t, weight_t>>(coo);
  }

  return G;
}

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return builder<space, build_views>(csr);
}

template <
    memory_space_t space,
    view_t build_views,
    typename edge_t,
    typename vertex_t,
    typename weight_t,
    typename std::enable_if<(space == memory_space_t::device)>::type* = nullptr>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
              format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  convert::offsets_to_indices<space>(
      csr.row_offsets.data().get(), csr.number_of_rows + 1,
      coo.row_indices.data().get(), csr.number_of_nonzeros);
  return builder<space, build_views>(csr, coo);
}

template <
    memory_space_t space,
    view_t build_views,
    typename edge_t,
    typename vertex_t,
    typename weight_t,
    typename std::enable_if<(space == memory_space_t::host)>::type* = nullptr>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
              format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  convert::offsets_to_indices<space>(
      csr.row_offsets.data(), csr.number_of_rows + 1, coo.row_indices.data(),
      csr.number_of_nonzeros);
  return builder<space, build_views>(csr, coo);
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock