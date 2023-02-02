/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-12-04
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/formats/formats.hxx>

namespace gunrock {
namespace graph {
namespace build {
namespace detail {

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  // Enable the types based on the different views required.
  // Enable CSR.
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using coo_v_t = empty_coo_t;
  using csc_v_t = empty_csc_t;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t>;
  graph_type G(properties);

  G.template set<csr_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                          memory::raw_pointer_cast(csr.row_offsets.data()),
                          memory::raw_pointer_cast(csr.column_indices.data()),
                          memory::raw_pointer_cast(csr.nonzero_values.data()));

  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  // Enable the types based on the different views required.
  using csr_v_t = empty_csr_t;

  //// Enable COO.
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  using csc_v_t = empty_csc_t;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, coo_v_t>;

  graph_type G(properties);

  G.template set<coo_v_t>(coo.number_of_rows, coo.number_of_nonzeros,
                          memory::raw_pointer_cast(coo.row_indices.data()),
                          memory::raw_pointer_cast(coo.column_indices.data()),
                          memory::raw_pointer_cast(coo.nonzero_values.data()));

  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  // Enable the types based on the different views required.
  using csr_v_t = empty_csr_t;

  using coo_v_t = empty_coo_t;

  //// Enable csc.
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csc_v_t>;

  graph_type G(properties);

  G.template set<csc_v_t>(csc.number_of_rows, csc.number_of_nonzeros,
                          memory::raw_pointer_cast(csc.column_offsets.data()),
                          memory::raw_pointer_cast(csc.row_indices.data()),
                          memory::raw_pointer_cast(csc.nonzero_values.data()));
  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
             format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  // Enable the types based on the different views required.
  //// Enable CSR.
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;

  //// Enable COO.
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  using csc_v_t = empty_csc_t;

  using graph_type =
      graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t, coo_v_t>;

  graph_type G(properties);

  G.template set<csr_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                          memory::raw_pointer_cast(csr.row_offsets.data()),
                          memory::raw_pointer_cast(csr.column_indices.data()),
                          memory::raw_pointer_cast(csr.nonzero_values.data()));

  G.template set<coo_v_t>(coo.number_of_rows, coo.number_of_nonzeros,
                          memory::raw_pointer_cast(coo.row_indices.data()),
                          memory::raw_pointer_cast(coo.column_indices.data()),
                          memory::raw_pointer_cast(coo.nonzero_values.data()));

  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
             format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  // Enable the types based on the different views required.
  //// Enable CSR.
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;

  using coo_v_t = empty_coo_t;

  //// Enable CSC.
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;

  using graph_type =
      graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t, csc_v_t>;

  graph_type G(properties);

  G.template set<csr_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                          memory::raw_pointer_cast(csr.row_offsets.data()),
                          memory::raw_pointer_cast(csr.column_indices.data()),
                          memory::raw_pointer_cast(csr.nonzero_values.data()));

  G.template set<csc_v_t>(csc.number_of_rows, csc.number_of_nonzeros,
                          memory::raw_pointer_cast(csc.column_offsets.data()),
                          memory::raw_pointer_cast(csc.row_indices.data()),
                          memory::raw_pointer_cast(csc.nonzero_values.data()));

  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
             format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  // Enable the types based on the different views required.
  using csr_v_t = empty_csr_t;

  //// Enable COO.
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  //// Enable CSC.
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;

  using graph_type =
      graph::graph_t<space, vertex_t, edge_t, weight_t, csc_v_t, coo_v_t>;

  graph_type G(properties);

  G.template set<coo_v_t>(coo.number_of_rows, coo.number_of_nonzeros,
                          memory::raw_pointer_cast(coo.row_indices.data()),
                          memory::raw_pointer_cast(coo.column_indices.data()),
                          memory::raw_pointer_cast(coo.nonzero_values.data()));

  G.template set<csc_v_t>(csc.number_of_rows, csc.number_of_nonzeros,
                          memory::raw_pointer_cast(csc.column_offsets.data()),
                          memory::raw_pointer_cast(csc.row_indices.data()),
                          memory::raw_pointer_cast(csc.nonzero_values.data()));

  return G;
}

template <memory_space_t space,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto builder(graph::graph_properties_t properties,
             format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
             format::coo_t<space, vertex_t, edge_t, weight_t>& coo,
             format::csc_t<space, vertex_t, edge_t, weight_t>& csc) {
  // Enable the types based on the different views required.
  //// Enable CSR.
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;

  //// Enable COO.
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  //// Enable CSC.
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t,
                                    csc_v_t, coo_v_t>;

  graph_type G(properties);

  G.template set<csr_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                          memory::raw_pointer_cast(csr.row_offsets.data()),
                          memory::raw_pointer_cast(csr.column_indices.data()),
                          memory::raw_pointer_cast(csr.nonzero_values.data()));

  G.template set<coo_v_t>(coo.number_of_rows, coo.number_of_nonzeros,
                          memory::raw_pointer_cast(coo.row_indices.data()),
                          memory::raw_pointer_cast(coo.column_indices.data()),
                          memory::raw_pointer_cast(coo.nonzero_values.data()));

  G.template set<csc_v_t>(csc.number_of_rows, csc.number_of_nonzeros,
                          memory::raw_pointer_cast(csc.column_offsets.data()),
                          memory::raw_pointer_cast(csc.row_indices.data()),
                          memory::raw_pointer_cast(csc.nonzero_values.data()));

  return G;
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock