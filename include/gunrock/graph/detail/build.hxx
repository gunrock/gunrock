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
#include <gunrock/algorithms/algorithms.hxx>

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
             vertex_t* row_indices,
             vertex_t* column_indices,
             edge_t* row_offsets,
             edge_t* column_offsets,
             weight_t* values) {
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
    G.template set<csr_v_t>(r, nnz, row_offsets, column_indices, values);
  }

  if constexpr (has(build_views, view_t::csc)) {
    G.template set<csc_v_t>(r, nnz, column_offsets, row_indices, values);
  }

  if constexpr (has(build_views, view_t::coo)) {
    G.template set<coo_v_t>(r, nnz, row_indices, column_indices, values);
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
              edge_t* row_offsets,
              vertex_t* column_indices,
              weight_t* values,
              vertex_t* row_indices = nullptr,
              edge_t* column_offsets = nullptr) {
  if constexpr (has(build_views, view_t::csc) &&
                has(build_views, view_t::csr)) {
    error::throw_if_exception(cudaErrorUnknown,
                              "CSC & CSR view not yet supported together.");
  }

  if constexpr (has(build_views, view_t::csc) ||
                has(build_views, view_t::coo)) {
    const edge_t size_of_offsets = r + 1;
    convert::offsets_to_indices<space>(row_offsets, size_of_offsets,
                                       row_indices, nnz);
  }

  if constexpr (has(build_views, view_t::csc)) {
    using execution_policy_t =
        std::conditional_t<space == memory_space_t::device,
                           decltype(thrust::device), decltype(thrust::host)>;
    execution_policy_t exec;
    thrust::sort_by_key(exec, column_indices, column_indices + nnz,
                        thrust::make_zip_iterator(
                            thrust::make_tuple(row_indices, values))  // values
    );

    const edge_t size_of_offsets = r + 1;
    convert::indices_to_offsets<space>(column_indices, nnz, column_offsets,
                                       size_of_offsets);
  }

  return builder<space,       // build for host
                 build_views  // supported views
                 >(r, c, nnz, row_indices, column_indices, row_offsets,
                   column_offsets, values);
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock