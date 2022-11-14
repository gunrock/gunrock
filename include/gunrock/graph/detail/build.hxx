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
  // using csr_v_t =
  //    std::conditional_t<has(build_views, view_t::csr),
  //                       graph::graph_csr_t<vertex_t, edge_t, weight_t>,
  //                       empty_csr_t>;

  //// Enable CSC.
  // using csc_v_t =
  //     std::conditional_t<has(build_views, view_t::csc),
  //                        graph::graph_csc_t<vertex_t, edge_t, weight_t>,
  //                        empty_csc_t>;

  //// Enable COO.
  // using coo_v_t =
  //     std::conditional_t<has(build_views, view_t::coo),
  //                        graph::graph_coo_t<vertex_t, edge_t, weight_t>,
  //                        empty_coo_t>;

  // using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t,
  // csr_v_t,
  //                                   csc_v_t, coo_v_t>;

  // graph_type G;

  // if constexpr (has(build_views, view_t::csr)) {
  //   G.template set<csr_v_t>(r, nnz, row_offsets, column_indices, values);
  // }

  // if constexpr (has(build_views, view_t::csc)) {
  //   G.template set<csc_v_t>(r, nnz, column_offsets, row_indices, values);
  // }

  // if constexpr (has(build_views, view_t::coo)) {
  //   G.template set<coo_v_t>(r, nnz, row_indices, column_indices, values);
  // }

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

template <memory_space_t space,
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
    G.template set<coo_v_t>(csr.number_of_rows, csr.number_of_nonzeros,
                            coo.row_indices.data().get(),
                            coo.column_indices.data().get(),
                            coo.nonzero_values.data().get());
  }

  return G;
}

// UNCOMMENT
// template <memory_space_t space,
//          view_t build_views,
//          typename edge_t,
//          typename vertex_t,
//          typename weight_t>
//// auto from_csr(vertex_t const& r,
////               vertex_t const& c,
////               edge_t const& nnz,
////               edge_t* row_offsets,
////               vertex_t* column_indices,
////               weight_t* values,
////               vertex_t* row_indices = nullptr,
////               edge_t* column_offsets = nullptr) {
// auto from_csr(csr_t csr, coo_t coo) {
//   // if constexpr (has(build_views, view_t::csc) &&
//   //               has(build_views, view_t::csr)) {
//   //   error::throw_if_exception(cudaErrorUnknown,
//   //                             "CSC & CSR view not yet supported
//   together.");
//   // }
//
//   // if constexpr (has(build_views, view_t::csc) ||
//   //               has(build_views, view_t::coo)) {
//   convert::offsets_to_indices<space>(
//       csr.row_offsets.data().get(), csr.number_of_rows + 1,
//       coo.row_indices.data().get(), csr.number_of_nonzeros);
//   //}
//
//   // if constexpr (has(build_views, view_t::csc)) {
//   //   using execution_policy_t =
//   //       std::conditional_t<space == memory_space_t::device,
//   //                          decltype(thrust::device),
//   decltype(thrust::host)>;
//   //   execution_policy_t exec;
//   //   thrust::sort_by_key(exec, column_indices, column_indices + nnz,
//   //                       thrust::make_zip_iterator(
//   //                           thrust::make_tuple(row_indices, values))  //
//   //                           values
//   //   );
//
//   //  const edge_t size_of_offsets = r + 1;
//   //  convert::indices_to_offsets<space>(column_indices, nnz, column_offsets,
//   //                                     size_of_offsets);
//   //}
//
//   return builder<space,       // build for host
//                  build_views  // supported views
//                  >(csr, coo);
// }

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
// auto from_csr(vertex_t const& r,
//               vertex_t const& c,
//               edge_t const& nnz,
//               edge_t* row_offsets,
//               vertex_t* column_indices,
//               weight_t* values,
//               vertex_t* row_indices = nullptr,
//               edge_t* column_offsets = nullptr) {
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr) {
  return builder<space,       // build for host
                 build_views  // supported views
                 >(csr);
}

template <memory_space_t space,
          view_t build_views,
          typename edge_t,
          typename vertex_t,
          typename weight_t>
auto from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr,
              format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
  convert::offsets_to_indices<space>(
      csr.row_offsets.data().get(), 
      csr.number_of_rows + 1,
      coo.row_indices.data().get(), 
      csr.number_of_nonzeros);

  return builder<space,       // build for host
                 build_views  // supported views
                 >(csr, coo);
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock