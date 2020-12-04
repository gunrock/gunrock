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

#include <tuple>

namespace gunrock {
namespace graph {

enum view_t : uint32_t {
  csr = 0 << 1,
  csc = 0 << 2,
  coo = 0 << 3,
  invalid = 0 << 0
};
inline constexpr view_t operator|(view_t lhs, view_t rhs) {
  return static_cast<view_t>(static_cast<uint32_t>(lhs) |
                             static_cast<uint32_t>(rhs));
}

namespace build {
namespace detail {

/**
 * @brief
 *
 * @tparam graph_type
 * @param I
 * @return graph
 */
template <view_t build_views, typename graph_type>
__device__ __host__ auto from_graph_t(graph_type& I) {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  using csr_v = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using csc_v = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  using coo_v = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  graph_type G;

  // if (build_views & view_t::csr)
  G.csr_v::set(I.csr_v::get_number_of_rows(),      // r
               I.csr_v::get_number_of_columns(),   // c
               I.csr_v::get_number_of_nonzeros(),  // nnz
               I.csr_v::get_row_offsets(),         // offsets
               I.csr_v::get_column_indices(),      // column indices
               I.csr_v::get_nonzero_values()       // nonzero values
  );

  // if (build_views & view_t::csc)
  G.csc_v::set(I.csc_v::get_number_of_rows(),      // r
               I.csc_v::get_number_of_columns(),   // c
               I.csc_v::get_number_of_nonzeros(),  // nnz
               I.csc_v::get_column_offsets(),      // offsets
               I.csc_v::get_row_indices(),         // row indices
               I.csc_v::get_nonzero_values()       // nonzero values
  );

  // if (build_views & view_t::coo)
  //   G.set<coo_v>(I.coo_v::get_number_of_rows(),      // r
  //                I.coo_v::get_number_of_columns(),   // c
  //                I.coo_v::get_number_of_nonzeros(),  // nnz
  //                I.coo_v::get_row_indices(),         // column indices
  //                I.coo_v::get_column_indices(),      // row indices
  //                I.coo_v::get_nonzero_values()       // nonzero values
  // );

  return G;
}

template <view_t build_views, typename graph_type>
__host__ __device__ void fix_virtual_inheritance(graph_type I, graph_type* O) {
#ifdef __CUDA_ARCH__
  auto G = from_graph_t<build_views>(I);
  memcpy(O, &G, sizeof(graph_type));
#else
  // On host this should be fine.
  memcpy(O, &I, sizeof(graph_type));
#endif
}

/**
 * @brief Instantiate polymorphic inhertance within the kernel & set the
 * existing data to it. No allocations allowed here.
 *
 * @tparam graph_type
 * @param I
 * @param O
 */
template <view_t build_views, typename graph_type>
__global__ void kernel_virtual_inheritance(graph_type I, graph_type* O) {
  fix_virtual_inheritance<build_views>(I, O);
}

/**
 * @brief Possible work around while keeping virtual (polymorphic behavior.)
 */
template <memory_space_t space, view_t build_views, typename graph_type>
void fix(graph_type I, graph_type* G) {
  if (space == memory_space_t::device)
    kernel_virtual_inheritance<build_views><<<1, 1>>>(I, G);
  else
    fix_virtual_inheritance<build_views>(I, G);
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
              weight_t* X) {
  // if ((build_views & view_t::csr) && (build_views & view_t::csc) &&
  // (build_views & view_t::coo)) {
  using csr_v = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using csc_v = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  using coo_v = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  using graph_type = graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v,
                                    csc_v /*, coo_v */>;

  // TODO: Build I and Aj.
  vertex_t* I = nullptr;
  edge_t* Aj = nullptr;

  graph_type G;
  G.csr_v::set(r, c, nnz, Ap, J, X);
  G.csc_v::set(r, c, nnz, I, Aj, X);
  // G.set<coo_v>(r, c, nnz, I, J, X);

  auto graph_deleter = [&](graph_type* ptr) { memory::free(ptr, space); };
  std::shared_ptr<graph_type> G_ptr(
      memory::allocate<graph_type>(sizeof(graph_type), space), graph_deleter);

  fix<space, build_views>(G, G_ptr.get());
  return G_ptr;
  // }
}

// template <memory_space_t space,
//           view_t build_views,
//           typename edge_t,
//           typename vertex_t,
//           typename weight_t>
// auto from_csc(vertex_t const& r,
//               vertex_t const& c,
//               edge_t const& nnz,
//               vertex_t* I,
//               edge_t* Aj,
//               weight_t* X) {
//   if ((build_views & view_t::csr) && (build_views & view_t::csc) &&
//       (build_views & view_t::coo)) {
//     using csr_v = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
//     using csc_v = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
//     using coo_v = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

//     using graph_type =
//         graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v, csc_v,
//         coo_v>;

//     // TODO: Build J and Ap.
//     vertex_t* J = nullptr;
//     edge_t* Ap = nullptr;

//     graph_type G;
//     G.set<csr_v>(r, c, nnz, Ap, J, X);
//     G.set<csc_v>(r, c, nnz, I, Aj, X);
//     G.set<coo_v>(r, c, nnz, I, J, X);

//     auto graph_deleter = [&](graph_type* ptr) { memory::free(ptr, space); };
//     std::shared_ptr<graph_type> G_ptr(
//         memory::allocate<graph_type>(sizeof(graph_type), space),
//         graph_deleter);

//     fix<space, build_views>(G, G_ptr.get());
//     return G_ptr;
//   }
// }

// template <memory_space_t space,
//           view_t build_views,
//           typename edge_t,
//           typename vertex_t,
//           typename weight_t>
// auto from_coo(vertex_t const& r,
//               vertex_t const& c,
//               edge_t const& nnz,
//               vertex_t* I,
//               vertex_t* J,
//               weight_t* X) {
//   // Build with all views.
//   if ((build_views & view_t::csr) && (build_views & view_t::csc) &&
//       (build_views & view_t::coo)) {
//     using csr_v = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
//     using csc_v = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
//     using coo_v = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

//     using graph_type =
//         graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v, csc_v,
//         coo_v>;

//     // TODO: Build Ap and Aj.
//     edge_t* Ap = nullptr;
//     edge_t* Aj = nullptr;

//     G.set<csr_v>(r, c, nnz, Ap, J, X);
//     G.set<csc_v>(r, c, nnz, I, Aj, X);
//     G.set<coo_v>(r, c, nnz, I, J, X);

//     auto graph_deleter = [&](graph_type* ptr) { memory::free(ptr, space); };
//     std::shared_ptr<graph_type> G_ptr(
//         memory::allocate<graph_type>(sizeof(graph_type), space),
//         graph_deleter);

//     fix<space, build_views>(G, G_ptr.get());
//     return G_ptr;
//   }
// }

}  // namespace detail

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
  // Host.
  edge_t* h_Ap = Ap;
  vertex_t* h_J = J;
  weight_t* h_X = X;

  // Device.
  edge_t* d_Ap = Ap;
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

  auto D = detail::from_csr<memory_space_t::device, build_views>(
      r, c, nnz, d_Ap, d_J, d_X);
  auto H = detail::from_csr<memory_space_t::host, build_views>(r, c, nnz, h_Ap,
                                                               h_J, h_X);
  graph_container_t G(D, H);
  return G;
}

// template <memory_space_t space,
//           view_t build_views,
//           typename edge_t,
//           typename vertex_t,
//           typename weight_t>
// auto from_csc(vertex_t const& r,
//               vertex_t const& c,
//               edge_t const& nnz,
//               vertex_t* I,
//               edge_t* Aj,
//               weight_t* X) {
//   vertex_t *h_I, *h_Aj, *d_I, *d_Aj;
//   weight_t *h_X, *d_X;

//   h_I = d_I = I;
//   h_Aj = d_Aj = Aj;
//   h_X = d_X = X;

//   // nullify space that's not needed.
//   if (space == memory_space_t::device) {
//     h_I = (vertex_t*)nullptr;
//     h_Aj = (vertex_t*)nullptr;
//     h_X = (weight_t*)nullptr;
//   } else {
//     d_I = (vertex_t*)nullptr;
//     d_Aj = (vertex_t*)nullptr;
//     d_X = (weight_t*)nullptr;
//   }

//   auto D = detail::from_csc<memory_space_t::device, build_views>(r, c, nnz,
//   d_I,
//                                                                  d_Aj, d_X);
//   auto H = detail::from_csc<memory_space_t::host, build_views>(r, c, nnz,
//   h_I,
//                                                                h_Aj, h_X);
//   graph_container_t G(D, H);
//   return G;
// }

// template <memory_space_t space,
//           view_t build_views,
//           typename edge_t,
//           typename vertex_t,
//           typename weight_t>
// auto from_coo(vertex_t const& r,
//               vertex_t const& c,
//               edge_t const& nnz,
//               vertex_t* I,
//               vertex_t* J,
//               weight_t* X) {
//   vertex_t *h_I, *h_J, *d_I, *d_J;
//   weight_t *h_X, *d_X;

//   h_I = d_I = I;
//   h_J = d_J = J;
//   h_X = d_X = X;

//   // nullify space that's not needed.
//   if (space == memory_space_t::device) {
//     h_I = (vertex_t*)nullptr;
//     h_J = (vertex_t*)nullptr;
//     h_X = (weight_t*)nullptr;
//   } else {
//     d_I = (vertex_t*)nullptr;
//     d_J = (vertex_t*)nullptr;
//     d_X = (weight_t*)nullptr;
//   }

//   auto D = detail::from_coo<memory_space_t::device, build_views>(r, c, nnz,
//   d_I,
//                                                                  d_J, d_X);
//   auto H = detail::from_coo<memory_space_t::host, build_views>(r, c, nnz,
//   h_I,
//                                                                h_J, h_X);
//   graph_container_t G(D, H);
//   return G;
// }

}  // namespace build
}  // namespace graph
}  // namespace gunrock