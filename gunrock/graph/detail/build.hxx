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

#include <gunrock/graph/detail/convert.hxx>

namespace gunrock {
namespace graph {
namespace build {
namespace detail {

#if 0
/**
 * @brief
 *
 * @tparam graph_type
 * @param I
 * @return graph
 */
template <typename graph_type>
__device__ __host__ auto from_graph_t(graph_type& I) {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  graph_type G;

  if constexpr (G.template contains_representation<csr_v_t>())
    G.template set<csr_v_t>(I.csr_v_t::get_number_of_rows(),      // r
                            I.csr_v_t::get_number_of_columns(),   // c
                            I.csr_v_t::get_number_of_nonzeros(),  // nnz
                            I.csr_v_t::get_row_offsets(),         // offsets
                            I.csr_v_t::get_column_indices(),  // column indices
                            I.csr_v_t::get_nonzero_values()   // nonzero values
    );

  if constexpr (G.template contains_representation<csc_v_t>())
    G.template set<csc_v_t>(I.csc_v_t::get_number_of_rows(),      // r
                            I.csc_v_t::get_number_of_columns(),   // c
                            I.csc_v_t::get_number_of_nonzeros(),  // nnz
                            I.csc_v_t::get_column_offsets(),      // offsets
                            I.csc_v_t::get_row_indices(),         // row indices
                            I.csc_v_t::get_nonzero_values()  // nonzero values
    );

  if constexpr (G.template contains_representation<coo_v_t>())
    G.template set<coo_v_t>(I.coo_v_t::get_number_of_rows(),      // r
                            I.coo_v_t::get_number_of_columns(),   // c
                            I.coo_v_t::get_number_of_nonzeros(),  // nnz
                            I.coo_v_t::get_row_indices(),         // row indices
                            I.coo_v_t::get_column_indices(),  // column indices
                            I.coo_v_t::get_nonzero_values()   // nonzero values
    );

  return G;
}

/**
 * @brief Instantiate polymorphic inhertance within the kernel & set the
 * existing data to it. No allocations allowed here.
 *
 * @tparam graph_type
 * @param I
 * @param O
 */
template <typename graph_type>
__global__ void kernel_virtual_inheritance(graph_type I, graph_type* O) {
  auto G = from_graph_t(I);
  memcpy(O, &G, sizeof(graph_type));
}

/**
 * @brief Possible work around while keeping virtual (polymorphic behavior.)
 */
template <memory_space_t space, typename graph_type>
void fix(graph_type I, graph_type* G) {
  if constexpr (space == memory_space_t::device) {
    kernel_virtual_inheritance<<<1, 1>>>(I, G);
  } else {
    memcpy(G, &I, sizeof(graph_type));
  }
}
#endif

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

  // auto graph_deleter = [&](graph_type* ptr) { memory::free(ptr, space); };
  // std::shared_ptr<graph_type> G_ptr(
  // memory::allocate<graph_type>(sizeof(graph_type), space), graph_deleter);

  // fix<space>(G, G_ptr.get());
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
    convert::generate_row_indices<space>(r, nnz, Ap, I);
  }

  // if constexpr (has(build_views, view_t::csc)) {
  //   convert::generate_column_offsets<space>(r, nnz, Ap, Aj);
  // }

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
    h_I = (vertex_t*)nullptr;
    h_J = (vertex_t*)nullptr;

    h_Ap = (edge_t*)nullptr;
    h_Aj = (edge_t*)nullptr;

    h_X = (weight_t*)nullptr;
  } else if (space == memory_space_t::host) {
    d_I = (vertex_t*)nullptr;
    d_J = (vertex_t*)nullptr;

    d_Ap = (edge_t*)nullptr;
    d_Aj = (edge_t*)nullptr;

    d_X = (weight_t*)nullptr;
  } else {
    error::throw_if_exception(cudaErrorUnknown);
  }

  auto D = builder<memory_space_t::device,  // build for device
                   build_views              // supported views
                   >(r, c, nnz, d_I, d_J, d_Ap, d_Aj, d_X);

  auto H = builder<memory_space_t::host,  // build for host
                   build_views            // supported views
                   >(r, c, nnz, h_I, h_J, h_Ap, h_Aj, h_X);
  graph_container_t G(&D, &H);
  return G;
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock