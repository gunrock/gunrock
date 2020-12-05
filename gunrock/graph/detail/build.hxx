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

namespace gunrock {
namespace graph {
namespace build {
namespace detail {

/**
 * @brief
 *
 * @tparam graph_type
 * @param I
 * @return graph
 */
template <uint32_t build_views, typename graph_type>
__device__ __host__ auto from_graph_t(graph_type& I) {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  using coo_v_t = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  graph_type G;

  if constexpr (has((view_t)build_views, view_t::csr))
    G.template set<csr_v_t>(I.csr_v_t::get_number_of_rows(),      // r
                            I.csr_v_t::get_number_of_columns(),   // c
                            I.csr_v_t::get_number_of_nonzeros(),  // nnz
                            I.csr_v_t::get_row_offsets(),         // offsets
                            I.csr_v_t::get_column_indices(),  // column indices
                            I.csr_v_t::get_nonzero_values()   // nonzero values
    );

  if constexpr (has((view_t)build_views, view_t::csc))
    G.template set<csc_v_t>(I.csc_v_t::get_number_of_rows(),      // r
                            I.csc_v_t::get_number_of_columns(),   // c
                            I.csc_v_t::get_number_of_nonzeros(),  // nnz
                            I.csc_v_t::get_column_offsets(),      // offsets
                            I.csc_v_t::get_row_indices(),         // row indices
                            I.csc_v_t::get_nonzero_values()  // nonzero values
    );

  if constexpr (has((view_t)build_views, view_t::coo))
    G.template set<coo_v_t>(I.coo_v_t::get_number_of_rows(),      // r
                            I.coo_v_t::get_number_of_columns(),   // c
                            I.coo_v_t::get_number_of_nonzeros(),  // nnz
                            I.coo_v_t::get_row_indices(),     // column indices
                            I.coo_v_t::get_column_indices(),  // row indices
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
template <uint32_t build_views, typename graph_type>
__global__ void kernel_virtual_inheritance(graph_type I, graph_type* O) {
  auto G = from_graph_t<build_views>(I);
  memcpy(O, &G, sizeof(graph_type));
}

/**
 * @brief Possible work around while keeping virtual (polymorphic behavior.)
 */
template <memory_space_t space, view_t build_views, typename graph_type>
void fix(graph_type I, graph_type* G) {
  if constexpr (space == memory_space_t::device) {
    kernel_virtual_inheritance<build_views><<<1, 1>>>(I, G);
  } else {
    memcpy(G, &I, sizeof(graph_type));
  }
}

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
    G.template set<csr_v_t>(r, c, nnz, Ap, J, X);
  }

  if constexpr (has(build_views, view_t::csc)) {
    G.template set<csc_v_t>(r, c, nnz, I, Aj, X);
  }

  if constexpr (has(build_views, view_t::coo)) {
    G.template set<coo_v_t>(r, c, nnz, I, J, X);
  }

  auto graph_deleter = [&](graph_type* ptr) { memory::free(ptr, space); };
  std::shared_ptr<graph_type> G_ptr(
      memory::allocate<graph_type>(sizeof(graph_type), space), graph_deleter);

  fix<space, build_views>(G, G_ptr.get());
  return G_ptr;
}
}  // namespace detail
}  // namespace build
}  // namespace graph
}  // namespace gunrock