#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

#include <gunrock/graph/coo.hxx>
#include <gunrock/graph/csc.hxx>
#include <gunrock/graph/csr.hxx>
#include <gunrock/graph/detail/graph_base.hxx>

// #include <gunrock/algorithms/search/binary_search.cuh>

namespace gunrock {
namespace graph {

using namespace format;
using namespace detail;
using namespace memory;

// Variadic inheritence, inherit only what you need
template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          class... graph_view_t>
class graph_t : public graph_view_t... {
  // Default view (graph representation) if no view is specified
  using first_view_t =
      typename std::tuple_element<0,  // get first type
                                  std::tuple<graph_view_t...>>::type;

 public:
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pointer_t = vertex_t*;
  using edge_pointer_t = edge_type*;
  using weight_pointer_t = weight_t*;

  using graph_type =
      graph_t<space, vertex_type, edge_type, weight_type, graph_view_t...>;

  // Base graph type, always exists.
  using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;

  // Different supported graph representation views.
  using graph_csr_view =
      graph_csr_t<space, vertex_type, edge_type, weight_type>;
  using graph_csc_view =
      graph_csc_t<space, vertex_type, edge_type, weight_type>;
  using graph_coo_view =
      graph_coo_t<space, vertex_type, edge_type, weight_type>;

  __host__ __device__ graph_t() : graph_view_t()... {}

  // template<typename csr_matrix_t>
  // graph_t(csr_matrix_t& rhs) :
  //   graph_base_type(rhs.num_rows,
  //                   rhs.num_nonzeros),
  //   graph_csr_view(rhs) {}

  template <typename edge_vector_t,
            typename vertex_vector_t,
            typename weight_vector_t>
  void set(typename vertex_vector_t::value_type const& r,
           typename vertex_vector_t::value_type const& c,
           typename edge_vector_t::value_type const& nnz,
           edge_vector_t& Ap,
           vertex_vector_t& Aj,
           weight_vector_t& Ax) {
    graph_csr_view::set(r, c, nnz, Ap, Aj, Ax);
  }

  __host__ __device__ void set(vertex_type const& r,
                               vertex_type const& c,
                               edge_type const& nnz,
                               edge_pointer_t Ap,
                               vertex_pointer_t Aj,
                               weight_pointer_t Ax) {
    graph_csr_view::set(r, c, nnz, Ap, Aj, Ax);
  }

  // XXX: add support for per-view based methods
  // template<typename view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_neighbor_list_length(vertex_type const& v) const override {
    return first_view_t::get_neighbor_list_length(v);
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const override {
    return first_view_t::get_source_vertex(e);
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const override {
    return first_view_t::get_edge_weight(e);
  }

  __host__ __device__ __forceinline__ std::size_t
  number_of_graph_representations() const {
    return number_of_formats_inherited;
  }

  template <typename view_t>
  constexpr bool contains_representation() {
    return std::disjunction_v<std::is_same<view_t, graph_view_t>...>;
  }

  constexpr memory_space_t memory_space() const { return space; }

 private:
  static constexpr std::size_t number_of_formats_inherited =
      sizeof...(graph_view_t);

};  // struct graph_t

namespace build {

/**
 * @brief
 *
 * @tparam graph_type
 * @param I
 * @return graph
 */
template <typename graph_type>
__device__ __host__ auto from_graph_t(graph_type& I) {
  graph_type G;
  G.set(I.get_number_of_rows(),      // r
        I.get_number_of_columns(),   // c
        I.get_number_of_nonzeros(),  // nnz
        I.get_row_offsets(),         // offsets
        I.get_column_indices(),      // column indices
        I.get_nonzero_values()       // nonzero values
  );

  return G;
}

template <typename graph_type>
__host__ __device__ void fix_virtual_inheritance(graph_type I, graph_type* O) {
  auto G = from_graph_t(I);
  memcpy(O, &G, sizeof(graph_type));
}

namespace device {
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
  fix_virtual_inheritance(I, O);
}

/**
 * @brief Possible work around while keeping virtual (polymorphic behavior.)
 *
 * @tparam graph_type
 * @param r
 * @param c
 * @param nnz
 * @param Ap
 * @param Aj
 * @param Ax
 * @return auto
 */
template <typename graph_type>
void csr_t(graph_type I, graph_type* G) {
  kernel_virtual_inheritance<graph_type><<<1, 1>>>(I, G);
}
}  // namespace device

namespace host {
template <typename graph_type>
void csr_t(graph_type I, graph_type* G) {
  fix_virtual_inheritance(I, G);
}
}  // namespace host

template <memory_space_t space,
          typename edge_vector_t,
          typename vertex_vector_t,
          typename weight_vector_t>
auto from_csr_t(typename vertex_vector_t::value_type const& r,
                typename vertex_vector_t::value_type const& c,
                typename edge_vector_t::value_type const& nnz,
                edge_vector_t& Ap,
                vertex_vector_t& Aj,
                weight_vector_t& Ax) {
  using vertex_type = typename vertex_vector_t::value_type;
  using edge_type = typename edge_vector_t::value_type;
  using weight_type = typename weight_vector_t::value_type;

  auto Ap_ptr = memory::raw_pointer_cast(Ap.data());
  auto Aj_ptr = memory::raw_pointer_cast(Aj.data());
  auto Ax_ptr = memory::raw_pointer_cast(Ax.data());

  using graph_type = graph::graph_t<
      space, vertex_type, edge_type, weight_type,
      graph::graph_csr_t<space, vertex_type, edge_type, weight_type>>;

  typename vector<graph_type, space>::type O(1);
  graph_type G;

  G.set(r, c, nnz, Ap_ptr, Aj_ptr, Ax_ptr);

  if (space == memory_space_t::device) {
    device::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));
  } else {
    host::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));
  }

  return O;
}

}  // namespace build

/**
 * @brief Get the average degree of a graph.
 *
 * @tparam graph_type
 * @param G
 * @return double
 */
template <typename graph_type>
__host__ __device__ double get_average_degree(graph_type const& G) {
  auto sum = 0;
  for (auto v = 0; v < G.get_number_of_vertices(); ++v)
    sum += G.get_neighbor_list_length(v);

  return (sum / G.get_number_of_vertices());
}

/**
 * @brief Get the degree standard deviation of a graph.
 * This method uses population standard deviation,
 * therefore measuring the standard deviation over
 * the entire population (all nodes). This can be
 * sped up by only taking a small sample and using
 * sqrt(accum / graph.get_number_of_vertices() - 1)
 * as the result.
 *
 * @tparam graph_type
 * @param G
 * @return double
 */
template <typename graph_type>
__host__ __device__ double get_degree_standard_deviation(const graph_type& G) {
  auto average_degree = get_average_degree(G);

  double accum = 0.0;
  for (auto v = 0; v < G.get_number_of_vertices(); ++v) {
    double d = G.get_neighbor_list_length(v);
    accum += (d - average_degree) * (d - average_degree);
  }
  return sqrt(accum / G.get_number_of_vertices());
}

/**
 * @brief build a log-scale degree histogram of a graph.
 *
 * @tparam graph_type
 * @tparam histogram_t
 * @param G
 * @return histogram_t*
 */
// template <typename graph_type, typename histogram_t>
// histogram_t* build_degree_histogram(graph_type &graph) {
//   using vertex_t = graph_type::vertex_t;
//   auto length = sizeof(vertex_t) * 8 + 1;

//   thrust::device_vector<vertex_t> histogram(length);

//   auto build_histogram = [graph] __device__ (vertex_t* counts, vertex_t i) {
//       auto degree = graph.get_neighbor_list_length(i);
//       while (num_neighbors >= (1 << log_length))
//         log_length++;

//       operation::atomic::add(&counts[log_length], (vertex_t)1);
//   };

//   auto begin = 0;
//   auto end = graph.get_number_of_vertices();
//   operator::for_all(thrust::device, histogram.data(), begin, end,
//   build_histogram);

//   return histogram.data.get();
// }

}  // namespace graph
}  // namespace gunrock