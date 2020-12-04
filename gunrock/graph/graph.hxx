#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

#include <gunrock/graph/detail/base.hxx>

#include <gunrock/graph/coo.hxx>
#include <gunrock/graph/csc.hxx>
#include <gunrock/graph/csr.hxx>

// #include <gunrock/algorithms/search/binary_search.cuh>

namespace gunrock {
namespace graph {

using namespace format;
using namespace detail;
using namespace memory;

/**
 * @brief Variadic inheritence based graph class, inherit only what you need
 * from the formats implemented. See detail/base.hxx for the graph_base_t
 * implementation. Things to consider:
 * - Coordinate graph view is equivalent to an edge list.
 * - Compressed sparse row/column view are equivalent to each other in terms of
 * complexity.
 *
 * | Operation     | Adjacency Matrix | COO  | Adj. List    | CSR/CSC |
 * |---------------|------------------|------|--------------|---------|
 * | scan          | O(n^2)           | O(m) | O(m+n)       | O(m+n)  |
 * | get neighbors | O(n)             | O(m) | O(d)         | O(d)    |
 * | is edge       | O(1)             | O(m) | O(d)         | O(d)    |
 * | insert edge   | O(1)             | O(1) | O(1) or O(d) | O(m+n)  | (x)
 * | delete edge   | O(1)             | O(m) | O(d)         | O(m+n)  | (x)
 *
 *
 * @tparam space
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam weight_t
 * @tparam graph_view_t
 */
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
  using graph_csr_view = graph_csr_t<vertex_type, edge_type, weight_type>;
  using graph_csc_view = graph_csc_t<vertex_type, edge_type, weight_type>;
  using graph_coo_view = graph_coo_t<vertex_type, edge_type, weight_type>;

  __host__ __device__ graph_t() : graph_view_t()... {}

  // template<typename csr_matrix_t>
  // graph_t(csr_matrix_t& rhs) :
  //   graph_base_type(rhs.num_rows,
  //                   rhs.num_nonzeros),
  //   graph_csr_view(rhs) {}

  // Override base class functions, CUDA does not support
  // virtual inheritance at all.
  __host__ __device__ __forceinline__ const vertex_type
  get_number_of_vertices() const {
    return first_view_t::get_number_of_vertices();
  }

  __host__ __device__ __forceinline__ const edge_type
  get_number_of_edges() const {
    return first_view_t::get_number_of_edges();
  }

  bool is_directed() { return first_view_t::is_directed(); }

  // template <class input_view_t = first_view_t, typename... T>
  // __host__ __device__ void set(T... args) {
  //   input_view_t::set(args...);
  // }

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const override {
    return input_view_t::get_number_of_neighbors(v);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const override {
    return input_view_t::get_source_vertex(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const override {
    return input_view_t::get_edge_weight(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const override {
    return input_view_t::get_destination_vertex(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const override {
    return input_view_t::get_starting_edge(v);
  }

  // graph_t specific methods.
  __host__ __device__ __forceinline__ std::size_t
  number_of_graph_representations() const {
    return number_of_formats_inherited;
  }

  template <typename input_view_t>
  constexpr bool contains_representation() {
    return std::disjunction_v<std::is_same<input_view_t, graph_view_t>...>;
  }

  constexpr memory_space_t memory_space() const { return space; }

 private:
  static constexpr std::size_t number_of_formats_inherited =
      sizeof...(graph_view_t);

};  // struct graph_t

/**
 * @brief Container to store graph struct pointers for both
 * host and device memory spaces.
 *
 * @tparam device_type
 * @tparam host_type
 */
template <typename device_type, typename host_type>
struct graph_container_t {
 private:
  std::shared_ptr<device_type> d;
  std::shared_ptr<host_type> h;

  // Calling .get() from __device__ isn't supported.
  device_type* d_ptr;

 public:
  using graph_type = device_type;

  graph_container_t(std::shared_ptr<device_type> _d,
                    std::shared_ptr<host_type> _h)
      : d(_d), h(_h), d_ptr(d.get()) {}

  __host__ __device__ __forceinline__ auto operator-> () const {
#ifdef __CUDA_ARCH__
    return d_ptr;
#else
    return h.get();
#endif
  }
};

/**
 * @brief Get the average degree of a graph.
 *
 * @tparam graph_container_type note that the input to this function is the
 * graph_container_type struct and not the graph_t, it takes both the device and
 * the host pointers to a graph.
 * @param G
 * @return double
 */
template <typename graph_container_type>
__host__ __device__ double get_average_degree(graph_container_type const& G) {
  auto sum = 0;
  for (auto v = 0; v < G->get_number_of_vertices(); ++v)
    sum += G->get_number_of_neighbors(v);

  return (sum / G->get_number_of_vertices());
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
 * @tparam graph_container_type note that the input to this function is the
 * graph_container_type struct and not the graph_t, it takes both the device and
 * the host pointers to a graph.
 * @param G
 * @return double
 */
template <typename graph_container_type>
__host__ __device__ double get_degree_standard_deviation(
    graph_container_type const& G) {
  auto average_degree = get_average_degree(G);

  double accum = 0.0;
  for (auto v = 0; v < G->get_number_of_vertices(); ++v) {
    double d = G->get_number_of_neighbors(v);
    accum += (d - average_degree) * (d - average_degree);
  }
  return sqrt(accum / G->get_number_of_vertices());
}

/**
 * @brief build a log-scale degree histogram of a graph.
 *
 * @tparam graph_container_type note that the input to this function is the
 * graph_container_t struct and not the graph_t, it takes both the device and
 * the host pointers to a graph.
 * @tparam histogram_t
 * @param G
 * @return histogram_t*
 */
// template <typename graph_container_type, typename histogram_t>
// histogram_t* build_degree_histogram(graph_container_type &graph) {
//   using vertex_t = typename graph_container_type::graph_type::vertex_type;

//   auto length = sizeof(vertex_t) * 8 + 1;

//   thrust::device_vector<vertex_t> histogram(length);

//   auto build_histogram = [graph] __device__ (vertex_t* counts, vertex_t i) {
//       auto degree = graph.get_neighbor_list_length(i);
//       while (num_neighbors >= (1 << log_length))
//         ++log_length;

//       operation::atomic::add(&counts[log_length], (vertex_t)1);
//   };

//   auto begin = 0;
//   auto end = graph.get_number_of_vertices();
//   operators::for_all(thrust::device, histogram.data(), begin, end,
//   build_histogram);

//   return histogram.data.get();
// }

}  // namespace graph
}  // namespace gunrock

// Build graph includes
#include <gunrock/graph/build.hxx>