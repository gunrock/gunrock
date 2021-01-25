#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/util/math.hxx>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

#include <gunrock/graph/coo.hxx>
#include <gunrock/graph/csc.hxx>
#include <gunrock/graph/csr.hxx>

namespace gunrock {
namespace graph {

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
      typename std::tuple_element<0,  // default: get first type
                                  std::tuple<graph_view_t...>>::type;

 public:
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;
  using vertex_pair_type = vertex_pair_t<vertex_type>;

  using vertex_pointer_t = vertex_t*;
  using edge_pointer_t = edge_type*;
  using weight_pointer_t = weight_t*;

  using graph_type =
      graph_t<space, vertex_type, edge_type, weight_type, graph_view_t...>;

  // Different supported graph representation views.
  using graph_csr_view_t = graph_csr_t<vertex_type, edge_type, weight_type>;
  using graph_csc_view_t = graph_csc_t<vertex_type, edge_type, weight_type>;
  using graph_coo_view_t = graph_coo_t<vertex_type, edge_type, weight_type>;

  __host__ __device__ graph_t()
      : number_of_vertices(0),
        number_of_edges(0),
        properties(),
        graph_view_t()... {}

  // template<typename csr_matrix_t>
  // graph_t(csr_matrix_t& rhs) :
  //   graph_base_type(rhs.num_rows,
  //                   rhs.num_nonzeros),
  //   graph_csr_view(rhs) {}

  // graph_t specific methods (base graph type)
  __host__ __device__ __forceinline__ const vertex_type
  get_number_of_vertices() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ const edge_type
  get_number_of_edges() const {
    return number_of_edges;
  }

  bool is_directed() { return properties.directed; }

  __host__ __device__ __forceinline__ std::size_t
  number_of_graph_representations() const {
    return number_of_formats_inherited;
  }

  template <typename input_view_t>
  constexpr bool contains_representation() {
    return std::disjunction_v<std::is_same<input_view_t, graph_view_t>...>;
  }

  __host__ __device__ __forceinline__ constexpr memory_space_t memory_space()
      const {
    return space;
  }

  template <class input_view_t = first_view_t, typename... T>
  __host__ __device__ void set(vertex_type const& _number_of_vertices,
                               edge_type const& _number_of_edges,
                               T... args) {
    this->number_of_vertices = _number_of_vertices;
    this->number_of_edges = _number_of_edges;
    input_view_t::set(_number_of_vertices, _number_of_edges, args...);
  }

  // Override pure virtual functions Must use [override] keyword to identify
  // functions that are overriding the derived class
  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const /* override */ {
    assert(v < this->get_number_of_vertices());
    return input_view_t::get_number_of_neighbors(v);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_source_vertex(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_destination_vertex(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const /* override */ {
    assert(v < this->get_number_of_vertices());
    return input_view_t::get_starting_edge(v);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_source_and_destination_vertices(e);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_edge(vertex_type const& source, vertex_type const& destination) const
  /* override */ {
    assert((source < this->get_number_of_vertices()) &&
           (destination < this->get_number_of_vertices()));
    return input_view_t::get_edge(source, destination);
  }

  template <typename input_view_t = first_view_t>
  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_edge_weight(e);
  }

 private:
  // TODO: fix this, it is including empty_t structs for now.
  // We can subtract those structs to get the real value.
  static constexpr std::size_t number_of_formats_inherited =
      sizeof...(graph_view_t);

  vertex_type number_of_vertices;
  edge_type number_of_edges;
  graph_properties_t properties;

};  // struct graph_t

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
    sum += G.get_number_of_neighbors(v);

  return (sum / G.get_number_of_vertices());
}

/**
 * @brief Get the degree standard deviation of a graph. This method uses
 * population standard deviation, therefore measuring the standard deviation
 * over the entire population (all nodes). This can be sped up by only taking a
 * small sample and using sqrt(accum / graph.get_number_of_vertices() - 1) as
 * the result.
 *
 * @tparam graph_type
 * @param G
 * @return double
 */
template <typename graph_type>
__host__ __device__ double get_degree_standard_deviation(graph_type const& G) {
  auto average_degree = get_average_degree(G);

  double accum = 0.0;
  for (auto v = 0; v < G.get_number_of_vertices(); ++v) {
    double d = G.get_number_of_neighbors(v);
    accum += (d - average_degree) * (d - average_degree);
  }
  return sqrt(accum / G.get_number_of_vertices());
}

/**
 * @brief build a log-scale degree histogram of a graph.
 * @todo maybe a faster implementation will maybe be creating a segment array
 * (which is just number of neighbors per vertex), and then sort and find the
 * end of each bin of values using an upper_bound search. Once that is achieved,
 * compute the adjacent_difference of the cumulative histogram.
 *
 * @tparam graph_type
 * @tparam histogram_t
 * @param G
 * @param histogram
 * @param context
 */
template <typename graph_type, typename histogram_t>
void build_degree_histogram(graph_type const& G,
                            histogram_t* histogram,
                            cuda::stream_t stream = 0) {
  using vertex_t = typename graph_type::vertex_type;
  auto length = sizeof(vertex_t) * 8 + 1;

  // Initialize histogram array to 0s.
  thrust::fill(thrust::cuda::par.on(stream),
               histogram + 0,       // iterator begin()
               histogram + length,  // iterator end()
               0                    // fill value
  );

  // Build the histogram count.
  auto build_histogram = [=] __device__(vertex_t const& v) {
    auto degree = G.get_number_of_neighbors(v);
    vertex_t log_length = 0;
    while (degree >= (1 << log_length))
      ++log_length;

    math::atomic::add(histogram + log_length, (histogram_t)1);
    return v;  // output is discarded, but we still need the return.
  };

  // Transform (count from 0...#_of_Vertices), and perform
  // the operation called build_histogram. Ignore output, as
  // we are interested in what goes in the pointer histogram.
  thrust::transform(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator((vertex_t)0),  // Begin: 0
      thrust::make_counting_iterator(
          (vertex_t)G.get_number_of_vertices()),  // End: # of Vertices
      thrust::make_discard_iterator(0),           // Discard Output Iterator
      build_histogram                             // Unary operation
  );
}

/**
 * @brief Utility to remove self-loops, so, if we have an edge between vertex_0
 * and vertex_0, that edge will be removed as it is a self-loop.
 *
 * @tparam graph_type
 * @param G
 */
template <typename graph_type>
void remove_self_loops(graph_type& G) {}

}  // namespace graph
}  // namespace gunrock

// Build graph includes
#include <gunrock/graph/build.hxx>