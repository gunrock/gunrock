#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

#include <gunrock/graph/coo.hxx>
#include <gunrock/graph/csc.hxx>
#include <gunrock/graph/csr.hxx>

namespace gunrock {
namespace graph {

using namespace memory;

struct empty_graph_t {};

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
 * @tparam space memory space to use for the graph (device or host).
 * @tparam vertex_t index type of the vertices, must be integral type.
 * @tparam edge_t index type of the edges, must be integral type.
 * @tparam weight_t type of the edge weights.
 * @tparam graph_view_t internal view of the graph, must be a class from among
 * the valid implementations, see graph/csr.hxx, graph/csc.hxx, graph/coo.hxx.
 */
template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          class... graph_view_t>
class graph_t : public graph_view_t... {
  // see: <gunrock/util/type_traits.hxx>
  using true_view_t =
      filter_tuple_t<std::tuple<empty_graph_t,
                                empty_csr_t,
                                empty_csc_t,
                                empty_coo_t>,  // filter these types.
                     std::tuple<graph_view_t...>>;

  // Default view (graph representation) if no view is specified
  using default_view_t = std::tuple_element_t<0, true_view_t>;

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

  /// Different supported graph representation views.
  using graph_csr_view_t = graph_csr_t<vertex_type, edge_type, weight_type>;
  using graph_csc_view_t = graph_csc_t<vertex_type, edge_type, weight_type>;
  using graph_coo_view_t = graph_coo_t<vertex_type, edge_type, weight_type>;

  /**
   * @brief Default constructor for the graph.
   */
  __host__ __device__ graph_t() : properties(), graph_view_t()... {}

  /**
   * @brief Get the number of vertices in the graph. Callable from both host and
   * device even if the memory space is device.
   *
   * @return vertex_type number of vertices in the graph.
   */
  template <class input_view_t = default_view_t>
  __host__ __device__ __forceinline__ const vertex_type
  get_number_of_vertices() const {
    return input_view_t::get_number_of_vertices();
  }

  /**
   * @brief Get the number of edges in the graph. Callable from both host and
   * device even if the memory space is device.
   *
   * @return edge_type number of edges in the graph.
   */
  template <class input_view_t = default_view_t>
  __host__ __device__ __forceinline__ const edge_type
  get_number_of_edges() const {
    return input_view_t::get_number_of_edges();
  }

  /**
   * @brief Returns if the constructed graph is directed or not (undirected).
   *
   * @return true
   * @return false
   */
  bool is_directed() { return properties.directed; }

  /**
   * @brief Number of valid graph representations inherited. Does not include
   * any empty representations.
   *
   * @return std::size_t number of valid graph representations.
   */
  __host__ __device__ __forceinline__ std::size_t
  number_of_graph_representations() const {
    return number_of_formats_inherited;
  }

  /**
   * @brief Pass in a graph representation view as a template parameter and
   * returns true if the graph representation is contains the view.
   *
   * @tparam input_view_t input graph view to check.
   * @return true contains the input view.
   * @return false does not contain the input view.
   */
  template <typename input_view_t>
  constexpr bool contains_representation() {
    return std::disjunction_v<std::is_same<input_view_t, graph_view_t>...>;
  }

  /**
   * @brief Return memory space of the graph.
   *
   * @return memory_space_t memory space of the graph.
   */
  __host__ __device__ __forceinline__ constexpr memory_space_t memory_space()
      const {
    return space;
  }

  /**
   * @brief Set the underlying data pointers and sizes of the graph views.
   *
   * @par Overview
   * This function is used to set the underlying data pointers and sizes of the
   * graph class. Note, this is important because this graph object does NOT own
   * the data. The data is passed to the graph_view_t objects by the user. So,
   * the user is responsible of creating the csr/csc/coo (for example) matrices,
   * passing the data pointers and sizes to the appropriate input graph_view_t
   * objects, and after the use is done, the user is responsible of freeing the
   * data.
   *
   * An example usage is:
   * \code
   *  // Create a graph object and set the data pointers and sizes.
   *  using view_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
   *  using graph_t = graph::graph_t<space, vertex_t, edge_t, weight_t, view_t>;
   *  graph_t G;
   *  G.template set<view_t>(r, nnz, row_offsets, column_indices, values);
   * \endcode
   *
   * @tparam input_view_t input graph view to set.
   * @tparam args_t Type of data pointers.
   * @param _number_of_vertices number of vertices in the graph.
   * @param _number_of_edges number of edges in the graph.
   * @param args data pointers of the view to set csr = (row_offsets,
   * column_indices, values), csc = (column_offsets, row_indices, values), coo =
   * (row_indices, column_indices, values).
   */
  template <class input_view_t = default_view_t, typename... args_t>
  __host__ __device__ void set(vertex_type const& _number_of_vertices,
                               edge_type const& _number_of_edges,
                               args_t... args) {
    input_view_t::set(_number_of_vertices, _number_of_edges, args...);
  }

  /**
   * @brief Get the number of neighbors for a given vertex.
   *
   * @tparam input_view_t specify a view (such as csr_view_t) to get the number
   * of neighbors using a specific underlying view/graph representation.
   * Otherwise, it defaults to the first valid view.
   * @param v vertex to get the number of neighbors.
   * @return edge_type number of neighbors.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const /* override */ {
    /// Override pure virtual functions Must use [override] keyword to identify
    /// functions that are overriding the derived class, however, there's some
    /// limited support within CUDA for virtual inheritance. Therefore, we will
    /// avoid using the keyword.
    assert(v < this->get_number_of_vertices());
    return input_view_t::get_number_of_neighbors(v);
  }

  /**
   * @brief Get the source vertex of an edge.
   *
   * @tparam input_view_t specify a view (such as csr_view_t) to get the source
   * vertex using a specific underlying view/graph representation. Otherwise, it
   * defaults to the first valid view.
   * @param e edge to get the source vertex.
   * @return vertex_type source vertex of the edge.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_source_vertex(e);
  }

  /**
   * @brief Get the destination vertex  of an edge.
   *
   * @tparam input_view_t specify a view (such as csr_view_t) to get the
   * destination vertex using a specific underlying view/graph representation.
   * Otherwise, it defaults to the first valid view.
   * @param e edge to get the destination vertex.
   * @return vertex_type destination vertex of the edge.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_destination_vertex(e);
  }

  /**
   * @brief Requires the format to be sorted. Get the starting index of the edge
   * for a given vertex. The starting edge + number of neighbors = ending edge.
   *
   * @tparam input_view_t specify a view (such as csr_view_t) to get the
   * starting edge using a specific underlying view/graph representation.
   * Otherwise, it defaults to the first valid view.
   * @param v vertex to get the starting edge.
   * @return edge_type starting edge of the vertex.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const /* override */ {
    assert(v < this->get_number_of_vertices());
    return input_view_t::get_starting_edge(v);
  }

  /**
   * @brief Get the source and destination vertices of an edge. Together, source
   * and destination vertex make up an edge by connecting to each other.
   *
   * @tparam input_view_t specify a view.
   * @param e edge to get the source and destination vertices.
   * @return vertex_pair_t source and destination vertices of the edge.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_source_and_destination_vertices(e);
  }

  /**
   * @brief Get the edge of a given source and destination vertices.
   *
   * @tparam input_view_t specify a view.
   * @param source source vertex of the edge.
   * @param destination destination vertex of the edge.
   * @return edge_type edge of the source and destination vertices.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ edge_type
  get_edge(vertex_type const& source, vertex_type const& destination) const
  /* override */ {
    assert((source < this->get_number_of_vertices()) &&
           (destination < this->get_number_of_vertices()));
    return input_view_t::get_edge(source, destination);
  }

  /**
   * @brief Get the edge weight of a given edge.
   *
   * @tparam input_view_t specify a view.
   * @param e edge to get the weight.
   * @return weight_type weight of the edge.
   */
  template <typename input_view_t = default_view_t>
  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const /* override */ {
    assert(e < this->get_number_of_edges());
    return input_view_t::get_edge_weight(e);
  }

 private:
  /// @note using `graph_view_t` here instead will cause problems as that pack
  /// includes empty structs, with true_view_t being the only valid views.
  static constexpr std::size_t number_of_formats_inherited =
      std::tuple_size_v<true_view_t>;

  graph_properties_t properties;

};  // namespace graph

/**
 * @brief Get the average degree of a graph.
 *
 * @tparam graph_type graph type.
 * @param G graph to get the average degree.
 * @return double average degree of the graph.
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
 * @tparam graph_type graph type.
 * @param G graph to get the degree standard deviation.
 * @return double degree standard deviation of the graph.
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
 * @tparam graph_type graph type.
 * @tparam histogram_t histogram type.
 * @param G graph to build the histogram.
 * @param histogram histogram pointer to build.
 * @param stream execution stream to use.
 */
template <typename graph_type, typename histogram_t>
void build_degree_histogram(graph_type const& G,
                            histogram_t* histogram,
                            gcuda::stream_t stream = 0) {
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
  };

  // For each (count from 0...#_of_Vertices), and perform
  // the operation called build_histogram. Ignore output, as
  // we are interested in what goes in the pointer histogram.
  thrust::for_each(thrust::cuda::par.on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),  // Begin: 0
                   thrust::make_counting_iterator<vertex_t>(
                       G.get_number_of_vertices()),  // End: # of Vertices
                   build_histogram                   // Unary operation
  );
}

/**
 * @brief Utility to remove self-loops, so, if we have an edge between vertex_0
 * and vertex_0, that edge will be removed as it is a self-loop.
 * @todo need an implementation.
 *
 * @tparam graph_type graph type.
 * @param G graph to remove self-loops.
 */
template <typename graph_type>
void remove_self_loops(graph_type& G) {}

}  // namespace graph
}  // namespace gunrock

// Build graph includes
#include <gunrock/graph/build.hxx>