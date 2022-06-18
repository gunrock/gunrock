#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/memory.hxx>
#include <gunrock/util/load_store.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/algorithms/search/binary_search.hxx>

namespace gunrock {
namespace graph {

struct empty_csr_t {};

using namespace memory;

// XXX: The ideal thing to do here is to inherit
// base class with virtual keyword specifier, therefore
// public virtual graph_base_t<...> {}, but according to
// cuda's programming guide, that is not allowerd.
// From my tests (see smart_struct) results in an illegal
// memory error. Another important thing to note is that
// virtual functions should also have undefined behavior,
// but they seem to work.
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_csr_t {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pair_type = vertex_pair_t<vertex_type>;

 public:
  __host__ __device__ graph_csr_t()
      : offsets(nullptr), indices(nullptr), values(nullptr) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/

  // Copy constructor
  // graph_csr_t(const graph_csr_t& rhs) = delete;
  // Copy assignment
  // graph_csr_t& operator=(const graph_csr_t& rhs) = delete;

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const {
    return (get_starting_edge(v + 1) - get_starting_edge(v));
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(edge_type const& e) const {
    auto keys = get_row_offsets();
    auto key = e;

    // returns `it` such that everything to the left is <= e.
    // This will be one element to the right of the node id.
    auto it = thrust::lower_bound(
        thrust::seq, thrust::counting_iterator<edge_t>(0),
        thrust::counting_iterator<edge_t>(this->number_of_vertices), key,
        [keys] __host__ __device__(const edge_t& pivot, const edge_t& key) {
          return keys[pivot] <= key;
        });

    return (*it) - 1;
  }

  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const {
    return thread::load(&indices[e]);
  }

  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const {
    return thread::load(&offsets[v]);
  }

  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(const edge_type& e) const {
    return {get_source_vertex(e), get_destination_vertex(e)};
  }

  __host__ __device__ __forceinline__ edge_type
  get_edge(const vertex_type& source, const vertex_type& destination) const {
    return (edge_type)search::binary::execute(get_column_indices(), destination,
                                              offsets[source],
                                              offsets[source + 1] - 1);
  }

  /**
   * @brief Count the number of vertices belonging to the set intersection
   * between the source and destination vertices adjacency lists. Executes a
   * function on each intersection.
   *
   * @param source Index of the source vertex
   * @param destination Index of the destination
   * @param on_intersection Lambda function executed at each intersection
   * @return Number of shared vertices between source and destination
   */
  template <typename operator_type>
  __host__ __device__ __forceinline__ vertex_type
  get_intersection_count(const vertex_type& source,
                         const vertex_type& destination,
                         operator_type on_intersection) const {
    vertex_type intersection_count = 0;

    auto source_neighbors_count = get_number_of_neighbors(source);
    auto destination_neighbors_count = get_number_of_neighbors(destination);

    auto source_offset = offsets[source];
    auto destination_offset = offsets[destination];

    // if (source_neighbors_count > destination_neighbors_count) {
    //   std::swap(source_offset, destination_offset);
    //   std::swap(source_neighbors_count, destination_neighbors_count);
    // }

    auto source_edges_iter = indices + source_offset;
    auto destination_edges_iter = indices + destination_offset;

    auto needle = *destination_edges_iter;
    auto source_search_start =
        search::binary::execute(source_edges_iter, needle, vertex_t{0},
                                source_neighbors_count, search::bound_t::lower);
    edge_type destination_search_start = 0;
    // printf("[%i -> %i] %i, [%i, %i], [%i, %i]\n", source, destination,
    // needle,
    //        source_search_start, destination_search_start,
    //        source_neighbors_count, destination_neighbors_count);

    while (source_search_start < source_neighbors_count &&
           destination_search_start < destination_neighbors_count) {
      auto cur_edge_src = source_edges_iter[source_search_start];
      auto cur_edge_dst = destination_edges_iter[destination_search_start];
      // printf("%i, %i | %i, %i\n", cur_edge_src, cur_edge_dst,
      //        source_search_start, destination_search_start);
      // if (source == 1 and destination == 2) {
      //   printf("%i, %i\n", cur_edge_src, cur_edge_dst);
      // }
      if (cur_edge_src == cur_edge_dst) {
        intersection_count++;
        source_search_start++;
        destination_search_start++;
        on_intersection(cur_edge_src);
        // printf("Triangle: %i, %i, %i\n", source, destination, cur_edge_src);
      } else if (cur_edge_src > cur_edge_dst) {
        destination_search_start++;
      } else {
        source_search_start++;
      }
    }

    return intersection_count;
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const {
    return thread::load(&values[e]);
  }

  // Representation specific functions
  // ...
  __host__ __device__ __forceinline__ auto get_row_offsets() const {
    return offsets;
  }

  __host__ __device__ __forceinline__ auto get_column_indices() const {
    return indices;
  }

  __host__ __device__ __forceinline__ auto get_nonzero_values() const {
    return values;
  }

  // Graph type (inherited from this class) has equivalents of this in graph
  // terminology (vertices and edges). Also include these for linear algebra
  // terminology
  __host__ __device__ __forceinline__ auto get_number_of_rows() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_columns() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_nonzeros() const {
    return number_of_edges;
  }

 protected:
  __host__ __device__ void set(vertex_type const& _number_of_vertices,
                               edge_type const& _number_of_edges,
                               edge_type* _row_offsets,
                               vertex_type* _column_indices,
                               weight_type* _values) {
    this->number_of_vertices = _number_of_vertices;
    this->number_of_edges = _number_of_edges;
    // Set raw pointers
    offsets = raw_pointer_cast<edge_type>(_row_offsets);
    indices = raw_pointer_cast<vertex_type>(_column_indices);
    values = raw_pointer_cast<weight_type>(_values);
  }

 private:
  // Underlying data storage
  vertex_type number_of_vertices;  // XXX: redundant
  edge_type number_of_edges;       // XXX: redundant

  edge_type* offsets;
  vertex_type* indices;
  weight_type* values;

};  // struct graph_csr_t

}  // namespace graph
}  // namespace gunrock