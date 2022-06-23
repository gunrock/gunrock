#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/algorithms/search/binary_search.hxx>

namespace gunrock {
namespace graph {

struct empty_csc_t {};

using namespace memory;

template <typename vertex_t, typename edge_t, typename weight_t>
class graph_csc_t {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pair_type = vertex_pair_t<vertex_t>;

 public:
  __host__ __device__ graph_csc_t()
      : offsets(nullptr), indices(nullptr), values(nullptr) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  // Copy constructor
  // graph_csc_t(const graph_csc_t& rhs) = delete;
  // Copy assignment
  // graph_csc_t& operator=(const graph_csc_t& rhs) = delete;

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(const vertex_type& v) const {
    return (offsets[v + 1] - offsets[v]);
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(const edge_type& e) const {
    // IMO, when a graph is stored as a CSC matrix, `indices` represents
    // all of the in-neighbors.  So `indices[e]` is the _source_ of the edge.
    return indices[e];
  }

  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(const edge_type& e) const {
    // Same justification as `get_source_vertex`

    auto keys = get_column_offsets();
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

  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const {
    return offsets[v];
  }

  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(const edge_type& e) const {
    return {get_source_vertex(e), get_destination_vertex(e)};
  }

  __host__ __device__ __forceinline__ edge_type
  get_edge(const vertex_type& source, const vertex_type& destination) const {
    return (edge_type)search::binary::execute(get_row_indices(), source,
                                              offsets[destination],
                                              offsets[destination + 1]) -
           1;
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const {
    return values[e];
  }

  // Representation specific functions
  // ...
  __host__ __device__ __forceinline__ auto get_column_offsets() const {
    return offsets;
  }

  __host__ __device__ __forceinline__ auto get_row_indices() const {
    return indices;
  }

  __host__ __device__ __forceinline__ auto get_nonzero_values() const {
    return values;
  }

  __host__ __device__ __forceinline__ auto get_number_of_vertices() const {
    return number_of_vertices;
  }

  __host__ __device__ __forceinline__ auto get_number_of_edges() const {
    return number_of_edges;
  }

 protected:
  __host__ __device__ void set(vertex_type const& _number_of_vertices,
                               edge_type const& _number_of_edges,
                               edge_type* _column_offsets,
                               vertex_type* _row_indices,
                               weight_type* _values) {
    this->number_of_vertices = _number_of_vertices;
    this->number_of_edges = _number_of_edges;
    // Set raw pointers
    offsets = raw_pointer_cast<edge_type>(_column_offsets);
    indices = raw_pointer_cast<vertex_type>(_row_indices);
    values = raw_pointer_cast<weight_type>(_values);
  }

 private:
  // Underlying data storage
  vertex_type number_of_vertices;
  edge_type number_of_edges;

  edge_type* offsets;
  vertex_type* indices;
  weight_type* values;

};  // struct graph_csc_t

}  // namespace graph
}  // namespace gunrock
