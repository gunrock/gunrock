#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/memory.hxx>
#include <gunrock/util/load_store.hxx>
#include <gunrock/util/type_traits.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/algorithms/search/binary_search.hxx>
#include <gunrock/formats/formats.hxx>

namespace gunrock {
namespace graph {

using namespace memory;

struct empty_coo_t {};

template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t>
class graph_coo_t {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pair_type = vertex_pair_t<vertex_type>;

 public:
  __host__ __device__ graph_coo_t()
      : row_indices(nullptr), column_indices(nullptr), values(nullptr) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  // Copy constructor
  // graph_coo_t(const graph_coo_t& rhs) = delete;
  // Copy assignment
  // graph_coo_t& operator=(const graph_coo_t& rhs) = delete;

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const {
    return get_starting_edge(v + 1) - get_starting_edge(v);
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(const edge_type& e) const {
    return thread::load(&row_indices[e]);
  }

  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(edge_type const& e) const {
    return thread::load(&column_indices[e]);
  }

  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const {
    auto ptr_row_indices = row_indices;
    // Returns `it` such that everything to the left is < `v`
    // This will be the offset of `v`
    auto it = thrust::lower_bound(
        thrust::seq, thrust::counting_iterator<edge_t>(0),
        thrust::counting_iterator<edge_t>(this->number_of_edges), v,
        [ptr_row_indices] __host__ __device__(const vertex_type& pivot,
                                              const vertex_type& key) {
          return ptr_row_indices[pivot] < key;
        });

    return (*it);
  }

  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(const edge_type& e) const {
    return {get_source_vertex(e), get_destination_vertex(e)};
  }

  __host__ __device__ __forceinline__ edge_type
  get_edge(const vertex_type& source, const vertex_type& destination) const {
    return (edge_type)search::binary::execute(get_column_indices(), destination,
                                              get_starting_edge(source),
                                              get_starting_edge(source + 1)) -
           1;
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const {
    return values[e];
  }

  // Representation specific functions
  // ...
  __host__ __device__ __forceinline__ auto get_row_indices() const {
    return row_indices;
  }

  __host__ __device__ __forceinline__ auto get_column_indices() const {
    return column_indices;
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
  __host__ void set(
      gunrock::format::coo_t<space, vertex_t, edge_t, weight_t>& coo) {
    this->number_of_vertices = coo.number_of_rows;
    this->number_of_edges = coo.number_of_nonzeros;
    // Set raw pointers
    row_indices = raw_pointer_cast(coo.row_indices.data());
    column_indices = raw_pointer_cast(coo.column_indices.data());
    values = raw_pointer_cast(coo.nonzero_values.data());
  }

 private:
  // Underlying data storage
  vertex_type number_of_vertices;
  edge_type number_of_edges;

  vertex_type* row_indices;
  vertex_type* column_indices;
  weight_type* values;
};  // struct graph_coo_t

}  // namespace graph
}  // namespace gunrock