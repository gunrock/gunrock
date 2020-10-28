#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/algorithms/search/binary_search.hxx>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/detail/base.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

namespace gunrock {
namespace graph {

using namespace format;
using namespace detail;

template <memory_space_t space,
          typename vertex_t,
          typename edge_t,
          typename weight_t>
class graph_coo_t : virtual public graph_base_t<vertex_t, edge_t, weight_t> {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  using vertex_pair_type = vertex_pair_t<vertex_t>;
  using properties_type = graph_properties_t;

  using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;

 public:
  graph_coo_t() : graph_base_type() {}

  // Override pure virtual functions
  // Must use [override] keyword to identify functions that are
  // overriding the derived class
  __host__ __device__ __forceinline__ edge_type
  get_number_of_neighbors(vertex_type const& v) const override {
    assert(v < graph_base_type::get_number_of_vertices());
    // XXX: ...
  }

  __host__ __device__ __forceinline__ vertex_type
  get_source_vertex(const edge_type& e) const override {
    assert(e < graph_base_type::get_number_of_edges());
    return row_indices[e];
  }

  __host__ __device__ __forceinline__ vertex_type
  get_destination_vertex(const edge_type& e) const override {
    assert(e < graph_base_type::get_number_of_edges());
    return column_indices[e];
  }

  __host__ __device__ __forceinline__ edge_type
  get_starting_edge(vertex_type const& v) const override {
    assert(v < graph_base_type::get_number_of_vertices());
    // XXX: ...
  }

  __host__ __device__ __forceinline__ vertex_pair_type
  get_source_and_destination_vertices(const edge_type& e) const override {
    assert(e < graph_base_type::get_number_of_edges());
    return {get_source_vertex(e), get_destination_vertex(e)};
  }

  __host__ __device__ __forceinline__ edge_type
  get_edge(const vertex_type& source,
           const vertex_type& destination) const override {
    assert((source < graph_base_type::get_number_of_vertices()) &&
           (destination < graph_base_type::get_number_of_vertices()));
    // XXX: ...
  }

  __host__ __device__ __forceinline__ weight_type
  get_edge_weight(edge_type const& e) const override {
    assert(e < graph_base_type::get_number_of_edges());
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

  __host__ __device__ __forceinline__ auto get_number_of_rows() const {
    return number_of_rows;
  }

  __host__ __device__ __forceinline__ auto get_number_of_columns() const {
    return number_of_columns;
  }

  __host__ __device__ __forceinline__ auto get_number_of_nonzeros() const {
    return number_of_nonzeros;
  }

 protected:
  template <typename vertex_vector_t, typename weight_vector_t>
  void set(vertex_type const& r,
           vertex_type const& c,
           edge_type const& nnz,
           vertex_vector_t& I,
           vertex_vector_t& J,
           weight_vector_t& X) {
    // Set number of verties & edges
    graph_base_type::set_number_of_vertices(r);
    graph_base_type::set_number_of_edges(nnz);

    number_of_rows = r;
    number_of_columns = c;
    number_of_nonzeros = nnz;

    // Set raw pointers
    row_indices = memory::raw_pointer_cast<edge_type>(I.data());
    column_indices = memory::raw_pointer_cast<vertex_type>(J.data());
    values = memory::raw_pointer_cast<weight_type>(X.data());
  }

  __host__ __device__ void set(vertex_type const& r,
                               vertex_type const& c,
                               edge_type const& nnz,
                               vertex_type* I,
                               vertex_type* J,
                               weight_type* X) {
    // Set number of verties & edges
    graph_base_type::set_number_of_vertices(r);
    graph_base_type::set_number_of_edges(nnz);

    number_of_rows = r;
    number_of_columns = c;
    number_of_nonzeros = nnz;

    // Set raw pointers
    row_indices = memory::raw_pointer_cast<edge_type>(I);
    column_indices = memory::raw_pointer_cast<vertex_type>(J);
    values = memory::raw_pointer_cast<weight_type>(X);
  }

 private:
  // Underlying data storage
  vertex_type number_of_rows;
  vertex_type number_of_columns;
  edge_type number_of_nonzeros;

  // XXX: Maybe use these to hold thrust pointers?
  // I don't know if this is safe, even when using
  // shared pointers.
  vertex_type* row_indices;
  vertex_type* column_indices;
  weight_type* values;
};  // struct graph_coo_t

}  // namespace graph
}  // namespace gunrock