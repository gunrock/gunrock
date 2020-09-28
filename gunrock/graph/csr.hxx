#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <thrust/iterator/counting_iterator.h>

#include <gunrock/algorithms/search/binary_search.hxx>

#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/graph/detail/graph_base.hxx>

namespace gunrock {
namespace graph {


using namespace format;
using namespace detail;

template <typename vertex_t, typename edge_t, typename weight_t> 
class graph_csr_t : public virtual graph_base_t<vertex_t, edge_t, weight_t> {
    
    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using csr_type        = csr_t<vertex_type, edge_type, weight_type>; // XXX: check type order

    public:
        graph_csr_t() : graph_base_type() {}

        graph_csr_t(edge_type* offsets, 
                    vertex_type* indices, 
                    weight_type* weights, 
                    vertex_type number_of_vertices, 
                    edge_type number_of_edges) : 
            graph_base_type(
                number_of_vertices, 
                number_of_edges),
            csr(
                number_of_vertices, 
                number_of_vertices, 
                number_of_edges, 
                offsets, 
                indices, 
                weights) {}
        
        // Override pure virtual functions
        // Must use [override] keyword to identify functions that are
        // overriding the derived class
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(const vertex_type& v) const override {
            assert(v < graph_base_type::_number_of_vertices);
            auto offsets = csr.row_offsets.get();
            return (offsets[v+1] - offsets[v]);
        }

        __host__ __device__ __forceinline__
        vertex_type get_source_vertex(const edge_type& e) const override {
            assert(e < graph_base_type::_number_of_edges);
            const edge_type* offsets = csr.row_offsets.get();
            // XXX: I am dumb, idk if this is upper or lower bound?
            return algo::search::binary::upper_bound(offsets, e, graph_base_type::_number_of_vertices);
        }
        
        // __host__ __device__ __forceinline__
        // vertex_type get_destination_vertex(const edge_type& e) const override { 

        // }
        // __host__ __device__ __forceinline__
        // vertex_pair_type get_source_and_destination_vertices(const edge_type& e) const override {

        // }
        
        // __host__ __device__ __forceinline__
        // edge_type get_edge(const vertex_type& source, 
        //                 const vertex_type& destination) const override {

        // }

        // Representation specific functions
        // ...

    protected:
        __host__ __device__ __forceinline__
        void set_row_offsets(std::shared_ptr<vertex_type>& offsets) {
            csr.row_offsets = offsets;
        }

        __host__ __device__ __forceinline__
        void set_column_indices(std::shared_ptr<edge_type>& indices) {
            csr.column_indices = indices;
        }

        __host__ __device__ __forceinline__
        void set_nonzero_values(std::shared_ptr<weight_type>& nonzeros) {
            csr.nonzero_values = nonzeros;
        }

        __host__ __device__ __forceinline__
        void set_num_rows(const vertex_type& num_rows) {
            csr.num_rows = num_rows;
        }

        __host__ __device__ __forceinline__
        void set_num_columns(const vertex_type& num_columns) {
            csr.num_columns = num_columns;
        }

        __host__ __device__ __forceinline__
        void set_num_nonzeros(const edge_type& num_nonzeros) {
            csr.num_nonzeros = num_nonzeros;
        }

        __host__ __device__ __forceinline__
        void set(const vertex_type& num_rows,
                 const vertex_type& num_columns,
                 const edge_type& num_nonzeros,
                 std::shared_ptr<vertex_type>& offsets, 
                 std::shared_ptr<edge_type>& indices, 
                 std::shared_ptr<weight_type>& nonzeros) {
            
            set_num_rows(num_rows);
            set_num_columns(num_columns);
            set_num_nonzeros(num_nonzeros);
            set_row_offsets(offsets);
            set_column_indices(indices);
            set_nonzero_values(nonzeros);
        }

    private:
        csr_type csr;
};  // struct graph_csr_t

}   // namespace graph
}   // namespace gunrock