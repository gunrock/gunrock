#pragma once

#include <cassert>
#include <tuple>

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
class graph_coo_t : public virtual graph_base_t<vertex_t, edge_t, weight_t> {

    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using coo_type        = coo_t<vertex_type, edge_type, weight_type>;
    
    public:
        graph_coo_t() : graph_base_type() {}

        graph_coo_t(edge_type* row_indices,
                    edge_type* column_indices, 
                    weight_type* weights,
                    vertex_type number_of_vertices,
                    edge_type number_of_edges) : 
            graph_base_type(
                number_of_vertices, 
                number_of_edges),
            coo(number_of_vertices, 
                number_of_vertices, 
                number_of_edges, 
                row_indices, 
                column_indices, 
                weights) {}
        
        // Override pure virtual functions
        // Must use [override] keyword to identify functions that are
        // overriding the derived class
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(const vertex_type& v) const override {

        }

        // __host__ __device__ __forceinline__
        // vertex_type get_source_vertex(const edge_type& e) const override {
        //     auto source_indices = coo.I.get();
        //     return source_indices[e];
        // }
        
        // __host__ __device__ __forceinline__
        // vertex_type get_destination_vertex(const edge_type& e) const override {
        //     auto destination_indices = coo.J.get(); 
        //     return destination_indices[e];
        // }

        // __host__ __device__ __forceinline__
        // vertex_pair_type get_source_and_destination_vertices(const edge_type& e) const override {
        //     auto source_indices = coo.I.get();
        //     auto destination_indices = coo.J.get();
        //     return {source_indices[e], destination_indices[e]};
        // }
        
        // __host__ __device__ __forceinline__
        // edge_type get_edge(const vertex_type& source, 
        //                 const vertex_type& destination) const override {
        //
        // }

        // Representation specific functions
        // ...

    private:
        coo_type coo;
};  // struct graph_coo_t

}   // namespace graph
}   // namespace gunrock