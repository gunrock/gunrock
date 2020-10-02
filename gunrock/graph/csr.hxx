#pragma once

#include <cassert>
#include <tuple>
#include <iterator>

#include <gunrock/algorithms/search/binary_search.hxx>

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>
#include <gunrock/graph/detail/graph_base.hxx>

namespace gunrock {
namespace graph {


using namespace format;
using namespace detail;
using namespace memory;

template <memory_space_t space, typename vertex_t, typename edge_t, typename weight_t> 
class graph_csr_t : virtual public graph_base_t<vertex_t, edge_t, weight_t> {
    
    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pointer_t  = vertex_t*;
    using edge_pointer_t    = edge_type*;
    using weight_pointer_t  = weight_t*;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;

    public:
        graph_csr_t() : graph_base_type() {}

        // Disable copy ctor and assignment operator.
        // We do not want to let user copy only a slice.
        // Explanation: https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
        // graph_csr_t(const graph_csr_t& rhs) = delete;               // Copy constructor
        // graph_csr_t& operator=(const graph_csr_t& rhs) = delete;    // Copy assignment
        
        // Override pure virtual functions
        // Must use [override] keyword to identify functions that are
        // overriding the derived class
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(vertex_type const& v) const override {
            // assert(v < graph_base_type::get_number_of_vertices());
            printf("Getting neighbor list length...\n");
            auto offsets = get_row_offsets();
            printf("*row_offsets = %p\n", offsets);
            printf("row_offsets[%i+1] = %i\n", v, offsets[v+1]);
            printf("row_offsets[%i] = %i\n", v, offsets[v]);
            return (offsets[v+1] - offsets[v]);
        }

        __host__ __device__ __forceinline__
        vertex_type get_source_vertex(edge_type const& e) const override {
            assert(e < graph_base_type::get_number_of_edges());

            // XXX: I am dumb, idk if this is upper or lower bound?
            return (vertex_type) algo::search::binary::upper_bound(
                get_row_offsets(), e, 
                graph_base_type::get_number_of_vertices());
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
        auto get_row_offsets() const {
            return row_offsets;
        }

        // __host__ __device__ __forceinline__
        // auto get_row_offsets_size() const {
        //     return csr->row_offsets.size();
        // }

        // __host__ __forceinline__
        // auto get_row_offsets_iterator() const {
        //     return csr->row_offsets.begin();
        // }

        // __host__ __forceinline__
        // auto get_row_offsets_iterator_end() const {
        //     return csr->row_offsets.end();
        // }

        template<typename vertex_vector_t, typename edge_vector_t, 
                 typename weight_vector_t>
        void set(vertex_type const& r, vertex_type const& c, edge_type const& nnz,
                 edge_vector_t& Ap, vertex_vector_t& Aj, weight_vector_t& Ax) {
            
            // Set number of verties & edges
            graph_base_type::set_number_of_vertices(r);
            graph_base_type::set_number_of_edges(nnz);

            // Set raw pointers
            row_offsets     = memory::raw_pointer_cast<edge_type>(Ap.data());
            column_indices  = memory::raw_pointer_cast<vertex_type>(Aj.data());
            nonzero_values  = memory::raw_pointer_cast<weight_type>(Ax.data());
        }

    private:
        // Underlying data storage

        // XXX: Maybe use these to hold thrust pointers?
        // I don't know if this is safe, even when using
        // shared pointers.
        edge_pointer_t     row_offsets;
        vertex_pointer_t   column_indices;
        weight_pointer_t   nonzero_values;
        
};  // struct graph_csr_t

}   // namespace graph
}   // namespace gunrock