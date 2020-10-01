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

template <typename vertex_t, typename edge_t, typename weight_t, 
          memory_space_t space = memory_space_t::host> 
class graph_csr_t : public virtual graph_base_t<vertex_t, edge_t, weight_t> {
    
    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using csr_type        = csr_t<vertex_type, edge_type, weight_type, space>;

    public:
        graph_csr_t() : 
            graph_base_type(),
            csr(std::make_shared<csr_type>()) {}

        // Disable copy ctor and assignment operator.
        // We do not want to let user copy only a slice.
        // Explanation: https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
        graph_csr_t(const graph_csr_t& rhs) = delete;               // Copy constructor
        graph_csr_t& operator=(const graph_csr_t& rhs) = delete;    // Copy assignment

        graph_csr_t(vertex_type number_of_vertices, 
                    edge_type number_of_edges,
                    std::shared_ptr<csr_type> rhs) : 
            graph_base_type(
                number_of_vertices, 
                number_of_edges) {
            csr = rhs;
        }
        
        // Override pure virtual functions
        // Must use [override] keyword to identify functions that are
        // overriding the derived class
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(const vertex_type& v) const override {
            assert(v < graph_base_type::_number_of_vertices);
            auto offsets = csr->row_offsets.data();
            return (offsets[v+1] - offsets[v]);
        }

        __host__ __device__ __forceinline__
        vertex_type get_source_vertex(const edge_type& e) const override {
            assert(e < graph_base_type::_number_of_edges);
            auto offsets = csr->row_offsets;
            auto comp = [] __host__ __device__ (const edge_type& key, 
                                                const edge_type& pivot) {
                                        return pivot < key;
            };
            // auto offsets = thrust::raw_pointer_cast(csr->row_offsets.data());
            // XXX: I am dumb, idk if this is upper or lower bound?
            // note that this returns an iterator, we need to dereference it to
            // return the vertex_type source_vertex.
            // return (vertex_type) algo::search::binary::upper_bound(offsets.data(), e, offsets.size());
            return (vertex_type) *(algo::search::binary::lower_bound(
                                    offsets.begin(), 
                                    offsets.end(), 
                                    e, 
                                    comp));
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
        void set(std::shared_ptr<csr_type> rhs) {
            csr = rhs;
        }

    private:
        std::shared_ptr<csr_type> csr;
};  // struct graph_csr_t

}   // namespace graph
}   // namespace gunrock