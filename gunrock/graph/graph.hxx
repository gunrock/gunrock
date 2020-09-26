#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/util/type_traits.hxx>
#include <gunrock/formats/formats.hxx>
// #include <gunrock/algorithms/search/binary_search.cuh>

namespace gunrock {
namespace graph {

using namespace format;

template <typename vertex_t>
struct vertex_pair_t {
    vertex_t source;
    vertex_t destination;
};

struct graph_properties_t {
    bool directed {false};
    bool weighted {false};
    graph_properties_t() = default;
};

template <typename vertex_t, typename edge_t, typename weight_t>
class graph_base_t {

    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;

    public:
        graph_base_t() : 
            _number_of_vertices(0), 
            _number_of_edges(0), 
            _properties() {}

        graph_base_t(vertex_type number_of_vertices, edge_type number_of_edges) :
            _number_of_vertices(number_of_vertices),
            _number_of_edges(number_of_edges),
            _properties() {}

        graph_base_t(vertex_type number_of_vertices, edge_type number_of_edges, properties_type properties) :
            _number_of_vertices(number_of_vertices),
            _number_of_edges(number_of_edges),
            _properties(properties) {}

        vertex_type get_number_of_vertices() { return _number_of_vertices; }
        edge_type get_number_of_edges() { return _number_of_edges; }
        bool is_directed() { return _properties.directed; }

        // Pure Virtual Functions:: must be implemented in derived classes
        __host__ __device__ __forceinline__
        virtual edge_type get_neighbor_list_length(const vertex_type& v) const = 0;
        
        // __host__ __device__ __forceinline__
        // virtual vertex_type get_source_vertex(const edge_type& e) const = 0;
        
        // __host__ __device__ __forceinline__
        // virtual vertex_type get_destination_vertex(const edge_type& e) const = 0;
        
        // __host__ __device__ __forceinline__
        // virtual vertex_pair_type get_source_and_destination_vertices(const edge_type& e) const = 0; // XXX: return type?
        
        // __host__ __device__ __forceinline__
        // virtual edge_type get_edge(const vertex_type& source, const vertex_type& destination) const = 0;

    protected:
        vertex_type     _number_of_vertices;
        edge_type       _number_of_edges;
        properties_type _properties;


}; // struct graph_base_t

template <typename vertex_t, typename edge_t, typename weight_t> 
class graph_csr_t : public virtual graph_base_t<vertex_t, edge_t, weight_t> {
    
    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using csr_type        = csr_t<edge_type, vertex_type, weight_type>; // XXX: check type order

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

        // __host__ __device__ __forceinline__
        // vertex_type get_source_vertex(const edge_type& e) const override {
        //     assert(e < graph_base_type::_number_of_edges);
        //     // return (algo::search::binary::device::block::rightmost(
        //     //     csr.row_offsets.get(), e, 
        //     //     graph_base_type::_number_of_vertices));
        // }
        
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
        void set_row_offsets(std::shared_ptr<vertex_type> offsets) {
            csr.row_offsets = offsets;
        }

        __host__ __device__ __forceinline__
        void set_column_indices(std::shared_ptr<edge_type> indices) {
            csr.column_indices = indices;
        }

        __host__ __device__ __forceinline__
        void set_nonzero_values(std::shared_ptr<weight_type> nonzeros) {
            csr.nonzero_values = nonzeros;
        }

        __host__ __device__ __forceinline__
        void set_num_rows(const vertex_type num_rows) {
            csr.num_rows = num_rows;
        }

        __host__ __device__ __forceinline__
        void set_num_columns(const vertex_type num_columns) {
            csr.num_columns = num_columns;
        }

        __host__ __device__ __forceinline__
        void set_num_nonzeros(const edge_type num_nonzeros) {
            csr.num_nonzeros = num_nonzeros;
        }

        __host__ __device__ __forceinline__
        void set(const vertex_type num_rows,
                 const vertex_type num_columns,
                 const edge_type num_nonzeros,
                 std::shared_ptr<vertex_type> offsets, 
                 std::shared_ptr<edge_type> indices, 
                 std::shared_ptr<weight_type> nonzeros) {
            
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

template <typename vertex_t, typename edge_t, typename weight_t> 
class graph_csc_t : public virtual graph_base_t<vertex_t, edge_t, weight_t> {

    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using vertex_pair_type = vertex_pair_t<vertex_t>;
    using properties_type = graph_properties_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using csc_type        = csc_t<edge_type, vertex_type, weight_type>;
    
    public:
        graph_csc_t() : graph_base_type() {}

        graph_csc_t(edge_type* offsets, 
                    vertex_type* indices, 
                    weight_type* weights, 
                    vertex_type number_of_vertices, 
                    edge_type number_of_edges) : 
            graph_base_type(
                number_of_vertices, 
                number_of_edges),
            csc(
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

        }

        // __host__ __device__ __forceinline__
        // vertex_type get_source_vertex(const edge_type& e) const override {

        // }
        
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

    private:
        csc_type csc;
};  // struct graph_csc_t

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

        // }

        // Representation specific functions
        // ...

    private:
        coo_type coo;
};  // struct graph_coo_t

// Empty type for conditional
// struct empty_t{};

// // Boolean based conditional inheritence
// // Relies on empty_t{};
// template <bool HAS_COO, bool HAS_CSR, bool HAS_CSC,
//           typename vertex_t, typename edge_t, typename weight_t> 
// class graph_t : 
//     public std::conditional_t<HAS_CSR, graph_csr_t<vertex_t, edge_t, weight_t>, empty_t>,
//     public graph_csc_t<vertex_t, edge_t, weight_t>,
//     public graph_coo_t<vertex_t, edge_t, weight_t> {

//     using vertex_type = vertex_t;
//     using edge_type   = edge_t;
//     using weight_type = weight_t;

//     using vertex_pair_type = vertex_pair_t<vertex_t>;

//     // using g_csr_t     = typename std::conditional<HAS_CSR, 
//     //                     graph_csr_t<vertex_type, edge_type, weight_type>, 
//     //                     std::nullptr_t>;
//     // using g_csc_t     = typename std::conditional<HAS_CSC, 
//     //                     graph_csc_t<vertex_type, edge_type, weight_type>, 
//     //                     std::nullptr_t>;
//     // using g_coo_t     = typename std::conditional<HAS_COO, 
//     //                     graph_coo_t<vertex_type, edge_type, weight_type>, 
//     //                     std::nullptr_t>;

// };  // struct graph_t

// Variadic inheritence, inherit only what you need
template<typename vertex_t, typename edge_t, typename weight_t, class... graph_view_t> 
class graph_t : public graph_view_t... {

    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using graph_base_type = graph_base_t<vertex_type, edge_type, weight_type>;
    using graph_csr_type = graph_csr_t<vertex_type, edge_type, weight_type>;
    using first_view_t = typename std::tuple_element<0, std::tuple<graph_view_t...> >::type;

    public:
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(const vertex_type& v) const override {
            return first_view_t::get_neighbor_list_length(v);
        }

        template<typename csr_matrix_t>
        void from_csr_t(const csr_matrix_t& _csr) {
            graph_base_type::_number_of_vertices = _csr.num_rows;
            graph_base_type::_number_of_edges = _csr.num_nonzeros;

            graph_csr_type::set(_csr.num_rows,
                                _csr.num_columns,
                                _csr.num_nonzeros,
                                _csr.row_offsets, 
                                _csr.column_indices, 
                                _csr.nonzero_values);
        }

        __host__ __device__ __forceinline__
        std::size_t number_of_graph_representations() const {
            return number_of_formats_inherited;
        }

        template<typename view_t>
        constexpr bool contains_representation() {
            return std::disjunction_v<std::is_same<view_t, graph_view_t>...>;
        }

    private:
        static constexpr std::size_t number_of_formats_inherited = sizeof...(graph_view_t);

};  // struct graph_t


} // namespace graph
} // namespace gunrock