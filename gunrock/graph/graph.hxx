#pragma once

#include <cassert>
#include <tuple>

#include <gunrock/util/type_traits.hxx>

#include <gunrock/formats/formats.hxx>

#include <gunrock/graph/properties.hxx>
#include <gunrock/graph/vertex_pair.hxx>

#include <gunrock/graph/detail/graph_base.hxx>
#include <gunrock/graph/csr.hxx>
#include <gunrock/graph/csc.hxx>
#include <gunrock/graph/coo.hxx>


// #include <gunrock/algorithms/search/binary_search.cuh>

namespace gunrock {
namespace graph {

using namespace format;
using namespace detail;

// // Empty type for conditional
// struct empty_t{};

// // Boolean based conditional inheritence
// // Relies on empty_t{};
// template <bool HAS_COO, bool HAS_CSR, bool HAS_CSC,
//           typename vertex_t, typename edge_t, typename weight_t> 
// class graph_t : 
//     public std::conditional_t<HAS_CSR, graph_csr_t<vertex_t, edge_t, weight_t>, empty_t>,
//     public graph_csc_t<vertex_t, edge_t, weight_t>,
//     public graph_coo_t<vertex_t, edge_t, weight_t> {

// };  // struct graph_t

// Variadic inheritence, inherit only what you need
template<typename vertex_t, typename edge_t, typename weight_t, class... graph_view_t> 
class graph_t : public graph_view_t... {

    using vertex_type = vertex_t;
    using edge_type   = edge_t;
    using weight_type = weight_t;

    using graph_base_type   = graph_base_t<vertex_type, edge_type, weight_type>;
    using graph_csr_type    = graph_csr_t<vertex_type, edge_type, weight_type>;
    using first_view_t      = typename std::tuple_element<0, // get first type
                                std::tuple<graph_view_t...> >::type;

    public:
        __host__ __device__ __forceinline__
        edge_type get_neighbor_list_length(const vertex_type& v) const override {
            return first_view_t::get_neighbor_list_length(v);
        }

        template<typename csr_matrix_t>
        void from_csr_t(csr_matrix_t& _csr) {
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