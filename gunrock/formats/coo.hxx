#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>

namespace gunrock {
namespace format {

/**
 * @brief Coordinate (COO) format.
 * 
 * @tparam index_t 
 * @tparam nz_size_t
 * @tparam value_t
 */
template <typename index_t,
          typename nz_size_t,
          typename value_t,
          memory_space_t space = memory_space_t::host>
struct coo_t {
    index_t num_rows;
    index_t num_columns;
    nz_size_t num_nonzeros;

    typename gunrock::vector<index_t, space>::type I; // row indices
    typename gunrock::vector<index_t, space>::type J; // column indices
    typename gunrock::vector<value_t, space>::type V; // nonzero values

}; // struct coo_t

} // namespace format
} // namespace gunrock