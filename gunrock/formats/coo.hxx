#pragma once

#include <gunrock/container/vector.cuh>

namespace gunrock {
namespace format {

/**
 * @brief Coordinate (COO) format.
 * 
 * @tparam index_t 
 * @tparam value_t
 */
template <typename index_t,
          typename value_t>
struct coo_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    thrust::host_vector<index_t> I; // row indices
    thrust::host_vector<index_t> J; // column indices
    thrust::host_vector<value_t> V; // nonzero values

    coo_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0) { }

    coo_t(index_t r, index_t c, index_t nnz) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        I(nnz),
        J(nnz),
        V(nnz) { }

}; // struct coo_t

} // namespace format
} // namespace gunrock