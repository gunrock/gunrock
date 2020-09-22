#pragma once

#include <gunrock/container/vector.cuh>

namespace gunrock {
namespace format {

/**
 * @brief Compressed Sparse Column (CSC) format.
 * 
 * @tparam index_t 
 * @tparam value_t
 */
template <typename offset_t,
          typename index_t,
          typename value_t>
struct csc_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    thrust::host_vector<offset_t> Aj;  // column offsets
    thrust::host_vector<index_t> Ap;   // row indices
    thrust::host_vector<value_t> Ax;   // nonzero values

    csc_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0) {}

    csc_t(index_t r, index_t c, index_t nnz) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        Aj(c+1),
        Ap(nnz),
        Ax(nnz) {}

}; // struct csc_t

} // namespace format
} // namespace gunrock