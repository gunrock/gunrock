#pragma once

#include <gunrock/container/vector.cuh>

namespace gunrock {
namespace format {

/**
 * @brief Compressed Sparse Row (CSR) format.
 * 
 * @tparam offset_t
 * @tparam index_t
 * @tparam value_t
 */
template <typename offset_t,
          typename index_t,
          typename value_t>
struct csr_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    thrust::host_vector<offset_t> Ap;  // row offsets
    thrust::host_vector<index_t> Aj;   // column indices
    thrust::host_vector<value_t> Ax;   // nonzero values

    csr_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0) {}

    csr_t(index_t r, index_t c, index_t nnz) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        Ap(r+1),
        Aj(nnz),
        Ax(nnz) {}

}; // struct csr_t

} // namespace format
} // namespace gunrock