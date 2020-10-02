#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>

namespace gunrock {
namespace format {

using namespace memory;

/**
 * @brief Compressed Sparse Row (CSR) format.
 * 
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::host>
struct csr_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    typename gunrock::vector<offset_t, space>::type row_offsets;   // Ap
    typename gunrock::vector<index_t, space>::type column_indices; // Aj
    typename gunrock::vector<value_t, space>::type nonzero_values; // Ax

    csr_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0) { }

    csr_t(index_t r, index_t c, index_t nnz) : 
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        row_offsets(r+1),
        column_indices(nnz),
        nonzero_values(nnz) { }

    ~csr_t() { std::cout << "object destructed" << std::endl; }

}; // struct csr_t

} // namespace format
} // namespace gunrock