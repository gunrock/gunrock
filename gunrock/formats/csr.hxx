#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

/**
 * @brief Compressed Sparse Row (CSR) format.
 * 
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <typename index_t,
          typename offset_t,
          typename value_t>
struct csr_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    std::shared_ptr<offset_t> row_offsets; // Ap
    std::shared_ptr<index_t> column_indices; // Aj
    std::shared_ptr<value_t> nonzero_values; // Ax
}; // struct csr_t

} // namespace format
} // namespace gunrock