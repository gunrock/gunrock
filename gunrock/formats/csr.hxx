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
template <memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
struct csr_t {
  index_t num_rows;
  index_t num_columns;
  index_t num_nonzeros;

  typename vector<offset_t, space>::type row_offsets;    // Ap
  typename vector<index_t, space>::type column_indices;  // Aj
  typename vector<value_t, space>::type nonzero_values;  // Ax

  csr_t()
      : num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        row_offsets(),
        column_indices(),
        nonzero_values() {}

  csr_t(index_t r, index_t c, index_t nnz)
      : num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        row_offsets(r + 1),
        column_indices(nnz),
        nonzero_values(nnz) {}

  ~csr_t() {}

};  // struct csr_t

}  // namespace format
}  // namespace gunrock