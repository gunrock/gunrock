#pragma once

#include <gunrock/container/vector.hxx>
#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

using namespace memory;

/**
 * @brief Compressed Sparse Column (CSC) format.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
struct csc_t {
  index_t num_rows;
  index_t num_columns;
  index_t num_nonzeros;

  typename vector<offset_t, space>::type column_offsets;  // Aj
  typename vector<index_t, space>::type row_indices;      // Ap
  typename vector<value_t, space>::type nonzero_values;   // Ax

  csc_t()
      : num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        column_offsets(),
        row_indices(),
        nonzero_values() {}

  csc_t(index_t r, index_t c, index_t nnz)
      : num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        column_offsets(c + 1),
        row_indices(nnz),
        nonzero_values(nnz) {}

  ~csc_t() {}

};  // struct csc_t

}  // namespace format
}  // namespace gunrock