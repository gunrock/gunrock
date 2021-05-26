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
  index_t number_of_rows;
  index_t number_of_columns;
  index_t number_of_nonzeros;

  vector_t<offset_t, space> column_offsets;  // Aj
  vector_t<index_t, space> row_indices;      // Ap
  vector_t<value_t, space> nonzero_values;   // Ax

  csc_t()
      : number_of_rows(0),
        number_of_columns(0),
        number_of_nonzeros(0),
        column_offsets(),
        row_indices(),
        nonzero_values() {}

  csc_t(index_t r, index_t c, index_t nnz)
      : number_of_rows(r),
        number_of_columns(c),
        number_of_nonzeros(nnz),
        column_offsets(c + 1),
        row_indices(nnz),
        nonzero_values(nnz) {}

  ~csc_t() {}

};  // struct csc_t

}  // namespace format
}  // namespace gunrock