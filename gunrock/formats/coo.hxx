#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>

namespace gunrock {
namespace format {

using namespace memory;

/**
 * @brief Coordinate (COO) format.
 *
 * @tparam index_t
 * @tparam nz_size_t
 * @tparam value_t
 */
template <memory_space_t space,
          typename index_t,
          typename nz_size_t,
          typename value_t>
struct coo_t {
  index_t num_rows;
  index_t num_columns;
  nz_size_t num_nonzeros;

  typename vector<index_t, space>::type row_indices;     // I
  typename vector<index_t, space>::type column_indices;  // J
  typename vector<value_t, space>::type nonzero_values;  // V

  coo_t()
      : num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        row_indices(),
        column_indices(),
        nonzero_values() {}

  coo_t(index_t r, index_t c, nz_size_t nnz)
      : num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        row_indices(nnz),
        column_indices(nnz),
        nonzero_values(nnz) {}

  ~coo_t() {}

};  // struct coo_t

}  // namespace format
}  // namespace gunrock