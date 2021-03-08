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
  index_t number_of_rows;
  index_t number_of_columns;
  nz_size_t number_of_nonzeros;

  vector_t<index_t, space> row_indices;     // I
  vector_t<index_t, space> column_indices;  // J
  vector_t<value_t, space> nonzero_values;  // V

  coo_t()
      : number_of_rows(0),
        number_of_columns(0),
        number_of_nonzeros(0),
        row_indices(),
        column_indices(),
        nonzero_values() {}

  coo_t(index_t r, index_t c, nz_size_t nnz)
      : number_of_rows(r),
        number_of_columns(c),
        number_of_nonzeros(nnz),
        row_indices(nnz),
        column_indices(nnz),
        nonzero_values(nnz) {}

  ~coo_t() {}

};  // struct coo_t

}  // namespace format
}  // namespace gunrock