#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>
#include <gunrock/formats/formats.hxx>

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
  index_t number_of_rows;
  index_t number_of_columns;
  offset_t number_of_nonzeros;

  typename vector<offset_t, space>::type row_offsets;    // Ap
  typename vector<index_t, space>::type column_indices;  // Aj
  typename vector<value_t, space>::type nonzero_values;  // Ax

  csr_t()
      : number_of_rows(0),
        number_of_columns(0),
        number_of_nonzeros(0),
        row_offsets(),
        column_indices(),
        nonzero_values() {}

  csr_t(index_t r, index_t c, offset_t nnz)
      : number_of_rows(r),
        number_of_columns(c),
        number_of_nonzeros(nnz),
        row_offsets(r + 1),
        column_indices(nnz),
        nonzero_values(nnz) {}

  ~csr_t() {}

  /**
   * @brief Convert a Coordinate Sparse Format into Compressed Sparse Row
   * Format.
   *
   * @tparam index_t
   * @tparam offset_t
   * @tparam value_t
   * @param coo
   * @return csr_t<memory_space_t::host, index_t, offset_t, value_t>&
   */
  csr_t<memory_space_t::host, index_t, offset_t, value_t>& operator=(
      const coo_t<memory_space_t::host, index_t, offset_t, value_t>& coo) {
    number_of_rows = coo.number_of_rows;
    number_of_columns = coo.number_of_columns;
    number_of_nonzeros = coo.number_of_nonzeros;

    // Allocate space for vectors
    row_offsets.resize(number_of_rows + 1);
    column_indices.resize(number_of_columns);
    nonzero_values.resize(number_of_nonzeros);

    // compute number of non-zero entries per row of A
    for (offset_t n = 0; n < number_of_nonzeros; n++) {
      row_offsets[coo.row_indices[n]]++;
    }

    // cumulative sum the nnz per row to get row_offsets[]
    for (index_t i = 0, sum = 0; i < number_of_rows; i++) {
      index_t temp = row_offsets[i];
      row_offsets[i] = sum;
      sum += temp;
    }
    row_offsets[number_of_rows] = number_of_nonzeros;

    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values
    for (offset_t n = 0; n < number_of_nonzeros; n++) {
      index_t row = coo.row_indices[n];
      index_t dest = row_offsets[row];

      column_indices[dest] = coo.column_indices[n];
      nonzero_values[dest] = coo.nonzero_values[n];

      row_offsets[row]++;
    }

    for (index_t i = 0, last = 0; i <= number_of_rows; i++) {
      index_t temp = row_offsets[i];
      row_offsets[i] = last;
      last = temp;
    }

    return *this;  // CSR representation (with possible duplicates)
  }

};  // struct csr_t

}  // namespace format
}  // namespace gunrock