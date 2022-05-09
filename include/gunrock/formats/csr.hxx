#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/container/vector.hxx>
#include <gunrock/formats/formats.hxx>

#include <thrust/transform.h>

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
  using index_type = index_t;
  using offset_type = offset_t;
  using value_type = value_t;

  index_t number_of_rows;
  index_t number_of_columns;
  offset_t number_of_nonzeros;

  vector_t<offset_t, space> row_offsets;    // Ap
  vector_t<index_t, space> column_indices;  // Aj
  vector_t<value_t, space> nonzero_values;  // Ax

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
   * @brief Copy constructor.
   * @param rhs
   */
  template <typename _csr_t>
  csr_t(const _csr_t& rhs)
      : number_of_rows(rhs.number_of_rows),
        number_of_columns(rhs.number_of_columns),
        number_of_nonzeros(rhs.number_of_nonzeros),
        row_offsets(rhs.row_offsets),
        column_indices(rhs.column_indices),
        nonzero_values(rhs.nonzero_values) {}

  /**
   * @brief Convert a Coordinate Sparse Format into Compressed Sparse Row
   * Format.
   *
   * @tparam index_t
   * @tparam offset_t
   * @tparam value_t
   * @param coo
   * @return csr_t<space, index_t, offset_t, value_t>&
   */
  csr_t<space, index_t, offset_t, value_t> from_coo(
      const coo_t<memory_space_t::host, index_t, offset_t, value_t>& coo) {
    number_of_rows = coo.number_of_rows;
    number_of_columns = coo.number_of_columns;
    number_of_nonzeros = coo.number_of_nonzeros;

    // Allocate space for vectors
    vector_t<offset_t, memory_space_t::host> Ap;
    vector_t<index_t, memory_space_t::host> Aj;
    vector_t<value_t, memory_space_t::host> Ax;

    // offset_t* Ap;
    // index_t* Aj;
    // value_t* Ax;

    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);
      // If returning csr_t on device, allocate temporary host vectors, build on
      // host and move to device.
      Ap.resize(number_of_rows + 1);
      Aj.resize(number_of_nonzeros);
      Ax.resize(number_of_nonzeros);

      // Ap = _Ap.data();
      // Aj = _Aj.data();
      // Ax = _Ax.data();

    } else {
      assert(space == memory_space_t::host);
      // If returning csr_t on host, use it's internal memory to build from COO.
      row_offsets.resize(number_of_rows + 1);
      column_indices.resize(number_of_nonzeros);
      nonzero_values.resize(number_of_nonzeros);

      // Ap = raw_pointer_cast(row_offsets.data());
      // Aj = raw_pointer_cast(column_indices.data());
      // Ax = raw_pointer_cast(nonzero_values.data());
    }

    // compute number of non-zero entries per row of A.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      ++Ap[coo.row_indices[n]];
    }

    // cumulative sum the nnz per row to get row_offsets[].
    for (index_t i = 0, sum = 0; i < number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = sum;
      sum += temp;
    }
    Ap[number_of_rows] = number_of_nonzeros;

    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values.
    for (offset_t n = 0; n < number_of_nonzeros; ++n) {
      index_t row = coo.row_indices[n];
      index_t dest = Ap[row];

      Aj[dest] = coo.column_indices[n];
      Ax[dest] = coo.nonzero_values[n];

      ++Ap[row];
    }

    for (index_t i = 0, last = 0; i <= number_of_rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = last;
      last = temp;
    }

    // If returning a device csr_t, move coverted data to device.
    if (space == memory_space_t::device) {
      row_offsets = Ap;
      column_indices = Aj;
      nonzero_values = Ax;
    }

    return *this;  // CSR representation (with possible duplicates)
  }

  void read_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "rb");

    // Read metadata
    assert(fread(&number_of_rows, sizeof(index_t), 1, file) != 0);
    assert(fread(&number_of_columns, sizeof(index_t), 1, file) != 0);
    assert(fread(&number_of_nonzeros, sizeof(offset_t), 1, file) != 0);

    row_offsets.resize(number_of_rows + 1);
    column_indices.resize(number_of_nonzeros);
    nonzero_values.resize(number_of_nonzeros);

    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_row_offsets(number_of_rows + 1);
      thrust::host_vector<index_t> h_column_indices(number_of_nonzeros);
      thrust::host_vector<value_t> h_nonzero_values(number_of_nonzeros);

      assert(fread(memory::raw_pointer_cast(h_row_offsets.data()),
                   sizeof(offset_t), number_of_rows + 1, file) != 0);
      assert(fread(memory::raw_pointer_cast(h_column_indices.data()),
                   sizeof(index_t), number_of_nonzeros, file) != 0);
      assert(fread(memory::raw_pointer_cast(h_nonzero_values.data()),
                   sizeof(value_t), number_of_nonzeros, file) != 0);

      // Copy data from host to device
      row_offsets = h_row_offsets;
      column_indices = h_column_indices;
      nonzero_values = h_nonzero_values;

    } else {
      assert(space == memory_space_t::host);

      assert(fread(memory::raw_pointer_cast(row_offsets.data()),
                   sizeof(offset_t), number_of_rows + 1, file) != 0);
      assert(fread(memory::raw_pointer_cast(column_indices.data()),
                   sizeof(index_t), number_of_nonzeros, file) != 0);
      assert(fread(memory::raw_pointer_cast(nonzero_values.data()),
                   sizeof(value_t), number_of_nonzeros, file) != 0);
    }
  }

  void write_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "wb");

    // Write metadata
    fwrite(&number_of_rows, sizeof(index_t), 1, file);
    fwrite(&number_of_columns, sizeof(index_t), 1, file);
    fwrite(&number_of_nonzeros, sizeof(offset_t), 1, file);

    // Write data
    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_row_offsets(row_offsets);
      thrust::host_vector<index_t> h_column_indices(column_indices);
      thrust::host_vector<value_t> h_nonzero_values(nonzero_values);

      fwrite(memory::raw_pointer_cast(h_row_offsets.data()), sizeof(offset_t),
             number_of_rows + 1, file);
      fwrite(memory::raw_pointer_cast(h_column_indices.data()), sizeof(index_t),
             number_of_nonzeros, file);
      fwrite(memory::raw_pointer_cast(h_nonzero_values.data()), sizeof(value_t),
             number_of_nonzeros, file);
    } else {
      assert(space == memory_space_t::host);

      fwrite(memory::raw_pointer_cast(row_offsets.data()), sizeof(offset_t),
             number_of_rows + 1, file);
      fwrite(memory::raw_pointer_cast(column_indices.data()), sizeof(index_t),
             number_of_nonzeros, file);
      fwrite(memory::raw_pointer_cast(nonzero_values.data()), sizeof(value_t),
             number_of_nonzeros, file);
    }

    fclose(file);
  }

};  // struct csr_t

}  // namespace format
}  // namespace gunrock