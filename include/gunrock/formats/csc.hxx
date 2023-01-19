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

  vector_t<offset_t, space> column_offsets;
  vector_t<index_t, space> row_indices;
  vector_t<value_t, space> nonzero_values;

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

  /**
   * @brief Convert CSR format into CSC
   * Format.
   *
   * @tparam index_t
   * @tparam index_t
   * @tparam value_t
   * @param csr
   * @return coo_t<space, index_t, index_t, value_t>&
   */
  // TODO: fix index_t -> offset_t
  csc_t<space, index_t, index_t, value_t> from_csr(
      const csr_t<memory_space_t::host, index_t, index_t, value_t>& csr) {
    number_of_rows = csr.number_of_rows;
    number_of_columns = csr.number_of_columns;
    number_of_nonzeros = csr.number_of_nonzeros;

    // Allocate space for vectors
    vector_t<index_t, memory_space_t::host> Ai;  // column offsets
    vector_t<index_t, memory_space_t::host> Aj;  // row indices
    vector_t<value_t, memory_space_t::host> Ax;  // values
    vector_t<index_t, memory_space_t::host> temp;

    Ai.resize(csr.number_of_columns + 1);
    Aj.resize(number_of_nonzeros);
    Ax.resize(number_of_nonzeros);
    temp.resize(number_of_nonzeros);

    Ax = csr.nonzero_values;
    temp = csr.column_indices;

    // CSC - column offsets and row indices
    gunrock::graph::convert::offsets_to_indices<memory_space_t::host>(
        csr.row_offsets.data(), csr.number_of_rows + 1, Aj.data(),
        number_of_nonzeros);

    using execution_policy_t =
        std::conditional_t<space == memory_space_t::device,
                           decltype(thrust::device), decltype(thrust::host)>;
    execution_policy_t exec;
    thrust::sort_by_key(
        exec, temp.data(),
        temp.data() + csr.number_of_nonzeros,
        thrust::make_zip_iterator(
            thrust::make_tuple(Aj.data(),
                               Ax.data()))  // values
    );

    gunrock::graph::convert::indices_to_offsets<space>(
        temp.data(), csr.number_of_nonzeros,
        Ai.data(), csr.number_of_rows + 1);

    column_offsets = Ai;
    row_indices = Aj;
    nonzero_values = Ax;

    return *this;  // CSC representation
  }

};  // struct csc_t

}  // namespace format
}  // namespace gunrock