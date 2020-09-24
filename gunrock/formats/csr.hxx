#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

/**
 * @brief Compressed Sparse Row (CSR) format.
 * 
 * @tparam offset_t
 * @tparam index_t
 * @tparam value_t
 */
template <typename offset_t,
          typename index_t,
          typename value_t>
struct csr_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    memory::memory_space_t location;

    std::shared_ptr<offset_t> row_offsets; // Ap
    std::shared_ptr<index_t> column_indices; // Aj
    std::shared_ptr<value_t> nonzero_values; // Ax

    // Default constructor
    csr_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        row_offsets(std::make_shared<offset_t>()),
        column_indices(std::make_shared<index_t>()),
        nonzero_values(std::make_shared<value_t>()) {}

    // Build using r, c, nnz
    csr_t(index_t r, index_t c, index_t nnz,
          memory::memory_space_t loc = memory::memory_space_t::device) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        row_offsets(
            memory::allocate<offset_t>((r+1) * sizeof(offset_t), location), 
            [&](offset_t* ptr) { memory::free(ptr, location); }),
        column_indices(
            memory::allocate<index_t>((nnz) * sizeof(index_t), location), 
            [&](index_t* ptr) { memory::free(ptr, location); }),
        nonzero_values(
            memory::allocate<value_t>((nnz) * sizeof(value_t), location), 
            [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the known memory space
    csr_t(index_t r, index_t c, index_t nnz,
          offset_t* Ap, index_t* Aj, value_t* Ax,
          memory::memory_space_t loc) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        row_offsets(Ap, [&](offset_t* ptr) { memory::free(ptr, location); }),
        column_indices(Aj, [&](index_t* ptr) { memory::free(ptr, location); }),
        nonzero_values(Ax, [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the unknown memory space
    csr_t(index_t r, index_t c, index_t nnz,
          offset_t* Ap, index_t* Aj, value_t* Ax) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(memory::memory_space_t::unknown),
        row_offsets(Ap),
        column_indices(Aj),
        nonzero_values(Ax) {}

}; // struct csr_t

} // namespace format
} // namespace gunrock