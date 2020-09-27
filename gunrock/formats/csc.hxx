#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

/**
 * @brief Compressed Sparse Column (CSC) format.
 * 
 * @tparam index_t 
 * @tparam offset_t
 * @tparam value_t
 */
template <typename index_t,
          typename offset_t,
          typename value_t>
struct csc_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    memory::memory_space_t location;

    std::shared_ptr<offset_t> column_offsets; // Aj
    std::shared_ptr<index_t> row_indices; // Ap
    std::shared_ptr<value_t> nonzero_values; // Ax

    // Default constructor
    csc_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        column_offsets(std::make_shared<offset_t>()),
        row_indices(std::make_shared<index_t>()),
        nonzero_values(std::make_shared<value_t>()) {}

    // Build using r, c, nnz
    csc_t(index_t r, index_t c, index_t nnz,
          memory::memory_space_t loc = memory::memory_space_t::device) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        column_offsets(
            memory::allocate<offset_t>((c+1) * sizeof(offset_t), location), 
            [&](offset_t* ptr) { memory::free(ptr, location); }),
        row_indices(
            memory::allocate<index_t>((nnz) * sizeof(index_t), location), 
            [&](index_t* ptr) { memory::free(ptr, location); }),
        nonzero_values(
            memory::allocate<value_t>((nnz) * sizeof(value_t), location), 
            [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the known (cudaMallocHost, cudaMalloc ...) memory space
    csc_t(index_t r, index_t c, index_t nnz,
          offset_t* Aj, index_t* Ap, value_t* Ax,
          memory::memory_space_t loc) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        column_offsets(Aj, [&](offset_t* ptr) { memory::free(ptr, location); }),
        row_indices(Ap, [&](index_t* ptr) { memory::free(ptr, location); }),
        nonzero_values(Ax, [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the unknown memory space
    csc_t(index_t r, index_t c, index_t nnz,
          offset_t* Aj, index_t* Ap, value_t* Ax) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(memory::memory_space_t::unknown),
        column_offsets(Aj),
        row_indices(Ap),
        nonzero_values(Ax) {}

}; // struct csc_t

} // namespace format
} // namespace gunrock