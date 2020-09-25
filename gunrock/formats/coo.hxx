#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

/**
 * @brief Coordinate (COO) format.
 * 
 * @tparam index_t 
 * @tparam nz_size_t
 * @tparam value_t
 */
template <typename index_t,
          typename nz_size_t,
          typename value_t>
struct coo_t {
    index_t num_rows;
    index_t num_columns;
    nz_size_t num_nonzeros;

    memory::memory_space_t location;

    std::shared_ptr<index_t> I; // row indices
    std::shared_ptr<index_t> J; // column indices
    std::shared_ptr<value_t> V; // nonzero values

    // Default constructor
    coo_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0),
        I(std::make_shared<index_t>()),
        J(std::make_shared<index_t>()),
        V(std::make_shared<value_t>()) {}

    // Build using r, c, nnz
    coo_t(index_t r, index_t c, nz_size_t nnz,
          memory::memory_space_t loc = memory::memory_space_t::device) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        I(memory::allocate<index_t>((nnz) * sizeof(index_t), location), 
          [&](index_t* ptr) { memory::free(ptr, location); }),
        J(memory::allocate<index_t>((nnz) * sizeof(index_t), location), 
          [&](index_t* ptr) { memory::free(ptr, location); }),
        V(memory::allocate<value_t>((nnz) * sizeof(value_t), location), 
          [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the known (cudaMallocHost, cudaMalloc ...) memory space
    coo_t(index_t r, index_t c, nz_size_t nnz,
          index_t* _I, index_t* _J, value_t* _V,
          memory::memory_space_t loc) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(loc),
        I(_I, [&](index_t* ptr) { memory::free(ptr, location); }),
        J(_J, [&](index_t* ptr) { memory::free(ptr, location); }),
        V(_V, [&](value_t* ptr) { memory::free(ptr, location); }) {}

    // Wrapper over existing pointers
    // allocated in the unknown memory space
    coo_t(index_t r, index_t c, nz_size_t nnz,
          index_t* _I, index_t* _J, value_t* _V) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        location(memory::memory_space_t::unknown),
        I(_I),
        J(_J),
        V(_V) {}

}; // struct coo_t

} // namespace format
} // namespace gunrock