#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

using namespace memory;

// Forward decleration
template <typename index_t,
          typename nz_size_t,
          typename value_t,
          memory_space_t space>
struct coo_t;

template <typename offset_t,
          typename index_t,
          typename value_t,
          memory_space_t space>
struct csr_t;

template <typename offset_t,
          typename index_t,
          typename value_t,
          memory_space_t space>
struct csc_t;

} // namespace format
} // namespace gunrock

#include <gunrock/formats/coo.hxx>
#include <gunrock/formats/csr.hxx>
#include <gunrock/formats/csc.hxx>