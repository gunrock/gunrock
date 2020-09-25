#pragma once

namespace gunrock {
namespace format {

// Forward decleration
template <typename index_t,
          typename nz_size_t,
          typename value_t>
struct coo_t;

template <typename offset_t,
          typename index_t,
          typename value_t>
struct csr_t;

template <typename offset_t,
          typename index_t,
          typename value_t>
struct csc_t;

} // namespace format
} // namespace gunrock

#include <gunrock/formats/coo.hxx>
#include <gunrock/formats/csr.hxx>
#include <gunrock/formats/csc.hxx>