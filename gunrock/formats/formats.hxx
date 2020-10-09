/**
 * @file formats.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/memory.hxx>

namespace gunrock {
namespace format {

using namespace memory;

// Forward decleration
template <memory_space_t space,
          typename index_t,
          typename nz_size_t,
          typename value_t>
struct coo_t;

template <memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
struct csr_t;

template <memory_space_t space,
          typename index_t,
          typename offset_t,
          typename value_t>
struct csc_t;

}  // namespace format
}  // namespace gunrock

#include <gunrock/formats/coo.hxx>
#include <gunrock/formats/csc.hxx>
#include <gunrock/formats/csr.hxx>