// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * utils.cuh
 *
 * @brief General graph-building utility routines
 */

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/random_bits.cuh>

#include <gunrock/coo.cuh>
#include <gunrock/csr.cuh>

namespace gunrock {
namespace graphio {


/**
 * Returns a random node-ID in the range of [0, num_nodes) 
 */
template<typename SizeT>
SizeT RandomNode(SizeT num_nodes) {
    SizeT node_id;
    util::RandomBits(node_id);
    if (node_id < 0) node_id *= -1;
    return node_id % num_nodes;
}


} // namespace graphio
} // namespace gunrock
