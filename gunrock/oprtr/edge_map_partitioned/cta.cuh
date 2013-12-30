// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------


/**
 * @file
 * cta.cuh
 *
 * @brief CTA tile-processing abstraction for Load balanced Edge Map
 */

#pragma once
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/io/load_tile.cuh>
#include <gunrock/util/operators.cuh>
#include <gunrock/util/soa_tuple.cuh>

#include <gunrock/util/scan/soa/cooperative_soa_scan.cuh>


//TODO: use CUB for SOA scan

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned {

// From Andrew Davidson's dStepping SSSP GPU implementation
// binary search on device, only works for arrays shorter
// than 1024

template <int NT>
__device__ int BinarySearch(unsigned int i, unsigned int *queue)
{
    int mid = (NT/2 - 1);

    if (NT > 512)
        mid = queue[mid] > i ? mid - 256 : mid + 256;
    if (NT > 256)
        mid = queue[mid] > i ? mid - 128 : mid + 128;
    if (NT > 128)
        mid = queue[mid] > i ? mid - 64 : mid + 64;

    mid = queue[mid] > i ? mid - 32 : mid + 32;
    mid = queue[mid] > i ? mid - 16 : mid + 16;
    mid = queue[mid] > i ? mid - 8 : mid + 8;
    mid = queue[mid] > i ? mid - 4 : mid + 4;
    mid = queue[mid] > i ? mid - 2 : mid + 2;
    mid = queue[mid] > i ? mid - 1 : mid + 1;

    mid = queue[mid] <= i ? mid + 1 : mid;

    return mid;
}

} //namespace edge_map_partitioned
} //namespace oprtr
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
