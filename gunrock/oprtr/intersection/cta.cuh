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
 * @brief CTA abstraction for Intersection
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


namespace gunrock {
namespace oprtr {
namespace intersection {

template<typename Value, typename VertexId, typename SizeT, bool RangeCheck, typename Comp>
__device__ int SerialSetIntersection(const Value* aData,
                                     const Value* bData,
                                     VertexId aBegin,
                                     VertexId aEnd,
                                     VertexId bBegin,
                                     VertexId bEnd,
                                     VertexId end,
                                     SizeT vt,
                                     Comp comp) {
                                     const int MinIterations = vt/2;
                                     int result = 0;

                                     #pragma unroll
                                     for (int i = 0; i < vt; ++i) {
                                       bool test = RangeCheck ?
                                       ((aBegin + bBegin < end) && (aBegin < aEnd) && (bBegin < bEnd)) :
                                       (i < MinIterations || (aBegin + bBegin < end));

                                       if (test) {
                                        Value aKey = aData[aBegin];
                                        Value bKey = bData[bBegin];

                                        bool pA = comp(aKey, bKey);
                                        bool pB = comp(bKey, aKey);

                                        if (!pA) ++aBegin;
                                        if (!pB) ++bBegin;
                                        if (pA == pB) ++result;
                                       }
                                     }
                                     return result;
                                    }

} //namespace intersection
} //namespace oprtr
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
