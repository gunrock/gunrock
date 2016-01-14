// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * trianglecounting.cuh
 *
 * @brief Triangle Counting Computation code
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/problem_base.cuh>

#include <cub/cub.cuh>

namespace gunrock {
namespace global_indicator {
namespace tc {

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct TcFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //Add computation here.
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //Add computation here.
    }

};

// Function to compute triangle count using intersection operator and 
// advance/filter operator.

template<class TcProblem, class VertexId, class SizeT, class Value=size_t>
unsigned int GetTriangleCount(
            SizeT       *d_row_offsets,
            VertexId    *d_column_indices,
            VertexId    *d_input_frontier,
            unsigned int *d_scanned_edges,
            gunrock::app::EnactorBase &enactor,
            ProblemData *problem,   // data_slice that contains community_ids, modularity_scores, d_degrees, and d_edge_num
            CudaContext &context)
            //TODO: add params needed by intersection operator.
            {
                //
                // Algorithm details:
                // 1) Advance for all vertices. Only preserve neighbors whose degree is
                // smaller than the source node's degree.
                // 2) Filter unchosen edge lists. Form the final edge list.
                // 3) Perform intersection on edge list.
                //
            }

} // namespace tc
} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
