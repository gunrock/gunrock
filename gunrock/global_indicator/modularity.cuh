// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * modularity.cuh
 *
 * @brief Modularity computation code
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/problem_base.cuh>

namespace gunrock {
namespace global_indicator {

//For now only compute unweighted graph.
//For each edge in the graph, use filter to find edges whose two end nodes belong
//to the same cluster, then compute the modularity accoring to the following equation:
//Q=sum_of_same_cluster_edges(A_ij - k_i*k_i/2m)/2m
//m:#edges, k_i:out degree of i, A_ij: 1/0
template<typename VertexId, typename SizeT>
float GetModularity(
            SizeT       *d_row_offsets,
            VertexId    *d_column_indices,
            SizeT       *d_degrees,
            VertexId    *d_input_frontier,
            SizeT       edge,
            float       *modularity_scores,
            VertexId    *community_ids)
            {

            }

} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
