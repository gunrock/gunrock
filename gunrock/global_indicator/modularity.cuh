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

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/problem_base.cuh>

#include <cub/cub.cuh>

namespace gunrock {
namespace global_indicator {
namespace modularity {

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct ModularityFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //should assign modularity_scores to 0 if not pass.
        return (&problem->d_community_ids[d_id] == &problem->d_community_ids[s_id]);
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        &problem->d_modularity_scores[e_id] = (&problem->d_edge_num << 1 - &problem->d_degrees[s_id]*&problem->d_degrees[d_id]);
    }

};

//An internal function to compute unweighted graph for CommunityDetection
//For each edge in the graph, use filter to find edges whose two end nodes belong
//to the same cluster, then compute the modularity accoring to the following equation:
//Q=sum_of_same_cluster_edges(A_ij - k_i*k_j/2m)/2m
//m:#edges, k_i:out degree of i, A_ij: 1/0
template<class ModularityProblem, class VertexId, class SizeT, class Value=float>
float GetModularity(
            SizeT       *d_row_offsets,
            VertexId    *d_column_indices,
            VertexId    *d_input_frontier,
            unsigned int *d_scanned_edges,
            gunrock::app::EnactorBase &enactor,
            ProblemData *problem,   // data_slice that contains community_ids, modularity_scores, d_degrees, and d_edge_num
            CudaContext &context)
            {
                typedef gunrock::oprtr::advance::KernelPolicy<
                    ModularityProblem,                  // Problem data type
                    300,                                // CUDA_ARCH
                    false,                              // INSTRUMENT
                    1,                                  // MIN_CTA_OCCUPANCY
                    10,                                 // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                                 // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;
                
                typename ModularityProblem::DataSlice *data_slice = problem->d_data_slices[0];
                
                typedef ModularityFunctor<
                    VertexId,
                    SizeT,
                    float,
                    ModularityProblem> MFunctor;

                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, ModularityProblem, MFunctor>(
                    NULL,
                    enactor.enactor_stats,
                    enactor.frontier_attribute,
                    data_slice,                     // contains community_ids, modularity_scores, degrees, edge_num
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    d_input_frontier,               // d_in_queue
                    NULL,                           // d_out_queue
                    (VertexId*)NULL,                // d_pred_in_queue
                    (VertexId*)NULL,                // d_pred_out_queue
                    d_row_offsets,
                    d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    problem->edges,
                    problem->edges,
                    enactor.work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);


                //do a global reduction to get the final modularity_score
                void    *d_temp_storage = NULL;
                size_t  temp_storage_bytes = 0;
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data_slice->modularity_scores, data_slice->modularity_scores, problem->edges);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data_slice->modularity_scores, data_slice->modularity_scores, problem->edges);
                // now the accumulated modularity_scores is stored in data_slice->modularity_scores[problem->edges-1]
                // should divide it by 4m^2
                // need to either keep it in device array or have a volatile var to get it afterwards.
            }

} // namespace modularity
} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
