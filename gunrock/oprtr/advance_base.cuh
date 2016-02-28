// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------


/**
 * @file
 * advance_base.cuh
 *
 * @brief common routines for advance kernels
 */

#pragma once

#include <gunrock/oprtr/advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {

/*
 * @brief Dispatch data structure.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE,
    gunrock::oprtr::advance::REDUCE_OP   R_OP>
static __device__ __forceinline__ void ProcessNeighbor(
    typename Problem::VertexId   v,
    typename Problem::VertexId   u,
    typename Problem::DataSlice *d_data_slice,
    typename Problem::SizeT      edge_id,
    typename Problem::SizeT      input_pos,
    typename Problem::VertexId   input_item,
    typename Problem::SizeT      output_pos,
    typename Functor::LabelT     label,
    typename Problem::VertexId  *d_out,
    typename Problem::Value     *d_value_to_reduce,
    typename Problem::Value     *d_reduce_frontier)
{
    if (Functor::CondEdge(
        v, u, d_data_slice, edge_id, input_item,
        label, input_pos, output_pos))
    {
        Functor::ApplyEdge(
            v, u, d_data_slice, edge_id, input_item,
            label, input_pos, output_pos);
        if (d_out != NULL)
        {
            if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V)
            {
                util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                        u,
                        d_out + output_pos);
            } else if (ADVANCE_TYPE == gunrock::oprtr::advance::V2E
                     ||ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
            {
                util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                        (typename Problem::VertexId)edge_id, // TODO: potential overflow if edge_id is large
                        d_out + output_pos);
            }
        }
        //printf("%d,%dCe\t", threadIdx.x, i);
        if (d_value_to_reduce != NULL)
        {
            if (R_TYPE == gunrock::oprtr::advance::VERTEX)
            {
                util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                        d_value_to_reduce[u],
                        d_reduce_frontier + output_pos);
            } else if (R_TYPE == gunrock::oprtr::advance::EDGE)
            {
                util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                        d_value_to_reduce[edge_id],
                        d_reduce_frontier + output_pos);
            }
        } else if (R_TYPE != gunrock::oprtr::advance::EMPTY)
        {
            // use user-specified function to generate value to reduce
        }
        //printf("%d,%dCf\t", threadIdx.x, i);

    } // end of if cond_success
    else {
        //printf("%d,%dCg\t", threadIdx.x, i);
        if (d_out != NULL)
        {
            util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                    util::InvalidValue<typename Problem::VertexId>(),
                    d_out + output_pos);
        }
        //printf("%d,%dCh\t", threadIdx.x, i);
        if (d_value_to_reduce != NULL)
        {
            util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                gunrock::oprtr::advance::Identity<typename Problem::Value, R_OP>()(),
                d_reduce_frontier + output_pos);
        }
       //printf("%d,%dCi\t", threadIdx.x, i);
    } // end of else cond_success
}

template <typename VertexId, typename SizeT>
__device__ __forceinline__ void PrepareQueue(
    bool   queue_reset,
    VertexId queue_index,
    SizeT &input_queue_length,
    SizeT *output_queue_length,
    util::CtaWorkProgress<SizeT> &work_progress)
{
    // Determine work decomposition
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // obtain problem size
        if (queue_reset)
        {
            work_progress.StoreQueueLength(input_queue_length, queue_index);
        }
        else
        {
            input_queue_length = work_progress.LoadQueueLength(queue_index);
        }

        work_progress.Enqueue(output_queue_length[0], queue_index + 1);

        // Reset our next outgoing queue counter to zero
        work_progress.StoreQueueLength(0, queue_index + 2);
        //work_progress.PrepResetSteal(queue_index + 1);
    }

    // Barrier to protect work decomposition
    __syncthreads();
}

} // oprtr
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
