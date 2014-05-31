// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Priority Queue Kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/priority_queue/near_far_pile.cuh>
#include <gunrock/priority_queue/kernel_policy.cuh>

#include <gunrock/util/test_utils.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace priority_queue {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    PriorityQueue,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::SizeT        SizeT;
    typedef typename ProblemData::DataSlice     DataSlice;
    typedef typename PriorityQueue::NearFarPile NearFarPile;

    static __device__ __forceinline__ void MarkVisit(
            VertexId     *&vertex_in,
            DataSlice       *&problem,
            SizeT &input_queue_length)
            {}

    static __device__ __forceinline__ void MarkNF(
            VertexId     *&vertex_in,
            NearFarPile     *&pq,
            DataSlice       *&problem,
            SizeT &input_queue_length,
            unsigned int &lower_priority_score_limit,
            unsigned int &upper_priority_score_limit)
    {
    }

    static __device__ __forceinline__ void Compact(
            VertexId        *&vertex_in,
            NearFarPile     *&pq,
            int             &selector,
            SizeT           &input_queue_length,
            VertexId        *&vertex_out,
            SizeT           &v_out_offset,
            SizeT           &far_pile_offset)
    {
    }
};

template<
    typename KernelPolicy,
    typename ProblemData,
    typename PriorityQueue,
    typename Functor>
struct Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor, true>
{
    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::SizeT        SizeT;
    typedef typename ProblemData::DataSlice     DataSlice;
    typedef typename PriorityQueue::NearFarPile NearFarPile;

    static __device__ __forceinline__ void MarkVisit(
                                            VertexId     *&vertex_in,
                                            DataSlice       *&problem,
                                            SizeT &input_queue_length)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int my_id = tid + bid*blockDim.x;

        if (my_id >= input_queue_length)
            return;

        unsigned int my_vert = vertex_in[my_id];
        problem->d_visit_lookup[my_vert] = my_id;
    }

    static __device__ __forceinline__ void MarkNF(
                                            VertexId     *&vertex_in,
                                            NearFarPile     *&pq,
                                            DataSlice       *&problem,
                                            SizeT &input_queue_length,
                                            unsigned int &lower_priority_score_limit,
                                            unsigned int &upper_priority_score_limit)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int my_id = tid + bid*blockDim.x;

        if (my_id >= input_queue_length)
            return;

        unsigned int bucket_max = UINT_MAX/problem->d_delta[0];
        unsigned int my_vert = vertex_in[my_id];
        unsigned int bucket_id = Functor::ComputePriorityScore(my_vert, problem);
        bool valid = (my_id == problem->d_visit_lookup[my_vert]);
        //printf(" valid:%d, my_id: %d, my_vert: %d\n", valid, my_id, my_vert, bucket_id);
        pq->d_valid_near[my_id] = (bucket_id < upper_priority_score_limit && bucket_id >= lower_priority_score_limit && valid) ? 1 : 0;
        pq->d_valid_far[my_id] = (bucket_id >= upper_priority_score_limit && bucket_id < bucket_max && valid) ? 1 : 0;
        //printf("valid near, far: %d, %d\n", pq->d_valid_near[my_id], pq->d_valid_far[my_id]);
    }

    static __device__ __forceinline__ void Compact(
                                            VertexId        *&vertex_in,
                                            NearFarPile     *&pq,
                                            int             &selector,
                                            SizeT           &input_queue_length,
                                            VertexId        *&vertex_out,
                                            SizeT           &v_out_offset,
                                            SizeT           &far_pile_offset)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int my_id = bid*blockDim.x + tid;
        if (my_id >= input_queue_length)
            return;

        unsigned int my_vert = vertex_in[my_id];
        unsigned int my_valid = pq->d_valid_near[my_id];

        if (my_valid == pq->d_valid_near[my_id+1]-1)
            vertex_out[my_valid+v_out_offset] = my_vert;

        my_valid = pq->d_valid_far[my_id];
        if (my_valid == pq->d_valid_far[my_id+1]-1)
            pq->d_queue[selector][my_valid+far_pile_offset] = my_vert;
        
    }
};

template<typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::BLOCKS)
    __global__
void MarkVisit(
        typename KernelPolicy::VertexId     *vertex_in,
        typename ProblemData::DataSlice     *problem,
        typename KernelPolicy::SizeT        input_queue_length)
{
    Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::MarkVisit(
            vertex_in,
            problem,
            input_queue_length);
}

template<typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::BLOCKS)
    __global__
void MarkNF(
        typename KernelPolicy::VertexId     *vertex_in,
        typename PriorityQueue::NearFarPile *pq,
        typename ProblemData::DataSlice     *problem,
        typename KernelPolicy::SizeT        input_queue_length,
        unsigned int                        lower_priority_score_limit,
        unsigned int                        upper_priority_score_limit)
{
    Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::MarkNF(
            vertex_in,
            pq,
            problem,
            input_queue_length,
            lower_priority_score_limit,
            upper_priority_score_limit);
}

template<typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::BLOCKS)
    __global__
void Compact(
        typename KernelPolicy::VertexId     *vertex_in,
        typename PriorityQueue::NearFarPile *pq,
        int                                 selector,
        typename KernelPolicy::SizeT        input_queue_length,
        typename KernelPolicy::VertexId     *vertex_out,
        typename KernelPolicy::SizeT        v_out_offset,
        typename KernelPolicy::SizeT        far_pile_offset)
{
    Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::Compact(
        vertex_in,
        pq,
        selector,
        input_queue_length,
        vertex_out,
        v_out_offset,
        far_pile_offset);
}

template <typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
    unsigned int Bisect(
        typename KernelPolicy::VertexId     *vertex_in,
        PriorityQueue                       *pq,
        typename KernelPolicy::SizeT        input_queue_length,
        typename ProblemData::DataSlice     *problem,
        typename KernelPolicy::VertexId     *vertex_out,
        typename KernelPolicy::SizeT        far_pile_offset,
        unsigned int                        lower_limit,
        unsigned int                        upper_limit,
        CudaContext                         &context)
{

    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::SizeT        SizeT;
    typename PriorityQueue::NearFarPile         *nf_pile = pq->d_nf_pile[0];

    int block_num = (input_queue_length + KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;
    unsigned int close_size[1];
    unsigned int far_size[1];
    close_size[0] = 0;
    far_size[0] = 0;
    if(input_queue_length > 0)
    {
        MarkVisit<KernelPolicy, ProblemData, PriorityQueue, Functor><<<block_num, KernelPolicy::THREADS>>>(vertex_in, problem, input_queue_length);
        // MarkNF
        MarkNF<KernelPolicy, ProblemData, PriorityQueue, Functor><<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, problem, input_queue_length, lower_limit, upper_limit);

        // Scan(near)
        // Scan(far)
        Scan<mgpu::MgpuScanTypeExc>(pq->nf_pile[0]->d_valid_near, input_queue_length+1, 0, mgpu::plus<VertexId>(), (VertexId*)0, (VertexId*)0, pq->nf_pile[0]->d_valid_near, context);
        Scan<mgpu::MgpuScanTypeExc>(pq->nf_pile[0]->d_valid_far, input_queue_length+1, 0, mgpu::plus<VertexId>(), (VertexId*)0, (VertexId*)0, pq->nf_pile[0]->d_valid_far, context);
        // Compact
        Compact<KernelPolicy, ProblemData, PriorityQueue, Functor><<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, pq->selector, input_queue_length, vertex_out, 0, far_pile_offset);
        // get output_near_length
        // get output_far_length
        cudaMemcpy(&close_size[0], pq->nf_pile[0]->d_valid_near+input_queue_length, sizeof(VertexId),
		    cudaMemcpyDeviceToHost);
		cudaMemcpy(&far_size[0], pq->nf_pile[0]->d_valid_far+input_queue_length, sizeof(VertexId),
		    cudaMemcpyDeviceToHost);

    }
    // Update near/far length
    pq->queue_length = far_pile_offset + far_size[0];
    return close_size[0];
}

} //priority_queue
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
