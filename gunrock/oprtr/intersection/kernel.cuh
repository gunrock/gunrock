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
 * @brief Intersection Kernel Entry point
 *
 * Expected inputs are two arrays of node IDs, each pair of nodes forms an edge.
 * The intersections of each node pair's neighbor lists are computed and returned
 * as a single usnigned int value. Can perform user-defined functors on each of
 * these intersection.
 */

// TODO: stream the intersection keys to output too.
//
// TODO: Par yesterday's discussion with Carl, should add one large list
// and one small list condition, since |n|log|N| would still be smaller
// than |n|+|N| if n is small enough and N is large enough.
//
// Notes: For per-block method, here is the rough schema:
// Input: two arrays with length: m stores node pairs.
//        row_offsets
//        column_indices
// Expected output: a single integer as triangle count
//

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/intersection/cta.cuh>

#include <gunrock/oprtr/intersection/kernel_policy.cuh>
#include <cub/cub.cuh>
#include <moderngpu.cuh>
#include <gunrock/util/test_utils.cuh>

namespace gunrock {
namespace oprtr {
namespace intersection {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef typename ProblemData::DataSlice DataSlice;

    // Get neighbor list sizes, scan to get both
    // fine_counts (for two per-thread methods) and
    // coarse_counts (for balanced-path per-block method)
    static __device__ void Inspect(
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            SizeT       *&d_flags,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       *&d_output_total,
                            SizeT       &input_length,
                            SizeT       &stride,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectTwoLargeNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       &input_length,
                            SizeT       &nv_per_block,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

};

/*
 * @brief Dispatch data structure.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true>
{
    typedef typename KernelPolicy::VertexId         VertexId;
    typedef typename KernelPolicy::SizeT            SizeT;
    typedef typename KernelPolicy::Value            Value;
    typedef typename ProblemData::DataSlice         DataSlice;

    // Get neighbor list sizes, scan to get both
    // fine_counts (for two per-thread methods) and
    // coarse_counts (for balanced-path per-block method)
    static __device__ __forceinline__ void Inspect(
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            SizeT       *&d_flags,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // Compute d_src_nl_sizes and d_dst_nl_sizes;
        VertexId idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_length) return;
        VertexId edge_id = d_edge_list[idx];
        VertexId src_node = d_src_node_ids[edge_id];
        VertexId dst_node = d_dst_node_ids[edge_id];
        // TODO: check -1 vertex id. for now let's just assume
        // that sane users won't input -1.
        SizeT src_nl_size = d_degrees[src_node];
        SizeT dst_nl_size = d_degrees[dst_node];
        d_flags[idx] = (src_nl_size > KernelPolicy::NL_SIZE_THRESHOLD
                     && dst_nl_size > KernelPolicy::NL_SIZE_THRESHOLD) ? 1 : 0;
    }

    static __device__ void IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       *&d_output_total,
                            SizeT       &input_length,
                            SizeT       &stride,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        VertexId start = threadIdx.x + blockIdx.x * blockDim.x;
        //VertexId end = (start + stride * KernelPolicy::THREADS > input_length)? input_length :
        //                (start + stride * KernelPolicy::THREADS);
        //typedef cub::BlockReduce<SizeT, KernelPolicy::THREADS> BlockReduceT;
        //__shared__ typename BlockReduceT::TempStorage temp_storage;

        for (VertexId idx = start; idx < input_length; idx += KernelPolicy::BLOCKS*KernelPolicy::THREADS) {
            SizeT count = 0;
            // get nls start and end index for two ids
            VertexId sid = __ldg(d_src_node_ids+idx);
            VertexId did = __ldg(d_dst_node_ids+idx);
            SizeT src_it = __ldg(d_row_offsets+sid);
            SizeT src_end = __ldg(d_row_offsets+sid+1);
            SizeT dst_it = __ldg(d_row_offsets+did);
            SizeT dst_end = __ldg(d_row_offsets+did+1);
            if (src_it == src_end || dst_it == dst_end) continue;
            SizeT src_nl_size = src_end - src_it;
            SizeT dst_nl_size = dst_end - dst_it;
            SizeT min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
            SizeT max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
            SizeT total = min_nl + max_nl;
            if ( min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl ) {
                // search
                SizeT min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
                SizeT min_end = min_it + min_nl;
                SizeT max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
                VertexId *keys = &d_column_indices[max_it];
                //printf("src:%d,dst:%d, src_it:%d, dst_it:%d, min_it:%d max_it:%d, min max nl size: %d, %d\n",sid, did, src_it, dst_it, min_it, max_it, min_nl, max_nl);
                while ( min_it < min_end) {
                    VertexId small_edge = d_column_indices[min_it++];
                    count += BinarySearch(keys, max_nl, small_edge);
                }
            } else {
                VertexId src_edge = __ldg(d_column_indices+src_it);
                VertexId dst_edge = __ldg(d_column_indices+dst_it);
                while (src_it < src_end && dst_it < dst_end) {
                    VertexId diff = src_edge - dst_edge;
                    src_edge = (diff <= 0) ? __ldg(d_column_indices+(++src_it)) : src_edge;
                    dst_edge = (diff >= 0) ? __ldg(d_column_indices+(++dst_it)) : dst_edge;
                    count += (diff == 0);
                }
            }
            d_output_total[idx] += total;
            d_output_counts[idx] += count;
        }
    }

    static __device__ void IntersectTwoLargeNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       &input_length,
                            SizeT       &nv_per_block,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        __shared__ typename KernelPolicy::SmemStorage smem_storage;
        // Each block uses balanced path to get intersect count
        // First, we divide the input node ids of length input_length into p
        // partitions, each block will compute the intersects of m/p node pairs.
        // Now let's first assume that p=blockDim.x
        
        // Starting and ending index for this block
        SizeT start = nv_per_block * blockIdx.x;
        SizeT end = (nv_per_block * (blockIdx.x+1) > input_length) ? input_length
        : nv_per_block*(blockIdx.x+1); 
        typedef cub::BlockReduce<SizeT, KernelPolicy::THREADS> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        
        // partition:
        // nv_per_partition = (acount + bcount)/THREADS
        // use FindSetPartitions, KernelSetPartitions (search.cuh)
        // and BalancedPath (ctasearch.cuh) to create a partition array
        // parition_device[THREADS] in smem
        //
        // intersect:
        // Each thread will check nv_per_partition values and accumulate tc number
        // if we only need tc number, we can modify
        // DeviceComputeSetAvailability
        //
        // block reduce is simple
        //
        if (threadIdx.x == 0) {
                d_output_counts[blockIdx.x] = 0;
                }

        //printf("blockId:%d, start:%d, end:%d\n", blockIdx.x, start, end);
        for (int i = start; i < end; ++i) {
            // For each of these pairs, a block will
            // perform the following phases:
            // 1) partition
            // 2) intersect per thread
            // 3) block reduce to get partial triangle counts
            //
            // partition:
            // get acount and bcount
            //printf("bid:%d, tid:%d, i:%d\n", blockIdx.x, threadIdx.x, i);
            VertexId edge_id = d_edge_list[i];
            VertexId src_node = d_src_node_ids[edge_id];
            VertexId dst_node = d_dst_node_ids[edge_id];
            SizeT a_rowoffset = d_row_offsets[src_node];
            SizeT b_rowoffset = d_row_offsets[dst_node];
            SizeT acount  = d_degrees[src_node];
            SizeT bcount  = d_degrees[dst_node];
            if (threadIdx.x == 0) printf("block:%d, eid:%d, sid:%d, did:%d, acount:%d, bcount:%d\n",
                    blockIdx.x, edge_id, src_node, dst_node, acount, bcount);
            VertexId *a_list = &d_column_indices[a_rowoffset];
            VertexId *b_list = &d_column_indices[b_rowoffset];
            int numPartitions = KernelPolicy::THREADS;
            int numSearches = numPartitions+1;
            int nv = max(1, (acount+bcount+numPartitions-1)/numPartitions);
            //Duplicate = T, comp = mgpu::less<VertexId>, numSearches=numPartitions+1
            int gid = threadIdx.x;
            int diag;
            if (gid < numSearches) {
                diag = min(acount+bcount, gid * nv);
                int2 bp = mgpu::BalancedPath<true, mgpu::int64>(a_list, acount,
                b_list, bcount, diag, 4, mgpu::less<VertexId>());
                //if (bp.y) bp.x += 1;
                smem_storage.s_partition_idx[gid] = bp.x;
            }
            __syncthreads();
            if (threadIdx.x == 0) printf("\n");

            // intersect per thread
            // for each thread, put a_list[a_rowoffset+diag] to a_list[a_rowoffset+diag+s_partition_idx[gid]]
            // and b_list[b_rowoffset+diag] to b_list[b_rowoffset+diag+nv-s_partition_idx[gid]] together.
            // Then do serial intersection
            VertexId aBegin = min(acount, smem_storage.s_partition_idx[gid]);
            VertexId bBegin = min(bcount, diag - smem_storage.s_partition_idx[gid]);
            VertexId aEnd = smem_storage.s_partition_idx[gid+1];
            VertexId bEnd = bBegin + nv - (smem_storage.s_partition_idx[gid+1] - smem_storage.s_partition_idx[gid]);
            VertexId end = acount+bcount;
            //if (gid*nv <= (acount+bcount)) printf("b:%d, diag:%d, divide:%d| acount:%d, bcount:%d", blockIdx.x, diag, smem_storage.s_partition_idx[gid], acount, bcount);
            SizeT result = SerialSetIntersection(a_list,
                                  b_list,
                                  aBegin,
                                  aEnd,
                                  bBegin,
                                  bEnd,
                                  nv,
                                  end,
                                  mgpu::less<VertexId>());

            // Block reduce to get total result.
            SizeT aggregate = BlockReduceT(temp_storage).Sum(result);

            if ( gid == 0)
                d_output_counts[blockIdx.x] += aggregate;
        }
    }

};


/**
 * @brief Kernel entry for Inspect function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier
 *                              queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier 
 *                              queue (destination node ids)
 * @param[in] d_edge_list       Device pointer of SizeT to the edge list index
 * @param[in] d_degrees         Device pointer of SizeT to the degree array
 * @param[out] d_flags          Device pointer of SizeT to the partition flag queue
 * @param[in] input_queue_len   Length of the incoming frontier queues(d_src_node_ids 
 *                              and d_dst_node_ids should have the same length)
 * @param[in] max_vertices      Maximum number of elements we can place into the
 *                              incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the 
 *                              outgoing frontier
 */
    template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void Inspect(
        typename KernelPolicy::VertexId         *d_src_node_ids,
        typename KernelPolicy::VertexId         *d_dst_node_ids,
        typename KernelPolicy::VertexId         *d_edge_list,
        typename KernelPolicy::SizeT            *d_degrees,
        typename KernelPolicy::SizeT            *d_flags,
        typename KernelPolicy::SizeT            input_queue_len,
        typename KernelPolicy::SizeT            max_vertices,
        typename KernelPolicy::SizeT            max_edges)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Inspect(
            d_src_node_ids,
            d_dst_node_ids,
            d_edge_list,
            d_degrees,
            d_flags,
            input_queue_len,
            max_vertices,
            max_edges);
}

/**
 * @brief Kernel entry for IntersectTwoSmallNL function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier queue (destination node ids)
 * @param[in] d_edge_list       Device pointer of VertexId to the edge list IDs
 * @param[in] d_degrees         Device pointer of SizeT to degree array
 * @param[in] problem           Device pointer to the problem object
 * @param[out] d_output_counts  Device pointer to the output counts array
 * @param[in] input_length      Length of the incoming frontier queues (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] num_vertex        Maximum number of elements we can place into the incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the outgoing frontier
 *
 */

  template<typename KernelPolicy, typename ProblemData, typename Functor>
  __launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
            typename KernelPolicy::SizeT        *d_row_offsets,
            typename KernelPolicy::VertexId     *d_column_indices,
            typename KernelPolicy::VertexId     *d_src_node_ids,
            typename KernelPolicy::VertexId     *d_dst_node_ids,
            typename KernelPolicy::SizeT        *d_degrees,
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
            typename KernelPolicy::SizeT        *d_output_total,
            typename KernelPolicy::SizeT        input_length,
            typename KernelPolicy::SizeT        stride,
            typename KernelPolicy::SizeT        num_vertex,
            typename KernelPolicy::SizeT        num_edge)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoSmallNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_degrees,
            problem,
            d_output_counts,
            d_output_total,
            input_length,
            stride,
            num_vertex,
            num_edge);
}

/**
 * @brief Kernel entry for IntersectTwoLargeNL function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier queue (destination node ids)
 * @param[in] d_src_nl_sizes    Device pointer of SizeT to the output neighbor list (nl) size for src node ids
 * @param[in] d_dst_nl_sizes    Device pointer of SizeT to the output neighbor list (nl) size for dst node ids
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_length      Length of the incoming frontier queues (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] nv_per_block      Number of pairs processed per block
 * @param[in] num_vertex        Maximum number of elements we can place into the incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the outgoing frontier
 *
 */

  template<typename KernelPolicy, typename ProblemData, typename Functor>
  __launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
  __global__
  void IntersectTwoLargeNL(
            typename KernelPolicy::SizeT        *d_row_offsets,
            typename KernelPolicy::VertexId     *d_column_indices,
            typename KernelPolicy::VertexId     *d_src_node_ids,
            typename KernelPolicy::VertexId     *d_dst_node_ids,
            typename KernelPolicy::VertexId     *d_edge_list,
            typename KernelPolicy::SizeT        *d_degrees,
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
            typename KernelPolicy::SizeT        input_length,
            typename KernelPolicy::SizeT        nv_per_block,
            typename KernelPolicy::SizeT        num_vertex,
            typename KernelPolicy::SizeT        num_edge)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoLargeNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_edge_list,
            d_degrees,
            problem,
            d_output_counts,
            input_length,
            nv_per_block,
            num_vertex,
            num_edge);
}

template <
    OprtrFlag FLAG,
    typename  GraphT,
    typename  FrontierInT,
    typename  FrontierOutT,
    typename  ParametersT,
    typename  AdvanceOpT,
    typename  FilterOpT>
cudaError_t Launch(
    const GraphT           graph,
    const FrontierInT    * frontier_in,
          FrontierOutT   * frontier_out,
          ParametersT     &parameters,
          AdvanceOpT       advance_op,
          FilterOpT        filter_op)
{
    typedef typename FrontierInT ::ValueT InKeyT;
    typedef typename FrontierOutT::ValueT OutKeyT;
    typedef typename ParametersT ::SizeT  SizeT;
    typedef typename ParametersT ::ValueT ValueT;
    typedef typename ParametersT ::LabelT LabelT;
    typedef typename Dispatch<FLAG, InKeyT, OutKeyT,
        SizeT, ValueT, LabelT, FilterOpT, true>
        ::KernelPolicyT KernelPolicyT;

    SizeT grid_size = (parameters.frontier -> queue_reset) ?
        (parameters.frontier -> queue_length / KernelPolicyT::THREADS + 1) :
        (parameters.cuda_props -> device_props.multiProcessorCount * KernelPolicyT::CTA_OCCUPANCY);
    Kernel<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT, FilterOpT>        <<<grid_size, KernelPolicyT::THREADS, 0, parameters.stream>>>(
        parameters.frontier -> queue_reset,
        (SizeT)(parameters.frontier -> queue_index),
        (frontier_in == NULL) ? ((InKeyT*)NULL)
            : (frontier_in -> GetPointer(util::DEVICE)),
        (parameters.values_in == NULL) ? ((ValueT*)NULL)
            : (parameters.values_in -> GetPointer(util::DEVICE)),
        parameters.frontier -> queue_length,
        parameters.label,
        (parameters.labels == NULL) ? ((LabelT*)NULL)
            : (parameters.labels -> GetPointer(util::DEVICE)),
        (parameters.visited_masks == NULL) ? ((unsigned char*)NULL)
            : (parameters.visited_masks -> GetPointer(util::DEVICE)),
        (frontier_out == NULL) ? ((OutKeyT*)NULL)
            : (frontier_out -> GetPointer(util::DEVICE)),
        parameters.frontier -> work_progress,
        filter_op);

    if (frontier_out != NULL)
    {
        parameters.frontier -> queue_index ++;
    }
    return cudaSuccess;
}

// Kernel Entry point for performing batch intersection computation
template <typename KernelPolicy, typename ProblemData, typename Functor>
    float LaunchKernel(
        gunrock::app::EnactorStats<typename KernelPolicy::SizeT>
                                                &enactor_stats,
        gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                                &frontier_attribute,
        typename ProblemData::DataSlice         *data_slice,
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_column_indices,
        typename KernelPolicy::VertexId         *d_src_node_ids,
        typename KernelPolicy::VertexId         *d_dst_node_ids,
        typename KernelPolicy::VertexId         *d_degrees,
        typename KernelPolicy::SizeT            *d_output_counts,
        typename KernelPolicy::SizeT            *d_output_total,
        typename KernelPolicy::SizeT            input_length,
        typename KernelPolicy::SizeT            max_vertex,
        typename KernelPolicy::SizeT            max_edge,
        util::CtaWorkProgress<typename KernelPolicy::SizeT> &work_progress,
        CudaContext                             &context,
        cudaStream_t                            stream)
{
    typedef typename KernelPolicy::SizeT        SizeT;
    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::Value        Value;

    /*void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    SizeT *d_total_count = NULL;
    SizeT total_counts[1] = {0};
    util::GRError(cudaMalloc(
                    &d_total_count,
                    sizeof(SizeT)),
                    "Total count cudaMalloc failed.", __FILE__, __LINE__);*/

    size_t stride = (input_length + KernelPolicy::BLOCKS * KernelPolicy::THREADS - 1)
                        >> (KernelPolicy::LOG_THREADS + KernelPolicy::LOG_BLOCKS);
    
    IntersectTwoSmallNL<KernelPolicy, ProblemData, Functor>
    <<<KernelPolicy::BLOCKS, KernelPolicy::THREADS>>>(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_degrees,
            data_slice,
            d_output_counts,
            d_output_total,
            input_length,
            stride,
            max_vertex,
            max_edge);

    /*util::DisplayDeviceResults(d_output_counts,10);
    util::DisplayDeviceResults(&d_output_counts[input_length/8],10);
    util::DisplayDeviceResults(&d_output_counts[input_length/4],10);
    util::DisplayDeviceResults(&d_output_counts[input_length/8*3],10);*/

    /*cub::DeviceReduce::Sum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_output_counts,
                                  &d_total_count[0],
                                  10);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceReduce::Sum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_output_counts,
                                  &d_total_count[0],
                                  10);*/

    /*util::GRError(cudaMemcpy( total_counts,
    &d_total_count[0], sizeof(SizeT),
    cudaMemcpyDeviceToHost),"Total count cudaMemcpy failed.", __FILE__,
    __LINE__);*/


    long total = mgpu::Reduce(d_output_total, input_length, context);
    long tc_count = mgpu::Reduce(d_output_counts, input_length, context);
    printf("tc_total:%ld\n, tc_count:%ld\n", total, tc_count);
    return (float)tc_count / (float)total;
    //return total_counts[0];
}

template <
    OprtrFlag FLAG,
    typename GraphT,
    typename FrontierInT,
    typename FrontierOutT,
    typename ParametersT,
    typename AdvanceOpT,
    typename FilterOpT>
cudaError_t Launch(
    const GraphT         &graph,
    const FrontierInT   * frontier_in,
          FrontierOutT  * frontier_out,
          ParametersT    &parameters,
          AdvanceOpT      advance_op,
          FilterOpT       filter_op)
{
    if (parameters.filter_mode == "CULL")
        return CULL::Launch<FLAG>(graph, frontier_in, frontier_out,
            parameters, advance_op, filter_op);
    if (parameters.filter_mode == "BY_PASS")
        return BP::Launch<FLAG>(graph, frontier_in, frontier_out,
            parameters, advance_op, filter_op);

    return util::GRError(cudaErrorInvalidValue,
        "FilterMode " + parameters.filter_mode + " undefined.", __FILE__, __LINE__);
}

template <
    OprtrFlag FLAG,
    typename GraphT,
    typename FrontierInT,
    typename FrontierOutT,
    typename ParametersT,
    typename OpT>
cudaError_t Launch(
    const GraphT         &graph,
    const FrontierInT   * frontier_in,
          FrontierOutT  * frontier_out,
          ParametersT    &parameters,
          OpT             op)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename FrontierInT::ValueT InKeyT;

    auto dummy_advance = []__host__ __device__ (
        const VertexT &src   , VertexT &dest, const SizeT &edge_id,
        const InKeyT  &key_in, const SizeT &input_pos, SizeT &output_pos) -> bool{
            return true;
        };

    return Launch<FLAG>(graph, frontier_in, frontier_out,
        parameters, dummy_advance, op);
}

}  // intersection
}  // oprtr
}  // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
