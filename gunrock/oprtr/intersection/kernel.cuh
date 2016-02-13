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
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
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
        d_flags[idx] = (src_nl_size > KernelPolicy::NL_THRESDHOLD
                     && dst_nl_size > KernelPolicy::NL_THRESHOLD) ? 1 : 0;
    }

    static __device__ void IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       &input_length,
                            SizeT       &stride,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        VertexId start = threadIdx.x + blockIdx.x * blockDim.x*stride;
        VertexId end = (start + stride * KernelPolicy::THREADS > input_length)? input_length :
                        (start + stride * KernelPolicy::THREADS);
        SizeT count = 0;

        typedef cub::BlockReduce<SizeT, KernelPolicy::THREADS> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;

        for (VertexId idx = start; idx < end; idx += KernelPolicy::THREADS) {
            // get nls start and end index for two ids
            VertexId eid = d_edge_list[idx];
            VertexId sid = d_src_node_ids[eid];
            VertexId did = d_dst_node_ids[eid];
            SizeT src_it = d_row_offsets[sid];
            SizeT src_end = d_row_offsets[sid+1];
            SizeT dst_it = d_row_offsets[did];
            SizeT dst_end = d_row_offsets[did+1];
            VertexId src_edge = d_column_indices[src_it];
            VertexId dst_edge = d_column_indices[dst_it];
            while (src_it < src_end && dst_it < dst_end) {
                VertexId diff = src_edge - dst_edge;
                src_edge = (diff <= 0) ? d_column_indices[++src_it] : src_edge;
                dst_edge = (diff >= 0) ? d_column_indices[++dst_it] : dst_edge;
                count += (diff == 0);
            }
        }

        SizeT aggregate = BlockReduceT(temp_storage).Sum(count);
        if (threadIdx.x == 0)
        {
            d_output_counts[blockIdx.x] += aggregate;
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
        if (threadIdx.x == 0)
                d_output_counts[blockIdx.x] = 0;
        for (int i = start; i < end; ++i) {
            // For each of these pairs, a block will
            // perform the following phases:
            // 1) partition
            // 2) intersect per thread
            // 3) block reduce to get partial triangle counts
            //
            // partition:
            // get acount and bcount
            VertexId edge_id = d_edge_list[i];
            VertexId src_node = d_src_node_ids[edge_id];
            VertexId dst_node = d_dst_node_ids[edge_id];
            SizeT a_rowoffset = d_row_offsets[src_node];
            SizeT b_rowoffset = d_row_offsets[dst_node];
            SizeT acount  = d_degrees[src_node];
            SizeT bcount  = d_degrees[dst_node];
            VertexId *a_list = &d_column_indices[a_rowoffset];
            VertexId *b_list = &d_column_indices[b_rowoffset];
            int numPartitions = KernelPolicy::THREADS;
            int numSearches = numPartitions+1;
            int nv = (acount+bcount+numPartitions-1)/numPartitions;
            //Duplicate = T, comp = mgpu::less<VertexId>, numSearches=numPartitions+1
            int gid = threadIdx.x;
            int diag;
            if (gid < numSearches) {
                diag = min(acount+bcount, gid * nv);
                int2 bp = mgpu::BalancedPath<true, mgpu::int64>(a_list, acount,
                b_list, bcount, diag, 4, mgpu::less<VertexId>());
                if (bp.y) bp.x |= 0x80000000;
                smem_storage.s_partition_idx[gid] = bp.x;
            } 
            __syncthreads();

            // intersect per thread
            // for each thread, put a_list[a_rowoffset+diag] to a_list[a_rowoffset+diag+s_partition_idx[gid]]
            // and b_list[b_rowoffset+diag] to b_list[b_rowoffset+diag+nv-s_partition_idx[gid]] together.
            // Then do serial intersection
            VertexId aBegin = 0;
            VertexId bBegin = 0;
            VertexId aEnd = smem_storage.s_partition_idx[gid];
            VertexId bEnd = nv - smem_storage.s_partition_idx[gid];
            SizeT result = SerialSetIntersection(a_list[a_rowoffset+diag],
                                  b_list[b_rowoffset+diag],
                                  aBegin,
                                  aEnd,
                                  bBegin,
                                  bEnd,
                                  nv,
                                  nv,
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
            typename KernelPolicy::VertexId     *d_edge_list,
            typename KernelPolicy::SizeT        *d_degrees,
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
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
            d_edge_list,
            d_degrees,
            problem,
            d_output_counts,
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
            typename KernelPolicy::VertexId     *&d_edge_list,
            typename KernelPolicy::SizeT        *&d_degrees,
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

// Kernel Entry point for performing batch intersection computation
template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(
        gunrock::app::EnactorStats              &enactor_stats,
        gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                                &frontier_attribute,
        typename ProblemData::DataSlice         *data_slice,
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_column_indices,
        typename KernelPolicy::VertexId         *d_src_node_ids,
        typename KernelPolicy::VertexId         *d_dst_node_ids,
        typename KernelPolicy::VertexId         *d_degrees,
        typename KernelPolicy::VertexId         *d_edge_list,
        typename KernelPolicy::VertexId         *d_edge_list_partitioned,
        typename KernelPolicy::SizeT            *d_flags,
        typename KernelPolicy::SizeT            *d_output_counts,
        typename KernelPolicy::SizeT            input_length,
        typename KernelPolicy::SizeT            max_vertex,
        typename KernelPolicy::SizeT            max_edge,
        util::CtaWorkProgress                   work_progress,
        CudaContext                             &context,
        cudaStream_t                            stream)
{

    typedef typename KernelPolicy::SizeT        SizeT;
    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::Value        Value;

    // Inspect 
    size_t block_num = (input_length + KernelPolicy::THREADS - 1)
                        >> KernelPolicy::LOG_THREADS;
    
    Inspect<KernelPolicy, ProblemData, Functor>
    <<<block_num, KernelPolicy::THREADS>>>(
            d_src_node_ids,
            d_dst_node_ids,
            d_edge_list,
            d_degrees,
            d_flags,
            input_length,
            max_vertex,
            max_edge);

    // Partition d_src_node_ids and d_dst_node_ids. Compute coarse_counts and
    // fine_counts.
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    SizeT *d_coarse_count;
    SizeT coarse_counts[1] = {0};
    util::GRError(cudaMalloc(
                    (void**)&d_coarse_count[0],
                    sizeof(SizeT)),
                    "Coarse count cudaMalloc failed.", __FILE__, __LINE__);
    
    cub::DevicePartition::Flagged(d_temp_storage,
                                  temp_storage_bytes,
                                  d_edge_list,
                                  d_flags, 
                                  d_edge_list_partitioned, 
                                  d_coarse_count, 
                                  input_length);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DevicePartition::Flagged(d_temp_storage,
                                  temp_storage_bytes,
                                  d_edge_list,
                                  d_flags, 
                                  d_edge_list_partitioned, 
                                  d_coarse_count, 
                                  input_length);
    util::GRError(cudaMemcpy(d_coarse_count, coarse_counts, sizeof(SizeT), cudaMemcpyHostToDevice),
                    "Coarse count cudaMemcpy failed.", __FILE__, __LINE__);

    SizeT fine_counts = input_length - coarse_counts[0];

    if (coarse_counts > 0) {
        SizeT pairs_per_block = (coarse_counts + KernelPolicy::BLOCKS - 1)
                                >> KernelPolicy::LOG_BLOCKS;
        // Use IntersectTwoLargeNL
        IntersectTwoLargeNL<KernelPolicy, ProblemData, Functor>
        <<<KernelPolicy::BLOCKS, KernelPolicy::THREADS>>>(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_edge_list_partitioned,
            d_degrees,
            data_slice,
            d_output_counts,
            coarse_counts,
            pairs_per_block,
            max_vertex,
            max_edge);
    } 

    size_t stride = (fine_counts + KernelPolicy::BLOCKS * KernelPolicy::THREADS - 1)
                        >> (KernelPolicy::LOG_THREADS + KernelPolicy::LOG_BLOCKS);
    
    // Use IntersectTwoSmallNL 
    IntersectTwoSmallNL<KernelPolicy, ProblemData, Functor>
    <<<KernelPolicy::BLOCKS, KernelPolicy::THREADS>>>(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            &d_edge_list_partitioned[coarse_counts[0]],
            d_degrees,
            data_slice,
            d_output_counts,
            fine_counts,
            stride,
            max_vertex,
            max_edge);
}

}  // intersection
}  // oprtr
}  // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
