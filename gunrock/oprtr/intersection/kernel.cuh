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
// Notes: For per-block method, here is the rough schema:
// Input: two arrays with length: m stores node pairs.
//        row_offsets
//        column_indices
// Expected output: a single integer as triangle count
//
// First, we divide the input arrays into p partitions, each block will compute
// the intersects of m/p node pairs. For each of these pairs, a block will
// perform the following phases:
// 1) partition
// 2) intersect per thread
// 3) block reduce to get partial triangle counts
//
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

    static __device__ __forceinline__ SizeT GetNeighborListLength(
                            SizeT       *&d_row_offsets,
                            VertexId    &d_vertex_id,
                            SizeT       &max_vertex,
                            SizeT       &max_edge)
    {
    }

    // Get neighbor list sizes, scan to get both
    // fine_counts (for two per-thread methods) and
    // coarse_counts (for balanced-path per-block method)
    static __device__ void Inspect(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
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
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectTwoLargeNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_src_nl_sizes,
                            SizeT       *&d_dst_nl_sizes,
                            DataSlice   *&problem,
                            SizeT       &input_length,
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

    static __device__ __forceinline__ SizeT GetNeighborListLength(
                            SizeT       *&d_row_offsets,
                            VertexId    &d_vertex_id,
                            SizeT       &max_vertex,
                            SizeT       &max_edge)
    {
        SizeT first = d_vertex_id >= max_vertex ? max_edge
                                                : d_row_offsets[d_vertex_id];
        SizeT second = (d_vertex_id + 1) >= max_vertex
                        ? max_edge : d_row_offsets[d_vertex_id+1];

        return (second > first) ? second - first : 0;
    }

    // Get neighbor list sizes, scan to get both
    // fine_counts (for two per-thread methods) and
    // coarse_counts (for balanced-path per-block method)
    static __device__ __forceinline__ void Inspect(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_flags,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // Compute d_src_nl_sizes and d_dst_nl_sizes;
        VertexId idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_length) return;
        SizeT src_nl_size = GetNeighborListLength(d_row_offsets, 
                                                  d_src_node_ids[idx], 
                                                  num_vertex, num_edge);
        SizeT dst_nl_size = GetNeighborListLength(d_row_offsets, 
                                                  d_dst_node_ids[idx], 
                                                  num_vertex, num_edge);
        d_flags[idx] = (src_nl_size > KernelPolicy::NL_THRESDHOLD
                     && dst_nl_size > KernelPolicy::NL_THRESHOLD) ? 1 : 0;
    }

    static __device__ void IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        VertexId idx = threadIdx.x + blockIdx.x * blockDim.x;
        SizeT count = 0;

        typedef cub::BlockReduce<SizeT, KernelPolicy::THREADS> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;

        for (;idx < input_length; idx += KernelPolicy::THREADS) {
            // get nls start and end index for two ids
            SizeT src_it = d_row_offsets[d_src_node_ids[idx]];
            SizeT src_end = d_row_offsets[d_src_node_ids[idx]+1];
            SizeT dst_it = d_row_offsets[d_dst_node_ids[idx]];
            SizeT dst_end = d_row_offsets[d_dst_node_ids[idx]+1];
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
            d_output_counts[blockIdx.x] = aggregate;
        } 
    }

    static __device__ void IntersectTwoLargeNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_src_nl_sizes,
                            SizeT       *&d_dst_nl_sizes,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // Each block uses balanced path to get intersect count
    }

};


/**
 * @brief Kernel entry for Inspect function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier
 *                              queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier 
 *                              queue (destination node ids)
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
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_src_node_ids,
        typename KernelPolicy::VertexId         *d_dst_node_ids,
        typename KernelPolicy::SizeT            *d_flags,
        typename KernelPolicy::SizeT            input_queue_len,
        typename KernelPolicy::SizeT            max_vertices,
        typename KernelPolicy::SizeT            max_edges)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Inspect(
            d_row_offsets,
            d_src_node_ids,
            d_dst_node_ids,
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
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
            typename KernelPolicy::SizeT        input_length,
            typename KernelPolicy::SizeT        num_vertex,
            typename KernelPolicy::SizeT        num_edge)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoSmallNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            problem,
            d_output_counts,
            input_length,
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
            typename KernelPolicy::SizeT        *d_src_nl_sizes,
            typename KernelPolicy::SizeT        *d_dst_nl_sizes,
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
            typename KernelPolicy::SizeT        input_length,
            typename KernelPolicy::SizeT        num_vertex,
            typename KernelPolicy::SizeT        num_edge)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoLargeNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_src_nl_sizes,
            d_dst_nl_sizes,
            problem,
            d_output_counts,
            input_length,
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
        typename KernelPolicy::VertexId         *d_src_node_ids_partitioned,
        typename KernelPolicy::VertexId         *d_dst_node_ids_partitioned,
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
            d_row_offsets,
            d_src_node_ids,
            d_dst_node_ids,
            d_flags,
            input_length,
            max_vertex,
            max_edge);

    // Partition d_src_node_ids and d_dst_node_ids. Compute coarse_counts and
    // fine_counts.
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    SizeT coarse_counts = 0;
    cub::DevicePartition::Flagged(d_temp_storage,
                                  temp_storage_bytes,
                                  d_src_node_ids,
                                  d_flags, 
                                  d_src_node_ids_partitioned, 
                                  coarse_counts, 
                                  input_length);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DevicePartition::Flagged(d_temp_storage, 
                                  temp_storage_bytes, 
                                  d_src_node_ids,
                                  d_flags, 
                                  d_src_node_ids_partitioned, 
                                  coarse_counts, 
                                  input_length);

    cub::DevicePartition::Flagged(d_temp_storage, 
                                  temp_storage_bytes, 
                                  d_src_node_ids,
                                  d_flags, 
                                  d_src_node_ids_partitioned, 
                                  coarse_counts, 
                                  input_length);

    SizeT fine_counts = input_length - coarse_counts;

    if (coarse_counts > 0) {
        // Use IntersectTwoLargeNL
    } 
    block_num = (fine_counts + KernelPolicy::THREADS - 1)
                        >> KernelPolicy::LOG_THREADS;
    
    // Use IntersectTwoSmallNL 
    IntersectTwoSmallNL<KernelPolicy, ProblemData, Functor>
    <<<block_num, KernelPolicy::THREADS>>>(
            d_row_offsets,
            d_column_indices,
            &d_src_node_ids_partitioned[coarse_counts],
            &d_dst_node_ids_partitioned[coarse_counts],
            data_slice,
            &d_output_counts[KernelPolicy::BLOCKS],
            fine_counts,
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
