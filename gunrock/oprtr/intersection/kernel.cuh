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

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/intersection/cta.cuh>

#include <gunrock/oprtr/intersection/kernel_policy.cuh>

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
                            SizeT       *&d_src_nl_sizes,
                            SizeT       *&d_dst_nl_sizes,
                            SizeT       &fine_counts,
                            SizeT       &coarse_counts,
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
                            SizeT       *&d_src_nl_sizes,
                            SizeT       *&d_dst_nl_sizes,
                            DataSlice   *&problem,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectOneSmallNL(
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
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_src_nl_sizes,
                            SizeT       *&d_dst_nl_sizes,
                            SizeT       &fine_counts,
                            SizeT       &coarse_counts,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // Compute d_src_nl_sizes and d_dst_nl_sizes;
        // Compute fine_counts and coarse_counts
    }

    static __device__ void IntersectTwoSmallNL(
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
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
    }

    static __device__ void IntersectOneSmallNL(
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
        // Same strategy as IntersectTwoSmallNL
        // for each column id in smaller NL, b-search in the larger column id list
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
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier queue (destination node ids)
 * @param[out] d_src_nl_sizes   Device pointer of SizeT to the output neighbor list (nl) size for src node ids
 * @param[out] d_dst_nl_sizes   Device pointer of SizeT to the output neighbor list (nl) size for dst node ids
 * @param[out] fine_counts      Counts of neighbor list pairs we will use per-thread intersect methods
 * @param[out] coarse_counts    Counts of neibhbor list pairs we will use per-block intersect methods
 * @param[in] input_queue_len   Length of the incoming frontier queues (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] max_vertices      Maximum number of elements we can place into the incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the outgoing frontier
 */
    template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void Inspect(
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_column_indices,
        typename KernelPolicy::VertexId         *d_src_node_ids,
        typename KernelPolicy::VertexId         *d_dst_node_ids,
        typename KernelPolicy::SizeT            *d_src_nl_sizes,
        typename KernelPolicy::SizeT            *d_dst_nl_sizes,
        typename KernelPolicy::SizeT            &fine_counts,
        typename KernelPolicy::SizeT            &coarse_counts,
        typename KernelPolicy::SizeT            input_queue_len,
        typename KernelPolicy::SizeT            max_vertices,
        typename KernelPolicy::SizeT            max_edges)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Inspect(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_src_nl_sizes,
            d_dst_nl_sizes,
            fine_counts,
            coarse_counts,
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
 * @param[in] d_src_nl_sizes    Device pointer of SizeT to the output neighbor list (nl) size for src node ids
 * @param[in] d_dst_nl_sizes    Device pointer of SizeT to the output neighbor list (nl) size for dst node ids
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_length      Length of the incoming frontier queues (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] num_vertex        Maximum number of elements we can place into the incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the outgoing frontier
 *
 */

  template<typename KernelPolicy, typeame ProblemData, typename Functor>
  __launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
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
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoSmallNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_src_nl_sizes,
            d_dst_nl_sizes,
            problem,
            d_output_counts,
            input__length,
            num_vertex,
            num_edge);
}

/**
 * @brief Kernel entry for IntersectOneSmallNL function
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

  template<typename KernelPolicy, typeame ProblemData, typename Functor>
  __launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
  __global__
  void IntersectOneSmallNL(
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
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectOneSmallNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_src_nl_sizes,
            d_dst_nl_sizes,
            problem,
            d_output_counts,
            input__length,
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

  template<typename KernelPolicy, typeame ProblemData, typename Functor>
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
            input__length,
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
        typename KernelPolicy::SizeT            *d_src_nl_sizes,
        typename KernelPolicy::SizeT            *d_dst_nl_sizes,
        typename KernelPolicy::SizeT            *d_output_counts,
        typename KernelPolicy::SizeT            input_length,
        typename KernelPolicy::SizeT            max_vertex,
        typename KernelPolicy::SizeT            max_edge,
        util::CtaWorkProgress                   work_progress,
        CudaContext                             &context,
        cudaStream_t                            stream)
{
}

}  // intersection
}  // oprtr
}  // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
