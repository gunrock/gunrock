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
 * @brief Load balanced Edge Map Kernel Entrypoint
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned {

// GetRowOffsets
//
// MarkPartitionSize
//
// RelaxPartitionedEdges

template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true>
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename ProblemData::DataSlice DataSlice;

    __device__ __forceinline__ void GetEdgeCount(
                            VertexId    *d_row_offsets,
                            VertexId    d_vertex_id,
                            SizeT       max_vertex,
                            SizeT       max_edge)
    {
        VertexId first = d_vertex_id >= max_vertex ? max_edge : d_row_offsets[d_vertex_id];
        VertexId second = (d_vertex_id + 1) >= max_vertex ? max_edge : d_row_offsets[d_vertex_id+1];

        return (second > first) ? second - first : 0;
    }

    static __device__ __forceinline__ void GetRowOffsets(
                                  )
    {
    }

    static __device__ __forceinline__ void MarkPartitionSizes(
                                  )
    {
    }

    static __device__ __forceinline__ void RelaxPartitionedEdges(
                                  )
    {
    }

};

} //edge_map_partitioned
} //oprtr
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
