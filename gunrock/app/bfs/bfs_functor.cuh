// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_functor.cuh
 *
 * @brief Device functions for BFS problem.
 */

#pragma once

#include <gunrock/util/track_utils.cuh>
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Structure contains device functions in BFS graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct BFSFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        bool result = true;
        if (!Problem::ENABLE_IDEMPOTENCE)
        {
            // Check if the destination node has been claimed as someone's child
//#if __GR_CUDA_ARCH__ >= 350
            VertexId old_label= atomicMin_(d_data_slice -> labels + d_id, /*new_*/label);
            result = label < old_label;
//#else
//            result = false;
//#endif
        }
        return result;
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        if (!Problem::ENABLE_IDEMPOTENCE)
       {
            //set preds[d_id] to be s_id
            if (Problem::MARK_PREDECESSORS)
            {
                if (d_data_slice -> original_vertex.GetPointer(util::DEVICE) != NULL)
                    s_id = d_data_slice -> original_vertex[s_id];
                util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                    s_id,
                    d_data_slice -> preds + d_id);
            }
        }
    }

    /**
     * @brief filter condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        return node != -1;
    }

    /**
     * @brief filter apply function. Doing nothing for BFS problem.
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        if (Problem::ENABLE_IDEMPOTENCE) {
            if (Problem::MARK_PREDECESSORS)
            {
                if (d_data_slice -> original_vertex.GetPointer(util::DEVICE) != NULL)
                    v = d_data_slice -> original_vertex[v];
                util::io::ModifiedStore<util::io::st::cg>::St(
                v, d_data_slice -> preds + node);
            }
        } else {
            // Doing nothing here
        }
    }
};

} // bfs
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
