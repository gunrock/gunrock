// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_functor.cuh
 *
 * @brief Device functions for SSSP problem.
 */

#pragma once
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <stdio.h>

namespace gunrock {
namespace app {
namespace sssp {

// TODO: 1) no atomics when in-degree is 1
// 2) if out-degree is 0 (1 in undirected graph), no enqueue and relaxation
// 3) first iteration no relaxation

/**
 * @brief Structure contains device functions in SSSP graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct SSSPFunctor {
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
        Value pred_distance, edge_weight;

        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            pred_distance, d_data_slice->distances + s_id);
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            edge_weight, d_data_slice->weights + edge_id);
        Value new_distance = pred_distance + edge_weight;

        // Check if the destination node has been claimed as someone's child
        Value old_distance = atomicMin(d_data_slice->distances + d_id, new_distance);
        bool result = (new_distance < old_distance);

        return result;
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set distance to its child
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
        if (Problem::MARK_PATHS)
        {
            if (d_data_slice -> original_vertex.GetPointer(util::DEVICE) != NULL)
                s_id = d_data_slice -> original_vertex[s_id];
            util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                s_id, d_data_slice->preds + d_id);
        }
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
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
        if (node == -1) return false;
        if (d_data_slice -> labels[node] == label)
        {
            return false;
        }
        d_data_slice -> labels[node] = label;
        return true;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for SSSP problem.
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
        // Doing nothing here
    }
};

template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct PQFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Compute Priority Score
     *
     * @param[in] node_id
     * @param[in] d_data_slice Data slice object
     *
     * \return Return the computed priority score
     */
    static __device__ __forceinline__ Value ComputePriorityScore(
        VertexId node_id, DataSlice *d_data_slice) {
        Value distance;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            distance, d_data_slice->distances + node_id);
        float delta;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            delta, d_data_slice->delta);
        return (delta == 0) ? distance : distance / delta;
    }
};


} // sssp
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
