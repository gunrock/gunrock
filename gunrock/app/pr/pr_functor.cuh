// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------
/**
 * @file
 * pr_functor.cuh
 *
 * @brief Device functions for PR problem.
 */

#pragma once

#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/pr/pr_problem.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief Structure contains device functions in PR graph traverse.
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
struct PRMarkerFunctor
{
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
        VertexId   s_id,
        VertexId   d_id,
        DataSlice *d_data_slice,
        SizeT      edge_id,
        VertexId   input_item,
        LabelT     label,
        SizeT      input_pos,
        SizeT     &output_pos)
    {
        return true;
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
        VertexId   s_id,
        VertexId   d_id,
        DataSlice *d_data_slice,
        SizeT      edge_id,
        VertexId   input_item,
        LabelT     label,
        SizeT      input_pos,
        SizeT     &output_pos)
    {
        d_data_slice -> markers[d_id] = 1;
    }
};

template <typename T>
struct Make4Vector
{
    typedef int4 V4;
};

template <>
struct Make4Vector<float>
{
    typedef float4 V4;
};

template <>
struct Make4Vector<double>
{
    typedef double4 V4;
};

/**
 * @brief Structure contains device functions in PR graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem>
struct PRFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef typename Make4Vector<Value>::V4 LabelT;

    /**
     * @brief Forward Edge Mapping condition function. Check if the
     * destination node has been claimed as someone else's child.
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
     * \return Whether to load the apply function for the edge and
     *         include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId   s_id,
        VertexId   d_id,
        DataSlice *d_data_slice,
        SizeT      edge_id,
        VertexId   input_item,
        LabelT     label,
        SizeT      input_pos,
        SizeT     &output_pos)
    {
        return true;
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
        VertexId   s_id,
        VertexId   d_id,
        DataSlice *d_data_slice,
        SizeT      edge_id,
        VertexId   input_item,
        LabelT     label,
        SizeT      input_pos,
        SizeT     &output_pos)
    {
        Value add_value = d_data_slice -> rank_curr[s_id];
        if (isfinite(add_value))
        {
            Value old_value = atomicAdd(d_data_slice->rank_next + d_id, add_value);
        }
    }

    /**
     * @brief filter condition function. Check if the Vertex Id
     *        is valid (not equal to -1). Personal PageRank feature will
     *        be activated when a source node ID is set.
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and
     *         include it in the outgoing vertex frontier.
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
        Value    old_value = d_data_slice -> rank_curr[node];
        Value    new_value = d_data_slice -> delta * d_data_slice -> rank_next[node];
        new_value = d_data_slice -> reset_value + new_value;
        if (d_data_slice -> degrees[node] != 0)
            new_value /= d_data_slice -> degrees[node];
        if (!isfinite(new_value)) new_value = 0;
        d_data_slice -> rank_curr[node] = new_value;
        return (fabs(new_value - old_value) > (d_data_slice->threshold * old_value));
    }

    /**
     * @brief filter apply function. Doing nothing for PR problem.
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

} // pr
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
