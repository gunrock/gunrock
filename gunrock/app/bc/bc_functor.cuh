// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_functor.cuh
 *
 * @brief Device functions for BC problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Structure contains device functions in forward traversal pass.
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
struct ForwardFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

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
     * \return Whether or not to load the apply function.
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
        // Check if the destination node has been claimed as someone's child
        VertexId old_label = atomicCAS(d_data_slice -> labels + d_id, -1, label);
        if (old_label != label && old_label != -1) return false;

        //Accumulate sigma value
        atomicAdd(d_data_slice->sigmas + d_id, d_data_slice->sigmas[s_id]);
        if (old_label == -1) 
        {
            return true;
        }
        else return false;
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
    }

    /**
     * @brief Forward vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether or not to load the apply function.
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
     * @brief Forward vertex mapping apply function. Doing nothing for BC.
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

/**
 * @brief Structure contains device functions in backward traversal pass.
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
struct BackwardFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
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
     * \return Whether to load the apply function for the edge.
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
        VertexId s_label = _ldg(d_data_slice -> labels + s_id);
        VertexId d_label = _ldg(d_data_slice -> labels + d_id);
        return (d_label == s_label + 1);
    }

    /**
     * @brief Backward Edge Mapping apply function. Compute delta value using
     * the formula: delta(s_id) = sigma(s_id)/sigma(d_id)(1+delta(d_id))
     * then accumulate to BC value of the source node.
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
        if (s_id == d_data_slice->src_node[0]) return;
        Value from_sigma = _ldg(d_data_slice -> sigmas + s_id);

        Value to_sigma = _ldg(d_data_slice -> sigmas + d_id);

        Value to_delta = _ldg(d_data_slice -> deltas + d_id);

        Value result = from_sigma / to_sigma * (1.0 + to_delta);

        //Accumulate bc value
        {
            Value old_delta = atomicAdd(d_data_slice->deltas + s_id, result);
            Value old_bc_value = atomicAdd(d_data_slice->bc_values + s_id, result);
        }
    }

    /**
     * @brief Backward vertex mapping condition function. Check if the Vertex Id is valid (equal to 0).
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
        return d_data_slice->labels + node == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
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

/**
 * @brief Structure contains device functions in backward traversal pass.
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
struct BackwardFunctor2 {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
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
     * \return Whether to load the apply function for the edge.
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

        VertexId s_label;
        VertexId d_label;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            s_label, d_data_slice->labels + s_id);
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            d_label, d_data_slice->labels + d_id);
        return (d_label == s_label + 1);
    }

    /**
     * @brief Backward Edge Mapping apply function. Compute delta value using
     * the formula: delta(s_id) = sigma(s_id)/sigma(d_id)(1+delta(d_id))
     * then accumulate to BC value of the source node.
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
    }

    /**
     * @brief Backward vertex mapping condition function. Check if the Vertex Id is valid (equal to 0).
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether or not to load the apply function.
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
        return d_data_slice->labels[node] == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
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

} // bc
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
