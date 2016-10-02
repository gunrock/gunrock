// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------- 
/**
 * @file
 * salsa_functor.cuh
 *
 * @brief Device functions for SALSA problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/salsa/salsa_problem.cuh>

namespace gunrock {
namespace app {
namespace salsa {

/**
 * @brief Structure contains device functions in SALSA graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Problem             Problem data type which contains data slice for SALSA problem
 * @tparam _LabelT             Vertex label type
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct HFORWARDFunctor
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            s_id, d_data_slice->hub_predecessors + edge_id);
    }

};

/**
 * @brief Structure contains device functions in SALSA graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for SALSA problem
 * @tparam _LabelT             Vertex label type
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT=VertexId>
struct HBACKWARDFunctor
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        VertexId v_id = d_data_slice->hub_predecessors[input_pos];
        bool flag = (d_data_slice->out_degrees[v_id] != 0);
        if (!flag) d_data_slice->hrank_next[v_id] = 0;
        return flag;
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        Value hrank_dst = (d_data_slice->in_degrees[s_id] == 0 || d_data_slice->out_degrees[d_id] == 0) ? 0 : d_data_slice->hrank_curr[d_id] / (d_data_slice->in_degrees[s_id] * d_data_slice->out_degrees[d_id]);
        VertexId v_id = d_data_slice->hub_predecessors[input_pos];
        atomicAdd(&d_data_slice->hrank_next[v_id], hrank_dst);
    }
};

/**
 * @brief Structure contains device functions in SALSA graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for SALSA problem
 * @tparam _LabelT             Vertex label type
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData, typename _LabelT=VertexId>
struct AFORWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            s_id, d_data_slice->auth_predecessors+edge_id);
    }

};

/**
 * @brief Structure contains device functions in SALSA graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for SALSA problem
 * @tparam _LabelT             Vertex label type
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData, typename _LabelT=VertexId>
struct ABACKWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        VertexId v_id = d_data_slice->auth_predecessors[input_pos];
        bool flag = (d_data_slice->in_degrees[v_id] != 0);
        if (!flag) d_data_slice->arank_next[v_id] = 0;
        return flag;
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
        VertexId    s_id,
        VertexId    d_id,
        DataSlice   *d_data_slice,
        SizeT       edge_id,
        VertexId    input_item,
        LabelT      label,
        SizeT       input_pos,
        SizeT       &output_pos)
    {
        Value arank_dst = (d_data_slice->out_degrees[s_id] == 0 || d_data_slice->in_degrees[d_id] == 0) ? 0 : d_data_slice->arank_curr[d_id] / (d_data_slice->out_degrees[s_id] * d_data_slice->in_degrees[d_id]);
        VertexId v_id = d_data_slice->auth_predecessors[input_pos];
        atomicAdd(&d_data_slice->arank_next[v_id], arank_dst);
    }
};

} // salsa
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

