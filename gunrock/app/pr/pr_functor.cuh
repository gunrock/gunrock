// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------- 
/**
 * @file
 * pr_functor.cuh
 *
 * @brief Device functions for PR problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/pr/pr_problem.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief Structure contains device functions in PR graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for PR problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PRFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return (problem->degrees[d_id] > 0 && problem->degrees[s_id] > 0);
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        atomicAdd(problem->rank_next + d_id, problem->rank_curr[s_id]/problem->degrees[s_id]);
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        Value delta = problem->delta;
        VertexId src_node = problem->src_node;
        Value threshold = (Value)problem->threshold;
        problem->rank_next[node] = (delta * problem->rank_next[node]) + (1.0-delta) * ((src_node == node || src_node == -1) ? 1 : 0);
        Value diff = fabs(problem->rank_next[node] - problem->rank_curr[node]);
 
        return (diff > threshold);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for PR problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions to remove zero degree node
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for PR problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RemoveZeroDegreeNodeFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return (problem->degrees[d_id] == 0);
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        atomicAdd(problem->degrees_pong + s_id, -1);
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        bool valid = (problem->degrees[node] == 0);
        problem->degrees_pong[node] = valid ? -1 : problem->degrees_pong[node];
        return (problem->degrees[node] > 0);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for PR problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
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
