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

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/pr/pr_problem.cuh>

// atomic addition from Jon Cohen at NVIDIA
__device__ static double atomicAdd(double *addr, double val)
{
    double old=*addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(
        atomicCAS((unsigned long long int*)addr,
               __double_as_longlong(assumed),
               __double_as_longlong(val + assumed)));
    } while( assumed!=old );
    return old; 
}

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief Structure contains device functions in PR graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename ProblemData >
struct PRMarkerFunctor {
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id Output edge index
     * @param[in] e_id_in Input edge index
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        //return (problem->degrees[d_id] > 0 && problem->degrees[s_id] > 0);
        return true;
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id Output edge index
     * @param[in] e_id_in Input edge index
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        //atomicAdd(problem->rank_next + d_id, problem->rank_curr[s_id]/problem->degrees[s_id]);
        problem->markers[d_id] = 1;
        //if (util::to_track(d_id))
        //    printf("%d\t marker[%lld] -> 1\n", problem->gpu_idx, (long long)d_id);
    }
};

/**
 * @brief Structure contains device functions in PR graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename ProblemData >
struct PRFunctor {
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the
     * destination node has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and
     *         include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        //return (problem->degrees[d_id] > 0 && problem->degrees[s_id] > 0);
        return true;
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        //if (TO_TRACK)
        //if (to_track(d_id)) printf("%d \tr[%d] \t+= %f\t from %d,%f\n", problem->gpu_idx, d_id, problem->rank_curr[s_id] / problem->degrees[s_id], s_id, problem->rank_curr[s_id]);
        Value add_value = problem->rank_curr[s_id] / (Value)problem->degrees[s_id];
        if (isfinite(add_value))
        {
            Value old_value = atomicAdd(problem->rank_next + d_id, add_value);
            //if (to_track(d_id))
            //{
            //    printf("%d\t rank_next[%d] += rank_curr[%d] (=%.8le) / %lld, old_value = %.8le\n",
            //        problem -> gpu_idx, d_id, s_id, problem->rank_curr[s_id], 
            //        (long long) problem->degrees[s_id], old_value);
            //}
        }
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id
     *        is valid (not equal to -1). Personal PageRank feature will
     *        be activated when a source node ID is set.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     * \return Whether to load the apply function for the node and
     *         include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        //Value    delta     = problem->delta    ;
        //VertexId src_node  = problem->src_node ;
        Value    old_value = problem -> rank_curr[node];
        Value    new_value = problem -> delta * problem->rank_next[node];
        if (!isfinite(new_value))
            new_value = 0;
        new_value = problem->reset_value + new_value;
        //problem->rank_next[node] = (delta * problem->rank_next[node]) + (1.0-delta) * ((src_node == node || src_node == -1) ? 1 : 0);
        //problem->rank_next[node] = problem->reset_value + delta * problem->rank_next[node];
        //Value diff = fabs(problem->rank_next[node] - problem->rank_curr[node]);
        //Value diff = fabs(new_value - old_value);
        //if (to_track(node))
        //    printf("%d\t rank_next[%d] %.8le -> %.8le + %.8le * %.8le = %.8le\n",
        //        problem->gpu_idx, node, old_value, problem->reset_value, problem->delta, 
        //        problem->rank_next[node], new_value);

        problem -> rank_curr[node] = new_value;
        return (fabs(new_value - old_value) > (problem->threshold * old_value));
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for PR problem.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions to remove zero degree node
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename ProblemData >
struct RemoveZeroDegreeNodeFunctor {
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id Output edge index
     * @param[in] e_id_in Input edge index
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
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
     * @param[in] e_id Output edge index
     * @param[in] e_id_in Input edge index
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id, VertexId d_id, DataSlice *problem,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        atomicAdd(problem->degrees_pong + s_id, -1);
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId node, DataSlice *problem, Value v = 0) {
        //SizeT degree = problem->degrees[node];
        //if (degree == 0)
        //    problem -> degrees_pong[node] = -1;
        //return (degree > 0);
        return (problem->degrees[node] > 0);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for PR problem.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
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
