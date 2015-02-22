// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------- 
/**
 * @file
 * mis_functor.cuh
 *
 * @brief Device functions for MIS problem.
 */


#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/mis/mis_problem.cuh>

namespace gunrock {
namespace app {
namespace mis {

/**
 * @brief Structure contains device functions in MIS graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for PR problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct MISFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //if destination node hasn't been colored, then reduce its label value
        //else the advance operator will assign an identity value for reduce
        //printf("%d\n", d_id);
        //printf("%d\n", problem->d_mis_ids[0]);
        return problem->d_mis_ids[d_id] == -1;
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
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return;
    }

    /**
     * @brief filter condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
    {
        //!IDEMPOTENCE && !MARK_PREDECESSOR, so v is iteration number, each iteration will get unique color id.
        problem->d_mis_ids[node] = (problem->d_labels[node] >= problem->d_reduced_values[nid]) ? v : -1;
        return problem->d_mis_ids[node] == -1;
    }

    /**
     * @brief filter apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
    {
        return;
    }
};

} // mis
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
