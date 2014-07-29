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

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Structure contains device functions in BFS graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor
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
        if (ProblemData::ENABLE_IDEMPOTENCE) {
            return true;
        } else {
            // Check if the destination node has been claimed as someone's child
            /*if (ProblemData::MARK_PREDECESSORS)
                return (atomicCAS( problem->preds + d_id , -2, s_id) == -2) ? true : false;
            else { 
                return (atomicCAS( problem->labels + d_id, -1, s_id+1) == -1) ? true : false;
            }*/
            Value label, new_weight;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                label, problem->labels + s_id);
            new_weight = label +1;
            return (new_weight < atomicMin(problem->labels + d_id, new_weight));
        }
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
        if (ProblemData::ENABLE_IDEMPOTENCE) {
            // do nothing here
        } else {
            //set d_labels[d_id] to be d_labels[s_id]+1
            if (ProblemData::MARK_PREDECESSORS) {
                /*VertexId label;
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        label, problem->labels + s_id);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        label+1, problem->labels + d_id);*/
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                    s_id, problem->preds + d_id);
                //if (s_id >= problem->nodes)
                //    printf("s_id = %d, d_id = %d \t", s_id, d_id);
            }
        }
    }

    /**
     * @brief filter condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return node != -1;
    }

    /**
     * @brief filter apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        if (ProblemData::ENABLE_IDEMPOTENCE) {
            util::io::ModifiedStore<util::io::st::cg>::St(
                    v, problem->labels + node);
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
