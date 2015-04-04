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

 template <typename VertexId>
    static __device__ bool to_track(VertexId node)
    {   
        const int num_to_track = 3;
        const VertexId node_to_track[] = {0,1,2};
        for (int i=0; i<num_to_track; i++)
            if (node == node_to_track[i]) return true;
        return false;
    }  

/**
 * @brief Structure contains device functions in SSSP graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for SSSP problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct SSSPFunctor
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
        Value label, weight;

        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        label, problem->labels + s_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        weight, problem->weights + e_id);
        Value new_weight = weight + label;
        
        // Check if the destination node has been claimed as someone's child
        Value old_weight = atomicMin(problem->labels + d_id, new_weight);
        if (to_track(s_id) || to_track(d_id))
            printf("lable[%d] : %d t-> %d + %d @ %d = %d \t", d_id, old_weight, weight, label, s_id, new_weight);
        return (new_weight < old_weight);
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
        if (ProblemData::MARK_PATHS)
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                    s_id, problem->preds + d_id); 
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
        return (node != -1);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for SSSP problem.
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

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PQFunctor
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
    static __device__ __forceinline__ Value ComputePriorityScore(VertexId node_id, DataSlice *problem)
    {
        Value weight;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        weight, problem->labels + node_id);
        float delta;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        delta, problem->delta);
        return (delta == 0) ? weight : weight/delta;
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
