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
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem >
struct SSSPFunctor {
    typedef typename Problem::DataSlice DataSlice;

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
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        Value distance, weight;

        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            distance, d_data_slice->distances + s_id);
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            weight, d_data_slice->weights + e_id);
        Value new_distance = weight + distance;

        // Check if the destination node has been claimed as someone's child
        Value old_distance = atomicMin(d_data_slice->distances + d_id, new_distance);
        //if (to_track(s_id) || to_track(d_id))
        //    printf("lable[%d] : %d t-> %d + %d @ %d = %d \t", d_id, old_weight, weight, distance, s_id, new_weight);
        return (new_distance < old_distance);
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set distance to its child
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
        VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
        VertexId e_id = 0, VertexId e_id_in = 0) {
        if (Problem::MARK_PATHS)
            util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                s_id, d_data_slice->preds + d_id);
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0) {
        if (node == -1) return false;
        return (atomicCAS(d_data_slice->sssp_marker + node, 0, 1) == 0);
        //return (node != -1);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for SSSP problem.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0) {
        // Doing nothing here
    }
};

template <
    typename VertexId, typename SizeT, typename Value, typename Problem>
struct PQFunctor {
    typedef typename Problem::DataSlice DataSlice;

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
