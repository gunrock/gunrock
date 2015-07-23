// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file vis_functor.cuh
 * @brief Device functions for Vertex-Induced Subgraph
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/vis/vis_problem.cuh>

namespace gunrock {
namespace app {
namespace vis {

/**
 * @brief Structure contains device functions
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice
 *
 */
template<typename VertexId, typename SizeT,
         typename Value, typename ProblemData>
struct VISFunctor {
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Advance condition function
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id Output edge id
     * @param[in] e_id_in Input edge id
     *
     * \return Whether to load the apply function for the edge and
     *         include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool
    CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem,
             VertexId e_id = 0, VertexId e_id_in = 0) {
        return problem->mask[d_id];
    }

    /**
     * @brief Advance apply function
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id Output edge id
     * @param[in] e_id_in Input edge id
     *
     */
    static __device__ __forceinline__ void
    ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem,
              VertexId e_id = 0, VertexId e_id_in = 0) {
        printf("select edges: sid: %d, did: %d, eid: %d\n", s_id, d_id, e_id);
    }

    /**
     * @brief filter condition function
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v Auxiliary value
     *
     * \return Whether to load the apply function for the node and
     *         include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool
    CondFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        return (node % 2) == 0;  // TODO: USER-DEFINED FILTER CONDITION HERE
    }

    /**
     * @brief filter apply function
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v Auxiliary value
     *
     */
    static __device__ __forceinline__ void
    ApplyFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, problem->mask + node);
    }
};

}  // namespace vis
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
