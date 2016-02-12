// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sample_functor.cuh
 * @brief Device functions
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/template/sample_problem.cuh>

namespace gunrock {
namespace app {
namespace sample {

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
         typename Value, typename Problem>
struct SampleFunctor {
    typedef typename Problem::DataSlice DataSlice;

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
    CondEdge(VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
             VertexId e_id = 0, VertexId e_id_in = 0) {
        return true;  // TODO(developer): advance condition function
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
    ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
              VertexId e_id = 0, VertexId e_id_in = 0) {
        // TODO(developer): advance apply function
    }

    /**
     * @brief filter condition function
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v Auxiliary value
     * @param[in] nid 
     *
     * \return Whether to load the apply function for the node and
     *         include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool
    CondFilter(VertexId node, DataSlice *d_data_slice, Value v = 0, SizeT nid = 0) {
        return true;  // TODO(developer): filter condition function
    }

    /**
     * @brief filter apply function
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v Auxiliary value
     * @param[in] nid
     *
     */
    static __device__ __forceinline__ void
    ApplyFilter(VertexId node, DataSlice *d_data_slice, Value v = 0, SizeT nid = 0) {
        // TODO(developer): filter apply function
    }
};

}  // namespace sample
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
