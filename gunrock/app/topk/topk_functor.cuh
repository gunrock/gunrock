// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * topk_functor.cuh
 *
 * @brief Device functions for TopK problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

namespace gunrock {
namespace app {
namespace topk {

/**
 * @brief Structure contains device functions in top k problem.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for top k problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct TOPKFunctor
{
  typedef typename ProblemData::DataSlice DataSlice;
  
  /**
   * @brief Forward Edge Mapping condition function.
   * find each vertex's neighbors
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   *
   * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, 
						  VertexId e_id = 0, VertexId e_id_in = 0)
  {
    return true;
  }
  
  /**
   * @brief Forward Edge Mapping apply function. Now we know the source node
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   *
   */
  static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, 
						   VertexId e_id = 0, VertexId e_id_in = 0)
  {
  
  }
  
  /**
   * @brief filter condition function. 
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   *
   * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
  {
    return true;
  }
  
  /**
   * @brief filter apply function. 
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
  {}
};
  
} // topk
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
