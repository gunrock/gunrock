// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.


/**
 * @file sm_functor.cuh
 * @brief Device functions
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sm/sm_problem.cuh>

namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief Structure contains device functions for doing initial filtering. 
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<typename VertexId, typename SizeT,typename Value, typename ProblemData>
struct SMInitFunctor {

    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Candidates initial filter condition function. 
     * Check if the data node has the same label and larger 
     * or equivalent degree as each query node.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v Auxiliary value
     *
     * \return Whether to load the apply function for the node and
     *         include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool
    CondFilter(VertexId node, DataSlice *problem, Value v=0 , SizeT nid = 0) {
        return true;  
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
	for(VertexId i=0; i < problem->nodes_query; i++){
		if ((problem->d_data_labels[node] == problem->d_query_labels[i]) && 
		    (problem->d_data_degrees[node] >= problem->d_query_degrees[i]))
			problem->d_c_set[node+i*problem->nodes_data]=1;
		else
			problem->d_c_set[node+i*problem->nodes_data]=0;
		
	}
    }
}; // SMInitFunctor

/**
 * @brief Structure contains device functions in computing query edge weights. 
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<
  typename VertexId,
  typename SizeT,
  typename Value,
  typename ProblemData>
struct EdgeWeightFunctor
{
  typedef typename ProblemData::DataSlice DataSlice;
  /**
   * @brief Forward Advance Kernel condition function.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   *
   * \return Whether to load the apply function for the edge and include
   * the destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
    VertexId s_id, VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
  {
    return true;
  }

   /**
   * @brief Forward Advance Kernel apply function.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   */
  static __device__ __forceinline__ void ApplyEdge(
    VertexId s_id,  VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
  {
    // add src node weight with dest node weight to get edge weight
    problem->d_edge_weight[e_id_in] = problem->d_temp_keys[s_id] + problem->d_temp_keys[d_id];
  }
}; // EdgeWeightFunctor

/**
 * @brief Structure contains device functions in pruning query node candidates. 
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<
  typename VertexId,
  typename SizeT,
  typename Value,
  typename ProblemData>
struct PruneFunctor
{
  typedef typename ProblemData::DataSlice DataSlice;
  /**
   * @brief Forward Advance Kernel condition function.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   *
   * \return Whether to load the apply function for the edge and include
   * the destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
    VertexId s_id, VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
  {
	return true;
  }

 /**
   * @brief Forward Advance Kernel apply function.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index
   * @param[in] e_id_in Input edge index
   */
  static __device__ __forceinline__ void ApplyEdge(
    VertexId s_id,  VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
  {
    int query_node;
    int temp;
    int neighbors;
    __shared__ int ones;
    if(threadIdx.x==0) ones=0;
    __syncthreads();
    int offset;
    int i,j;
    bool save=false;

    for(i=0; i<problem->nodes_query; i++){
	query_node = problem->d_temp_keys[i];
	// check if there's a candidate for each neighbor of query node 
        // TODO: add the 1s together and compare with query node's #neigbors, if smaller prune out
	neighbors = problem->d_row_offsets[query_node+1] - problem->d_row_offsets[query_node];

	if(problem->d_c_set[query_node * problem->nodes_data+s_id]){
	    for(j=0; j < neighbors; j++){
		offset = problem->d_row_offsets[query_node]+j;
		if(problem->d_c_set[d_id+problem->nodes_data * problem->d_column_indices[offset]])
		{
			save=true;
		}
		__syncthreads();
		if(save) continue;
		else break;
		
	    }
	
	   if(i<neighbors) problem->d_c_set[s_id + query_node * problem->nodes_data]=0;
	   __syncthreads();
	}

	if(problem->d_c_set[query_node * problem->nodes_data+s_id]){
	   for(j=0; j<problem->nodes_query; j++){
		temp = problem->d_temp_keys[i];
		if(problem->d_c_set[d_id+temp*problem->nodes_data]) atomicAdd(&ones,1);
	   	__syncthreads();
	
	   }
	   if(ones<neighbors) problem->d_c_set[s_id+query_node*problem->nodes_data]=0;
	}
	
    }
  }

}; // PruneFunctor

}  // namespace sm
}  // nimespacewapp
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
