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
     * Check if each data node has the same label and larger 
     * or equivalent degree as every query node.
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
        for(VertexId i=0; i<problem->nodes_query; i++)
	    if((problem->d_data_labels[node]==problem->d_query_labels[i]) &&
		    (problem->d_data_degrees[node] >= problem->d_query_degrees[i]))
	        return true;  
	return false;
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

//TODO: sort the query nodes based on edge weight.

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
	VertexId i,j;
	int offset;
	
	for(i=0; i<problem->nodes_query; i++)
	    if(problem->d_c_set[s_id + i*problem->nodes_data]==1) // s_id is a candidate of i
		// loop for i's neighbors (j) to check if d_id is a candidate of any j
		for(offset = problem->d_query_row[i]; offset<problem->d_query_row[i+1]; ++offset)
		{
		    j = problem->d_query_col[offset];
		    if(problem->d_c_set[d_id + j*problem->nodes_data]==1)
			return true;
		}
	__syncthreads();		
	// If d_id is no candidate of any i's neighbors, subtract 1 from s_id's degree
	atomicSub(problem->d_data_degrees+s_id,1);	
	return false;
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
    int neighbors;
    int offset;
    VertexId i,j;
    int num_iterations=3; // the more iterations the more accurate
	
    for(int it=0; it<num_iterations; it++){
        for(i=0; i<problem->nodes_query; i++){
	//? do I need to store the ids of row offsets in a seperate array
	//	query_node = problem->d_temp_keys[i]; 
		
	    // s_id is one of i's candidates
	    if(problem->d_c_set[i * problem->nodes_data+s_id]==1){
		neighbors = problem->d_query_row[i+1] - problem->d_query_row[i];
		// If s_id's degree is smaller than i's degree
		if(problem->d_data_degrees[s_id]<neighbors){ 
		    problem->d_c_set[i * problem->nodes_data+s_id]=0;
		    atomicSub(problem->d_data_degrees+d_id,1);
		    continue;
		}
		// check if a candidate exits for each neighbor of i among s_id's neighbors
	    	for(offset=problem->d_query_row[i]; offset<problem->d_query_row[i+1]; offset++){
		    j = problem->d_query_col[offset];
		    if(problem->d_c_set[d_id+problem->nodes_data * j]==1)
		    	problem->d_temp_keys[s_id]++; // Competitive add
		
		    __syncthreads();
	  	    if(problem->d_temp_keys[s_id]>0) continue;
		    else break;
		
	    	}
	        // break happens: Not every i's neighbor can find a candidate among s_id's neighbors
	        if(offset<problem->d_query_row[i+1]) 
		    problem->d_c_set[s_id + i * problem->nodes_data]=0;
		    atomicSub(problem->d_data_degrees+d_id,1);
		}
	    __syncthreads();
	}
	__syncthreads();
   }
  }

}; // PruneFunctor


/**
 * @brief Structure contains device functions in joining candidates into candidate subgraphs. 
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
struct CountFunctor
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
    if(s_id < d_id){
	for(int i=0; i<problem->edges_query; i++){
	    VertexId source=problem->froms_query[i];
	    VertexId dest=problem->tos_query[i];
	    if(problem->d_c_set[s_id+source*problem->nodes_data]==1 && problem->d_c_set[d_id+dest*problem->nodes_data]==1)	 
		return true;
	}    
	__syncthreads();
	return false;
    }
    else return false;
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
	for(int i=0; i<problem->edges_query; i++){
	    VertexId source=problem->froms_query[i];
	    VertexId dest=problem->tos_query[i];
	    if(problem->d_c_set[s_id+source*problem->nodes_data]==1 && problem->d_c_set[d_id+dest*problem->nodes_data]==1)	 
	    {
 	    	atomicAdd(problem->d_temp_keys+i,1);
	        break;
 	    }
	}
  }

}; // CountFunctor

/**
 * @brief Structure contains device functions in joining candidates into candidate subgraphs. 
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
struct CollectFunctor
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
	__shared__ int pos;
        for(int i=0; i<problem->edges_query; i++){
	    if(blockIdx.x * blockDim.x + threadIdx.x==0){ 
		problem->d_temp_keys[i+1]+=problem->d_temp_keys[i];
		pos=problem->d_temp_keys[i];
	    }
	    __syncthreads();
	    VertexId source=problem->froms_query[problem->d_query_edgeId[i]];
	    VertexId dest=problem->tos_query[problem->d_query_edgeId[i]];
	    if(problem->d_c_set[s_id+source*problem->nodes_data]==1 && problem->d_c_set[d_id+dest*problem->nodes_data]==1)
	    {
		problem->froms_data[pos-1]=s_id;
		problem->tos_data[pos-1]=d_id;
		atomicSub(&pos,1);
	    }
        }
  }

}; // CollectFunctor


}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
