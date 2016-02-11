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

// Loop the first two kernels for a few iterations in order to prune 
// candidate nodes based on degrees and labels
// Rember to initialized d_temp_keys before every iteration
/////////////////////////////////////////////////////////////////////

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
    	if(node>=0 && node<problem->nodes_data)	return true;
	else return false;
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
    #pragma unroll
	for(VertexId i=0; i < problem->nodes_query; i++){
		if ((problem->d_data_labels[node] == problem->d_query_labels[i]) && 
		    (problem->d_data_degrees[node] >= problem->d_query_degrees[i]))
		{
			problem->d_c_set[node+i*problem->nodes_data]=1;
			problem->d_temp_keys[node]=1;
		}
		else
			problem->d_c_set[node+i*problem->nodes_data]=0;
		
	}
    }
}; // SMInitFunctor


/**
 * @brief Structure contains device functions in computing valid degree for each node. 
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
struct UpdateDegreeFunctor
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
    if(s_id<problem->nodes_data && d_id<problem->nodes_data)
    {
	#pragma unroll
	for(VertexId i=0; i < problem->nodes_query; i++)
	    if(problem->d_c_set[s_id+i*problem->nodes_data]==1)
	    	return true;
    }
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
	if(problem->d_temp_keys[d_id]==0) 
	   atomicSub(problem->d_data_degrees+s_id,1);
  }
}; // UpdateDegreeFunctor
/// End of recursion
//==================================================================================
// EdgeWeightFunctor currently not used
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
//==========================================================================================
// Loop the following functor alone for several iterations
////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in pruning query node candidates 
 * based on the number of neighbors of each candidate node. 
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
    if(s_id >= 0 && d_id >= 0 && s_id < problem->nodes_data && d_id < problem->nodes_data)
	return true;
    else return false;
  }

 /**
   * @brief Forward Advance Kernel apply function.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id Output edge index (index of row_offset idx)
   * @param[in] e_id_in Input edge index
   */
  static __device__ __forceinline__ void ApplyEdge(
    VertexId s_id,  VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
  {
    //if(e_id<problem->edges_data)	printf("e_id:%d s_id:%d->d_id:%d\n",e_id, s_id, d_id);

	for(VertexId i=0; i<problem->nodes_query; i++)
	{
	    if(problem->d_c_set[s_id+i*problem->nodes_data]!=0)
	    {

		for(SizeT offset = problem->d_query_row[i]; offset<problem->d_query_row[i+1]; ++offset)
		{
		    VertexId j = problem->d_query_col[offset];
		    if(problem->d_c_set[d_id + j*problem->nodes_data]!=0)
			problem->d_c_set[s_id + i*problem->nodes_data]+=1;
		}		
		__syncthreads();

		// d_c_set[i,s_id] now stores the number of neighbors 
		// of s_id corresponding to i's neighbors+1
   		// If the prior is less than the latter s_id is not a candidate of i 
		if(problem->d_c_set[s_id+i*problem->nodes_data]-1 <
		   problem->d_query_row[i+1]-problem->d_query_row[i])
		    problem->d_c_set[s_id + i*problem->nodes_data]=0;
		else 
		    problem->d_c_set[s_id + i*problem->nodes_data]=1;
			
	    }
	}


  }

}; // PruneFunctor


/**
 * @brief Structure contains device functions in labeling candidate edges based on query edge ids. 
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
struct LabelEdgeFunctor
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
   // if(e_id<problem->edges_data)	printf("e_id:%d s_id:%d->d_id:%d\n",e_id, s_id, d_id);
    if(s_id < d_id && e_id<problem->edges_data) 
	return true;
    
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
	    if(	problem->d_c_set[s_id+source*problem->nodes_data]==1 && 
		problem->d_c_set[d_id+dest*problem->nodes_data]==1)
		problem->d_temp_keys[e_id]|=(1<<i); // label the candidate edge with the index of its query edge plus 1
	}
  }

}; // LabelEdgeFunctor

/**
 * @brief Structure contains device functions in collecting candidate edges 
 * corresponding to the same query edge. 
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
    if(s_id<d_id && e_id<problem->edges_data)
	return true;
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
	if(problem->d_data_degrees[e_id]>=1 && (e_id==0 || problem->d_data_degrees[e_id]>problem->d_data_degrees[e_id-1])){
	    problem->froms_data[problem->d_data_degrees[e_id]-1 +
				problem->d_query_col[problem->edges_query-2]]=s_id;
	    problem->tos_data[problem->d_data_degrees[e_id]-1 + 
			      problem->d_query_col[problem->edges_query-2]]=d_id;
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
