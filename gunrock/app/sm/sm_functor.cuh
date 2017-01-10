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
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace sm {


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
  typename Problem,
  typename _LabelT = VertexId>
struct SMInitFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;
 
    /** 
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
	// d_query_col stores the query node labels; d_in stores the data node labels
        #pragma unroll
	    for(VertexId i=0; i < d_data_slice -> nodes_query; i++)
            {
   	        if (((VertexId)d_data_slice -> d_in[s_id] == d_data_slice->d_query_col[i]) && 
		        (d_data_slice -> d_row_offsets[s_id+1] - d_data_slice -> d_row_offsets[s_id] >=
			(d_data_slice -> d_query_row[     i+1] - d_data_slice -> d_query_row[     i])))
		{
			d_data_slice -> d_c_set[s_id] |= 1 << i;
			
		}
	    }
	return true;
    }

    /** 
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set distance to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
	    return;
    }
}; //SMInitFunctor

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
  typename Problem,
  typename _LabelT = VertexId>
struct SMFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {

 	    bool res = false;

	    for (SizeT i=0; i < d_data_slice -> edges_query; i++)
	    {
		unsigned int from = d_data_slice -> froms_query[i];
		unsigned int to   = d_data_slice -> tos_query[i];
		res = ((d_data_slice -> d_c_set[s_id] >> from) % 2 == 1) && 
		      ((d_data_slice -> d_c_set[d_id] >> to  ) % 2 == 1) && 
		      (s_id < d_id); // only for triangle counting

		if(res) break;
	    }
	    // use froms to store if the edge is kept or not
	    d_data_slice -> d_in[edge_id] = (res) ? (VertexId)1 : 0;
	    return res; 
    }

    /** 
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set distance to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
	    return;
    }

}; // SMFunctor


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
    typename Problem>
struct EdgeWeightFunctor
{
    typedef typename Problem::DataSlice DataSlice;
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
        VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
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
        VertexId s_id,  VertexId d_id, DataSlice *d_data_slice,
        VertexId e_id = 0, VertexId e_id_in = 0)
    {
        // add src node weight with dest node weight to get edge weight
        d_data_slice -> edge_weight[e_id_in] = 
            d_data_slice -> temp_keys[s_id] + d_data_slice -> temp_keys[d_id];
    }
}; // EdgeWeightFunctor


}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
