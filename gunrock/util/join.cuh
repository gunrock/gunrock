// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file: join.cuh
 *
 */

#pragma once

namespace gunrock {
namespace util {



template<typename Value, typename SizeT>
__global__ void Update(
    	  Value*          indices,
	  Value*          pos,
    const SizeT           edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = pos[0];
    while(x < size)
    {
	if(x==(size-1) || indices[x]/edges_data < indices[x+1]/edges_data)
	{ 
	    pos[indices[x]/edges_data] = x+1;
	}
	__syncthreads();
	indices[x] %= edges_data;
	x += STRIDE;
    }
}
 

__global__ void debug(
    unsigned long long* counts)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) printf("======Num of subgraph matches: %d=======\n", counts[0]);
}

template<typename VertexId, typename SizeT>
__global__ void MaskOut(
    const SizeT  nodes_query,
    const SizeT  size,
    VertexId* d_partial)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while(x<size){
        if(d_partial[x]<0){
            int mode = x%nodes_query;
            #pragma unroll
            for(int idx = 0; idx < nodes_query; idx++){
                if(x+idx-mode>=size) printf("====Error:mask out mem access=====\n");
                d_partial[x+idx-mode] = -1;
            }
        }
        x += STRIDE;
    }
}

template<typename VertexId, typename SizeT>
__global__ void WriteToPartial(
    const SizeT* const src_node,
    const SizeT* const dest_node,
    const SizeT* const flag, // only valid when level is greater than 2
    const SizeT* const count,
    const SizeT  const iter,
    const SizeT  const flag_size, // only valid when level is greater than 2
    const SizeT  const nodes_query,
    VertexId*          d_partial)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT bound = count[0];
    if(iter==0){
    if(x==0)    printf("iteration:%d, bound:%d\n", iter, bound);
        while(x<bound){
    //        printf("%d: src:%d, dest:%d\n", x, src_node[x], dest_node[x]);
            d_partial[x*nodes_query] = src_node[x];
            d_partial[x*nodes_query+iter+1] = dest_node[x]; 
            x += STRIDE;
        }
    }
    else{
    if(x==0) printf("iteration:%d, bound:%d\n", iter, flag_size);
        while(x<flag_size){
            if(flag[x]>0)
                d_partial[x*nodes_query+iter+1] = flag[x]; 
            x += STRIDE;
        }
    }
}
/*
 * @brief Join Kernel function.
 *
 * @tparam VertexID
 * @tparam SizeT
 * 
 * @param[in] edges     number of query edges
 * @param[in] pos       address of the first candidate for each query edge
 * @param[in] froms     source node id for each candidate edge
 * @param[in] tos       destination node id for each candidate edge
 * @param[in] flag      flags to show positions of intersection nodes
 * @param[out]froms_out output edge list source node id
 * @param[out]tos_out   output edge list destination node id
 */
/*template <
    typename VertexId, typename SizeT, typename Value>
__global__ void Join(
    const SizeT*                 row_offset,
    const SizeT*                 column_indices,
    const VertexId* const        query_ng,// query node sequence
    const bool*     const        isValid_data, // mark valid data nodes
    const SizeT     const        query_nodes,
    const SizeT     const        data_nodes,
          VertexId*              partial_results, 
          unsigned long long*  	 counts,// store the number of matches
    	  VertexId*              results,// store candidate edges in query edge order
          SizeT                  depth) // current recursive level
{
    unsigned long long size = pos[0];
    // #intermediate results = products of all #candidate of edges of query edges
    for(int i=0; i<edges_query-1; i++)
        size *= (pos[i+1] - pos[i]);
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned long long edge_id[3]; //specific for triangles 
    SizeT iter;
    SizeT offset;
    SizeT edge;
    unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) printf("====Number of intermediate results:%llu, STRIDE size:%llu========\n", size, STRIDE);
    while (x < size)
    {
	unsigned long long id = x;
	for(iter = edges_query-1; iter > 0; iter--)    // query edge 1+'s candidate edge ids
	{
	    edge_id[iter] = pos[iter-1] + id % (pos[iter]-pos[iter-1]);
	    id /= (pos[iter]-pos[iter-1]);
	    edge_id[iter] = edges[edge_id[iter]];
	}
	edge_id[iter] = edges[id];
	#pragma unroll
	for(iter=0; iter < edges_query-1; iter++)
	{
	   offset = iter*(iter+1)/2;
	   #pragma unroll 
	   for(edge = 0; edge < iter+1; edge++)
	   {
	    if(edge_id[iter+1]==edge_id[edge]) break; //two edges in one combination have the same id

	    VertexId c = intersect[(offset+edge)*2];
	    VertexId d = intersect[(offset+edge)*2+1];
  	    if(c!=0)  
 	    {
	        if(c%2==1){
		    if(froms[edge_id[edge]] != froms[edge_id[iter+1]])     break;
	        }
	        else{ 
		    if(tos[  edge_id[edge]] != froms[edge_id[iter+1]])	   break;
	        }
	    }
	    else 
	    {
	        if(froms[edge_id[iter+1]] == froms[edge_id[edge]] 
			|| froms[edge_id[iter+1]] == tos[edge_id[edge]])   break;
	    }

	    if(d!=0)
	    {
	        if(d%2==1){
		    if(froms[edge_id[edge]] != tos[edge_id[iter+1]])       break;
		}
	        else{
		    if(tos[  edge_id[edge]] != tos[edge_id[iter+1]])       break;
		}
	    }
	    else
	    { 
	        if(tos[edge_id[iter+1]] == froms[edge_id[edge]] 
			|| tos[edge_id[iter+1]] == tos[edge_id[edge]])     break;
	    }

	   }

	   if(edge!=iter+1) // the current join fails
	       break;
     }
     if(iter == edges_query - 1)  // the current join succeeds for all query edges
     {
        #pragma unroll
	for(iter = 0; iter<edges_query; iter++)
	    output[counts[0]*edges_query + iter] = edge_id[iter];

	atomicAdd(counts, 1);
     }
     x += STRIDE;

    }
} // Join Kernel
*/

template<typename VertexId, typename SizeT, typename Value>
__global__ void Label(
    const Value*    const      froms_data,
    const VertexId* const      tos_data,
    const VertexId* const      froms_query,
    const VertexId* const      tos_query,
    const VertexId* const      d_c_set,
          Value* 	       label,
    const VertexId* const      d_query_row,
    const SizeT                edges_data,
    const SizeT                edges_query)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = d_query_row[0];
    while (x < size)
    {
	#pragma unroll
	for(int i=0; i<edges_query; i++)
	{
	    VertexId src  = froms_query[i];
	    VertexId dest = tos_query[i];
	    if( ((d_c_set[froms_data[x]] >> src ) % 2 == 1) &&
	        ((d_c_set[tos_data[  x]] >> dest) % 2 == 1) )
	       label[x + i*edges_data/2] = 1; 
	    
	}
	x += STRIDE;
    }   
} // Label Kernel

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variable
// mode:c++
// c-file-style: "NVIDIA"
// End:
           
