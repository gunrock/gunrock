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



template<typename SizeT, typename T>
__global__ void Update(
    	  T*              indices,
	  SizeT*          pos,
    const SizeT           edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = pos[0];
//    if(x==0) printf("size1=%d\n", size);
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
 

template<typename SizeT>
__global__ void debug_init(
    const char* const d_c_set,
    const SizeT        nodes_query,
    const SizeT        nodes_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < nodes_query * nodes_data)
    {
	if(d_c_set[x]==1) printf("node %d's candidate node: %d\n", x/nodes_data, x%nodes_data);
	x += STRIDE;
    }
}

template<typename SizeT>
__global__ void debug_label(
   const unsigned long long* const d_temp_keys,
   const SizeT edges)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < edges){
        printf("e_id:%d  edge_label:%lld \n", x,d_temp_keys[x]);
	x += STRIDE;
    }
}

template<typename VertexId, typename SizeT, typename Value, typename T>
__global__ void debug(
    const Value*    const froms_data,
    const VertexId* const tos_data,
    const VertexId* const froms_query,
    const VertexId* const tos_query,
    const T*        const tos,
    const SizeT*    const d_query_col,
    const SizeT	    edges_query,
    const SizeT     edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) {for(int i=0; i<edges_query; i++) printf("%d ", d_query_col[i]); printf("\n");}
  /*  __syncthreads();
    while (x < d_query_col[edges_query-1])
    {
	#pragma unroll
	for(int j=0; j<edges_query; j++){
	    if(j==0 && x<d_query_col[j])
		 printf("thread:%d edges:%d candidate e_id:%lld Edge %d: %d -> %d 's candidate:	%d -> %d\n", 
					x, d_query_col[j], tos[x], j,froms_query[j], 
					tos_query[j], froms_data[tos[x]], tos_data[tos[x]]);
	    else if(j>0 && x<d_query_col[j] && x>=d_query_col[j-1]) 
		printf("thread:%d edges:%d candidate e_id:%lld Edge %d: %d -> %d 's candidate:	%d -> %d\n", 
					x, d_query_col[j], tos[x], j,froms_query[j],
					tos_query[j], froms_data[tos[x]], tos_data[tos[x]]);
	}
	x += STRIDE;
    }*/
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
template <
    typename VertexId, typename SizeT, typename Value, typename T>
__global__ void Join(
    const SizeT                 edges_data,
    const SizeT                 edges_query,
    const SizeT*    const       pos,// store the start positions for each query edge's candidates
          SizeT*        	counts,// store the number of matches
    const VertexId* const	intersect,
    const Value*    const       froms,
    const VertexId* const       tos,
    const T*        const       edges,
    	  VertexId*             output)  // store candidate edges in query edge order
{
    unsigned long long size = pos[0];

//printf("pos[0]:%d, pos[1]-pos[0]:%d, pos[2]-pos[1]:%d\n", pos[0], pos[1]-pos[0], pos[2]-pos[1]);

    for(int i=0; i<edges_query-1; i++)
        size *= (pos[i+1] - pos[i]);
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned long long edge_id[3];
    SizeT iter;
    SizeT offset;
    SizeT edge;
    unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
if(x==0) printf("===size=%lld===\n",size);
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
//printf("edge_id[0]:%lld, edge_id[1]:%lld, edge_id[2]:%lld\n", edge_id[0], edge_id[1], edge_id[2]);
	for(iter=0; iter < edges_query-1; iter++)
	{
	   offset = iter*(iter+1)/2;
	   
	   for(edge = 0; edge < iter+1; edge++)
	   {
	    if(edge_id[iter+1]==edge_id[edge]) break; //two edges in one combination have the same id

	    VertexId c = intersect[(offset+edge)*2];
	    VertexId d = intersect[(offset+edge)*2+1];
//printf("iter:%d, edge:%d, iter+1:%d, c:%d, d:%d, edge_id[%d]:%d, edge_id[%d]:%d\n", iter,edge,iter+1, c, d, edge, edge_id[edge], iter+1, edge_id[iter+1]);
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
//printf("=====current join succeeds====\n");
	for(iter = 0; iter<edges_query; iter++)
	    output[counts[0]*edges_query + iter] = edge_id[iter];

	atomicAdd(counts, 1);
     }
     x += STRIDE;

    }
} // Join Kernel


template<typename SizeT,
typename VertexId>
__global__ void debug_0(
    const char* const flag,
    const SizeT* const pos,
    const SizeT* const counts,
    const VertexId* const froms,
    const VertexId* const tos,
    const VertexId* const froms_out,
    const VertexId* const tos_out,
    const SizeT edges,
    const SizeT iter)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
    if(x==0) printf(" size=%d\n", size);
    while (x < size)
    {
	SizeT a = (x%((iter==0)?pos[iter]:counts[0]))*edges;
	SizeT b = pos[iter]+x/((iter==0)?pos[iter]:counts[0]); // edge iter+1 e_id
        if(flag[x]!=0) printf("After Join: froms[%d]:%d -> tos[%d]:%d  froms[%d]:%d -> tos[%d]:%d flag[%d]:%d\n",froms_out[a], froms[froms_out[a]], froms_out[a], tos[froms_out[a]], tos_out[b], froms[tos_out[b]], tos_out[b], tos[tos_out[b]], x,flag[x]);
	x += STRIDE;
    }
}



template<typename VertexId, typename SizeT, typename Value, typename T>
__global__ void debug_1(
    const T*        const froms,
    const Value*    const froms_data,
    const VertexId* const tos_data,
    const SizeT*    const pos,
    const SizeT*    const counts,
    const SizeT	    edges_data,
    const SizeT	    edges_query)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < counts[0])
    {
        if(x==0) printf("Number of matched tris: %d\n", counts[0]);
	VertexId edgeId = froms[x];
	    for(int edge = 0; edge< edges_query; edge++)
	    {
		printf("edges[%d]: froms[%d]: %d -> tos[%d]: %d	\n", edge, x,froms_data[edgeId%edges_data],x,tos_data[edgeId%edges_data]);	
	    }
	x += STRIDE;
    }
}

template<typename SizeT, typename VertexId>
__global__ void debug_before_select(
    const VertexId* const d_out,
    const SizeT        num_selected)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x==0) printf("# of elements: %d\n",num_selected);
    while (x < num_selected)
    {
        printf("element %d's flag:%d\n",x,d_out[x]); 
	x += STRIDE;
    }
}

template<typename SizeT, typename T>
__global__ void debug_select(
    const T*     const d_out,
    const SizeT* const num_selected)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x==0) printf("# of selected elements: %d\n",num_selected[0]);
    while (x < num_selected[0])
    {
        printf("elements with flag 1: d_out[%d]:%d\n",x,d_out[x]); 
	x += STRIDE;
    }
}



template<typename VertexId, typename SizeT, typename Value>
__global__ void Label(
    const Value*    const      froms_data,
    const VertexId* const      tos_data,
    const VertexId* const      froms_query,
    const VertexId* const      tos_query,
    const VertexId* const      d_c_set,
          VertexId*            label,
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
           
