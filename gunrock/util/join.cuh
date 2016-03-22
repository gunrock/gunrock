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

template<typename SizeT>
__global__ void Mark(
    const SizeT		edges,
    const SizeT* const 	keys,
          bool* 	flag)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < edges)
    {
	if(keys[x]%2==1) 
	    flag[x]=1;
	else flag[x]=0;
	x += STRIDE;
    }
}

template<typename SizeT>
__global__ void Update1(
    const unsigned long long* const    froms,
	  SizeT*          d_query_col,
    const SizeT           edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = d_query_col[0];
    if(x==0) printf("size1=%d\n", size);
    while(x < size)
    {
	if(x==(size-1) || froms[x]/edges_data < froms[x+1]/edges_data)
	{printf("x:%d, froms[x]/edges_data=%d\n", x, froms[x]/edges_data);
	    d_query_col[froms[x]/edges_data] = x+1;
	}
	x += STRIDE;
    }
}
 
template<typename SizeT>
__global__ void Update2(
          unsigned long long*        froms,
    const SizeT* const  d_query_col,
    const SizeT         edges_query,
    const SizeT         edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = d_query_col[edges_query-1];
    if(x==0) printf("size2=%d\n", size);
    while(x < size)
    {
	froms[x] %= edges_data;
	x += STRIDE;
    }
}


template<typename VertexId, typename SizeT>
__global__ void debug_before_init(
    const VertexId* const froms_data,
    const VertexId* const tos_data,
    const SizeT           edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < edges_data)
    {
	printf("%d: froms_data:%d tos_data:%d\n", x, froms_data[x], tos_data[x]);
	x += STRIDE;
    }
}

template<typename SizeT>
__global__ void debug_init(
    const bool* const d_c_set,
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

template<typename VertexId, typename SizeT>
__global__ void debug(
    const VertexId* const froms_data,
    const VertexId* const tos_data,
    const VertexId* const froms_query,
    const VertexId* const tos_query,
    const unsigned long long*    const tos,
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
    typename VertexId,
    typename SizeT>
__global__ void Join(
    const SizeT                 edges_data,
    const SizeT 		iter,
    const SizeT*    const       pos,
          SizeT*        	counts,
          bool*    	        flag,
    const VertexId* const	intersect,
    const VertexId* const       froms,
    const VertexId* const       tos,
    const unsigned long long*    const       d_data_degrees)
{
    counts[1]=counts[0]; // store the previous iteration's number of matches in counts[1]
    const SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
    const SizeT STRIDE = gridDim.x * blockDim.x;

    // x: is the number of matched middle results * edges_query
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) printf(" size=%d\n", size);
    if(x==0) printf("iter=%d pos[0]=%d, pos[1]=%d, pos[2]=%d, counts[0]=%d counts[1]=%d\n",iter, pos[0], pos[1], pos[2], counts[0], counts[1]);
//    if(iter==0 && x<pos[iter]) printf("Collected candidate edges: froms[%d]:%d->tos[%d]:%d\n", x, froms[x], x, tos[x]);

    while (x < size)
    {

	SizeT a = (x%((iter==0)?pos[iter]:counts[0]));
	SizeT b = ((iter==0)?pos[iter]:counts[0])+x/((iter==0)?pos[iter]:counts[0]); // edge iter+1 e_id

	SizeT offset = iter*(iter+1)/2;
	SizeT edge;
        unsigned long long edgeId = d_data_degrees[a];

	for(edge = 0; edge < iter+1; edge++)
	{
	    VertexId c = intersect[(offset+edge)*2];
	    VertexId d = intersect[(offset+edge)*2+1];

  	    if(c!=0)  
 	    {
	        if(c%2==1){
		    if(froms[edgeId % edges_data] != froms[d_data_degrees[b]])     {flag[x] = 0;  break;}
	        }
	        else{ 
		    if(tos[  edgeId % edges_data] != froms[d_data_degrees[b]])	   {flag[x] = 0;  break;}
	        }
	    }
	    else 
	    {
	        if(froms[d_data_degrees[b]] == froms[edgeId % edges_data] 
			|| froms[d_data_degrees[b]] == tos[edgeId % edges_data])
	        {
	      	    flag[x] = 0;
		    break;
		}    
	    }

	    if(d!=0)
	    {
	        if(d%2==1){
		    if(froms[edgeId % edges_data] != tos[d_data_degrees[b]])       {flag[x] = 0;  break;}
		}
	        else{
		    if(tos[  edgeId % edges_data] != tos[d_data_degrees[b]])       {flag[x] = 0;  break;}
		}
	    }
	    else
	    { 
	        if(tos[d_data_degrees[b]] == froms[edgeId % edges_data] 
			|| tos[d_data_degrees[b]] == tos[edgeId % edges_data])
	        {
	    	    flag[x] = 0; 
		    break;
		}
	    }

	    edgeId /= edges_data;
	}

	if(edge==iter+1)  flag[x]=1;
	x += STRIDE;

    }
} // Join Kernel


template<typename SizeT,
typename VertexId>
__global__ void debug_0(
    const bool* const flag,
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
/*
 * @brief Collect Kernel function.
 *
 * @tparam VertexID
 * @tparam SizeT
 * 
 * @param[in] edges	number of query edges
 * @param[in] iter	iteration number
 * @param[in] flag      index of each valid middle result
 * @param[in] froms     source node id for each candidate edge
 * @param[in] tos       destination node id for each candidate edge
 * @param[in] pos       address of the first candidate for each query edge
 */
template <
    typename VertexId,
    typename SizeT>
__global__ void Collect(
    const SizeT                 edges_query,
    const SizeT                 edges_data,
    const SizeT 		iter,
    const unsigned long long*    const 	d_data_degrees,
    const VertexId* const	froms_data,
    const VertexId* const	tos_data,
    	  unsigned long long* 	        flag, // store selected indices
    	  SizeT*     	        pos,
	  SizeT*		counts)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;

    //if(x==0) printf("iter=%d, counts[0]=%d, counts[1]=%d, edges = %d, counts[0]*edges=%d, pos[iter]=%d\n",iter, counts[0], counts[1], edges, counts[0]*edges, pos[iter]);

    while (x < counts[0])
    {

	SizeT a = (flag[x]%((iter==0)?pos[iter]:counts[1]));
	SizeT b = pos[iter]+flag[x]/((iter==0)?pos[iter]:counts[1]);

	__syncthreads();

    	flag[x] = d_data_degrees[b] * pow(edges_data, iter+1) + d_data_degrees[a];
	x += STRIDE;
    }

    while (x >= counts[0] && x < counts[0]+ pos[edges_query-1]-pos[iter]) 
    {
	flag[x] = d_data_degrees[x-counts[0]+pos[iter]];
	x += STRIDE;
    }
	
} // Collect Kernel



template<typename VertexId, typename SizeT>
__global__ void debug_1(
    const unsigned long long*    const froms,
    const VertexId* const froms_data,
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
	unsigned long long edgeId = froms[x];
	    for(int edge = 0; edge< edges_query; edge++)
	    {
		printf("edges[%d]: froms[%d]: %d -> tos[%d]: %d	\n", edge, x,froms_data[edgeId%edges_data],x,tos_data[edgeId%edges_data]);	
	    }
	x += STRIDE;
    }
}

template<typename SizeT>
__global__ void debug_before_select(
    const unsigned long long* const d_out,
    const SizeT        num_selected)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x==0) printf("# of elements: %d\n",num_selected);
    while (x < num_selected)
    {
        printf("element %d's flag:%lld\n",x,d_out[x]); 
	x += STRIDE;
    }
}

template<typename SizeT>
__global__ void debug_select(
    const unsigned long long* const d_out,
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

template<typename VertexId, typename SizeT>
__global__ void Label(
    const VertexId* const      froms_data,
    const VertexId* const      tos_data,
    const VertexId* const      froms_query,
    const VertexId* const      tos_query,
    const bool*     const      d_c_set,
          unsigned long long*  froms,
          unsigned long long*  d_data_degrees,
    const SizeT                edges_data,
    const SizeT                nodes_data,
    const SizeT                edges_query)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = froms[edges_data-1];
    if(x==0) printf("size=%d, edges_data/2=%d\n", size, edges_data/2);
    // size < edges_data/2
 //   if(x < edges_data) froms[x]=0;
    while (x < size)
    {
	#pragma unroll
	for(int i=0; i<edges_query; i++)
	{
	    VertexId src  = froms_query[i];
	    VertexId dest = tos_query[i];
	    if(d_c_set[froms_data[x] + src  * nodes_data] == 1 &&
	       d_c_set[tos_data[  x] + dest * nodes_data] == 1)
	       d_data_degrees[x + i*edges_data/2] = 1; 
	    
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
           
