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
          SizeT* 	flag)
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
__global__ void Update(
    const SizeT 	iter,
    const SizeT		edges_query,
    const SizeT		edges_data,
    const SizeT* const  pos,
    	  SizeT*	d_temp_keys,
    	  SizeT*	d_query_col)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < edges_data)
    {
	d_temp_keys[x]>>=1;    
        if(x==0){
     	    for(SizeT i=edges_query-iter-1; i<edges_query-1; i++) d_query_col[i]=d_query_col[i+1];
     	    d_query_col[edges_query-1] = d_query_col[edges_query-2] + pos[edges_data-1];// number of candidate edges for query edge iter
     	}
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

template<typename VertexId, typename SizeT>
__global__ void debug_label(
   const VertexId* const d_temp_keys,
   const SizeT edges)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < edges){
        if(d_temp_keys[x]!=0) printf("e_id:%d  edge_label:%d \n", x,d_temp_keys[x]);
	x += STRIDE;
    }
}

template<typename VertexId, typename SizeT>
__global__ void debug(
    const int iter,
    const VertexId* const froms_data,
    const VertexId* const tos_data,
    const VertexId* const froms_query,
    const VertexId* const tos_query,
    const SizeT*    const d_query_col,
    const SizeT	    edges_query,
    const SizeT     edges_data)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < d_query_col[edges_query-1])
    {
    	if(x==0) {for(int i=0; i<edges_query; i++) printf("%d ", d_query_col[i]); printf("\n");}
	#pragma unroll
	for(int j=0; j<edges_query; j++){
	    if(j==0 && x<d_query_col[j])
		 printf("thread:%d edges:%d Edge %d: %d -> %d 's candidate:	%d -> %d\n", 
					x, d_query_col[j], j,froms_query[j], 
					tos_query[j], froms_data[x], tos_data[x]);
	    else if(j>0 && x<d_query_col[j] && x>=d_query_col[j-1]) 
		printf("thread:%d edges:%d Edge %d: %d -> %d 's candidate:	%d -> %d\n", 
					x, d_query_col[j], j,froms_query[j],
					tos_query[j], froms_data[x], tos_data[x]);
	}
	x += STRIDE;
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
template <
    typename VertexId,
    typename SizeT>
__global__ void Join(
    const SizeT                 edges,
    const SizeT 		iter,
    const SizeT*    const       pos,
          SizeT*        	counts,
          bool*    	        flag,
    const VertexId* const	intersect,
    const VertexId* const       froms,
    const VertexId* const       tos,
          VertexId*             froms_out,
          VertexId*             tos_out)
{
    counts[1]=counts[0];
    const SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
    const SizeT STRIDE = gridDim.x * blockDim.x;

    // x: is the number of matched middle results * edges_query
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) printf(" size=%d\n", size);
   // if(x==0) printf("iter=%d pos[0]=%d, pos[1]=%d, pos[2]=%d, counts[0]=%d counts[1]=%d\n",iter, pos[0], pos[1], pos[2], counts[0], counts[1]);
//    if(iter==0 && x<pos[iter]) printf("Collected candidate edges: froms[%d]:%d->tos[%d]:%d\n", x, froms[x], x, tos[x]);

    while (x < size)
    {

	SizeT a = (x%((iter==0)?pos[iter]:counts[0]))*edges;
	SizeT b = pos[iter]+x/((iter==0)?pos[iter]:counts[0]); // edge iter+1 e_id
//if(a/edges==2311 && b==2312+pos[iter]) printf("here: iter:%d %d->%d %d->%d stride:%d\n", iter, froms[a/edges],tos[a/edges], froms[b], tos[b], x/STRIDE);
//if(a/edges==8423) printf("682: froms[%d]=%d tos[%d]=%d  b=%d\n", a/edges, froms[a/edges], a/edges, tos[a/edges],b);
//if(b==8424+pos[iter]) printf("683: froms[%d]=%d tos[%d]=%d a:%d\n", b, froms[b], b, tos[b],a/edges);

	if(iter==0){ 
	// In the first iteration, put pos[0] candidate edges for edge 0 to out_put list
	    	froms_out[a]=froms[a/edges];
	    	tos_out[a]=tos[a/edges];
	}
	VertexId c = intersect[iter*2];
	VertexId d = intersect[iter*2+1];
	//if(x==0) printf("iter:%d, c=%d, d=%d\n",iter,c,d);
//if(a/edges==2311 && b==2312+pos[iter]) printf("here1: iter:%d %d->%d %d->%d stride:%d\n", iter, froms[a/edges],tos[a/edges], froms[b], tos[b], x/STRIDE);

	if(c!=0)  
 	{
	    SizeT edge = c/2; // the edge that edge iter has intersect node with
	    if(c%2==1){
		if(froms_out[a+edge]!=froms[b]) {flag[x]=0;  x += STRIDE; continue;}
	    }
	    else{ 
		if(tos_out[a+edge-1]!=froms[b])	{ flag[x]=0;  x += STRIDE; continue;}
	    }
	}
	else 
	{
	    SizeT edge;
	    for(edge = 0; edge<iter+1; edge++){
	        if(froms[b]==froms_out[a+edge] || froms[b]==tos_out[a+edge])
	        {
	      	    flag[x]=0;
		    break;
		}    
	    }
	    if(edge!=iter+1) {x+=STRIDE; continue;}
	}

//if(a/edges==2311 && b==2312+pos[iter]) printf("here2: iter:%d %d->%d %d->%d stride:%d\n", iter, froms[a/edges],tos[a/edges], froms[b], tos[b], x/STRIDE);

	if(d!=0)
	{
	    SizeT edge = d/2;
	    if(d%2==1){
		if(froms_out[a+edge]!=tos[b])   {flag[x]=0;  x += STRIDE; continue;}}
	    else{
		if(tos_out[a+edge-1]!=tos[b])   {flag[x]=0;  x += STRIDE; continue;}}
	}
	else
	{ 
//if(a/edges==2311 && b==2312+pos[iter]) printf("here2.5: iter:%d %d->%d %d->%d stride:%d\n", iter, froms[a/edges],tos[a/edges], froms[b], tos[b], x/STRIDE);
	    SizeT edge;
	    for(edge=0; edge<iter+1; edge++){
	        if(tos[b]==froms_out[a+edge] || tos[b]==tos_out[a+edge])
	        {
	    	    flag[x]=0; 
		    break;
		}
	    }
	    if(edge!=iter+1) {x+=STRIDE; continue;}
	}

	flag[x]=1;
//if(a/edges==2310 && b==2311+pos[iter]) printf("here3: iter:%d %d->%d %d->%d flag[%d]=%d, stride:%d\n", iter, froms[a/edges],tos[a/edges], froms[b], tos[b], x/edges, flag[x/edges], x/STRIDE);
//if(iter==0) printf("STRIDE:%d 	froms[%d]:%d -> tos[%d]:%d(%d:%d %d:%d)	froms[%d]:%d -> tos[%d]:%d   flag[%d]:%d\n",x/STRIDE, a,froms_out[a],a,tos_out[a], a/edges, froms[a/edges], a/edges, tos[a/edges], b,froms[b],b,tos[b], x,flag[x]);
//if(iter==1) printf("froms_out[%d]:%d -> tos_out[%d]:%d	froms_out[%d]:%d -> tos_out[%d]:%d	froms[%d]:%d -> tos[%d]:%d   flag[%d]:%d\n",a,froms_out[a],a,tos_out[a], a+1, froms_out[a+1], a+1, tos_out[a+1], b,froms[b],b,tos[b], x/edges,flag[x/edges]);
//if(iter==1) printf("Join: froms[%d]:%d -> tos[%d]:%d\n",a+x%edges, froms_out[a+x%edges], a+x%edges, tos_out[a+x%edges]);
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
    const SizeT edges,
    const SizeT iter)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
    if(x==0) printf(" size=%d\n", size);
    while (x < size)
    {
	SizeT a = (x%((iter==0)?pos[iter]:counts[0]));
	SizeT b = pos[iter]+x/((iter==0)?pos[iter]:counts[0]); // edge iter+1 e_id
__syncthreads();
        if(flag[x]!=0) printf("After Join: froms[%d]:%d -> tos[%d]:%d  froms[%d]:%d -> tos[%d]:%d flag[%d]:%d\n",a, froms[a], a, tos[a], b, froms[b], b, tos[b], x,flag[x]);
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
    const SizeT                 edges,
    const SizeT 		iter,
    const SizeT*    const 	flag,
    const VertexId* const	froms_data,
    const VertexId* const	tos_data,
    	  VertexId* 	        froms,
    	  VertexId* 	        tos,
    	  SizeT*     	        pos,
	  SizeT*		counts)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x==0) printf("iter=%d, counts[0]=%d, counts[1]=%d, edges = %d, counts[0]*edges=%d, pos[iter]=%d\n",iter, counts[0], counts[1], edges, counts[0]*edges, pos[iter]);

    while (x < counts[0] * edges)
    {
//    printf("x:%d counts[0]=%d, counts[1]=%d, edges = %d, counts[0]*edges=%d\n", x, counts[0], counts[1], edges, counts[0]*edges);
	//SizeT a = (x/edges%((iter==0)?pos[iter]:counts[1]))*edges+x%edges;
	//SizeT b = pos[iter]+x/(edges*((iter==0)?pos[iter]:counts[0])); // edge iter+1 e_id

	SizeT a = (flag[x/edges]%((iter==0)?pos[iter]:counts[1]))*edges+x%edges;
	SizeT b = pos[iter]+flag[x/edges]/((iter==0)?pos[iter]:counts[1]);
//if(iter==1) printf("Collect: froms[%d]:%d -> tos[%d]:%d x:%d\n", a, froms[a], a, tos[a],x);
// printf("0:large group:%d small group: %d  iter:%d froms_out[%d]:%d->tos_out[%d]:%d flag[%d]=%d\n",flag[x/edges]/((iter==0)?pos[iter]:counts[1]), flag[x/edges]%((iter==0)?pos[iter]:counts[1]),iter,a,froms[a],a,tos[a],x/edges,flag[x/edges]);
    	VertexId from = froms[a];
    	VertexId to = tos[a];
//if(iter==1) printf("Collect: froms[%d]:%d(%d) -> tos[%d]:%d(%d)   x:%d\n", a, froms[a],from, a, tos[a],to, x);
	__syncthreads();
//if(iter==1) printf("Collect: froms[%d]:%d -> tos[%d]:%d x:%d\n", a, from, a, to,x);
// printf("1:large group:%d small group: %d  iter:%d froms_out[%d]:%d->tos_out[%d]:%d flag[%d]=%d\n",flag[x/edges]/((iter==0)?pos[iter]:counts[1]), flag[x/edges]%((iter==0)?pos[iter]:counts[1]),iter,a,from,a,to,x/edges,flag[x/edges]);

	if(x%edges==iter+1){
	    froms[x] = froms_data[b];
	    tos[x] = tos_data[b];
//if(iter==1) printf("iter:%d edge:%d	added in by this iter froms[%d](froms_data[%d]):%d -> tos[%d](tos_data[%d]):%d	flag[%d]:%d\n",iter,x%edges,x,b,froms_data[b],x,b, tos_data[b],x/edges,flag[x/edges]);
	}
	else{
	    froms[x]=from;
	    tos[x]=to;
//__syncthreads();
//if(iter==1) printf("iter:%d edge:%d from prev iter	froms[%d]:%d -> tos[%d]:%d(froms[%d]:%d->tos[%d]:%d)	flag[%d]:%d  x:%d\n",iter,x%edges,x,froms[x],x,tos[x],a,from, a, to, x/edges,flag[x/edges],x);
	}
__syncthreads();
//if(iter==1) printf("iter:%d edge:%d stride:%d 	froms[%d]:%d -> tos[%d]:%d	flag[%d]:%d   x:%d\n",iter,x%edges,x/STRIDE,x,froms[x],x,tos[x],x/edges,flag[x/edges],x);

	x += STRIDE;
    } 
//    __syncthreads();
//    counts[1] = counts[0];
	
} // Collect Kernel



template<typename VertexId, typename SizeT>
__global__ void debug_1(
    const VertexId* const froms,
    const VertexId* const tos,
    const SizeT*    const flag,
    const SizeT*    const pos,
    const SizeT*    const counts,
    const SizeT	    edges_query)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < counts[0]*edges_query)
    {
        if(x==0) printf("Number of matched tris: %d\n", counts[0]);
	    printf("edges[%d]: froms[%d]: %d -> tos[%d]: %d	flag[%d]: %d\n", x%edges_query, x,froms[x],x,tos[x],x/edges_query,flag[x/edges_query]);	
	x += STRIDE;
    }
}

template<typename InputT, typename SizeT, typename FlagT>
__global__ void debug_select(
    const SizeT  const  iter,
    const InputT* const d_in,
    const SizeT         nodes,
    const SizeT         edges,
    const FlagT* const flag,
    const SizeT* const d_out,
    const SizeT* const num_selected)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x==0) printf("iter:%d # of selected elements: %d\n",iter, num_selected[0]);
//    while (x < num_selected[0])
//    {
//        printf("elements with flag 1: d_out[%d]:%d\n",x,d_out[x]); 
//	x += STRIDE;
//    }
}
/*
template<typename FlagT, typename VertexId, typename SizeT>
__global__ void flag(
     FlagT*           d_flag,
    const SizeT       num_elements)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
	if(x%2==0) d_flag[x]=1;
	else d_flag[x]=0;
	x += STRIDE;
    }   
}*/

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variable
// mode:c++
// c-file-style: "NVIDIA"
// End:
           
