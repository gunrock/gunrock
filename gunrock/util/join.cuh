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
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<edges)
    {
	if(keys[x]%2==1) {
	    flag[x]=1;

	}
	else flag[x]=0;

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
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < edges_data)
    	d_temp_keys[x]>>=1;
    else if(x==edges_data){
	for(SizeT i=edges_query-iter-1; i<edges_query-1; i++) d_query_col[i]=d_query_col[i+1];
    	d_query_col[edges_query-1] = d_query_col[edges_query-2] + pos[edges_data-1];// number of candidate edges for query edge iter
    }
}

template<typename SizeT>
__global__ void debug_init(
    const SizeT* const d_c_set,
    const SizeT        nodes_query,
    const SizeT        nodes_data)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<nodes_query * nodes_data)
    {
	if(d_c_set[x]==1) printf("node %d's candidate node: %d\n", x/nodes_data, x%nodes_data);
    } 
}

template<typename SizeT>
__global__ void debug_label(
   const SizeT* const d_temp_keys,
   const SizeT edges)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<edges) printf("e_id:%d  edge_label:%d \n", x,d_temp_keys[x]);
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
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) {for(int i=0; i<edges_query; i++) printf("%d ", d_query_col[i]); printf("\n");}
    if(x<d_query_col[edges_query-1]){
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
    const SizeT*    const	counts,
          SizeT*    	        flag,
    const VertexId* const	intersect,
    const VertexId* const       froms,
    const VertexId* const       tos,
          VertexId*             froms_out,
          VertexId*             tos_out)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

    // x: is the number of matched middle results * edges_query
    SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
//if(x==0) printf("iter:%d counts[0]:%d, edges:%d size:%d\n",iter,counts[0],edges,size);
    if(x>=0 && x<size*edges)
    {
	SizeT a = (x/edges%((iter==0)?pos[iter]:counts[0]))*edges;
	SizeT b = pos[iter]+x/(edges*((iter==0)?pos[iter]:counts[0])); // edge iter+1 e_id

	if(iter==0){ // In the first iteration, put pos[0] candidate edges for edge 0 to out_put list
	   // if(x%edges==0 && x<edges*pos[iter]) 
	   // {
	   // 	froms_out[x]=froms[x/edges%pos[iter]];
	   // 	tos_out[x]=tos[x/edges%pos[iter]];
	    	froms_out[a]=froms[x/edges%pos[iter]];
	    	tos_out[a]=tos[x/edges%pos[iter]];
//printf("The first candidate edges: group: %d  froms_out[%d]:%d->tos_out[%d]:%d\n",x/edges,a,froms_out[a],a,tos_out[a]);
   	   // }
	}
	    __syncthreads();
	//else {
	//    froms_out[x]=froms_out[x%(counts[0]*edges)];
	//    tos_out[x]=tos_out[x%(counts[0]*edges)];
	//}

//if(iter==0 && x<edges*size/(pos[iter+1]-pos[iter])) printf("group: %d  iter:%d froms_out[%d]:%d->tos_out[%d]:%d \n",x/edges,iter,x,froms_out[x],x,tos_out[x]);
	VertexId c = intersect[iter*2];
	VertexId d = intersect[iter*2+1];
//	if(x==0) printf("iter:%d, c=%d, d=%d\n",iter,c,d);
	{if(c!=0)  
 	{
	    SizeT edge = c/2; // the edge that edge iter has intersect node with
	    if(c%2==1){
		//if(froms_out[x/edges*edges+edge]!=froms[b]) {flag[x/edges]=0; return;}}
//if(iter==0) printf("group: %d  iter:froms_out[%d]:%d->tos_out[%d]:%d iter+1:froms[%d]:%d->tos[%d]:%d x:%d\n",x/edges,a+edge,froms_out[a+edge],a+edge,tos_out[a+edge],b,froms[b],b,tos[b],x);
		if(froms_out[a+edge]!=froms[b]) {flag[x/edges]=0; return;}}
	    else{ 
		//if(tos_out[x/edges*edges+edge-1]!=froms[b]) {flag[x/edges]=0; return;}}
		if(tos_out[a+edge-1]!=froms[b]) {flag[x/edges]=0; return;}}
	}
	else 
	{
		for(SizeT edge = 0; edge<iter+1; edge++){
		    //if(froms[b]==froms_out[x/edges*edges+edge] || froms[b]==tos_out[x/edges*edges+edge])
		    if(froms[b]==froms_out[a+edge] || froms[b]==tos_out[a+edge])
		    {
		    	flag[x/edges]=0;
		      	return;
		    }    
		}
	}}

	{if(d!=0)
	{
	    SizeT edge = d/2;
	    if(d%2==1){
		//if(froms_out[x/edges*edges+edge]!=tos[b])   {flag[x/edges]=0; return;}}
		if(froms_out[a+edge]!=tos[b])   {flag[x/edges]=0; return;}}
	    else{
		//if(tos_out[x/edges*edges+edge-1]!=tos[b])   {flag[x/edges]=0; return;}}
		if(tos_out[a+edge-1]!=tos[b])   {flag[x/edges]=0; return;}}
	}
	else
	{
	    for(SizeT edge=0; edge<iter+1; edge++){
	        //if(tos[b]==froms_out[x/edges*edges+edge] || tos[b]==tos_out[x/edges*edges+edge])
	        if(tos[b]==froms_out[a+edge] || tos[b]==tos_out[a+edge])
	        {
	    	    flag[x/edges]=0; 
		    return;
		}
	    }
	   // printf("d==0 && pass for loop: froms[%d]:%d -> tos[%d]:%d	flag[%d]=1\n",b,froms[b],b,tos[b],x/edges);
	}}
	flag[x/edges]=1;
	//froms_out[x/edges*edges+iter+1] = froms[b];
	//tos_out[x/edges*edges+iter+1] = tos[b];
    }
} // Join Kernel


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
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT size = ((iter==0) ? pos[iter]:counts[0]) * (pos[iter+1]-pos[iter]);
    if(x>=0 && x<size*edges)
    {
	//SizeT a = x%(((iter==0)?pos[iter]:counts[0]) * edges);
	SizeT a = (x/edges%((iter==0)?pos[iter]:counts[0]))*edges+x%edges;
	SizeT b = pos[iter]+x/(edges*((iter==0)?pos[iter]:counts[0])); // edge iter+1 e_id

	if(flag[x/edges]>=1 && (x/edges==0 || flag[x/edges]>flag[x/edges-1]))
	{
// printf("large group:%d small group: %d  iter:%d froms_out[%d]:%d->tos_out[%d]:%d flag[%d]=%d\n",x/edges%(pos[iter+1]-pos[iter]), x/edges/(pos[iter+1]-pos[iter]),iter,a,froms[a],a,tos[a],x/edges,flag[x/edges]);
	    	VertexId from = froms[a];
	    	VertexId to = tos[a];
	    	//VertexId from = froms[x];
	    	//VertexId to = tos[x];
	    	__syncthreads();
		if(x%edges!=iter+1){
	    	froms[(flag[x/edges]-1)*edges+x%edges]=from;
		tos[(flag[x/edges]-1)*edges+x%edges]=to;}
		else{
		froms[(flag[x/edges]-1)*edges+iter+1] = froms_data[b];
		tos[(flag[x/edges]-1)*edges+iter+1] = tos_data[b];}
//printf("iter:%d 	froms[%d]:%d -> tos[%d]:%d	flag[%d]:%d\n",iter,(flag[x/edges]-1)*edges+x%edges,froms[(flag[x/edges]-1)*edges+x%edges],(flag[x/edges]-1)*edges+x%edges, tos[(flag[x/edges]-1)*edges+x%edges],x/edges,flag[x/edges]);
//printf("iter:%d 	froms[%d]:%d -> tos[%d]:%d	flag[%d]:%d\n",iter,(flag[x/edges]-1)*edges+x%edges,from,(flag[x/edges]-1)*edges+x%edges, to,x/edges,flag[x/edges]);
		counts[0] = flag[size-1];
	}
    } 
	
} // Collect Kernel

template<typename SizeT>
__global__ void debug_0(
    const SizeT* const flag,
    const SizeT 	 edges)
{

    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<edges*edges) printf("flag[%d]:%d\n",x,flag[x]);
}

template<typename VertexId, typename SizeT>
__global__ void debug_1(
    const VertexId* const froms,
    const VertexId* const tos,
    const SizeT*    const flag,
    const SizeT*    const pos,
    const SizeT*    const counts,
    const SizeT	    edges_query)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x==0) printf("Number of matched tris: %d\n", counts[0]);

    if(x>=0 && x<counts[0])
    {
	for(int i=0; i<edges_query; i++)
	    printf("edges[%d]: %d -> %d	flag: %d	x:%d\n", i, froms[x*edges_query+i],tos[x*edges_query+i],flag[x],x);	
    }
}

template<typename SizeT>
__global__ void debug_select(
    const SizeT* const d_c_set,
    const SizeT* const flag,
    const SizeT* const d_out,
    const SizeT* const num_elements)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<num_elements) printf("d_out[%d]:%d	num_elements:%d\n",x,d_out[x],num_elements); 
}


} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variable
// mode:c++
// c-file-style: "NVIDIA"
// End:
           
