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


template<typename VertexId, typename SizeT>
__global__ void MaskOut(
    const SizeT  iter,
    const SizeT  nodes_query, 
    const SizeT  size, // d_partial's size
    VertexId* d_partial)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    while(x<size){
        if(d_partial[x]<0 && x%nodes_query < iter+2){
            #pragma unroll
            for(int i = x - x%nodes_query; i < x; i++){
                if(d_partial[i] >=0 ) d_partial[i] = -1;
            }
        }
        x += STRIDE;
    }
}

template<typename VertexId, typename SizeT>
__global__ void WriteToPartial(
    const SizeT* const src_node,
    const SizeT* const dest_node,
    const SizeT* const flag, // index when iter>1
    const SizeT* const count,
    const SizeT  const iter,
    const SizeT  const nodes_query,
    VertexId*          d_partial)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT bound = count[0];
    if(iter==0){
            while(x<bound){
                d_partial[x*nodes_query] = src_node[x];
                d_partial[x*nodes_query+iter+1] = dest_node[x]; 
                x += STRIDE;
            }
    }else{
            while(x<bound*nodes_query){
                VertexId temp = d_partial[flag[x/nodes_query]*nodes_query+x%nodes_query];
                if(x%nodes_query==iter+1) temp = dest_node[x/nodes_query];
                __syncthreads();
                d_partial[x] = temp;
                x += STRIDE;
            }
    }
}

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variable
// mode:c++
// c-file-style: "NVIDIA"
// End:
           
