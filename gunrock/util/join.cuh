// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file: join.cuh
 *
 * @brief Simple Join Kernel
 */
#pragma once

namespace gunrock {
namespace util {

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
    const SizeT*    const       pos,
    const VertexId* const       froms,
    const VertexId* const       tos,
    const VertexId* const       flag,
          VertexId*             froms_out,
          VertexId*             tos_out)
{
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int num_matches;
    num_matches = pos[0];
    int i,j;
    if(x<pos[0]) // allocate all edge cadidate for the first query edge to each bucket of size edges
    {
        froms_out[x*edges] = froms[x];
        tos_out[x*edges] = tos[x];
    }
    __syncthreads();

    for(i=0; i<edges; i++){
        for(j=0; j<pos[0]; j++){
          if(x>=pos[i-1] && x<pos[i]){
            if(flag[i*2]==0 && flag[i*2+1]==0){
                atomicAdd(&num_matches,1);
                froms_out[j*edges+i] = froms[x];
                tos_out[j*edges+i] = tos[x];
                continue;
            }
            if(flag[i*2]%2==1)
                if(froms[x]!=froms_out[j*edges+flag[i*2]/2]) continue;
            else if(flag[i*2]!=0 && flag[i*2]%2==0)
                if(froms[x]!=tos_out[j*edges+flag[i*2]/2-1]) continue;
            if(flag[i*2+1]%2==1)
                if(tos[x]!=froms_out[j*edges+flag[i*2]/2]) continue;
            else if(flag[i*2+1]!=0 && flag[i*2+1]%2==0)
                if(tos[x]!=tos_out[j*edges+flag[i*2+1]/2-1]) continue;

            atomicAdd(&num_matches,1);
            froms_out[j*edges+i] = froms[x];
            tos_out[j*edges+i] = tos[x];
          }
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
           
