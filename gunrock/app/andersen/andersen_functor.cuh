#pragma once

#include <stdio.h>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/andersen/andersen_problem.cuh>

namespace gunrock {
namespace app {
namespace andersen {

    template <typename VertexId>
    static __device__ __host__ bool to_track(VertexId node)
    {
        const int num_to_track = 22;
        const VertexId node_to_track[] = {6167, 6168, 6176, 6183, 6246, 1668, 2204, 2244,
                   2457, 2458, 2444, 2447, 9938, 6249, 6247, 6248, 6169, 6181, 6171, 6170, 10542, 10876};
        for (int i=0; i<num_to_track; i++)
            if (node == node_to_track[i]) return true;
        return false;
    }

    template <typename VertexId, typename SizeT>
    static __device__ __host__ VertexId hash_function(VertexId x, VertexId y, SizeT stride, SizeT m)
    {
        return (((x%m)*(stride%m) + y)%m);
    }
 
/**
 * @brief Structure contains device functions for doing mask update.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for Andersen problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RuleKernelFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("RuleKernelFunctor CondEdge %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, s_id, d_id);
        if (problem->num_gpus == 1) 
        {
            //printf(" blockId = %d threadId = %d : %d->%d return true\n", blockIdx.x, threadIdx.x, s_id, d_id);
            return true;
        } else {
            //printf(" blockId = %d threadId = %d : %d->%d num_gpus = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, problem->num_gpus);
        }
        
        int local_gpu;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        local_gpu, problem->partition_table + s_id);
        return (local_gpu == 0);
    }

    /**
     * @brief Forward Edge Mapping apply function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        SizeT s_offset_start, s_offset_end, length, offset;
        //util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
        //    s_offset_start, problem->s_offsets + d_id);
        //util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
        //    s_offset_end  , problem->s_offsets + d_id+1);
        s_offset_start = problem->s_offsets[d_id];
        s_offset_end   = problem->s_offsets[d_id+1];
        length = 0; offset = 0;
        if (to_track(s_id) || to_track(d_id))
            printf("RuleKernelFunctor ApplyEdge %d,%d : %d -> %d, edges [%d->%d) @ %d, e_id_in = %d, e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, s_offset_start, s_offset_end, s_offset_end-s_offset_start, e_id_in, e_id);
        if (problem->r_offsets2!=NULL)
        {
            offset = problem->r_offsets2[e_id];
            if (to_track(s_id + offset) || to_track(s_id - offset) || to_track(d_id + offset) || to_track(d_id - offset))
                printf("RuleKernelFunctor ApplyEdge %d,%d : %d -> %d, edges [%d->%d) @ %d, e_id_in = %d, e_id = %d, offset = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, s_offset_start, s_offset_end, s_offset_end-s_offset_start, e_id_in, e_id, offset);
        }
        //printf("blockId = %d threadId = %d %d->%d org start=%d end=%d length=%d\n", blockIdx.x, threadIdx.x, s_id, d_id, s_offset_start, s_offset_end, s_offset_end-s_offset_start);
        for (SizeT i=s_offset_start; i<s_offset_end; i++)
        {
            VertexId y, old_value;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                y, problem->s_indices + i);
            y += offset;
            old_value=atomicCAS(problem->t_hash + hash_function(s_id, y, problem->h_stride, problem->h_size), 0, y+1);
            if (old_value==0) {printf("%d->%d adding\t", s_id, y);length ++;}
            else if (old_value!=y+1) {printf("%d->%d conflict\t", s_id, y);problem->t_conflict = true;}
        }
        atomicAdd(problem->t_length + s_id, length);
    }
};

/**
 * @brief Structure contains device functions for doing mask update.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for Andersen problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RuleKernelFunctor2
{
    typedef typename ProblemData::DataSlice DataSlice;

   /**
     * @brief Forward Edge Mapping condition function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("RuleKernelFunctor2 CondEdge %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, s_id, d_id);
        if (problem->num_gpus == 1) return true;
        
        int local_gpu;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        local_gpu, problem->partition_table + s_id);
        return (local_gpu == 0);
    }

    /**
     * @brief Forward Edge Mapping apply function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        SizeT s_offset_start, s_offset_end, t_offset, offset;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            s_offset_start, problem->s_offsets + d_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            s_offset_end  , problem->s_offsets + d_id+1);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            t_offset      , problem->t_offsets + s_id);
        if (to_track(s_id) || to_track(d_id))
            printf("RuleKernelFunctor2 ApplyEdge %d,%d : %d -> %d, edges [%d->%d) @ %d, e_id_in = %d, e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, s_offset_start, s_offset_end, s_offset_end-s_offset_start, e_id_in, e_id);
        offset = 0;
        if (problem->r_offsets2 !=NULL)
            offset = problem->r_offsets2[e_id];
        for (SizeT i=s_offset_start; i<s_offset_end; i++)
        {
            VertexId y;
            SizeT pos;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                y, problem->s_indices + i);
            y+=offset;
            if (atomicCAS(problem->t_marker + hash_function(s_id, y, problem->h_stride, problem->h_size), 0, 1)==0)
            {
                pos = atomicAdd(problem->t_length + s_id, 1);
                pos += t_offset;
                printf("RuleKernelFunctor2 ApplyEdge %d,%d : adding %d -> %d\n", blockIdx.x, threadIdx.x, s_id, y);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                    y, problem->t_indices + pos);
            }
        }
    }
};

/**
 * @brief Structure contains device functions for doing mask update.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for Andersen problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct MakeHashFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("MakeHashFunctor CondEdge %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, s_id, d_id);
        if (problem->num_gpus == 1) return true;
        
        int local_gpu;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                        local_gpu, problem->partition_table + s_id);
        return (local_gpu == 0);
    }

    /**
     * @brief Forward Edge Mapping apply function.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("MakeHashFunctor ApplyEdge %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, s_id, d_id);
        VertexId old_value = atomicCAS(problem->t_hash + hash_function(s_id, d_id, problem->h_stride, problem->h_size), 0, d_id+1);
        if (old_value != 0 && old_value != d_id +1) problem->t_conflict = true;
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct MakeInverstFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("MakeInverstFunctor CondEdge %d,%d : %d->%d, e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, e_id);
        return true;
    }
 
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //SizeT pos;
        if (to_track(s_id) || to_track(d_id))
            printf("MakeInverstFunctor ApplyEdge %d,%d : %d->%d, e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, e_id);
        atomicAdd(problem->t_length + d_id, 1);
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct MakeInverstFunctor2
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("MakeInverstFunctor2 CondEdge %d,%d : %d->%d e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, e_id);
        return true;
    }
 
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        if (to_track(s_id) || to_track(d_id))
            printf("MakeInverstFunctor2 ApplyEdge %d,%d : %d->%d e_id = %d\n", blockIdx.x, threadIdx.x, s_id, d_id, e_id);
        SizeT pos;
        pos = atomicAdd(problem->t_length + d_id, 1);
        pos += problem->t_offsets[d_id];
        problem->t_indices[pos] = s_id;
    }
};

template<typename VertexId, typename SizeT, typename Value>
__global__ void InverstGraph(
    const SizeT           num_nodes,
    const SizeT*    const r_offsets,
    const VertexId* const r_indices,
    const SizeT*    const t_offsets,
          SizeT*          t_lengths,
          VertexId*       t_indices)
{
    const VertexId STRIDE = blockDim.x * gridDim.x;
    VertexId x = threadIdx.x + blockIdx.x * blockDim.x;

    while (x < num_nodes)
    {
        for (SizeT i = r_offsets[x]; i< r_offsets[x+1]; i++)
        {
            VertexId y = r_indices[i];
            if (to_track(x) || to_track(y))
                printf("InverstGraph %d,%d : %d -> %d, e_id = %d\n", blockIdx.x, threadIdx.x, x, y, i);
            SizeT pos = atomicAdd(t_lengths + y, 1);
            pos += t_offsets[y];
            t_indices[pos] = x;
        }
        x += STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT>
__global__ void UnionGraphs(
    const SizeT           num_nodes,
    const SizeT*    const g1_offsets,
    const VertexId* const g1_indices,
    const SizeT*    const g2_offsets,
    const VertexId* const g2_indices,
          SizeT*          g3_offsets,
          VertexId*       g3_indices)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT y, end_index, i, x = blockIdx.x * blockDim.x + threadIdx.x;

    while (x < num_nodes) 
    {
        y = g1_offsets[x] + g2_offsets[x];
        end_index = g1_offsets[x+1];
        for (i=g1_offsets[x]; i<end_index; i++)
        {
            g3_indices[y] = g1_indices[i];
            if (to_track(x) || to_track(g1_indices[i]))
              printf("UnionGraphs %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, x, g1_indices[i]);
            y++;
        }
        end_index = g2_offsets[x+1];
        for (i=g2_offsets[x]; i<end_index; i++)
        {
            g3_indices[y] = g2_indices[i];
            if (to_track(x) || to_track(g2_indices[i]))
              printf("UnionGraphs %d,%d : %d -> %d\n", blockIdx.x, threadIdx.x, x, g2_indices[i]);
            y++;
        }
        g3_offsets[x+1] = y;
        x+=STRIDE;
    }
}

} // andersen
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
