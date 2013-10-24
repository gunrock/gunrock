#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>

namespace gunrock {
namespace app {
namespace bfs {

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem)
    {
        // Check if the destination node has been claimed as someone's child
        return (atomicCAS(&problem->d_preds[d_id], -2, s_id) == -2) ? true : false;

    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem)
    {
        //for simple BFS, Apply doing nothing
        //set d_labels[d_id] to be d_labels[s_id]+1
        VertexId label;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            label, problem->d_labels + s_id);
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            label+1, problem->d_labels + d_id);
    }

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return node != -1;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        // Doing nothing here
    }
};

} // bfs
} // app
} // gunrock
