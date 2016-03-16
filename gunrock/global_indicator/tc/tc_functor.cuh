#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/global_indicator/tc/tc_problem.cuh>

namespace gunrock {
namespace global_indicator {
namespace tc {

template<
typename VertexId,
typename SizeT,
typename Value,
typename ProblemData>
struct TCFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
    
    static __device__ __forceinline__ bool CondEdge(
    VertexId s_id, VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
    {
        bool res =  (problem->d_degrees[s_id] > problem->d_degrees[d_id]
                || (problem->d_degrees[s_id] == problem->d_degrees[d_id] && s_id < d_id));
        problem->d_src_node_ids[e_id] = (res) ? 1:0;
        return res;
    }

    static __device__ __forceinline__ void ApplyEdge(
    VertexId s_id, VertexId d_id, DataSlice *problem,
    VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return;
    }

    static __device__ __forceinline__ bool CondFilter(
    VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
    {
            return (node!=-1);
    }

    static __device__ __forceinline__ void ApplyFilter(
    VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
    {
        return;
    }
};

}
}
}
