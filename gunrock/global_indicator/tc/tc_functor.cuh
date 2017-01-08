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
typename ProblemData,
typename _LabelT = VertexId>
struct TCFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
    typedef _LabelT LabelT;

    static __device__ __forceinline__ bool CondEdge(
            VertexId s_id,
            VertexId d_id,
            DataSlice *problem,
            SizeT edge_id,
            VertexId input_item,
            LabelT label,
            SizeT input_pos,
            SizeT &output_pos)
    {
        //bool res =  (problem->d_degrees[s_id] > problem->d_degrees[d_id]
        //        || (problem->d_degrees[s_id] == problem->d_degrees[d_id] && s_id < d_id));
        bool res = s_id < d_id;
        problem->d_src_node_ids[edge_id] = (res) ? 1:0;
        return res;
    }

    static __device__ __forceinline__ void ApplyEdge(
            VertexId s_id,
            VertexId d_id,
            DataSlice *problem,
            SizeT edge_id,
            VertexId input_item,
            LabelT label,
            SizeT input_pos,
            SizeT &output_pos)
    {
        return;
    }

    static __device__ __forceinline__ bool CondFilter(
        VertexId v,
        VertexId node,
        DataSlice *d_data_slice,
        SizeT nid,
        LabelT label,
        SizeT input_pos,
        SizeT output_pos)
    {
        return (node!=-1);
    }

    static __device__ __forceinline__ void ApplyFilter(
        VertexId v,
        VertexId node,
        DataSlice *d_data_slice,
        SizeT nid,
        LabelT label,
        SizeT input_pos,
        SizeT output_pos)
    {
        return;
    }
};

}
}
}
