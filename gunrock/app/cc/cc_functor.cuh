#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>

namespace gunrock {
namespace app {
namespace cc {

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMinFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return !problem->d_marks[node];
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        VertexId parent_from = problem->d_component_ids[problem->d_froms[node]];
        VertexId parent_to = problem->d_component_ids[problem->d_tos[node]];
        VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
        VertexId min_node = parent_from + parent_to - max_node;
        if (max_node == min_node)
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, problem->d_marks + node);
        else
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            max_node, problem->d_component_ids + min_node);
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMaxFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return !problem->d_marks[node];
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        VertexId parent_from = problem->d_component_ids[problem->d_froms[node]];
        VertexId parent_to = problem->d_component_ids[problem->d_tos[node]];
        VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
        VertexId min_node = parent_from + parent_to - max_node;
        if (max_node == min_node)
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, problem->d_marks + node);
        else
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            min_node, problem->d_component_ids + max_node);
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return problem->d_masks[node];
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        VertexId parent = problem->d_component_ids[node];
        VertexId grand_parent = problem->d_component_ids[parent];
        if (parent != grand_parent)
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            grand_parent, problem->d_component_ids + node);
        else
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            false, problem->d_masks + node);
    }
};

} // cc
} // app
} // gunrock
