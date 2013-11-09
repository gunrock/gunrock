#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>

namespace gunrock {
namespace app {
namespace cc {

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct UpdateMaskFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return true;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        problem->d_masks[node] = (problem->d_component_ids[node] == node)?0:1;
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookInitFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        return true; 
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        
        VertexId max_node = problem->d_froms[node] > problem->d_tos[node] ? problem->d_froms[node] : problem->d_tos[node];
        VertexId min_node = problem->d_froms[node] + problem->d_tos[node] - max_node;
        problem->d_component_ids[max_node] = min_node;
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMinFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        if (!problem->d_marks[node])
        {
            VertexId from_node = problem->d_froms[node];
            VertexId to_node = problem->d_tos[node];
            VertexId parent_from = problem->d_component_ids[from_node];
            VertexId parent_to = problem->d_component_ids[to_node];
            VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
            VertexId min_node = parent_from + parent_to - max_node;
            if (max_node == min_node)
                problem->d_marks[node] = true;
            else
                problem->d_component_ids[min_node] = max_node;
            return true;
        }
        else
            return false;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
       //do nothing 
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMaxFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        if (!problem->d_marks[node])
        {
            VertexId from_node = problem->d_froms[node];
            VertexId to_node = problem->d_tos[node];
            VertexId parent_from = problem->d_component_ids[from_node];
            VertexId parent_to = problem->d_component_ids[to_node];
            VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
            VertexId min_node = parent_from + parent_to - max_node;
            if (max_node == min_node)
                problem->d_marks[node] = true;
            else
                problem->d_component_ids[max_node] = min_node;
            return true;
        }
        else
            return false;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        //do nothing
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        VertexId parent = problem->d_component_ids[node];
        VertexId grand_parent = problem->d_component_ids[parent];
        if (parent != grand_parent) {
            problem->d_component_ids[node] = grand_parent;
            return true;
        }
        else
            return false;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    { 
        //do nothing
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpMaskFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        if (problem->d_masks[node] == 0)
        {
            VertexId parent = problem->d_component_ids[node];
            VertexId grand_parent = problem->d_component_ids[parent];
            if (parent != grand_parent) {
                problem->d_component_ids[node] = grand_parent;
                return true;
            }
            else
                problem->d_masks[node] = -1;
        }
        else
            return false;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
       //do nothing 
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpUnmaskFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem)
    {
        if (problem->d_masks[node] == 1)
        {
            VertexId parent = problem->d_component_ids[node];
            VertexId grand_parent = problem->d_component_ids[parent];
            problem->d_component_ids[node] = grand_parent;
        }
            return true;
    }

    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem)
    {
        //do nothing
    }
};

} // cc
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
