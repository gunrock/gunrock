#pragma once
#include <wrapper/app/mad/mad_problem.cuh>

namespace wrapper {
namespace app {
namespace mad {

template<typename Value, typename ProblemData>
struct MADFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Apply(int node_id, DataSlice *problem)
    {
        Value v = problem->d_results[node_id];
        problem->d_results[node_id] = v*2+1;
    }
};

} //mad
} //app
} //wrapper
