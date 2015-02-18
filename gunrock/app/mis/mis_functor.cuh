// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------- 
/**
 * @file
 * mis_functor.cuh
 *
 * @brief Device functions for MIS problem.
 */


#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/mis/mis_problem.cuh>

namespace gunrock {
namespace app {
namespace mis {

/**
 * @brief Structure contains device functions in MIS graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for PR problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct MISFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
};

} // mis
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
