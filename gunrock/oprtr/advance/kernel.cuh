#pragma once
#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/cta.cuh>
#include <gunrock/oprtr/edge_map_backward/cta.cuh>
#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

//TODO: finish LaucnKernel, should load diferent kernels according to their AdvanceMode
//AdvanceType is the argument to send into each kernel call
template <typename KernelPolicy, typename ProblemData, typename Functor, TYPE AdvanceType>
LaunchKernel()
{
}

} //advance
} //oprtr
} //gunrock/
