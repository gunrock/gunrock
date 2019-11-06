// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel_policy.cuh
 *
 * @brief Kernel configuration policy for Priority Queue Kernel
 */

#pragma once
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace priority_queue {

template <typename _ProblemData, int _CUDA_ARCH, bool _INSTRUMENT,
          int _LOG_THREADS, int _LOG_BLOCKS>

struct KernelPolicy {
  typedef _ProblemData ProblemData;
  typedef typename ProblemData::VertexId VertexId;
  typedef typename ProblemData::SizeT SizeT;

  enum {

    CUDA_ARCH = _CUDA_ARCH,
    INSTRUMENT = _INSTRUMENT,

    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
  };
};

}  // namespace priority_queue
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
