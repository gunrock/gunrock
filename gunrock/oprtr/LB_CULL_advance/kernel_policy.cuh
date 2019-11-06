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
 * @brief Kernel configuration policy for Load balanced Edge Expansion Kernel
 */

#pragma once

//#include <cub/cub.cuh>
#include <gunrock/util/scan/block_scan.cuh>

namespace gunrock {
namespace oprtr {
namespace LB_CULL {

/**
 * @brief Kernel configuration policy for partitioned edge mapping kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by
 * parameterizing them with different performance-tuned parameterizations of
 * this type.  By incorporating this type into the kernel code itself, we guide
 * the compiler in expanding/unrolling the kernel code for specific
 * architectures and problem types.
 *
 * @tparam _ProblemData                 Problem data type.
 * @tparam _CUDA_ARCH                   CUDA SM architecture to generate code
 * for.
 * @tparam _INSTRUMENT                  Whether or not we want instrumentation
 * logic generated
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have
 * resident per SM (influences per-CTA smem cache sizes and register
 * allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 */
template <
    // typename _ProblemData,
    OprtrFlag _FLAG,
    typename _VertexT,  // Data types
    typename _InKeyT, typename _OutKeyT, typename _SizeT, typename _ValueT,
    typename _LabelT,
    // int _CUDA_ARCH,         // Machine parameters
    // int _MIN_CTA_OCCUPANCY, // Tunable parameters
    int _MAX_CTA_OCCUPANCY, int _LOG_THREADS, int _LOG_BLOCKS,
    int _LIGHT_EDGE_THRESHOLD>

class KernelPolicy {
 public:
  //---------------------------------------------------------------------
  // Constants and typedefs
  //---------------------------------------------------------------------

  typedef _VertexT VertexT;
  typedef _InKeyT InKeyT;
  typedef _OutKeyT OutKeyT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  typedef _LabelT LabelT;

  enum {
    FLAG = _FLAG,
    // CUDA_ARCH                       = _CUDA_ARCH,
    // INSTRUMENT                      = _INSTRUMENT,

    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
    LIGHT_EDGE_THRESHOLD = _LIGHT_EDGE_THRESHOLD,
    WARP_SIZE = GR_WARP_THREADS(CUDA_ARCH),
    LOG_WARP_SIZE = 5,
    WARP_SIZE_MASK = WARP_SIZE - 1,
    WARPS = THREADS / WARP_SIZE,
    LOG_OUTPUT_PER_THREAD = 2,
    OUTPUT_PER_THREAD = 1 << LOG_OUTPUT_PER_THREAD,
    // OUTPUT_PER_THREAD               = 4,
  };

  enum {
    // Amount of storage we can use for hashing scratch space under target
    // occupancy
    // MAX_SCRATCH_BYTES_PER_CTA       = (GR_SMEM_BYTES(CUDA_ARCH) /
    // _MAX_CTA_OCCUPANCY)
    //                                    - 128, // Fudge-factor to guarantee
    //                                    occupancy

    // SCRATCH_ELEMENT_SIZE            = sizeof(SizeT) * 2 + sizeof(VertexId) *
    // 2,

    SCRATCH_ELEMENTS =
        256,  //= (THREADS > MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE) ?
              //MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE : THREADS,
  };

  // typedef cub::BlockScan<SizeT, THREADS, cub::BLOCK_SCAN_RAKING> BlockScanT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;
  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    // Scratch elements
    // struct {
    SizeT output_offset[SCRATCH_ELEMENTS];
    SizeT row_offset[SCRATCH_ELEMENTS];
    VertexT
        vertices[((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0)
                     ? 1
                     : SCRATCH_ELEMENTS];  // SCRATCH_ELEMENTS];
    InKeyT input_queue[SCRATCH_ELEMENTS];
    SizeT block_offset;
    SizeT *output_counter;
    LabelT *labels;
    unsigned char *visited_masks;
    // VertexT                    *column_indices;
    SizeT block_output_start;
    SizeT block_output_end;
    SizeT block_output_size;
    SizeT block_input_start;
    SizeT block_input_end;
    SizeT iter_input_start;
    SizeT iter_input_size;
    SizeT iter_input_end;
    SizeT iter_output_end;
    SizeT iter_output_size;
    SizeT iter_output_end_offset;
    OutKeyT thread_output_vertices[THREADS * OUTPUT_PER_THREAD];
    // MaskT                       tex_mask_bytes[THREADS * OUTPUT_PER_THREAD];
    // bool                        warps_cond[WARPS];
    // bool                        block_cond;
    SizeT block_count;
    SizeT block_first_v_skip_count;
    typename BlockScanT::Temp_Space scan_space;
    // union {
    //    typename BlockScanT::TempStorage scan_space;
    //} cub_storage;
    //};

    __device__ __forceinline__ void Init(
        const VertexT &queue_index, const SizeT &num_inputs,
        const SizeT *&block_input_starts, unsigned char *&_visited_masks,
        LabelT *&_labels, const SizeT *&output_offsets,
        const SizeT &num_outputs, util::CtaWorkProgress<SizeT> &work_progress) {
      output_counter =
          work_progress.template GetQueueCounter<VertexT>(queue_index + 1);
      // int lane_id = threadIdx.x & KernelPolicy::WARP_SIZE_MASK;
      // int warp_id = threadIdx.x >> KernelPolicy::LOG_WARP_SIZE;
      labels = _labels;
      visited_masks = ((FLAG & OprtrOption_Idempotence) != 0)
                          ? _visited_masks
                          : ((unsigned char *)NULL);

      if (block_input_starts != NULL) {
        SizeT outputs_per_block = (num_outputs + gridDim.x - 1) / gridDim.x;
        block_output_start = (SizeT)blockIdx.x * outputs_per_block;
        if (block_output_start >= num_outputs) return;
        block_output_end =
            min(block_output_start + outputs_per_block, num_outputs);
        block_output_size = block_output_end - block_output_start;
        block_input_end =
            (blockIdx.x + 1 == gridDim.x)
                ? num_inputs
                : min(block_input_starts[blockIdx.x + 1], num_inputs);
        if (block_input_end < num_inputs &&
            block_output_end >
                (block_input_end > 0 ? output_offsets[block_input_end - 1] : 0))
          block_input_end++;

        block_input_start = block_input_starts[blockIdx.x];
        block_first_v_skip_count =
            (block_output_start != output_offsets[block_input_start])
                ? block_output_start -
                      (block_input_start > 0
                           ? output_offsets[block_input_start - 1]
                           : 0)
                : 0;
        iter_input_start = block_input_start;
      }

      /*printf("(%4d, %4d) : block_input = %d ~ %d, %d "
          "block_output = %d ~ %d, %d\n",
          blockIdx.x, threadIdx.x,
          block_input_start, block_input_end, block_input_end -
         block_input_start + 1, block_output_start, block_output_end,
         block_output_size);*/
    }
  };

  enum {
    THREAD_OCCUPANCY = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
    SMEM_OCCUPANCY = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
    CTA_OCCUPANCY = GR_MIN(_MAX_CTA_OCCUPANCY,
                           GR_MIN(GR_SM_CTAS(CUDA_ARCH),
                                  GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

    VALID = (CTA_OCCUPANCY > 0),
    // ELEMENT_ID_MASK	                = ~(1ULL<<(sizeof(VertexT)*8-2)),
  };
};

}  // namespace LB_CULL
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
