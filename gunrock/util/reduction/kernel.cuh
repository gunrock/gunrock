// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Load balanced Edge Map Kernel Entry point
 */

#pragma once

#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/scan/block_scan.cuh>

namespace gunrock {
namespace util {
namespace reduce {

template <typename T, /*int _CUDA_ARCH,*/ int _LOG_THREADS>
struct BlockReduce {
  enum {
    // CUDA_ARCH         = _CUDA_ARCH,
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_WARP_THREADS = 5,  // GR_LOG_WARP_THREADS(CUDA_ARCH),
    WARP_THREADS = 1 << LOG_WARP_THREADS,
    WARP_THREADS_MASK = WARP_THREADS - 1,
    LOG_BLOCK_WARPS = _LOG_THREADS - LOG_WARP_THREADS,
    BLOCK_WARPS = 1 << LOG_BLOCK_WARPS,
  };

  struct TempSpace {
    T warp_results[BLOCK_WARPS];
  };

  template <typename InputT, typename ReduceT>
  static __device__ __forceinline__ T WarpReduce(const InputT &thread_in,
                                                 ReduceT reduce_op) {
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    T lane_local = thread_in;
    T lane_recv;

    lane_recv = _shfl_xor(lane_local, 1, WARPSIZE, 0xFFFFFFFFu);
    if ((lane_id & 1) == 0) {
      lane_local = reduce_op(lane_local, lane_recv);
      lane_recv = _shfl_xor(lane_local, 2, WARPSIZE, 0x55555555u);
    }

    if ((lane_id & 3) == 0) {
      lane_local = reduce_op(lane_local, lane_recv);
      lane_recv = _shfl_xor(lane_local, 4, WARPSIZE, 0x11111111u);
    }

    if ((lane_id & 7) == 0) {
      lane_local = reduce_op(lane_local, lane_recv);
      lane_recv = _shfl_xor(lane_local, 8, WARPSIZE, 0x01010101u);
    }

    if ((lane_id & 0xF) == 0) {
      lane_local = reduce_op(lane_local, lane_recv);
      lane_recv = _shfl_xor(lane_local, 0x10, WARPSIZE, 0x00010001u);
    }

    if (lane_id == 0) {
      lane_local = reduce_op(lane_local, lane_recv);
    }
    _all(1);
    _shfl(lane_local, 0, WARPSIZE, 0xFFFFFFFFu);
    return lane_local;
  }

  template <typename InputT, typename ReduceT>
  static __device__ __forceinline__ T Reduce(const InputT &thread_in,
                                             ReduceT reduce_op, T init_val,
                                             TempSpace &temp_space,
                                             bool to_print = false) {
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    T warp_result = WarpReduce(thread_in, reduce_op);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0) {
      temp_space.warp_results[warp_id] = warp_result;
      if (to_print) printf("warp %d, sum = %d\n", warp_id, warp_result);
    }
    __syncthreads();

    if (warp_id == 0) {
      // if (!to_print)
      warp_result =
          (threadIdx.x < BLOCK_WARPS ? temp_space.warp_results[threadIdx.x]
                                     : init_val);
      if (to_print)
        printf("thread %d, warp_result = %d, BLOCK_WARPS = %d\n", threadIdx.x,
               warp_result, BLOCK_WARPS);
      warp_result = WarpReduce(warp_result, reduce_op);
      if (to_print) printf("result = %d\n", warp_result);
      if (threadIdx.x == 0) temp_space.warp_results[0] = warp_result;
    }
    __syncthreads();

    return temp_space.warp_results[0];
  }
};

using ReduceFlag = uint32_t;
enum : ReduceFlag {
  LOG_THREADS_ = 9,
  BLOCK_SIZE_ = 1 << LOG_THREADS_,
  MAX_GRID_SIZE = 160,
  WARP_THRESHOLD = 64,
  BLOCK_THRESHOLD = BLOCK_SIZE_ * 2,
  GRID_THRESHOLD = MAX_GRID_SIZE * BLOCK_THRESHOLD * 2,
};

template <typename InputT, typename OutputT, typename SizeT,
          typename ReductionOp>
__global__ void SegReduce_Kernel(InputT *keys_in, OutputT *keys_out,
                                 SizeT num_segments, SizeT *segment_offsets,
                                 ReductionOp reduce_op, OutputT init_value,
                                 SizeT *num_grid_segments,
                                 SizeT *grid_segments) {
  typedef Block_Scan<SizeT, LOG_THREADS_> BlockScanT;
  typedef BlockReduce<OutputT, LOG_THREADS_> BlockReduceT;
  __shared__ SizeT s_seg_starts[BLOCK_SIZE_];
  __shared__ SizeT s_seg_sizes[BLOCK_SIZE_];
  __shared__ SizeT s_seg_idxs[BLOCK_SIZE_];
  __shared__ union {
    typename BlockScanT::Temp_Space scan;
    typename BlockReduceT::TempSpace reduce;
  } s_temp_space;

  SizeT block_seg_idx = (SizeT)blockIdx.x * blockDim.x;
  SizeT seg_idx = 0, seg_start = 0, seg_end = 0, seg_size = 0;
  int warp_offset = (threadIdx.x >> BlockReduceT::LOG_WARP_THREADS)
                    << BlockReduceT::LOG_WARP_THREADS;
  int lane_id = threadIdx.x & BlockReduceT::WARP_THREADS_MASK;
  bool seg_active = false;

  while (block_seg_idx < num_segments) {
    seg_idx = block_seg_idx + threadIdx.x;

    if (seg_idx < num_segments) {
      seg_start = segment_offsets[seg_idx];
      seg_end = segment_offsets[seg_idx + 1];
      seg_active = true;
    } else {
      seg_start = 0;
      seg_end = 0;
      seg_active = false;
    }

    seg_size = seg_end - seg_start;
    // if (seg_active)
    //    printf("Seg %d, [%d, %d) size = %d\n",
    //        seg_idx, seg_start, seg_end, seg_size);

    if (seg_active && seg_size < WARP_THRESHOLD) {
      OutputT val = init_value;
      for (SizeT pos = seg_start; pos < seg_end; pos++)
        val = reduce_op(val, keys_in[pos]);
      keys_out[seg_idx] = val;
      seg_active = false;
    }

    SizeT store_pos = 0, num_segs_in_store = 0;
    BlockScanT::Warp_LogicScan(
        (seg_active && seg_size < BLOCK_THRESHOLD) ? 1 : 0, store_pos,
        num_segs_in_store);
    if (seg_active && seg_size < BLOCK_THRESHOLD) {
      store_pos += warp_offset;
      s_seg_starts[store_pos] = seg_start;
      s_seg_sizes[store_pos] = seg_size;
      s_seg_idxs[store_pos] = seg_idx;
      seg_active = false;
    }
    for (int i = 0; i < num_segs_in_store; i++) {
      SizeT warp_seg_start = s_seg_starts[warp_offset + i];
      SizeT warp_seg_size = s_seg_sizes[warp_offset + i];
      OutputT val = init_value;
      for (SizeT j = lane_id; j < warp_seg_size;
           j += BlockReduceT::WARP_THREADS) {
        val = reduce_op(val, keys_in[j + warp_seg_start]);
      }
      val = BlockReduceT::WarpReduce(val, reduce_op);
      if (lane_id == 0) keys_out[s_seg_idxs[warp_offset + i]] = val;
    }
    __syncthreads();

    BlockScanT::LogicScan((seg_active && seg_size < GRID_THRESHOLD) ? 1 : 0,
                          store_pos, s_temp_space.scan, num_segs_in_store);
    if (seg_active && seg_size < GRID_THRESHOLD) {
      s_seg_starts[store_pos] = seg_start;
      s_seg_sizes[store_pos] = seg_size;
      s_seg_idxs[store_pos] = seg_idx;
      seg_active = false;
      // printf("Block.Pos %d: Seg %d, [%d, %d)\n",
      //    store_pos, seg_idx, seg_start, seg_end);
    }
    __syncthreads();
    for (int i = 0; i < num_segs_in_store; i++) {
      SizeT block_seg_start = s_seg_starts[i];
      SizeT block_seg_size = s_seg_sizes[i];
      OutputT val = init_value;
      for (SizeT j = threadIdx.x; j < block_seg_size; j += BLOCK_SIZE_) {
        val = reduce_op(val, keys_in[j + block_seg_start]);
      }
      val =
          BlockReduceT::Reduce(val, reduce_op, init_value, s_temp_space.reduce);
      if (threadIdx.x == 0) keys_out[s_seg_idxs[i]] = val;
    }
    __syncthreads();

    if (seg_active) {
      store_pos = atomicAdd(num_grid_segments, 1);
      grid_segments[store_pos] = seg_idx;
      keys_out[seg_idx] = init_value;
      // printf("Pos %d <- Seg %d, [%d, %d)\n",
      //    store_pos, seg_idx, seg_start, seg_end);
    }
    block_seg_idx += (SizeT)gridDim.x * blockDim.x;
  }
}

template <typename OutputT, typename SizeT>
__global__ void SegReduce_GInit(SizeT *num_grid_segments, SizeT *grid_segments,
                                OutputT *keys_out, OutputT init_value) {
  SizeT num_grids = num_grid_segments[0];

  for (SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x; i < num_grids;
       i += (SizeT)blockDim.x * gridDim.x) {
    keys_out[grid_segments[i]] = init_value;
  }
}

template <typename InputT, typename OutputT, typename SizeT,
          typename ReductionOp>
__global__ void SegReduce_GKernel(InputT *keys_in, OutputT *keys_out,
                                  SizeT num_segments, SizeT *segment_offsets,
                                  ReductionOp reduce_op, OutputT init_value,
                                  SizeT *num_grid_segments,
                                  SizeT *grid_segments) {
  typedef BlockReduce<OutputT, LOG_THREADS_> BlockReduceT;
  __shared__ typename BlockReduceT::TempSpace s_temp_space;
  SizeT num_segs = num_grid_segments[0];
  SizeT thread_id = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

  for (SizeT i = 0; i < num_segs; i++) {
    SizeT seg_idx = grid_segments[i];
    SizeT seg_start = segment_offsets[seg_idx];
    SizeT seg_end = segment_offsets[seg_idx + 1];
    SizeT seg_size = seg_end - seg_start;

    // if (threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("Seg %d: [%d, %d), size = %d\n",
    //        seg_idx, seg_start, seg_end, seg_size);

    OutputT val = init_value;
    for (SizeT j = thread_id; j < seg_size; j += (SizeT)blockDim.x * gridDim.x)
      val = reduce_op(val, keys_in[j + seg_start]);
    // printf("thread %d, val = %d\n",
    //    threadIdx.x, val);

    val = BlockReduceT::Reduce(val, reduce_op, init_value, s_temp_space, false);
    if (threadIdx.x == 0) {
      // printf("block %d, val = %d\n",
      //    blockIdx.x, val);
      OutputT old_val = keys_out[seg_idx];
      OutputT expected;
      do {
        expected = old_val;
        old_val =
            atomicCAS(keys_out + seg_idx, expected, reduce_op(old_val, val));
      } while (expected != old_val);
    }
    __syncthreads();
  }
}

}  // namespace reduce
}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
