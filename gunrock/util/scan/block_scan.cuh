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

namespace gunrock {
namespace util {

template <typename T, /*int _CUDA_ARCH,*/ int _LOG_THREADS>
struct Block_Scan {
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

  struct Temp_Space {
    T warp_counter_offset[BLOCK_WARPS];
    T block_sum;
  };

  static __device__ __forceinline__ void Warp_Scan(T thread_in, T &thread_out,
                                                   T &sum) {
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    T &lane_local = thread_out;
    T lane_recv;

    lane_local = thread_in;
    // UpSweep<int, LOG_WARP_THREADS>::Sweep(lane_local, lane_recv, lane_id);
    // UpSweep LOG_WIDTH = 5
    // if ((lane_id & 1) == 0)
    lane_recv = _shfl_xor(lane_local, 1);
    if ((lane_id & 1) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 4
      lane_recv = _shfl_xor(lane_local, 2, WARPSIZE, 0x55555555u);
    }

    if ((lane_id & 3) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 3
      lane_recv = _shfl_xor(lane_local, 4, WARPSIZE, 0x11111111u);
    }

    if ((lane_id & 7) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 2
      lane_recv = _shfl_xor(lane_local, 8, WARPSIZE, 0x01010101u);
    }

    if ((lane_id & 0xF) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 1
      lane_recv = _shfl_xor(lane_local, 0x10, WARPSIZE, 0x00010001u);
    }

    if (lane_id == 0) {
      lane_local += lane_recv;
    }
    _all(1);
    sum = _shfl(lane_local, 0);
    if (lane_id == 0) {
      lane_recv = 0;
    }
    lane_local = lane_recv;

    // DownSweep<int, LOG_WARP_THREADS-2>::Sweep(lane_local, lane_recv,
    // lane_id); DownSweep LOG_WIDTH = 3
    lane_recv = _shfl_up(lane_local, 8);
    if ((lane_id & 0xF) == 8) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 2
    lane_recv = _shfl_up(lane_local, 4);
    if ((lane_id & 7) == 4) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 1
    lane_recv = _shfl_up(lane_local, 2);
    if ((lane_id & 3) == 2) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 0
    lane_recv = _shfl_up(lane_local, 1);
    if ((lane_id & 1) == 1) lane_local += lane_recv;
  }

  static __device__ __forceinline__ void Warp_Scan(T thread_in, T &thread_out) {
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    T &lane_local = thread_out;
    T lane_recv;

    lane_local = thread_in;
    // UpSweep<int, LOG_WARP_THREADS>::Sweep(lane_local, lane_recv, lane_id);
    // UpSweep LOG_WIDTH = 5
    // if ((lane_id & 1) == 0)
    lane_recv = _shfl_xor(lane_local, 1);
    if ((lane_id & 1) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 4
      lane_recv = _shfl_xor(lane_local, 2, WARPSIZE, 0x55555555u);
    }

    if ((lane_id & 3) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 3
      lane_recv = _shfl_xor(lane_local, 4, WARPSIZE, 0x11111111u);
    }

    if ((lane_id & 7) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 2
      lane_recv = _shfl_xor(lane_local, 8, WARPSIZE, 0x01010101u);
    }

    if ((lane_id & 0xF) == 0) {
      lane_local += lane_recv;
      // UpSweep LOG_WIDTH = 1
      lane_recv = _shfl_xor(lane_local, 0x10, WARPSIZE, 0x00010001u);
    }

    if (lane_id == 0) {
      lane_local += lane_recv;
      // if (lane_id == 0)
      lane_recv = 0;
    }
    lane_local = lane_recv;
    _all(1);

    // DownSweep<int, LOG_WARP_THREADS-2>::Sweep(lane_local, lane_recv,
    // lane_id); DownSweep LOG_WIDTH = 3
    lane_recv = _shfl_up(lane_local, 8);
    if ((lane_id & 15) == 8) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 2
    lane_recv = _shfl_up(lane_local, 4);
    if ((lane_id & 7) == 4) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 1
    lane_recv = _shfl_up(lane_local, 2);
    if ((lane_id & 3) == 2) lane_local += lane_recv;

    // DownSweep LOG_WIDTH = 0
    lane_recv = _shfl_up(lane_local, 1);
    if ((lane_id & 1) == 1) lane_local += lane_recv;
  }

  static __device__ __forceinline__ void Warp_LogicScan(int thread_in,
                                                        T &thread_out) {
    unsigned int warp_flag = _ballot(thread_in);
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    unsigned int lane_mask = (1 << lane_id) - 1;
    thread_out = __popc(warp_flag & lane_mask);
  }

  static __device__ __forceinline__ void Warp_LogicScan(int thread_in,
                                                        T &thread_out, T &sum) {
    unsigned int warp_flag = _ballot(thread_in);
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    unsigned int lane_mask = (1 << lane_id) - 1;
    thread_out = __popc(warp_flag & lane_mask);
    sum = __popc(warp_flag);
  }

  static __device__ __forceinline__ void Scan(T thread_in, T &thread_out,
                                              Temp_Space &temp_space) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_Scan(thread_in, thread_out);
    if ((threadIdx.x & WARP_THREADS_MASK) == WARP_THREADS_MASK) {
      warp_sum = thread_in + thread_out;
      // if (warp_sum != 0)
      //    printf("(%4d, %4d) : warp_sum = %d\n",
      //        blockIdx.x, threadIdx.x, warp_sum);
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    }
    // if (thread_in != 0)
    //    printf("(%4d, %4d) : thread_out = %4d, warp_sum = %4d\n",
    //        blockIdx.x, threadIdx.x, thread_out, warp_sum);
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
    }
    __syncthreads();

    thread_out += temp_space.warp_counter_offset[warp_id];
    // if (thread_in != 0)
    //    printf("(%4d, %4d) : thread_out2 = %4d\n",
    //        blockIdx.x, threadIdx.x, thread_out);
  }

  static __device__ __forceinline__ void Scan(T thread_in, T &thread_out,
                                              Temp_Space &temp_space,
                                              T &block_sum) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_Scan(thread_in, thread_out, warp_sum);
    // printf("(%4d, %4d) : WARP_THREADS_MASK = %d\n",
    //    blockIdx.x, threadIdx.x, WARP_THREADS_MASK);
    if ((threadIdx.x & WARP_THREADS_MASK) == WARP_THREADS_MASK) {
      // printf("(%4d, %4d, %d) : warp_sum = %d\n",
      //    blockIdx.x, threadIdx.x, threadIdx.x >> LOG_WARP_THREADS, warp_sum);
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    }
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum, block_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
      if (threadIdx.x == 0) {
        temp_space.block_sum = block_sum;
        // printf("(%4d, %4d) : block_sum = %d\n",
        //    blockIdx.x, threadIdx.x, block_sum);
      }
      // printf("(%4d, %4d) : warp_offset = %d\n",
      //    blockIdx.x, threadIdx.x, warp_sum);
    }
    __syncthreads();

    thread_out += temp_space.warp_counter_offset[warp_id];
    block_sum = temp_space.block_sum;
  }

  static __device__ __forceinline__ void LogicScan(int thread_in, T &thread_out,
                                                   Temp_Space &temp_space) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_LogicScan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0)
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
    }
    __syncthreads();

    thread_out += temp_space.warp_counter_offset[warp_id];
  }

  static __device__ __forceinline__ void LogicScan(int thread_in, T &thread_out,
                                                   Temp_Space &temp_space,
                                                   T &block_sum) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_LogicScan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0)
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum, block_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
      if (threadIdx.x == 0) temp_space.block_sum = block_sum;
    }
    __syncthreads();

    thread_out += temp_space.warp_counter_offset[warp_id];
    block_sum = temp_space.block_sum;
  }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
