// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * global_barrier.cuh
 *
 * @brief Software Global Barrier
 */

#pragma once

#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace util {

/**
 * Manages device storage needed for implementing a global software barrier
 * between CTAs in a single grid
 */
class GlobalBarrier {
 public:
  typedef unsigned int SyncFlag;

 protected:
  // Counters in global device memory
  SyncFlag* d_sync;

  // Simple wrapper for returning a CG-loaded SyncFlag at the specified pointer
  __device__ __forceinline__ SyncFlag LoadCG(SyncFlag* d_ptr) const {
    SyncFlag retval;
    util::io::ModifiedLoad<util::io::ld::cg>::Ld(retval, d_ptr);
    return retval;
  }

 public:
  /**
   * Constructor
   */
  GlobalBarrier() : d_sync(NULL) {}

  /**
   * Synchronize
   */
  __device__ __forceinline__ void Sync() const {
    volatile SyncFlag* d_vol_sync = d_sync;

    // Threadfence and syncthreads to make sure global writes are visible before
    // thread-0 reports in with its sync counter
    __threadfence();
    __syncthreads();

    if (blockIdx.x == 0) {
      // Report in ourselves
      if (threadIdx.x == 0) {
        d_vol_sync[blockIdx.x] = 1;
      }

      __syncthreads();

      // Wait for everyone else to report in
      for (int peer_block = threadIdx.x; peer_block < gridDim.x;
           peer_block += blockDim.x) {
        while (LoadCG(d_sync + peer_block) == 0) {
          __threadfence_block();
        }
      }

      __syncthreads();

      // Let everyone know it's safe to read their prefix sums
      for (int peer_block = threadIdx.x; peer_block < gridDim.x;
           peer_block += blockDim.x) {
        d_vol_sync[peer_block] = 0;
      }

    } else {
      if (threadIdx.x == 0) {
        // Report in
        d_vol_sync[blockIdx.x] = 1;

        // Wait for acknowledgement
        while (LoadCG(d_sync + blockIdx.x) == 1) {
          __threadfence_block();
        }
      }

      __syncthreads();
    }
  }
};

/**
 * Version of global barrier with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base GlobalBarrier
 * as parameters to kernels.
 */
class GlobalBarrierLifetime : public GlobalBarrier {
 protected:
  // Number of bytes backed by d_sync
  size_t sync_bytes;

 public:
  // Constructor
  GlobalBarrierLifetime() : GlobalBarrier(), sync_bytes(0) {}

  // Deallocates and resets the progress counters
  cudaError_t HostReset() {
    cudaError_t retval = cudaSuccess;
    if (d_sync) {
      retval = util::GRError(cudaFree(d_sync),
                             "GlobalBarrier cudaFree d_sync failed: ", __FILE__,
                             __LINE__);
      d_sync = NULL;
    }
    sync_bytes = 0;
    return retval;
  }

  // Destructor
  virtual ~GlobalBarrierLifetime() { HostReset(); }

  // Sets up the progress counters for the next kernel launch (lazily
  // allocating and initializing them if necessary)
  cudaError_t Setup(int sweep_grid_size) {
    cudaError_t retval = cudaSuccess;
    do {
      size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
      if (new_sync_bytes > sync_bytes) {
        if (d_sync) {
          if (retval =
                  util::GRError(cudaFree(d_sync),
                                "GlobalBarrierLifetime cudaFree d_sync failed",
                                __FILE__, __LINE__))
            break;
        }

        sync_bytes = new_sync_bytes;

        if (retval =
                util::GRError(cudaMalloc((void**)&d_sync, sync_bytes),
                              "GlobalBarrierLifetime cudaMalloc d_sync failed",
                              __FILE__, __LINE__))
          break;

        // Initialize to zero
        util::MemsetKernel<SyncFlag>
            <<<(sweep_grid_size + 128 - 1) / 128, 128>>>(d_sync, 0,
                                                         sweep_grid_size);
        if (retval = util::GRError(
                cudaDeviceSynchronize(),
                "GlobalBarrierLifetime MemsetKernel d_sync failed", __FILE__,
                __LINE__))
          break;
      }
    } while (0);

    return retval;
  }
};

}  // namespace util
}  // namespace gunrock
