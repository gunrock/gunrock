// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cta_work_progress.cuh
 *
 * @brief Management of temporary device storage needed for implementing
 * work-stealing progress between CTAs in a single grid.
 */

#pragma once

#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

/**
 * Manages device storage needed for implementing work-stealing
 * and queuing progress between CTAs in a single grid.
 *
 * Can be used for:
 *
 * (1) Work-stealing. Consists of a pair of counters in
 *     global device memory, optionally, a host-managed selector for
 *     indexing into the pair.
 *
 * (2) Device-managed queue.  Consists of a quadruplet of counters
 *     in global device memory and selection into the counters is made
 *     based upon the supplied iteration count.
 *          Current iteration: incoming queue length
 *          Next iteration: outgoing queue length
 *          Next next iteration: needs to be reset before next iteration
 *     Can be used with work-stealing counters to for work-stealing
 *     queue operation
 *
 * For work-stealing passes, the current counter provides an atomic
 * reference of progress, and the current pass will typically reset
 * the counter for the next.
 *
 */
template <typename SizeT>
class CtaWorkProgress {
 protected:
  enum {
    QUEUE_COUNTERS = 4,
    STEAL_COUNTERS = 2,
    OVERFLOW_COUNTERS = 1,
  };

  // Seven pointer-sized counters in global device memory (we may not use
  // all of them, or may only use 32-bit versions of them)
  SizeT* d_counters;
  util::Array1D<int, SizeT, PINNED,
                cudaHostAllocMapped | cudaHostAllocPortable>* p_counters;

  // Host-controlled selector for indexing into d_counters.
  int progress_selector;

 public:
  enum { COUNTERS = QUEUE_COUNTERS + STEAL_COUNTERS + OVERFLOW_COUNTERS };

  /**
   * Constructor
   */
  CtaWorkProgress()
      : d_counters(NULL), progress_selector(0), p_counters(NULL) {}

  virtual ~CtaWorkProgress() {}

  //---------------------------------------------------------------------
  // Work-stealing
  //---------------------------------------------------------------------

  // Steals work from the host-indexed progress counter, returning
  // the offset of that work (from zero) and incrementing it by count.
  // Typically called by thread-0
  // template <typename SizeT>
  __device__ __forceinline__ SizeT Steal(SizeT count) {
    SizeT* d_steal_counters = d_counters + QUEUE_COUNTERS;
    return util::AtomicInt<SizeT>::Add(d_steal_counters + progress_selector,
                                       count);
  }

  // Steals work from the specified iteration's progress counter, returning the
  // offset of that work (from zero) and incrementing it by count.
  // Typically called by thread-0
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ SizeT Steal(SizeT count, IterationT iteration) {
    SizeT* d_steal_counters = d_counters + QUEUE_COUNTERS;
    return util::AtomicInt<SizeT>::Add(d_steal_counters + (iteration & 1),
                                       count);
  }

  // Resets the work progress for the next host-indexed work-stealing
  // pass.  Typically called by thread-0 in block-0.
  // template <typename SizeT>
  __device__ __forceinline__ void PrepResetSteal() {
    SizeT reset_val = 0;
    SizeT* d_steal_counters = d_counters + QUEUE_COUNTERS;
    util::io::ModifiedStore<util::io::st::cg>::St(
        reset_val, d_steal_counters + (progress_selector ^ 1));
  }

  // Resets the work progress for the specified work-stealing iteration.
  // Typically called by thread-0 in block-0.
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ void PrepResetSteal(IterationT iteration) {
    SizeT reset_val = 0;
    SizeT* d_steal_counters = d_counters + QUEUE_COUNTERS;
    util::io::ModifiedStore<util::io::st::cg>::St(
        reset_val, d_steal_counters + (iteration & 1));
  }

  //---------------------------------------------------------------------
  // Queuing
  //---------------------------------------------------------------------

  // Get counter for specified iteration
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ SizeT* GetQueueCounter(
      IterationT iteration) const {
    return d_counters + (iteration & 0x3);
  }

  // Load work queue length for specified iteration
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ SizeT LoadQueueLength(IterationT iteration) {
    SizeT queue_length;
    util::io::ModifiedLoad<util::io::ld::cg>::Ld(queue_length,
                                                 GetQueueCounter(iteration));
    return queue_length;
  }

  // Store work queue length for specified iteration
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ void StoreQueueLength(SizeT queue_length,
                                                   IterationT iteration) {
    util::io::ModifiedStore<util::io::st::cg>::St(queue_length,
                                                  GetQueueCounter(iteration));
  }

  // Enqueues work from the specified iteration's queue counter, returning the
  // offset of that work (from zero) and incrementing it by count.
  // Typically called by thread-0
  template </*typename SizeT,*/ typename IterationT>
  __device__ __forceinline__ SizeT Enqueue(SizeT count, IterationT iteration) {
    SizeT old_value =
        util::AtomicInt<SizeT>::Add(GetQueueCounter(iteration), count);
    // printf("d_counters = %p, iteration = %lld, old_value = %lld, count =
    // %lld, blockIdx.x = %d\n",
    //    d_counters, (long long) iteration, (long long) old_value, (long
    //    long)count, blockIdx.x);
    return old_value;
  }

  // Sets the overflow counter to non-zero
  // template <typename SizeT>
  __device__ __forceinline__ void SetOverflow() {
    d_counters[QUEUE_COUNTERS + STEAL_COUNTERS] = 1;
  }

  cudaError_t Reset_(SizeT reset_val = 0, cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    for (SizeT i = 0; i < COUNTERS; i++) (*p_counters)[i] = reset_val;
    if (retval =
            p_counters->Move(util::HOST, util::DEVICE, COUNTERS, 0, stream))
      return retval;
    progress_selector = 0;
    return retval;
  }

  cudaError_t Init() {
    cudaError_t retval = cudaSuccess;
    if (retval = p_counters->Allocate(COUNTERS, util::HOST | util::DEVICE))
      return retval;
    d_counters = p_counters->GetPointer(util::DEVICE);
    progress_selector = 0;
    return retval;
  }

  cudaError_t Release() {
    progress_selector = 0;
    d_counters = NULL;
    return p_counters->Release();
  }
};

/**
 * Version of work progress with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base CtaWorkProgress
 * as parameters to kernels.
 */
template <typename SizeT>
class CtaWorkProgressLifetime : public CtaWorkProgress<SizeT> {
 protected:
  // GPU d_counters was allocated on
  int gpu;
  util::Array1D<int, SizeT, PINNED, cudaHostAllocMapped | cudaHostAllocPortable>
      counters;

 public:
  /**
   * Constructor
   */
  CtaWorkProgressLifetime() : CtaWorkProgress<SizeT>(), gpu(GR_INVALID_DEVICE) {
    counters.SetName("counters");
    this->p_counters = &counters;
  }

  /**
   * Destructor
   */
  virtual ~CtaWorkProgressLifetime() {
    // Release();
  }

  // Deallocates and resets the progress counters
  cudaError_t Release() {
    cudaError_t retval = cudaSuccess;

    if (gpu != GR_INVALID_DEVICE) {
      // Save current gpu
      int current_gpu;
      if (retval = util::GRError(
              cudaGetDevice(&current_gpu),
              "CtaWorkProgress cudaGetDevice failed: ", __FILE__, __LINE__))
        return retval;

      // Deallocate
      if (retval = util::GRError(
              cudaSetDevice(gpu),
              "CtaWorkProgress cudaSetDevice failed: ", __FILE__, __LINE__))
        return retval;

      // if (retval = util::GRError(cudaFree(d_counters),
      //    "CtaWorkProgress cudaFree d_counters failed: ", __FILE__, __LINE__))
      //    return retval;
      if (retval = CtaWorkProgress<SizeT>::Release()) return retval;

      // d_counters = NULL;
      gpu = GR_INVALID_DEVICE;

      // Restore current gpu
      if (retval = util::GRError(
              cudaSetDevice(current_gpu),
              "CtaWorkProgress cudaSetDevice failed: ", __FILE__, __LINE__))
        return retval;
    }
    return retval;
  }

  // Sets up the progress counters for the next kernel launch (lazily
  // allocating and initializing them if necessary)
  // template <typename SizeT>
  cudaError_t Init() {
    cudaError_t retval = cudaSuccess;

    // Make sure that our progress counters are allocated
    if (this->counters.GetPointer(util::DEVICE) == NULL) {
      // Allocate and initialize
      if (retval = util::GRError(
              cudaGetDevice(&gpu),
              "CtaWorkProgress cudaGetDevice failed: ", __FILE__, __LINE__))
        return retval;

      if (retval = CtaWorkProgress<SizeT>::Init()) return retval;

      if (retval = CtaWorkProgress<SizeT>::Reset_()) return retval;
    }

    // Update our progress counter selector to index the next progress counter
    // progress_selector ^= 1;

    return retval;
  }

  // Checks if overflow counter is set
  // template <typename SizeT>
  cudaError_t CheckOverflow(bool& overflow,
                            cudaStream_t stream = 0)  // out param
  {
    cudaError_t retval = cudaSuccess;

    // SizeT counter;
    if (retval =
            counters.Move(util::DEVICE, util::HOST, 1,
                          this->QUEUE_COUNTERS + this->STEAL_COUNTERS, stream))
      return retval;

    // if (retval = util::GRError(cudaMemcpy(
    //        &counter,
    //        ((SizeT*) d_counters) + QUEUE_COUNTERS + STEAL_COUNTERS,
    //        1 * sizeof(SizeT),
    //        cudaMemcpyDeviceToHost),
    //    "CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__))
    //    break;

    overflow = counters[this->QUEUE_COUNTERS + this->STEAL_COUNTERS];

    return retval;
  }

  // Acquire work queue length
  template <typename IterationT /*, typename SizeT*/>
  cudaError_t GetQueueLength(IterationT iteration, SizeT& queue_length,
                             bool DEBUG = false, cudaStream_t stream = 0,
                             bool skip_sync = false)  // out param
  {
    cudaError_t retval = cudaSuccess;

    IterationT queue_length_idx = iteration & 0x3;

    if (stream == 0) {
      if (!DEBUG)
        cudaMemcpy(&queue_length, this->d_counters + queue_length_idx,
                   sizeof(SizeT), cudaMemcpyDeviceToHost);
      else if (retval = util::GRError(
                   cudaMemcpy(&queue_length,
                              this->d_counters + queue_length_idx,
                              sizeof(SizeT), cudaMemcpyDeviceToHost),
                   "CtaWorkProgress cudaMemcpy d_counters failed", __FILE__,
                   __LINE__))
        return retval;
    } else {
      // printf("GetQueueLength using MemcpyAsync\n");
      if (!DEBUG)
        cudaMemcpyAsync(&queue_length, this->d_counters + queue_length_idx,
                        sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
      else if (retval = util::GRError(
                   cudaMemcpyAsync(
                       &queue_length, this->d_counters + queue_length_idx,
                       sizeof(SizeT), cudaMemcpyDeviceToHost, stream),
                   "CtaWorkProgress cudaMemcpyAsync d_counter failed.",
                   __FILE__, __LINE__))
        return retval;
      if (!skip_sync) {
        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "CtaWorkProgress GetQueueLength failed",
                                   __FILE__, __LINE__))
          return retval;
      }
    }
    /*if (retval = counters.Move(util::DEVICE, util::HOST,
        1, queue_length_idx, stream))
        return retval;
    if (!skip_sync)
    {
        if (retval = util::GRError(cudaStreamSynchronize(stream),
            "CtaWorkProgress GetQueueLength failed", __FILE__, __LINE__))
            return retval;
    }
    queue_length = counters[queue_length_idx];*/
    return retval;
  }

  template <typename IndexT /*, typename SizeT*/>
  SizeT* GetQueueLengthPointer(IndexT index) {
    IndexT queue_length_idx = index & 0x3;
    return counters.GetPointer(util::DEVICE) + queue_length_idx;
  }

  // Set work queue length
  template <typename IterationT /*, typename SizeT*/>
  cudaError_t SetQueueLength(IterationT iteration, SizeT queue_length,
                             bool DEBUG = false, cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;

    IterationT queue_length_idx = iteration & 0x3;
    /*if (stream == 0)
    {
        if (!DEBUG)
            cudaMemcpy(((SizeT*) d_counters) + queue_length_idx, &queue_length,
    sizeof(SizeT), cudaMemcpyHostToDevice); else if (retval =
    util::GRError(cudaMemcpy(
            ((SizeT*) d_counters) + queue_length_idx,
            &queue_length,
            1 * sizeof(SizeT),
            cudaMemcpyHostToDevice),
            "CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__))
    break; } else {
       // printf("gpu = %d, queue_idx = %d, d_counters = %p, stream = %d,
    queue_length = %d\n",gpu, queue_length_idx, d_counters, stream,
    queue_length);fflush(stdout);
        //util::MemsetKernel<<<1,1,0,stream>>>(((SizeT*) d_counters) +
    queue_length_idx, queue_length, 1); cudaMemcpyAsync(((SizeT*) d_counters) +
    queue_length_idx, &queue_length, sizeof(SizeT),
    cudaMemcpyHostToDevice,stream); if (DEBUG)
        {
            cudaStreamSynchronize(stream);
            retval = util::GRError("CtaWorkProgress MemsetKernel d_counters
    failed", __FILE__, __LINE__);
        }
    }*/
    this->counters[queue_length_idx] = queue_length;
    if (retval = this->counters.Move(util::HOST, util::DEVICE, 1,
                                     queue_length_idx, stream))
      return retval;
    if (DEBUG) {
      if (retval = util::GRError(cudaStreamSynchronize(stream),
                                 "CtaWorkProgress SetQueuelength failed",
                                 __FILE__, __LINE__))
        return retval;
    }

    return retval;
  }
};

}  // namespace util
}  // namespace gunrock
