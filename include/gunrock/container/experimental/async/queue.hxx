#pragma once

#include <inttypes.h>
#include <limits>
#include <assert.h>

#include <gunrock/util/experimental/async/util.hxx>

namespace gunrock {
namespace experimental {
namespace async {

#define PADDING_SIZE (64)  // 32 gets bad performance

template <typename T, typename CounterT>
struct Queue {
  T* queue;
  CounterT capacity;
  volatile CounterT *start, *end, *start_alloc, *end_alloc, *end_max,
      *end_count;
  volatile CounterT* stop;
  uint32_t min_iter;
  uint32_t queue_id = 0;  // remove queue_id change the structure layout and
                          // gets bad performance
  int num_queues = 1;

  Queue() {}

  __host__ void init(CounterT _capacity,
                     T* _queue,
                     volatile CounterT* _start,
                     volatile CounterT* _end,
                     volatile CounterT* _start_alloc,
                     volatile CounterT* _end_alloc,
                     volatile CounterT* _end_max,
                     volatile CounterT* _end_count,
                     volatile CounterT* _stop,
                     int _num_queues,
                     uint32_t _queue_id,
                     uint32_t _min_iter = 800) {
    capacity = _capacity;
    queue = _queue;

    start = _start;
    end = _end;
    start_alloc = _start_alloc;
    end_alloc = _end_alloc;
    end_max = _end_max;
    end_count = _end_count;
    stop = _stop;

    num_queues = _num_queues;
    queue_id = _queue_id;
    min_iter = _min_iter;
  }

  __host__ void release() {
    if (queue != NULL)
      cudaFree(queue);
  }

  __host__ void reset() {
    cudaMemset((void*)queue, -1, sizeof(T) * capacity);
    cudaMemset((void*)start, 0, sizeof(CounterT));
    cudaMemset((void*)start_alloc, 0, sizeof(CounterT));
    cudaMemset((void*)end, 0, sizeof(CounterT));
    cudaMemset((void*)end_alloc, 0, sizeof(CounterT));
    cudaMemset((void*)end_max, 0, sizeof(CounterT));
    cudaMemset((void*)end_count, 0, sizeof(CounterT));
    cudaMemset((void*)stop, 0, sizeof(CounterT));
  }

  __forceinline__ __device__ T get(CounterT) const;
  __forceinline__ __device__ CounterT next() const;
  __forceinline__ __device__ void update_end() const;

  template <typename Functor, typename... Args>
  __host__ void launch_thread(int numBlock,
                              int numThread,
                              cudaStream_t stream,
                              Functor f,
                              Args... arg);
};

template <typename T, typename CounterT>
__forceinline__ __device__ T Queue<T, CounterT>::get(CounterT index) const {
  return ((volatile T*)queue)[index];
}

template <typename T, typename CounterT>
__forceinline__ __device__ CounterT Queue<T, CounterT>::next() const {
  unsigned mask = __activemask();                    // active threads in warp
  uint32_t total = __popc(mask);                     // num threads in warp
  unsigned int rank = __popc(mask & lanemask_lt());  // rank of current thread
  int leader = __ffs(mask) - 1;                      // ID of leader

  CounterT alloc;
  if (rank == 0) {
    alloc = atomicAdd((CounterT*)start_alloc,
                      total);  // Get + increment index to read from
  }

  __syncwarp(mask);  // Why do we need this?

  alloc = __shfl_sync(
      mask, alloc, leader);  // Share starting index to read from among threads
  return alloc + rank;       // Return index for current thread
}

template <typename T, typename CounterT>
__forceinline__ __device__ void Queue<T, CounterT>::update_end() const {
  if (LANE_ == 0) {
    CounterT end_count_now = *(end_count);

    // // >> Testing
    // CounterT end_ = *(end);
    // if(end_ != end_count_now) assert(end_ < end_count_now);
    // // <<

    if ((*(end_max) ==
         end_count_now) &&  // if everything up to max position is valid AND
        (*(end) != end_count_now)  // hasn't already been updated / are more
                                   // items in queue
    ) {
      // printf("update end\n");
      atomicMax((CounterT*)end, end_count_now);  // increment end
    }
  }
}

template <typename T, typename CounterT, typename Functor, typename... Args>
__launch_bounds__(512, 2) __global__
    void _launch_thread(Queue<T, CounterT> q, Functor F, Args... args) {
  CounterT index = TID;
  CounterT end = *(q.end);
  uint32_t iter = 0;

  do {
    // Run
    while (index <
           end) {  // Consume while there are valid elements in the queue
      T task = q.get(index);
      assert(task != -1);
      F(task, args...);
      __syncwarp();
      index = q.next();
    }

    __syncwarp();
    end = *(q.end);

    if (index >= end) {
      if (*(q.stop) == blockDim.x * gridDim.x / 32 * q.num_queues)
        break;  // All threads have finished?

      iter++;
      if (iter == q.min_iter) {  // If we've done enough iterations?
        if (LANE_ < q.num_queues) {
          atomicAdd((CounterT*)(q.stop + LANE_ * 64),
                    1);  // Record that LANE has finished?
        }
      }

      q.update_end();  // Is this redundant w/ end update in `push`? Seems like
                       // it...
      __syncwarp();
    }

  } while (true);

  __syncwarp();
}

template <typename T, typename CounterT>
template <typename Functor, typename... Args>
void Queue<T, CounterT>::launch_thread(int numBlock,
                                       int numThread,
                                       cudaStream_t stream,
                                       Functor f,
                                       Args... arg) {
  _launch_thread<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
// Queues

template <typename T, typename CounterT = uint32_t>
struct Queues {
  T* queue;
  CounterT capacity;
  uint32_t num_queues;

  volatile CounterT* counters;
  volatile CounterT *start, *end, *start_alloc, *end_alloc, *end_max,
      *end_count, *stop;
  int num_counters = 7;
  int num_block;
  int num_thread;

  uint32_t min_iter;
  Queue<T, CounterT>* worklist;
  cudaStream_t* streams;

  __host__ void init(CounterT _capacity,
                     uint32_t _num_q = 8,
                     int _num_block = 280,
                     int _num_thread = 256,
                     uint32_t _min_iter = 800) {
    capacity = _capacity;
    num_queues = _num_q;
    min_iter = _min_iter;
    num_block = _num_block;
    num_thread = _num_thread;

    // Allocate queue memory
    auto queue_size = sizeof(T) * capacity * num_queues;
    cudaMalloc(&queue, queue_size);
    cudaMemset((void*)queue, -1, queue_size);

    // Allocate counter memory
    auto counter_size =
        sizeof(CounterT) * num_counters * num_queues * PADDING_SIZE;
    cudaMalloc(&counters, counter_size);
    cudaMemset((void*)counters, 0, counter_size);

    start = &counters[0 * num_queues * PADDING_SIZE];
    start_alloc = &counters[1 * num_queues * PADDING_SIZE];
    end_alloc = &counters[2 * num_queues * PADDING_SIZE];
    end = &counters[3 * num_queues * PADDING_SIZE];
    end_max = &counters[4 * num_queues * PADDING_SIZE];
    end_count = &counters[5 * num_queues * PADDING_SIZE];
    stop = &counters[6 * num_queues * PADDING_SIZE];

    worklist =
        (Queue<T, CounterT>*)malloc(sizeof(Queue<T, CounterT>) * num_queues);
    streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_queues);

    for (uint64_t q_id = 0; q_id < num_queues; q_id++) {
      auto q_offset = q_id * capacity;
      auto c_offset = q_id * PADDING_SIZE;
      worklist[q_id].init(
          capacity, queue + q_offset, start + c_offset, end + c_offset,
          start_alloc + c_offset, end_alloc + c_offset, end_max + c_offset,
          end_count + c_offset, stop, num_queues, q_id, min_iter);

      cudaStreamCreateWithFlags(&streams[q_id], cudaStreamNonBlocking);
    }
  }

  __host__ void release() {
    if (queue != NULL)
      cudaFree(queue);
    if (counters != NULL)
      cudaFree((void*)counters);
    if (streams != NULL)
      free(streams);
    if (worklist != NULL)
      free(worklist);
  }

  __device__ void push(T item) {
    // printf("push | %d\n", item);

    unsigned mask =
        __activemask();  // 32-bit mask indicating active threads in warp
    uint32_t total = __popc(mask);  // Number of active threads
    unsigned int rank = __popc(mask & lanemask_lt());  // Rank of current thread
    int leader =
        __ffs(mask) -
        1;  // Position of least sig. bit -> whether thread is min active

    uint64_t q_id = WARPID % num_queues;
    uint64_t c_offset = q_id * PADDING_SIZE;

    CounterT alloc;
    if (rank == 0) {
      alloc = atomicAdd((CounterT*)(end_alloc + c_offset),
                        total);  // Get + increment index to write to
    }

    alloc =
        __shfl_sync(mask, alloc, leader);  // copy alloc to all active threads
    assert(alloc + total <= capacity);

    queue[q_id * capacity + (alloc + rank)] =
        item;  // Insert item into queue at alloc + rank
    __threadfence();

    if (rank == 0) {
      CounterT end_count_old =
          atomicAdd((CounterT*)(end_count + c_offset),
                    total);  // Update number of items in queue
      CounterT end_max_old =
          atomicMax((CounterT*)(end_max + c_offset),
                    alloc + total);  // Update maximum position of item in queue

      // <<
      // ?? This might be redundant, given `update_end` in `_launch_thread`
      CounterT end_count_new = end_count_old + total;
      CounterT end_max_new = max(end_max_old, alloc + total);
      if (end_count_new == end_max_new) {  // If queue is full up to last
                                           // position, bump end counter
        __threadfence();
        atomicMax((CounterT*)(end + c_offset), end_max_new);
      }
      // >>
    }
    __syncwarp(mask);
  }

  template <typename Functor>
  __host__ void launch_thread(Functor f) {
    for (int i = 0; i < num_queues; i++)
      worklist[i].launch_thread(num_block / num_queues, num_thread, streams[i],
                                f, *this);
  }

  __host__ void reset() {
    for (int i = 0; i < num_queues; i++)
      worklist[i].reset();

    cudaDeviceSynchronize();
  }

  __host__ void sync() {
    for (int i = 0; i < num_queues; i++)
      cudaStreamSynchronize(streams[i]);
  }
};

}  // namespace async
}  // namespace experimental
}  // namespace gunrock