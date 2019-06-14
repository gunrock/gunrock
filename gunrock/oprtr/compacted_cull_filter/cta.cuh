// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cta.cuh
 *
 * @brief block level operation for compacted cull filter
 */

#pragma once

#include <gunrock/oprtr/compacted_cull_filter/kernel_policy.cuh>
#include <gunrock/oprtr/cull_filter/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace compacted_cull_filter {

template <typename KernelPolicy, typename Problem, typename Functor>
struct ThreadWork {
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::Value Value;
  typedef typename Problem::DataSlice DataSlice;
  typedef typename Functor::LabelT LabelT;
  typedef typename KernelPolicy::SmemStorage SmemStorageT;

  VertexId vertices[KernelPolicy::GLOBAL_LOAD_SIZE];
  int num_elements;
  int block_num_elements;
  int block_input_start;
  int warp_id;
  int lane_id;
  VertexId *warp_hash;
  VertexId *d_keys_in;
  VertexId *d_keys_out;
  SizeT input_queue_length;
  SizeT *d_output_counter;
  unsigned char *d_visited_mask;
  VertexId *d_labels;
  DataSlice *d_data_slice;
  LabelT label;
  SmemStorageT &smem;

  __device__ __forceinline__ ThreadWork(
      // VertexId     **block_warp_hash,
      SmemStorageT &_smem, VertexId *_d_keys_in, VertexId *_d_keys_out,
      SizeT _input_queue_length, SizeT *_d_output_counter,
      SizeT _block_input_start, unsigned char *_d_visited_mask,
      DataSlice *_d_data_slice, LabelT _label)
      : smem(_smem),
        num_elements(0),
        block_num_elements(0),
        block_input_start(_block_input_start),
        warp_id(threadIdx.x >> KernelPolicy::LOG_WARP_SIZE),
        lane_id(threadIdx.x & KernelPolicy::WARP_SIZE_MASK),
        d_keys_in(_d_keys_in),
        d_keys_out(_d_keys_out),
        input_queue_length(_input_queue_length),
        d_output_counter(_d_output_counter),
        d_visited_mask(_d_visited_mask),
        d_data_slice(_d_data_slice),
        label(_label) {
    // warp_hash = smem.warp_hash[warp_id];
    d_labels = d_data_slice->labels.GetPointer(util::DEVICE);
  }
};

template <typename KernelPolicy, typename Problem, typename Functor>
struct Cta {
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::Value Value;
  typedef typename Functor::LabelT LabelT;
  typedef Cta<KernelPolicy, Problem, Functor> CtaT;
  typedef ThreadWork<KernelPolicy, Problem, Functor> ThreadWorkT;
  typedef typename KernelPolicy::SmemStorage SmemStorageT;
  typedef typename KernelPolicy::BlockScanT BlockScanT;
  typedef typename KernelPolicy::BlockLoadT BlockLoadT;

  SmemStorageT smem;

  __device__ __forceinline__ void Init(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    /*for (int i = threadIdx.x;
        i <  KernelPolicy::BLOCK_HASH_LENGTH;
        i += KernelPolicy::THREADS)
        smem.block_hash[i] = util::InvalidValue<VertexId>();

    for (int i = thread_work.lane_id;
        i <  KernelPolicy::WARP_HASH_LENGTH;
        i += KernelPolicy::WARP_SIZE)
        smem.warp_hash[thread_work.warp_id][i] =
    util::InvalidValue<VertexId>();*/
  }

  __device__ __forceinline__ void Load_from_Global(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    // typedef typename util::VectorType<VertexId,
    // KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD>::Type LoadT;
    if (thread_work.block_input_start +
            (KernelPolicy::GLOBAL_LOAD_SIZE << KernelPolicy::LOG_THREADS) <=
        thread_work.input_queue_length) {
      BlockLoadT(smem.cub_storage.load_space)
          .Load(thread_work.d_keys_in + thread_work.block_input_start,
                thread_work.vertices);
      thread_work.num_elements = KernelPolicy::GLOBAL_LOAD_SIZE;
    } else {
      SizeT thread_input_pos =
          thread_work.block_input_start +
          thread_work.warp_id *
              (KernelPolicy::GLOBAL_LOAD_SIZE << KernelPolicy::LOG_WARP_SIZE) +
          thread_work.lane_id;
      thread_work.num_elements = 0;

#pragma unroll
      for (int i = 0; i < KernelPolicy::GLOBAL_LOAD_SIZE; i++) {
        if (thread_input_pos >= thread_work.input_queue_length) {
          thread_work.vertices[i] = util::InvalidValue<VertexId>();
        } else {
          thread_work.vertices[i] = thread_work.d_keys_in[thread_input_pos];
          thread_work.num_elements++;
        }
        thread_input_pos += KernelPolicy::WARP_SIZE;
      }
    }

    /*LoadT *keys_in = (LoadT*)(d_keys_in + thread_input_pos);
    *((LoadT*)thread_data.vertices) = *keys_in;
    if (thread_input_start + KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD >=
    input_queue_length)
    {
        thread_data.num_elements = input_queue_length - thread_input_start;
        for (int i=thread_data.num_elements;
            i < KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD; i++)
            thread_data.vertices[i] = util::InvalidValue<VertexId>();
    } else thread_data.num_elements =
    KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD;*/
  }

  __device__ __forceinline__ void Load_from_Shared(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    /*typedef typename util::VectorType<VertexId,
    KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD>::Type LoadT; SizeT
    thread_input_start = threadIdx.x *
    KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD; if (thread_input_start >=
    thread_data.block_num_elements)
    {
        thread_data.num_elements = 0;
    } else {
        LoadT *keys_in = (LoadT*)(vertices + thread_input_start);
        *((LoadT*)thread_data.vertices) = *keys_in;
        if (thread_input_start + KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD >=
    thread_data.block_num_elements)
        {

        } else thread_data.num_elements =
    KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD;
    }*/
    int thread_pos = thread_work.warp_id * (KernelPolicy::GLOBAL_LOAD_SIZE
                                            << KernelPolicy::LOG_WARP_SIZE) +
                     thread_work.lane_id;
    thread_work.num_elements = 0;
#pragma unroll
    for (int i = 0; i < KernelPolicy::GLOBAL_LOAD_SIZE; i++) {
      if (thread_pos < smem.num_elements) {
        thread_work.vertices[i] = smem.vertices[thread_pos];
        thread_work.num_elements++;
      } else
        thread_work.vertices[i] = util::InvalidValue<VertexId>();
      thread_pos += KernelPolicy::WARP_SIZE;
    }
  }

  __device__ __forceinline__ void Store_to_Global(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    // temp_space[threadIdx.x] = thread_data.num_elements;
    /*if (threadIdx.x == 0) num_elements = 0;
    __syncthreads();

    WarpScanT(cub_storage.scan_space[thread_data.warp_id]).ExclusiveSum(thread_data.num_elements,
    thread_offset); int warp_offset; if (thread_data.lane_id ==
    KernelPolicy::WARP_SIZE_MASK)
    {
        warp_offset = atomicAdd(&num_elements, thread_offset +
    thread_data.num_elements);
        //num_elements = thread_data.block_num_elements;
    }
    __syncthreads();*/
    SizeT thread_offset = 0;  //, block_size = 0;
    BlockScanT(smem.cub_storage.scan_space)
        .ExclusiveSum(thread_work.num_elements,
                      thread_offset);  //, block_size);
    if (threadIdx.x == KernelPolicy::THREADS - 1) {
      smem.block_offset =
          atomicAdd(thread_work.d_output_counter,
                    thread_offset + thread_work.num_elements);  // block_size);
      // if (//block_offset > thread_data.input_queue_length ||
      //    num_elements > KernelPolicy::THREADS *
      //    KernelPolicy::GLOBAL_LOAD_SIZE) printf("(%4d, %4d) : num_elements =
      //    %d, block_offset = %d, input_queue_length = %d\n", blockIdx.x,
      //    threadIdx.x, num_elements, block_offset,
      //    thread_data.input_queue_length);
    }
    __syncthreads();
    // if (thread_data.lane_id == KernelPolicy::WARP_SIZE_MASK)
    //    warp_offset += block_offset;
    // warp_offset = cub::ShuffleIndex(warp_offset,
    // KernelPolicy::WARP_SIZE_MASK);
    thread_offset += smem.block_offset;
    //__syncthreads();
    //#pragma unroll
    for (int i = 0;
         i < /*KernelPolicy::GLOBAL_LOAD_SIZE*/ thread_work.num_elements; i++) {
      // if (i == thread_data.num_elements) break;
      // thread_data.d_keys_out[thread_offset + i] = thread_data.vertices[i];
      util::io::ModifiedStore<util::io::st::cg>::St(
          thread_work.vertices[i],
          thread_work.d_keys_out + (thread_offset + i));
    }
  }

  __device__ __forceinline__ void Store_to_Shared(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    SizeT thread_offset = 0, block_num_elements;
    // thread_work.num_elements = 0;
    //#pragma unroll
    // for (int i=0; i<thread_work.num_elements;i++)
    // if (util::isValid(thread_work.vertices[i])) thread_work.num_elements ++;
    BlockScanT(smem.cub_storage.scan_space)
        .ExclusiveSum(thread_work.num_elements, thread_offset,
                      block_num_elements);
    if (threadIdx.x == 0) smem.num_elements = block_num_elements;
    // thread_work.num_elements = 0;
    //#pragma unroll
    for (int i = 0; i < thread_work.num_elements; i++) {
      // if (!util::isValid(thread_work.vertices[i])) continue;
      smem.vertices[thread_offset + i] = thread_work.vertices[i];
      // thread_work.num_elements ++;
    }
    __syncthreads();
  }

  __device__ __forceinline__ void Local_Compact(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    int temp_size = 0;
#pragma unroll
    for (int i = 0;
         i < /*thread_data.num_elements*/ KernelPolicy::GLOBAL_LOAD_SIZE; i++) {
      if (!util::isValid(thread_work.vertices[i])) continue;
      // if (temp_size != i)
      thread_work.vertices[temp_size] = thread_work.vertices[i];
      temp_size++;
    }
    thread_work.num_elements = temp_size;
  }

  __device__ __forceinline__ void BitMask_Cull(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
#pragma unroll
    for (int i = 0;
         i < KernelPolicy::GLOBAL_LOAD_SIZE /* thread_data.num_elements*/;
         i++) {
      if (!util::isValid(thread_work.vertices[i])) continue;
      // Location of mask byte to read
      SizeT mask_byte_offset =
          (thread_work.vertices[i]  //& KernelPolicy::ELEMENT_ID_MASK
           ) >>
          3;

      // Bit in mask byte corresponding to current vertex id
      unsigned char mask_bit = 1 << (thread_work.vertices[i] & 7);

      // Read byte from visited mask in tex
      // unsigned char tex_mask_byte = tex1Dfetch(
      //    gunrock::oprtr::cull_filter::BitmaskTex<unsigned
      //    char>::ref,//cta->t_bitmask[0], mask_byte_offset);
      // unsigned char tex_mask_byte = cta->d_visited_mask[mask_byte_offset];
      unsigned char tex_mask_byte =
          _ldg(thread_work.d_visited_mask + mask_byte_offset);

      if (mask_bit & tex_mask_byte) {
        // Seen it
        thread_work.vertices[i] = util::InvalidValue<VertexId>();
      } else {
        // unsigned char mask_byte = tex_mask_byte;
        // util::io::ModifiedLoad<util::io::ld::cg>::Ld(
        //    mask_byte, cta->d_visited_mask + mask_byte_offset);
        // mask_byte = cta->d_visited_mask[mask_byte_offset];

        // mask_byte |= tex_mask_byte;

        // if (mask_bit & mask_byte) {
        // Seen it
        //    tile->element_id[LOAD][VEC] = util::InvalidValue<VertexId>();
        //} else {
        // Update with best effort
        // mask_byte |= mask_bit;

        tex_mask_byte |= mask_bit;
        util::io::ModifiedStore<util::io::st::cg>::St(
            tex_mask_byte,  // mask_byte,
            thread_work.d_visited_mask + mask_byte_offset);
        // thread_work.d_visited_mask[mask_byte_offset] |= mask_bit;
        /// thread_data.d_visited_mask [mask_byte_offset] = tex_mask_byte;
        //}
      }
    }
  }

  template <typename DummyT, bool ENABLE_IDEMPOTENCE>
  struct VertexC {
    static __device__ __forceinline__ void Cull(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {}
  };

  template <typename DummyT>
  struct VertexC<DummyT, true> {
    static __device__ __forceinline__ void Cull(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {
#pragma unroll
      for (int i = 0;
           i < KernelPolicy::GLOBAL_LOAD_SIZE /*thread_data.num_elements*/;
           i++) {
        if (!util::isValid(thread_work.vertices[i])) continue;
        VertexId row_id =
            thread_work.vertices[i];  //& KernelPolicy::ELEMENT_ID_MASK;
        if (thread_work.d_labels[row_id] != util::MaxValue<LabelT>())
          thread_work.vertices[i] = util::InvalidValue<VertexId>();
      }
    }
  };

  template <typename DummyT>
  struct VertexC<DummyT, false> {
    static __device__ __forceinline__ void Cull(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {}
  };

  __device__ __forceinline__ void Vertex_Cull(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    VertexC<SizeT, Problem::ENABLE_IDEMPOTENCE>::Cull(thread_work);
  }

  __device__ __forceinline__ void History_Cull(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
#pragma unroll
    for (int i = 0;
         i < KernelPolicy::GLOBAL_LOAD_SIZE /*thread_data.num_elements*/; i++) {
      if (!util::isValid(thread_work.vertices[i])) continue;
      int hash = (thread_work.vertices[i]) & KernelPolicy::BLOCK_HASH_MASK;
      VertexId retrieved = smem.block_hash[hash];

      if (retrieved == thread_work.vertices[i])
        // Seen it
        thread_work.vertices[i] = util::InvalidValue<VertexId>();
      else  // Update it
        smem.block_hash[hash] = thread_work.vertices[i];
    }
  }

  __device__ __forceinline__ void Warp_Cull(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
#pragma unroll
    for (int i = 0;
         i < KernelPolicy::GLOBAL_LOAD_SIZE /* thread_data.num_elements*/;
         i++) {
      if (!util::isValid(thread_work.vertices[i])) continue;
      // int warp_id = threadIdx.x >> 5;
      int hash = thread_work.vertices[i] & (KernelPolicy::WARP_HASH_MASK);

      smem.warp_hash[thread_work.warp_id][hash] = thread_work.vertices[i];
      // thread_work.warp_hash[hash] = thread_work.vertices[i];
      VertexId retrieved = smem.warp_hash[thread_work.warp_id][hash];
      // VertexId retrieved = thread_work.warp_hash[hash];

      if (retrieved == thread_work.vertices[i]) {
        smem.warp_hash[thread_work.warp_id][hash] = threadIdx.x;
        // thread_work.warp_hash[hash] = threadIdx.x;
        VertexId tid = smem.warp_hash[thread_work.warp_id][hash];
        // VertexId tid = thread_work.warp_hash[hash];
        if (tid != threadIdx.x)
          thread_work.vertices[i] = util::InvalidValue<VertexId>();
      }
    }
  }

  template <typename DummyT, bool ENABLE_IDEMPOTENCE, bool MARK_PREDECESSORS>
  struct VertexP {
    static __device__ __forceinline__ void Process(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {}
  };

  template <typename DummyT>
  struct VertexP<DummyT, true, false> {
    static __device__ __forceinline__ void Process(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {
      //#pragma unroll
      for (int i = 0; i < /*KernelPolicy::GLOBAL_LOAD_SIZE*/
                      thread_work.num_elements;
           i++) {
        if (!util::isValid(thread_work.vertices[i])) continue;
        VertexId row_id =
            thread_work.vertices[i];  // & KernelPolicy::ELEMENT_ID_MASK;

        if (thread_work.d_labels[row_id] != util::MaxValue<LabelT>()) {
          thread_work.vertices[i] = util::InvalidValue<VertexId>();
        } else {
          if (Functor::CondFilter(
                  util::InvalidValue<VertexId>(), row_id,
                  thread_work.d_data_slice, util::InvalidValue<SizeT>(),
                  thread_work.label, util::InvalidValue<SizeT>(),
                  util::InvalidValue<SizeT>())) {
            Functor::ApplyFilter(util::InvalidValue<VertexId>(), row_id,
                                 thread_work.d_data_slice,
                                 util::InvalidValue<SizeT>(), thread_work.label,
                                 util::InvalidValue<SizeT>(),
                                 util::InvalidValue<SizeT>());
          } else
            thread_work.vertices[i] = util::InvalidValue<VertexId>();
        }
      }
    }
  };

  template <typename DummyT, bool MARK_PREDECESSORS>
  struct VertexP<DummyT, false, MARK_PREDECESSORS> {
    static __device__ __forceinline__ void Process(ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {
#pragma unroll
      for (int i = 0;
           i < KernelPolicy::GLOBAL_LOAD_SIZE /* thread_data.num_elements*/;
           i++) {
        if (!util::isValid(thread_work.vertices[i])) continue;
        if (Functor::CondFilter(
                util::InvalidValue<VertexId>(), thread_work.vertices[i],
                thread_work.d_data_slice, util::InvalidValue<SizeT>(),
                thread_work.label, util::InvalidValue<SizeT>(),
                util::InvalidValue<SizeT>())) {
          Functor::ApplyFilter(
              util::InvalidValue<VertexId>(), thread_work.vertices[i],
              thread_work.d_data_slice, util::InvalidValue<SizeT>(),
              thread_work.label, util::InvalidValue<SizeT>(),
              util::InvalidValue<SizeT>());
        } else
          thread_work.vertices[i] = util::InvalidValue<VertexId>();
      }
    }
  };

  __device__ __forceinline__ void Vertex_Process(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    VertexP<SizeT, Problem::ENABLE_IDEMPOTENCE,
            Problem::MARK_PREDECESSORS>::Process(thread_work);
  }

  template <typename DummyT, bool ENABLE_IDEMPOTENCE>
  struct Kernel_ {
    static __device__ __forceinline__ void Invoke(CtaT &cta,
                                                  ThreadWorkT &thread_work) {}
  };

  template <typename DummyT>
  struct Kernel_<DummyT, true> {
    static __device__ __forceinline__ void Invoke(CtaT &cta,
                                                  ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {
      cta.Load_from_Global(thread_work);
      // cta.Warp_Cull       (thread_work);
      // cta.History_Cull    (thread_work);
      cta.BitMask_Cull(thread_work);
      cta.Local_Compact(thread_work);
      cta.Store_to_Shared(thread_work);
      cta.Load_from_Shared(thread_work);
      if (thread_work.num_elements > 0) {
        // cta.Vertex_Cull     (thread_work);
        // cta. Local_Compact   (thread_data);
        cta.Vertex_Process(thread_work);
        cta.Local_Compact(thread_work);
      }
      cta.Store_to_Global(thread_work);
    }
  };

  template <typename DummyT>
  struct Kernel_<DummyT, false> {
    static __device__ __forceinline__ void Invoke(CtaT &cta,
                                                  ThreadWorkT &thread_work)
    // SmemStorageT &smem)
    {
      cta.Load_from_Global(thread_work);
      // cta.Vertex_Cull     (thread_work);
      cta.Vertex_Process(thread_work);
      cta.Local_Compact(thread_work);
      cta.Store_to_Global(thread_work);
    }
  };

  __device__ __forceinline__ void Kernel(ThreadWorkT &thread_work)
  // SmemStorageT &smem)
  {
    Kernel_<SizeT, Problem::ENABLE_IDEMPOTENCE>::Invoke(*this, thread_work);
    /*Load_from_Global();
    if (Problem::ENABLE_IDEMPOTENCE)
    {
        BitMask_Cull    ();
        Vertex_Cull     ();
        History_Cull    ();
        Warp_Cull       ();
    } else Vertex_Cull  ();
    Vertex_Process  ();
    Local_Compact   ();
    Store_to_Global ();*/
  }
};  // end of Cta

}  // namespace compacted_cull_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
