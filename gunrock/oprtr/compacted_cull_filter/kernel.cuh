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
 * @brief compacted cull filter
 */

#pragma once

#include <cub/cub.cuh>
#include <gunrock/oprtr/filter/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace compacted_cull_filter {

template <typename Problem>
struct KernelPolicy
{
    enum {
        LOG_THREADS          = 8,
        THREADS              = 1 << LOG_THREADS,
        MAX_BLOCKS           = 1024,
        BLOCK_HASH_BITS      = 8,
        BLOCK_HASH_LENGTH    = 1 << BLOCK_HASH_BITS,
        BLOCK_HASH_MASK      = BLOCK_HASH_LENGTH -1,
        GLOBAL_LOAD_SIZE     = 4,
        SHARED_LOAD_SIZE     = 8 / sizeof(typename Problem::VertexId),
        WARP_SIZE            = GR_WARP_THREADS(CUDA_ARCH),
        LOG_WARP_SIZE        = 5,
        WARP_SIZE_MASK       = WARP_SIZE -1,
        WARP_HASH_BITS       = 7,
        WARP_HASH_LENGTH     = 1 << WARP_HASH_BITS,
        WARP_HASH_MASK       = WARP_HASH_LENGTH -1,
        WARPS                = THREADS / WARP_SIZE,
        ELEMENT_ID_MASK      = ~(1<<(sizeof(typename Problem::VertexId)*8-2)),

    };
};

template <typename KernelPolicy, typename Problem, typename Functor>
struct ThreadStorage{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::Value    Value;
    typedef typename Problem::DataSlice DataSlice;
    typedef typename Functor::LabelT    LabelT;

    VertexId vertices[KernelPolicy::GLOBAL_LOAD_SIZE];
    int       num_elements;
    int       block_num_elements;
    int       block_input_start;
    int       warp_id;
    int       lane_id;
    VertexId *warp_hash;
    VertexId *d_keys_in;
    VertexId *d_keys_out;
    SizeT     input_queue_length;
    SizeT    *d_output_counter;
    unsigned char *d_visited_mask;
    VertexId *d_labels;
    DataSlice *d_data_slice;
    LabelT     label;

    __device__ __forceinline__ ThreadStorage(
        VertexId     **block_warp_hash,
        VertexId      *_d_keys_in,
        VertexId      *_d_keys_out,
        SizeT          _input_queue_length,
        SizeT         *_d_output_counter,
        SizeT          _block_input_start,
        unsigned char *_d_visited_mask,
        DataSlice     *_d_data_slice,
        LabelT         _label) :
        num_elements        (0),
        block_num_elements  (0),
        block_input_start   (_block_input_start),
        warp_id             (threadIdx.x >> KernelPolicy::LOG_WARP_SIZE),
        lane_id             (threadIdx.x &  KernelPolicy::WARP_SIZE_MASK),
        d_keys_in           (_d_keys_in),
        d_keys_out          (_d_keys_out),
        input_queue_length  (_input_queue_length),
        d_output_counter    (_d_output_counter),
        d_visited_mask      (_d_visited_mask),
        d_data_slice        (_d_data_slice),
        label               (_label)
    {
        warp_hash = block_warp_hash[warp_id];
        d_labels = d_data_slice -> labels.GetPointer(util::DEVICE);
    }
};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
struct Cta
{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::Value    Value;
    typedef typename Functor::LabelT   LabelT;
    typedef Cta<KernelPolicy, Problem, Functor> CtaT;
    typedef ThreadStorage<KernelPolicy, Problem, Functor> ThreadStorageT;
    //typedef cub::BlockScan<int, KernelPolicy::THREADS, cub::BLOCK_SCAN_RAKING /*cub::BLOCK_SCAN_WARP_SCANS*/> BlockScanT;
    typedef cub::WarpScan<int> WarpScanT;

    VertexId vertices  [KernelPolicy::THREADS * KernelPolicy::GLOBAL_LOAD_SIZE];
    VertexId block_hash[KernelPolicy::BLOCK_HASH_LENGTH];
    VertexId warp_hash [KernelPolicy::WARPS][KernelPolicy::WARP_HASH_LENGTH];
    int    temp_space[KernelPolicy::THREADS];
    union {
        //typename cub::BlockLoadT ::TempStorage load_space;
        //typename cub::BlockStoreT::TempStorage store_space;
        //typename BlockScanT::TempStorage scan_space;
        typename WarpScanT::TempStorage scan_space[KernelPolicy::WARPS];
    } cub_storage;

    int    num_elements;
    SizeT  block_offset;

    __device__ __forceinline__ Cta()
    {}

    __device__ __forceinline__ void Init(
        ThreadStorageT &thread_data)
    {
        for (int i = threadIdx.x;
            i <  KernelPolicy::BLOCK_HASH_LENGTH;
            i += KernelPolicy::THREADS)
            block_hash[i] = util::InvalidValue<VertexId>();

        for (int i = thread_data.lane_id;
            i <  KernelPolicy::WARP_HASH_LENGTH;
            i += KernelPolicy::WARP_SIZE)
            warp_hash[thread_data.warp_id][i] = util::InvalidValue<VertexId>();
    }

    __device__ __forceinline__ void Load_from_Global(
        ThreadStorageT &thread_data)
    {
        //typedef typename util::VectorType<VertexId, KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD>::Type LoadT;
        SizeT thread_input_pos = thread_data.block_input_start +
            thread_data.warp_id * KernelPolicy::WARP_SIZE * KernelPolicy::GLOBAL_LOAD_SIZE + thread_data.lane_id;
        thread_data.num_elements = 0;
        //if (threadIdx.x == 0) printf("(%4d, %4d) : block_input_start = %d, input_queue_length = %d\n",
        //    blockIdx.x, threadIdx.x, thread_data.block_input_start, thread_data.input_queue_length);
        #pragma unroll
        for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
        {
            if (thread_input_pos >= thread_data.input_queue_length)
            {
                thread_data.vertices[i] = util::InvalidValue<VertexId>();
            } else {
                thread_data.vertices[i] = thread_data.d_keys_in[thread_input_pos];
                thread_data.num_elements ++;
                //printf("(%4d, %4d) : reading pos = %d, item = %d\n",
                //    blockIdx.x, threadIdx.x, thread_input_pos, thread_data.vertices[i]);
            }

            thread_input_pos += KernelPolicy::WARP_SIZE;
        }

        /*LoadT *keys_in = (LoadT*)(d_keys_in + thread_input_pos);
        *((LoadT*)thread_data.vertices) = *keys_in;
        if (thread_input_start + KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD >= input_queue_length)
        {
            thread_data.num_elements = input_queue_length - thread_input_start;
            for (int i=thread_data.num_elements;
                i < KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD; i++)
                thread_data.vertices[i] = util::InvalidValue<VertexId>();
        } else thread_data.num_elements = KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD;*/
    }

    __device__ __forceinline__ void Load_from_Shared(
        ThreadStorageT &thread_data)
    {
        /*typedef typename util::VectorType<VertexId, KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD>::Type LoadT;
        SizeT thread_input_start = threadIdx.x * KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD;
        if (thread_input_start >= thread_data.block_num_elements)
        {
            thread_data.num_elements = 0;
        } else {
            LoadT *keys_in = (LoadT*)(vertices + thread_input_start);
            *((LoadT*)thread_data.vertices) = *keys_in;
            if (thread_input_start + KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD >= thread_data.block_num_elements)
            {

            } else thread_data.num_elements = KernelPolicy::NUM_ELEMENT_PER_SHARED_LOAD;
        }*/
    }

    __device__ __forceinline__ void Store_to_Global(
        ThreadStorageT &thread_data)
    {
        //temp_space[threadIdx.x] = thread_data.num_elements;
        if (threadIdx.x == 0) num_elements = 0;
        __syncthreads();

        //BlockScanT(cub_storage.scan_space).ExclusiveSum(temp_space, temp_space, thread_data.block_num_elements);
        SizeT thread_offset = 0;
        WarpScanT(cub_storage.scan_space[thread_data.warp_id]).ExclusiveSum(thread_data.num_elements, thread_offset);
        int warp_offset;
        if (thread_data.lane_id == KernelPolicy::WARP_SIZE_MASK)
        {
            warp_offset = atomicAdd(&num_elements, thread_offset + thread_data.num_elements);
            //num_elements = thread_data.block_num_elements;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            block_offset = atomicAdd(thread_data.d_output_counter, num_elements);
        }
        __syncthreads();
        //if (thread_data.lane_id == KernelPolicy::WARP_SIZE_MASK)
        //    warp_offset += block_offset;
        warp_offset = cub::ShuffleIndex(warp_offset, KernelPolicy::WARP_SIZE_MASK);
        thread_offset += warp_offset + block_offset;
        //__syncthreads();
        for (int i=0; i<thread_data.num_elements; i++)
            thread_data.d_keys_out[thread_offset + i] = thread_data.vertices[i];
    }

    __device__ __forceinline__ void Store_to_Shared(
        ThreadStorageT &thread_data)
    {

    }

    __device__ __forceinline__ void Local_Compact(
        ThreadStorageT &thread_data)
    {
        int temp_size = 0;
        for (int i=0; i< thread_data.num_elements; i++)
        if (util::isValid(thread_data.vertices[i]))
        {
            thread_data.vertices[temp_size] = thread_data.vertices[i];
            temp_size ++;
        }
        thread_data.num_elements = temp_size;
    }

    __device__ __forceinline__ void BitMask_Cull(
        ThreadStorageT &thread_data)
    {
        #pragma unroll
        for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
        {
            if (!util::isValid(thread_data.vertices[i])) continue;
            // Location of mask byte to read
            SizeT mask_byte_offset = (thread_data.vertices[i] & KernelPolicy::ELEMENT_ID_MASK) >> 3;

            // Bit in mask byte corresponding to current vertex id
            unsigned char mask_bit = 1 << (thread_data.vertices[i] & 7);

            // Read byte from visited mask in tex
            unsigned char tex_mask_byte = tex1Dfetch(
                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,//cta->t_bitmask[0],
                mask_byte_offset);
            //unsigned char tex_mask_byte = cta->d_visited_mask[mask_byte_offset];

            if (mask_bit & tex_mask_byte)
            {
                // Seen it
                thread_data.vertices[i] = util::InvalidValue<VertexId>();
            } else {
                //unsigned char mask_byte = tex_mask_byte;
                //util::io::ModifiedLoad<util::io::ld::cg>::Ld(
                //    mask_byte, cta->d_visited_mask + mask_byte_offset);
                //mask_byte = cta->d_visited_mask[mask_byte_offset];

                //mask_byte |= tex_mask_byte;

                //if (mask_bit & mask_byte) {
                    // Seen it
                //    tile->element_id[LOAD][VEC] = util::InvalidValue<VertexId>();
                //} else {
                    // Update with best effort
                    //mask_byte |= mask_bit;
                    tex_mask_byte |= mask_bit;
                    util::io::ModifiedStore<util::io::st::cg>::St(
                        tex_mask_byte, //mask_byte,
                        thread_data.d_visited_mask + mask_byte_offset);
                //}
            }
        }
    }

    template <typename DummyT, bool ENABLE_IDEMPOTENCE>
    struct VertexC
    {
        static __device__ __forceinline__ void Cull(
            ThreadStorageT &thread_data)
        {}
    };

    template <typename DummyT>
    struct VertexC<DummyT, true>
    {
        static __device__ __forceinline__ void Cull(
            ThreadStorageT &thread_data)
        {
            #pragma unroll
            for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
            {
                if (!util::isValid(thread_data.vertices[i])) continue;
                VertexId row_id = thread_data.vertices[i] & KernelPolicy::ELEMENT_ID_MASK;
                if (thread_data.d_labels[row_id] != util::MaxValue<LabelT>())
                    thread_data.vertices[i] = util::InvalidValue<VertexId>();
            }
        }
    };

    template <typename DummyT>
    struct VertexC<DummyT, false>
    {
        static __device__ __forceinline__ void Cull(
            ThreadStorageT &thread_data)
        {}
    };

    __device__ __forceinline__ void Vertex_Cull(
        ThreadStorageT &thread_data)
    {
        VertexC<SizeT, Problem::ENABLE_IDEMPOTENCE>::Cull(thread_data);
    }

    __device__ __forceinline__ void History_Cull(
        ThreadStorageT &thread_data)
    {
        #pragma unroll
        for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
        {
            if (!util::isValid(thread_data.vertices[i])) continue;
            int hash = (thread_data.vertices[i]) & KernelPolicy::BLOCK_HASH_MASK;
            VertexId retrieved = block_hash[hash];

            if (retrieved == thread_data.vertices[i])
                // Seen it
                thread_data.vertices[i] = util::InvalidValue<VertexId>();
            else // Update it
                block_hash[hash] = thread_data.vertices[i];
        }
    }

    __device__ __forceinline__ void Warp_Cull(
        ThreadStorageT &thread_data)
    {
        #pragma unroll
        for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
        {
            if (!util::isValid(thread_data.vertices[i])) continue;
            //int warp_id = threadIdx.x >> 5;
            int hash    = thread_data.vertices[i] & (KernelPolicy::WARP_HASH_MASK);

            warp_hash[thread_data.warp_id][hash] = thread_data.vertices[i];
            VertexId retrieved = warp_hash[thread_data.warp_id][hash];

            if (retrieved == thread_data.vertices[i])
            {
                warp_hash[thread_data.warp_id][hash] = threadIdx.x;
                VertexId tid = warp_hash[thread_data.warp_id][hash];
                if (tid != threadIdx.x)
                    thread_data.vertices[i] = util::InvalidValue<VertexId>();
            }
        }
    }

    template <typename DummyT, bool ENABLE_IDEMPOTENCE, bool MARK_PREDECESSORS>
    struct VertexP
    {
        static __device__ __forceinline__ void Process(
            ThreadStorageT &thread_data)
        {}
    };

    template <typename DummyT>
    struct VertexP<DummyT, true, false>
    {
        static __device__ __forceinline__ void Process(
            ThreadStorageT &thread_data)
        {
            #pragma unroll
            for (int i=0; i<KernelPolicy::GLOBAL_LOAD_SIZE; i++)
            {
                if (!util::isValid(thread_data.vertices[i])) continue;
                VertexId row_id = thread_data.vertices[i] & KernelPolicy::ELEMENT_ID_MASK;

                if (thread_data.d_labels[row_id] != util::MaxValue<LabelT>())
                {
                    thread_data.vertices[i] = util::InvalidValue<VertexId>();
                } else {
                    if (Functor::CondFilter(
                        util::InvalidValue<VertexId>(),
                        row_id,
                        thread_data.d_data_slice,
                        util::InvalidValue<SizeT>(),
                        thread_data.label,
                        util::InvalidValue<SizeT>(),
                        util::InvalidValue<SizeT>()))
                    {
                        Functor::ApplyFilter(
                            util::InvalidValue<VertexId>(),
                            row_id,
                            thread_data.d_data_slice,
                            util::InvalidValue<SizeT>(),
                            thread_data.label,
                            util::InvalidValue<SizeT>(),
                            util::InvalidValue<SizeT>());
                    } else thread_data.vertices[i] = util::InvalidValue<VertexId>();
                }
            }
        }
    };

    template <typename DummyT, bool MARK_PREDECESSORS>
    struct VertexP<DummyT, false, MARK_PREDECESSORS>
    {
        static __device__ __forceinline__ void Process(
            ThreadStorageT &thread_data)
        {
            #pragma unroll
            for (int i=0; i < KernelPolicy::NUM_ELEMENT_PER_GLOBAL_LOAD; i++)
            {
                if (!util::isValid(thread_data.vertices[i])) continue;
                if (Functor::CondFilter(
                    util::InvalidValue<VertexId>(),
                    thread_data.vertices[i],
                    thread_data.d_data_slice,
                    util::InvalidValue<SizeT>(),
                    thread_data.label,
                    util::InvalidValue<SizeT>(),
                    util::InvalidValue<SizeT>()))
                {
                    Functor::ApplyFilter(
                        util::InvalidValue<VertexId>(),
                        thread_data.vertices[i],
                        thread_data.d_data_slice,
                        util::InvalidValue<SizeT>(),
                        thread_data.label,
                        util::InvalidValue<SizeT>(),
                        util::InvalidValue<SizeT>());
                } else thread_data.vertices[i] = util::InvalidValue<VertexId>();
            }
        }
    };

    __device__ __forceinline__ void Vertex_Process(
        ThreadStorageT &thread_data)
    {
        VertexP<SizeT, Problem::ENABLE_IDEMPOTENCE, Problem::MARK_PREDECESSORS>::Process(thread_data);
    }

    template <typename DummyT, bool ENABLE_IDEMPOTENCE>
    struct Kernel_{
        static __device__ __forceinline__ void Invoke(
            CtaT            &cta,
            ThreadStorageT  &thread_data)
        {}
    };

    template <typename DummyT>
    struct Kernel_<DummyT, true>
    {
        static __device__ __forceinline__ void Invoke(
            CtaT            &cta,
            ThreadStorageT  &thread_data)
        {
            cta. Load_from_Global(thread_data);
            cta. BitMask_Cull    (thread_data);
            cta. Vertex_Cull     (thread_data);
            cta. History_Cull    (thread_data);
            cta. Warp_Cull       (thread_data);
            cta. Vertex_Process  (thread_data);
            cta. Local_Compact   (thread_data);
            cta. Store_to_Global (thread_data);
        }
    };

    template <typename DummyT>
    struct Kernel_<DummyT, false>
    {
        static __device__ __forceinline__ void Invoke(
            CtaT            &cta,
            ThreadStorageT  &thread_data)
        {
            cta. Load_from_Global(thread_data);
            cta. Vertex_Cull     (thread_data);
            cta. Vertex_Process  (thread_data);
            cta. Local_Compact   (thread_data);
            cta. Store_to_Global (thread_data);
        }
    };

    __device__ __forceinline__ void Kernel(
        ThreadStorageT &thread_data)
    {
        Kernel_<Cta, Problem::ENABLE_IDEMPOTENCE>::Invoke(*this, thread_data);
    }
};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
__global__ void LaunchKernel(
    typename Functor::LabelT     label,
    bool                         queue_reset,
    typename Problem::VertexId   queue_index,
    typename Problem::SizeT      num_elements,
    typename Problem::VertexId  *d_keys_in,
    typename Problem::Value     *d_values_in,
    typename Problem::VertexId  *d_keys_out,
    typename Problem::DataSlice *d_data_slice,
    unsigned char               *d_visited_mask,
    util::CtaWorkProgress<typename Problem::SizeT>
                                 work_progress,
    typename Problem::SizeT      max_in_frontier,
    typename Problem::SizeT      max_out_frontier,
    util::KernelRuntimeStats     kernel_stats)
{
    if (threadIdx.x == 0)
    {
        if (queue_reset)
        {
            work_progress.StoreQueueLength(num_elements, queue_index);
        } else {
            num_elements = work_progress.LoadQueueLength(queue_index);
        }
        if (blockIdx.x == 0)
        {
            work_progress.StoreQueueLength(0, queue_index + 2);
            //printf("queue_reset = %s, num_elements = %d\n",
            //    queue_reset ? "true" : "false", num_elements);
        }
    }
    __syncthreads();

    __shared__ Cta<KernelPolicy, Problem, Functor> cta;
    ThreadStorage<KernelPolicy, Problem, Functor> thread_data(
        (typename Problem::VertexId**)cta.warp_hash,
        d_keys_in,
        d_keys_out,
        num_elements,
        work_progress.template GetQueueCounter<typename Problem::VertexId>(queue_index+1),
        (typename Problem::SizeT)blockIdx.x * KernelPolicy::THREADS * KernelPolicy::GLOBAL_LOAD_SIZE,
        d_visited_mask,
        d_data_slice,
        label);

    cta.Init  (thread_data);
    while (thread_data.block_input_start < num_elements)
    {
        cta.Kernel(thread_data);
        thread_data.block_input_start += KernelPolicy::THREADS * KernelPolicy::GLOBAL_LOAD_SIZE;
    }
}

} // namespace compacted_cull_filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
