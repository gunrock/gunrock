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
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/device_intrinsics.cuh>

#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>
#include <gunrock/oprtr/cull_filter/cta.cuh>

#include <gunrock/oprtr/edge_map_partitioned_cull/kernel_policy.cuh>
#include <gunrock/oprtr/advance_base.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned_cull {

/**
 * Arch dispatch
 */
template <
    typename VertexId, 
    typename SizeT>
struct LoadRowOffset {};

template <typename VertexId>
struct LoadRowOffset<VertexId, int>
{
    static __device__ __forceinline__ int Load 
        (int *&d_row_offsets, VertexId &pos)
    {
        //return tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<int>::row_offsets, pos);
        return _ldg(d_row_offsets + pos);
    }    
};

template <typename VertexId>
struct LoadRowOffset<VertexId, long long>
{
    static __device__ __forceinline__ long long Load 
        (long long *&d_row_offsets, VertexId &pos)
    {    
        return _ldg(d_row_offsets + pos);
    }    
};

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE,
    gunrock::oprtr::advance::REDUCE_OP   R_OP,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{};

template <
    typename _Problem,
    int _CUDA_ARCH,
    int _MAX_CTA_OCCUPANCY,
    int _LOG_THREADS,
    int _LOG_BLOCKS,
    int _LIGHT_EDGE_THRESHOLD>
__device__ __forceinline__ void KernelPolicy
    <_Problem, _CUDA_ARCH, _MAX_CTA_OCCUPANCY,
    _LOG_THREADS, _LOG_BLOCKS, _LIGHT_EDGE_THRESHOLD>
    ::SmemStorage::Init(
    typename _Problem::VertexId   &queue_index,
    typename _Problem::DataSlice *&d_data_slice,
    typename _Problem::SizeT     *&d_scanned_edges,
    typename _Problem::SizeT     *&partition_starts,
    typename _Problem::SizeT      &partition_size,
    typename _Problem::SizeT      &input_queue_len,
    typename _Problem::SizeT     *&output_queue_len,
    util::CtaWorkProgress<typename _Problem::SizeT>
                                  &work_progress)
{
    d_output_counter = work_progress.template GetQueueCounter<VertexId>(queue_index + 1);
    //int lane_id = threadIdx.x & KernelPolicy::WARP_SIZE_MASK;
    //int warp_id = threadIdx.x >> KernelPolicy::LOG_WARP_SIZE;
    d_labels = d_data_slice -> labels.GetPointer(util::DEVICE);
    d_visited_mask = (_Problem::ENABLE_IDEMPOTENCE) ? d_data_slice -> visited_mask.GetPointer(util::DEVICE) : NULL;

    if (partition_starts != NULL)
    {
        block_output_start = (SizeT)blockIdx.x * partition_size;
        if (block_output_start >= output_queue_len[0]) return;
        block_output_end   = min(
            block_output_start + partition_size, output_queue_len[0]);
        block_output_size  = block_output_end - block_output_start;
        block_input_end    = (blockIdx.x == gridDim.x - 1) ?
            input_queue_len : min(
            partition_starts[blockIdx.x + 1] , input_queue_len);
        if (block_input_end < input_queue_len &&
            block_output_end > (block_input_end > 0 ? d_scanned_edges[block_input_end-1] : 0))
            block_input_end ++;

        block_input_start = partition_starts[blockIdx.x];
        block_first_v_skip_count =
            (block_output_start != d_scanned_edges[block_input_start]) ?
                block_output_start -
                    (block_input_start > 0 ? d_scanned_edges[block_input_start -1] : 0)
                : 0;
        iter_input_start = block_input_start;
    }

    /*printf("(%4d, %4d) : block_input = %d ~ %d, %d "
        "block_output = %d ~ %d, %d\n",
        blockIdx.x, threadIdx.x,
        block_input_start, block_input_end, block_input_end - block_input_start + 1,
        block_output_start, block_output_end, block_output_size);*/
}

/*
 * @brief Dispatch data structure.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE,
    gunrock::oprtr::advance::REDUCE_OP   R_OP>
struct Dispatch<KernelPolicy, Problem, Functor,
    ADVANCE_TYPE, R_TYPE, R_OP, true>
{
    typedef typename KernelPolicy::VertexId         VertexId;
    typedef typename KernelPolicy::SizeT            SizeT;
    typedef typename KernelPolicy::Value            Value;
    typedef typename Problem::DataSlice             DataSlice;
    typedef typename Functor::LabelT                LabelT;
    typedef typename KernelPolicy::BlockScanT       BlockScanT;
    typedef typename Problem::MaskT                 MaskT;
    //typedef typename KernelPolicy::BlockLoadT       BlockLoadT;

    static __device__ __forceinline__ SizeT GetNeighborListLength(
        SizeT     *&d_row_offsets,
        VertexId  *&d_column_indices,
        VertexId   &d_vertex_id,
        SizeT      &max_vertex,
        SizeT      &max_edge)
        //gunrock::oprtr::advance::TYPE &ADVANCE_TYPE)
    {
        SizeT first  = LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets, d_vertex_id);
            /*(d_vertex_id >= max_vertex) ?
            max_edge :*/ //d_row_offsets[d_vertex_id];
            //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  d_vertex_id);
            //_ldg(d_row_offsets + (d_vertex_id));
        SizeT second = LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets, d_vertex_id + 1);
            /*(d_vertex_id + 1 >= max_vertex) ?
            max_edge :*/ //d_row_offsets[d_vertex_id+1];
            //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  d_vertex_id + 1);
            //_ldg(d_row_offsets + (d_vertex_id+1));

        //printf(" d_vertex_id = %d, max_vertex = %d, max_edge = %d, first = %d, second = %d\n",
        //       d_vertex_id, max_vertex, max_edge, first, second);
        return /*(second > first) ?*/ second - first/* : 0*/;
    }

    static __device__ __forceinline__ void GetEdgeCounts(
        SizeT    *&d_row_offsets,
        VertexId *&d_column_indices,
        SizeT    *&d_column_offsets,
        VertexId *&d_row_indices,
        VertexId *&d_queue,
        SizeT    *&d_scanned_edges,
        SizeT     &num_elements,
        SizeT     &max_vertex,
        SizeT     &max_edge,
        //gunrock::oprtr::advance::TYPE &ADVANCE_TYPE,
        bool      &in_inv,
        bool      &out_inv)
    {
        //int tid = threadIdx.x;
        //int bid = blockIdx.x;

        VertexId thread_pos = (VertexId)blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_pos > num_elements)// || my_id >= max_edge)
            return;

        VertexId v_id;
        //printf("in_inv:%d, my_id:%d, column_idx:%d\n", in_inv, my_id, column_index);
        if (thread_pos < num_elements)
        {
            if (ADVANCE_TYPE == gunrock::oprtr::advance::V2E ||
                ADVANCE_TYPE == gunrock::oprtr::advance::V2V)
            {
                v_id = d_queue[thread_pos];
            } else {
                v_id = (in_inv) ?
                    d_row_indices[thread_pos] : d_column_indices[thread_pos];
            }
        } else v_id = (VertexId) -1;
        if (v_id < 0 || v_id > max_vertex)
        {
            d_scanned_edges[thread_pos] = 0;
            return;
        }

        //printf("my_id:%d, vid bef:%d\n", my_id, v_id);
        // add a zero length neighbor list to the end (this for getting both exclusive and inclusive scan in one array)
        //SizeT ncount = (!out_inv) ?
        d_scanned_edges[thread_pos] = (!out_inv) ?
            GetNeighborListLength(d_row_offsets, d_column_indices, v_id, max_vertex, max_edge):
            GetNeighborListLength(d_column_offsets, d_row_indices, v_id, max_vertex, max_edge);
        //printf("my_id:%d, out_inv:%d, vid:%d, ncount:%d\n", my_id, out_inv, v_id, ncount);
        //SizeT num_edges = (thread_pos == num_elements) ? 0 : ncount;
        //printf("%d, %d, %dG\t", my_id, v_id, num_edges);
        //d_scanned_edges[thread_pos] = num_edges;
    }

    static __device__ __forceinline__ void MarkPartitionSizes(
        SizeT        *&d_needles,
        SizeT         &split_val,
        SizeT         &num_elements,
        SizeT         &output_queue_length)
    {
        VertexId thread_pos = (VertexId) blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_pos >= num_elements) return;
        SizeT try_output = (SizeT) split_val * thread_pos;
        d_needles[thread_pos] = (try_output > output_queue_length) ?
            output_queue_length : try_output;
    }

    static __device__ __forceinline__ void Write_Global_Output(
        typename KernelPolicy::SmemStorage &smem_storage,
        SizeT                     &output_pos,
        SizeT                     &thread_output_count,
        LabelT                    &label,
        VertexId                 *&d_keys_out)
    {
        /*if (threadIdx.x == 0)
            smem_storage.block_count = 0;
        __syncthreads();
        if (thread_output_count != 0)
            output_pos = atomicAdd(&smem_storage.block_count, thread_output_count);
        __syncthreads();
        if (threadIdx.x == 0)
            smem_storage.block_offset = atomicAdd(smem_storage.d_output_counter, smem_storage.block_count);
        __syncthreads();*/
        KernelPolicy::BlockScanT::Scan(thread_output_count, output_pos, smem_storage.scan_space);

        //KernelPolicy::BlockScanT(smem_storage.cub_storage.scan_space)
        //    .ExclusiveSum(thread_output_count, output_pos);
        if (threadIdx.x == KernelPolicy::THREADS -1)
        {
            if (output_pos + thread_output_count != 0)
            smem_storage.block_offset = atomicAdd(smem_storage.d_output_counter, output_pos + thread_output_count);
        }
        __syncthreads();

        if (thread_output_count != 0)
        {
            output_pos += smem_storage.block_offset;
            SizeT temp_pos = (threadIdx.x << KernelPolicy::LOG_OUTPUT_PER_THREAD);
            for (int i=0; i<thread_output_count; i++)
            {
                VertexId u = smem_storage.thread_output_vertices[temp_pos];
                if (d_keys_out != NULL)
                {
                    util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                        u,
                        d_keys_out + output_pos);
                }

                if (Problem::ENABLE_IDEMPOTENCE)
                {
                    //util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                    //    label, smem_storage.d_labels + u);
                    //smem_storage.d_labels[u] = label;

                    //util::io::ModifiedStore<util::io::st::cg>::St(
                    //    smem_storage.tex_mask_bytes[temp_pos], //mask_byte,
                    //    smem_storage.d_visited_mask + ((u & KernelPolicy::ELEMENT_ID_MASK)>> 3));
                    //smem_storage.d_visited_mask[(u & KernelPolicy::ELEMENT_ID_MASK)>> (2 + sizeof(MaskT))] = 
                    //    smem_storage.tex_mask_bytes[temp_pos];
                }

                output_pos ++;
                temp_pos ++;
            }
            thread_output_count = 0;
        }
    }

    static __device__ __forceinline__ void RelaxPartitionedEdges2(
        bool                     &queue_reset,
        VertexId                 &queue_index,
        LabelT                   &label,
        SizeT                   *&d_row_offsets,
        SizeT                   *&d_inverse_row_offsets,
        VertexId                *&d_column_indices,
        VertexId                *&d_inverse_column_indices,
        SizeT                   *&d_scanned_edges,
        SizeT                   *&partition_starts,
        SizeT                    &num_partitions,
        VertexId                *&d_queue,
        VertexId                *&d_keys_out,
        Value                   *&d_values_out,
        DataSlice               *&d_data_slice,
        SizeT                    &input_queue_len,
        SizeT                   *&output_queue_len,
        SizeT                    &partition_size,
        SizeT                    &max_vertices,
        SizeT                    &max_edges,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats &kernel_stats,
        bool                     &input_inverse_graph,
        bool                     &output_inverse_graph,
        Value                   *&d_value_to_reduce,
        Value                   *&d_reduce_frontier)
   {
        //PrepareQueue(queue_reset, queue_index, input_queue_len, output_queue_len, work_progress);
        __shared__ typename KernelPolicy::SmemStorage smem_storage;

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            // obtain problem size
            if (queue_reset)
            {
                work_progress.StoreQueueLength(input_queue_len, queue_index);
            } else {
                input_queue_len = work_progress.LoadQueueLength(queue_index);
            }
            //if (!Problem::ENABLE_IDEMPOTENCE)
            //work_progress.Enqueue(output_queue_len[0], queue_index + 1);

            // Reset our next outgoing queue counter to zero
            work_progress.StoreQueueLength(0, queue_index + 2);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            smem_storage.Init(
                queue_index,
                d_data_slice,
                d_scanned_edges,
                partition_starts,
                partition_size,
                input_queue_len,
                output_queue_len,
                work_progress);
            smem_storage.d_column_indices = (output_inverse_graph) ?
                d_inverse_column_indices : d_column_indices;
            /*printf("(%4d, %4d) : block_input = %6d ~ %6d, %6d, block_output = %9d ~ %9d, %9d\n",
                blockIdx.x, threadIdx.x,
                smem_storage.block_input_start,
                smem_storage.block_input_end,
                smem_storage.block_input_end - smem_storage.block_input_start,
                smem_storage.block_output_start,
                smem_storage.block_output_end,
                smem_storage.block_output_size);*/
        }
        __syncthreads();
        if (smem_storage.block_output_start >= output_queue_len[0]) return;

        SizeT block_output_processed = 0;
        //SizeT iter_input_start  = partition_starts[blockIdx.x];

        //VertexId* column_indices = (output_inverse_graph)?
        //    d_inverse_column_indices:
        //    d_column_indices        ;
        SizeT thread_output_count = 0;
        //SizeT* row_offsets = (output_inverse_graph) ?
        //    d_inverse_row_offsets :
        //    d_row_offsets         ;

        while (block_output_processed < smem_storage.block_output_size &&
            smem_storage.iter_input_start < smem_storage.block_input_end)
        {
            if (threadIdx.x == 0)
            {
                smem_storage.iter_input_size  = min(
                    (SizeT)KernelPolicy::SCRATCH_ELEMENTS-1,
                    smem_storage.block_input_end - smem_storage.iter_input_start);
                smem_storage.iter_input_end = smem_storage.iter_input_start +
                    smem_storage.iter_input_size;
                smem_storage.iter_output_end =
                    (smem_storage.iter_input_end < input_queue_len) ?
                    d_scanned_edges[smem_storage.iter_input_end] :
                    output_queue_len[0];
                smem_storage.iter_output_end = min(
                    smem_storage.iter_output_end,
                    smem_storage.block_output_end);
                smem_storage.iter_output_size = min(
                    smem_storage.iter_output_end -
                    smem_storage.block_output_start,
                    smem_storage.block_output_size);
                smem_storage.iter_output_size -= block_output_processed;
                smem_storage.iter_output_end_offset = smem_storage.iter_output_size + block_output_processed;

                /*printf("(%4d, %4d) : iter_input = %6d ~ %6d, %6d iter_output = %9d ~ %9d, %9d offset = %9d ~ %9d, %9d\n",
                    blockIdx.x, threadIdx.x,
                    smem_storage.iter_input_start,
                    smem_storage.iter_input_end,
                    smem_storage.iter_input_size,
                    smem_storage.block_output_start + block_output_processed,
                    smem_storage.iter_output_end,
                    smem_storage.iter_output_size,
                    block_output_processed,
                    smem_storage.iter_output_end_offset,
                    smem_storage.iter_output_end_offset - block_output_processed);*/
            }
            __syncthreads();

            //volatile SizeT block_output_start = smem_storage.block_output_start;
            //SizeT iter_output_end_offset = smem_storage.iter_output_size + block_output_processed;
            VertexId thread_input = smem_storage.iter_input_start + threadIdx.x;
            if (threadIdx.x < KernelPolicy::SCRATCH_ELEMENTS)
            {
                if (thread_input <= smem_storage.block_input_end &&
                    thread_input < input_queue_len)
                {
                    VertexId input_item = d_queue[thread_input];
                    smem_storage.output_offset[threadIdx.x] =
                        d_scanned_edges [thread_input] - smem_storage.block_output_start;
                    smem_storage.input_queue  [threadIdx.x] = input_item;

                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                    {
                        //smem_storage.vertices [threadIdx.x] = input_item;
                        if (input_item >= 0)
                            smem_storage.row_offset[threadIdx.x]= (output_inverse_graph) ?
                                _ldg(d_inverse_row_offsets + input_item) :
                                LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets, input_item);
                            
                                //row_offsets[input_item];
                                //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  input_item);
                                //_ldg(((output_inverse_graph) ? d_inverse_row_offsets :
                                //     d_row_offsets) + input_item);
                        else smem_storage.row_offset[threadIdx.x] = util::MaxValue<SizeT>();
                    }
                    else if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                    {
                        if (input_inverse_graph) {
                            smem_storage.vertices[threadIdx.x] = d_inverse_column_indices[input_item];
                            smem_storage.row_offset[threadIdx.x] =
                                (output_inverse_graph)?
                                d_inverse_row_offsets [d_inverse_column_indices[input_item]]:
                                d_row_offsets [d_inverse_column_indices[input_item]];;
                        } else {
                            smem_storage.vertices[threadIdx.x] = d_column_indices        [input_item];
                            smem_storage.row_offset[threadIdx.x] =
                                (output_inverse_graph)?
                                d_inverse_row_offsets [d_column_indices[input_item]] :
                                d_row_offsets         [d_column_indices[input_item]];
                        }
                    }

                    /*if (TO_TRACK && util::pred_to_track(d_data_slice -> gpu_idx, input_item))
                        printf("(%4d, %4d) : Load Src %7d, label = %2d, "
                            "v_index = %3d, input_pos = %8d, output_offset = %8d, row_offset = %8d, degree = %5d"
                            " iter_input_start = %8d, iter_input_end = %8d, iter_output_start_offset = %8d, iter_output_end_offset = %8d, block_output_start = %8d, skip_count = %4d\n",
                            blockIdx.x, threadIdx.x, input_item, label-1, threadIdx.x, thread_input,
                            smem_storage.output_offset[threadIdx.x],
                            smem_storage.row_offset[threadIdx.x],
                            d_row_offsets[input_item + 1] - d_row_offsets[input_item],
                            smem_storage.iter_input_start,
                            smem_storage.iter_input_end,
                            block_output_processed,
                            smem_storage.iter_output_end_offset,
                            smem_storage.block_output_start,
                            smem_storage.block_first_v_skip_count);*/
                } else {
                    smem_storage.output_offset[threadIdx.x] = util::MaxValue<SizeT>(); //max_edges;
                    //smem_storage.vertices     [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.input_queue  [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.row_offset   [threadIdx.x] = util::MaxValue<SizeT>();
                }
            }
            __syncthreads();
            if (threadIdx.x < KernelPolicy::SCRATCH_ELEMENTS)
            smem_storage.row_offset[threadIdx.x] -= (threadIdx.x == 0) ?
                (block_output_processed - smem_storage.block_first_v_skip_count) :
                smem_storage.output_offset[threadIdx.x -1];
            __syncthreads();

            //SizeT v_index    = 0;
            VertexId v          = 0;
            VertexId input_item = 0;
            SizeT next_v_output_start_offset = 0;
            //SizeT v_output_start_offset = 0;
            SizeT row_offset_v = 0;
            VertexId v_index = 0;
            for (SizeT thread_output_offset = threadIdx.x + block_output_processed;
                thread_output_offset - threadIdx.x < smem_storage.iter_output_end_offset;
                thread_output_offset += KernelPolicy::THREADS)
            {
                //SizeT thread_output_offset = small_iter_block_output_offset + threadIdx.x;
                //SizeT edge_id = 0;
                //VertexId u = 0;
                bool to_process = false;
                SizeT output_pos = 0;
                MaskT tex_mask_byte;

                if (thread_output_offset < smem_storage.iter_output_end_offset)
                {
                    if (thread_output_offset >= next_v_output_start_offset)
                    {
                        v_index    = util::BinarySearch<KernelPolicy::SCRATCH_ELEMENTS>(
                            thread_output_offset, smem_storage.output_offset);
                        input_item = smem_storage.input_queue[v_index];
                        if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                            ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                            v = input_item;
                        else v = smem_storage.vertices[v_index];
                        row_offset_v = smem_storage.row_offset[v_index];
                        next_v_output_start_offset = smem_storage.output_offset[v_index];
                        //if (v_index > 0)
                        //{
                            //block_first_v_skip_count = 0;
                        //    v_output_start_offset = smem_storage.output_offset[v_index-1];
                        //} else v_output_start_offset = block_output_processed - smem_storage.block_first_v_skip_count;
                    }

                    //edge_id = smem_storage.row_offset[v_index] + thread_output_offset + block_first_v_skip_count - v_output_start_offset;
                    SizeT edge_id = row_offset_v + thread_output_offset; /*+ ((v_index == 0) ? smem_storage.block_first_v_skip_count : 0)*/ //- v_output_start_offset;
                    //VertexId u = (output_inverse_graph) ?
                    //    d_inverse_column_indices[edge_id] :
                    //    d_column_indices[edge_id];
                    VertexId u = smem_storage.d_column_indices [edge_id];
                    //output_pos = block_output_start + thread_output_offset;
                    /*if (TO_TRACK && (util::to_track(d_data_slice -> gpu_idx, u)))// || util::pred_to_track(d_data_slice -> gpu_idx, v)))
                        printf("(%4d, %4d) : Expand %4d, label = %d, "
                            "src = %d, v_index = %d, edge_id = %d, thread_offset = %d,"
                            " iter_output_start = %d, iter_output_end = %d\n",
                            blockIdx.x, threadIdx.x, u, label, v, v_index,
                            edge_id, thread_output_offset,
                            block_output_processed,
                            smem_storage.iter_output_end_offset);*/

                    if (Problem::ENABLE_IDEMPOTENCE)
                    {
                        // Location of mask byte to read
                        //SizeT mask_byte_offset = (u & KernelPolicy::ELEMENT_ID_MASK) >> 3;
                        output_pos = (u & KernelPolicy::ELEMENT_ID_MASK) >> (2 + sizeof(MaskT));

                        // Bit in mask byte corresponding to current vertex id
                        MaskT mask_bit = 1 << (u & ((1 << (2+sizeof(MaskT)))-1));

                        // Read byte from visited mask in tex
                        //tex_mask_byte = tex1Dfetch(
                        //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,//cta->t_bitmask[0],
                        //    output_pos);//mask_byte_offset);
                        //tex_mask_byte = smem_storage.d_visited_mask[output_pos];
                        tex_mask_byte = _ldg(smem_storage.d_visited_mask + output_pos);                        

                        if (!(mask_bit & tex_mask_byte))
                        {
                            //do 
                            {
                                tex_mask_byte |= mask_bit;
                                util::io::ModifiedStore<util::io::st::cg>::St(
                                    tex_mask_byte, //mask_byte,
                                    smem_storage.d_visited_mask + output_pos);
                                //tex_mask_byte = smem_storage.d_visited_mask[output_pos];
                            }// while (!(mask_bit & tex_mask_byte));
                            
                            if (smem_storage.d_labels[u] == util::MaxValue<LabelT>())
                            //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels,  u) == util::MaxValue<LabelT>())
                            {
                                to_process = true;
                                d_data_slice -> labels[u] = label;
                            }
                        }
                    } else to_process = true;

                    if (to_process)
                    {
                        //ProcessNeighbor
                        //    <KernelPolicy, Problem, Functor,
                        //    ADVANCE_TYPE, R_TYPE, R_OP>(
                        //    v, u, d_data_slice, edge_id,
                        //    iter_input_start + v_index, input_item,
                        //    output_pos,
                        //    label, d_keys_out, d_values_out,
                        //    d_value_to_reduce, d_reduce_frontier);
                        //SizeT input_pos = iter_input_start + v_index;
                        output_pos = util::InvalidValue<SizeT>();
                        if (Functor::CondEdge(
                            v, u, d_data_slice, edge_id, input_item, label,
                            util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                            output_pos))
                        {
                            Functor::ApplyEdge(
                                v, u, d_data_slice, edge_id, input_item, label,
                                util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                                output_pos);
                        } else to_process = false;
                    }

                    if (to_process)
                    {
                        if (Functor::CondFilter(
                            v, u, d_data_slice, input_item, label,
                            util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                            output_pos))
                        {
                            Functor::ApplyFilter(
                                v, u, d_data_slice, input_item, label,
                                util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                                output_pos);
                        } else to_process = false;
                    }

                    if (to_process)
                    {
                        output_pos = (threadIdx.x << KernelPolicy::LOG_OUTPUT_PER_THREAD) + thread_output_count;
                        smem_storage.thread_output_vertices[output_pos] =
                            (ADVANCE_TYPE == gunrock::oprtr::advance::V2E ||
                             ADVANCE_TYPE == gunrock::oprtr::advance::E2E) ?
                             ((VertexId) edge_id) : u;
                        //smem_storage.tex_mask_bytes[output_pos] = tex_mask_byte;
                        thread_output_count ++;
                    }
                }

                if (__syncthreads_or(thread_output_count == KernelPolicy::OUTPUT_PER_THREAD))
                {
                    Write_Global_Output(smem_storage, output_pos,
                        thread_output_count, label, d_keys_out);
                }
            } // for

            __syncthreads();
            block_output_processed += smem_storage.iter_output_size;

            if (threadIdx.x == 0)
            {
                smem_storage.block_first_v_skip_count = 0;
                smem_storage.iter_input_start += smem_storage.iter_input_size;
            }
            __syncthreads();
        }

        if (__syncthreads_or(thread_output_count != 0))
        {
            SizeT output_pos;
            Write_Global_Output(smem_storage, output_pos,
                thread_output_count, label, d_keys_out);
        }
    }

    static __device__ __forceinline__ void RelaxLightEdges(
        bool                     &queue_reset,
        VertexId                 &queue_index,
        LabelT                   &label,
        SizeT                   *&d_row_offsets,
        SizeT                   *&d_inverse_row_offsets,
        VertexId                *&d_column_indices,
        VertexId                *&d_inverse_column_indices,
        SizeT                   *&d_scanned_edges,
        VertexId                *&d_queue,
        VertexId                *&d_keys_out,
        Value                   *&d_values_out,
        DataSlice               *&d_data_slice,
        SizeT                    &input_queue_length,
        SizeT                   *&d_output_queue_length,
        SizeT                    &max_vertices,
        SizeT                    &max_edges,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats &kernel_stats,
        bool                     &input_inverse_graph,
        bool                     &output_inverse_graph,
        Value                   *&d_value_to_reduce,
        Value                   *&d_reduce_frontier)
    {
        //PrepareQueue(queue_reset, queue_index, input_queue_length, d_output_queue_length, work_progress);
        __shared__ typename KernelPolicy::SmemStorage smem_storage;

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            // obtain problem size
            if (queue_reset)
            {
                work_progress.StoreQueueLength(input_queue_length, queue_index);
            } else {
                input_queue_length = work_progress.LoadQueueLength(queue_index);
            }
            //if (!Problem::ENABLE_IDEMPOTENCE)
            //work_progress.Enqueue(output_queue_len[0], queue_index + 1);

            // Reset our next outgoing queue counter to zero
            work_progress.StoreQueueLength(0, queue_index + 2);
        }
        if (threadIdx.x == 0)
        {
            /*smem_storage.d_output_counter = work_progress.template GetQueueCounter<VertexId>(queue_index + 1);
            //int lane_id = threadIdx.x & KernelPolicy::WARP_SIZE_MASK;
            //int warp_id = threadIdx.x >> KernelPolicy::LOG_WARP_SIZE;
            smem_storage.d_labels = d_data_slice -> labels.GetPointer(util::DEVICE);
            smem_storage.d_visited_mask = d_data_slice -> visited_mask.GetPointer(util::DEVICE);
            printf("(%4d, %4d)\n", blockIdx.x, threadIdx.x);*/
            SizeT *partition_starts = NULL;
            SizeT partition_size = 0;
            smem_storage.Init(
                queue_index,
                d_data_slice,
                d_scanned_edges,
                partition_starts,
                partition_size,
                input_queue_length,
                d_output_queue_length,
                work_progress);
        }
        __syncthreads();
        //printf("(%3d, %3d) 1\n", blockIdx.x, threadIdx.x);

        SizeT* row_offsets = (output_inverse_graph) ?
            d_inverse_row_offsets :
            d_row_offsets         ;
        VertexId* column_indices = (output_inverse_graph)?
            d_inverse_column_indices:
            d_column_indices        ;
        //SizeT partition_start    = (long long)input_queue_length * blockIdx.x / gridDim.x;
        //SizeT partition_end      = (long long)input_queue_length * (blockIdx.x + 1) / gridDim.x;
        VertexId input_item      = 0;
        SizeT block_input_start  = (SizeT) blockIdx.x * KernelPolicy::SCRATCH_ELEMENTS;//partition_start;
        SizeT thread_output_count = 0;

        //while (block_input_start < partition_end)
        {
            SizeT block_input_end    =
            (block_input_start + KernelPolicy::SCRATCH_ELEMENTS >= input_queue_length) ?
                input_queue_length - 1 :
                block_input_start + KernelPolicy::SCRATCH_ELEMENTS - 1;
            SizeT thread_input       = block_input_start + threadIdx.x;
            SizeT block_output_start = (block_input_start >= 1) ?
                d_scanned_edges[block_input_start - 1] : 0;
            SizeT block_output_end  = d_scanned_edges[block_input_end];
            SizeT block_output_size = block_output_end - block_output_start;

            if (threadIdx.x < KernelPolicy::SCRATCH_ELEMENTS)
            {
                if (thread_input <= block_input_end+1 && thread_input < input_queue_length) // input_queue_length)
                {
                    input_item = d_queue[thread_input];
                    //printf("%d input_item = queue[%d] = %d\n",
                    //    threadIdx.x, thread_input, input_item);
                    smem_storage.input_queue  [threadIdx.x] = input_item;
                    smem_storage.output_offset[threadIdx.x] = d_scanned_edges[thread_input] - block_output_start;
                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                    {
                        //smem_storage.vertices[threadIdx.x] = input_item;
                        if (input_item >= 0)
                            smem_storage.row_offset[threadIdx.x] =  (output_inverse_graph) ?
                                _ldg(d_inverse_row_offsets + input_item) :
                                LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets, input_item);
                                //row_offsets[input_item];
                                //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  input_item);
                                //_ldg(row_offsets + input_item);
                        else smem_storage.row_offset[threadIdx.x] = util::MaxValue<SizeT>();
                    } else if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                    {
                        if (input_inverse_graph)
                        {
                            smem_storage.vertices  [threadIdx.x] = d_inverse_column_indices[input_item];
                            smem_storage.row_offset[threadIdx.x] = row_offsets[d_inverse_column_indices[input_item]];
                        } else {
                            smem_storage.vertices  [threadIdx.x] = d_column_indices[        input_item];
                            smem_storage.row_offset[threadIdx.x] = row_offsets[d_column_indices[input_item]];
                        }
                    }

                    //printf("%d\t %d\t (%4d, %4d) : %d, %d ~ %d, %d\n",
                    //    d_data_slice -> gpu_idx, label, blockIdx.x, threadIdx.x, smem_storage.vertices[threadIdx.x],
                    //    smem_storage.row_offset[threadIdx.x],
                    //    d_row_offsets[input_item + 1],
                    //    smem_storage.output_offset[threadIdx.x]);
                } // end of if thread_input < input_queue_length
                else {
                    smem_storage.output_offset[threadIdx.x] = util::MaxValue<SizeT>();//max_edges; // - block_output_start?
                    //smem_storage.vertices     [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.input_queue  [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.row_offset   [threadIdx.x] = util::MaxValue<SizeT>();
                }
            }

            __syncthreads();
            //printf("(%3d, %3d) 2\n", blockIdx.x, threadIdx.x);


            //printf("(%4d, %4d)  : block_input = %d ~ %d, block_output = %d ~ %d, %d\n",
            //    blockIdx.x, threadIdx.x,
            //    block_input_start, block_input_end,
            //    block_output_start, block_output_end, block_output_size);

            int v_index = 0;
            VertexId v = 0;
            SizeT next_v_output_start_offset = 0;
            SizeT v_output_start_offset = 0;
            SizeT row_offset_v = 0;

            for (SizeT thread_output = threadIdx.x; thread_output  - threadIdx.x< block_output_size;
                thread_output += KernelPolicy::THREADS)
            {
                bool to_process = true;
                SizeT output_pos = 0;
                MaskT tex_mask_byte = 0;
                if (thread_output < block_output_size)
                {
                    if (thread_output >= next_v_output_start_offset)
                    {
                        v_index    = util::BinarySearch<KernelPolicy::SCRATCH_ELEMENTS>(thread_output, smem_storage.output_offset);
                        //v          = smem_storage.vertices   [v_index];
                        input_item = smem_storage.input_queue[v_index];
                        v          = input_item;
                        v_output_start_offset = (v_index > 0) ? smem_storage.output_offset[v_index-1] : 0;
                        next_v_output_start_offset = (v_index < KernelPolicy::SCRATCH_ELEMENTS) ?
                            smem_storage.output_offset[v_index] : util::MaxValue<SizeT>();
                        row_offset_v = smem_storage.row_offset[v_index];
                    }

                    SizeT edge_id = row_offset_v - v_output_start_offset + thread_output;
                    VertexId u = column_indices [edge_id];
                    //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                    //    u, column_indices + edge_id);
                    //ProcessNeighbor<KernelPolicy, Problem, Functor,
                    //    ADVANCE_TYPE, R_TYPE, R_OP>(
                    //    v, u, d_data_slice, edge_id,
                    //    block_input_start + v_index, input_item,
                    //    block_output_start + thread_output,
                    //    label, d_keys_out, d_values_out,
                    //    d_value_to_reduce, d_reduce_frontier);
                    //printf("%d\t %d\t (%4d, %4d) : %d -> %d\n",
                    //    d_data_slice -> gpu_idx, label, blockIdx.x, threadIdx.x, v, u);
                    if (Problem::ENABLE_IDEMPOTENCE)
                    {
                        //output_pos = (u & KernelPolicy::ELEMENT_ID_MASK) >> 3;
                        output_pos = (u & KernelPolicy::ELEMENT_ID_MASK) >> (2 + sizeof(MaskT));

                        // Bit in mask byte corresponding to current vertex id
                        //MaskT mask_bit = 1 << (u & 7);
                        MaskT mask_bit = 1 << (u & ((1 << (2 + sizeof(MaskT)))-1));

                        // Read byte from visited mask in tex
                        //tex_mask_byte = tex1Dfetch(
                        //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,//cta->t_bitmask[0],
                        //    output_pos);//mask_byte_offset);
                        tex_mask_byte = smem_storage.d_visited_mask[output_pos];

                        if (!(mask_bit & tex_mask_byte))
                        {
                            do {
                                tex_mask_byte |= mask_bit;
                                util::io::ModifiedStore<util::io::st::cg>::St(
                                    tex_mask_byte, //mask_byte,
                                    smem_storage.d_visited_mask + output_pos);
                                tex_mask_byte = smem_storage.d_visited_mask[output_pos];
                            } while (!(mask_bit & tex_mask_byte));

                            //if (smem_storage.d_labels[u] != util::MaxValue<LabelT>())
                            //    to_process = false;
                            if (d_data_slice -> labels[u] != util::MaxValue<LabelT>())
                            //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels,  u) != util::MaxValue<LabelT>())
                                to_process = false;
                            else d_data_slice -> labels[u] = label;
                        } else to_process = false;
                    }

                    if (to_process)
                    {
                        if (Functor::CondEdge(
                            v, u, d_data_slice, edge_id, input_item, label,
                            util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                            output_pos))
                        {
                            Functor::ApplyEdge(
                                v, u, d_data_slice, edge_id, input_item, label,
                                util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                                output_pos);
                        } else to_process = false;
                    }

                    if (to_process)
                    {
                        if (Functor::CondFilter(
                            v, u, d_data_slice, input_item, label,
                            util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                            output_pos))
                        {
                            Functor::ApplyFilter(
                                v, u, d_data_slice, input_item, label,
                                util::InvalidValue<SizeT>(),//smem_storage.iter_input_start + v_index, //input_pos,
                                output_pos);
                        } else to_process = false;
                    }

                    if (to_process)
                    {
                        output_pos = (threadIdx.x << KernelPolicy::LOG_OUTPUT_PER_THREAD) + thread_output_count;
                        smem_storage.thread_output_vertices[output_pos] =
                            (ADVANCE_TYPE == gunrock::oprtr::advance::V2E ||
                             ADVANCE_TYPE == gunrock::oprtr::advance::E2E) ?
                             ((VertexId) edge_id) : u;
                        //smem_storage.tex_mask_bytes[output_pos] = tex_mask_byte;
                        thread_output_count ++;
                    }
                }

                if (__syncthreads_or(thread_output_count == KernelPolicy::OUTPUT_PER_THREAD))
                    Write_Global_Output(smem_storage, output_pos,
                        thread_output_count, label, d_keys_out);
            } // end of for thread_output

            //block_input_start += KernelPolicy::SCRATCH_ELEMENTS;
            //__syncthreads();
        }

        if (__syncthreads_or(thread_output_count !=0))
        {
            SizeT output_pos = util::InvalidValue<SizeT>();
            Write_Global_Output(smem_storage, output_pos,
                thread_output_count, label, d_keys_out);
        }
    } // end of RelaxLightEdges
};


/**
 * @brief Kernel entry for relax partitioned edge function
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             label value to use in functor
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_scanned_edges   Device pointer of scanned neighbor list queue of the current frontier
 * @param[in] partition_starts  Device pointer of partition start index computed by sorted search in moderngpu lib
 * @param[in] num_partitions    Number of partitions in the current frontier
 * @param[in] d_queue           Device pointer of VertexId to the incoming frontier queue
 * @param[out] d_out            Device pointer of VertexId to the outgoing frontier queue
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_queue_len   Length of the incoming frontier queue
 * @param[in] output_queue_len  Length of the outgoing frontier queue
 * @param[in] partition_size    Size of workload partition that one block handles
 * @param[in] max_vertices      Maximum number of elements we can place into the incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the outgoing frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT is set)
 * @param[in] ADVANCE_TYPE      enumerator which shows the advance type: V2V, V2E, E2V, or E2E
 * @param[in] inverse_graph     Whether this iteration's advance operator is in the opposite direction to the previous iteration
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void RelaxPartitionedEdges2(
    bool                                     queue_reset,
    typename KernelPolicy::VertexId          queue_index,
    typename Functor     ::LabelT            label,
    typename KernelPolicy::SizeT            *d_row_offsets,
    typename KernelPolicy::SizeT            *d_inverse_row_offsets,
    typename KernelPolicy::VertexId         *d_column_indices,
    typename KernelPolicy::VertexId         *d_inverse_column_indices,
    typename KernelPolicy::SizeT            *d_scanned_edges,
    typename KernelPolicy::SizeT            *partition_starts,
    typename KernelPolicy::SizeT             num_partitions,
    typename KernelPolicy::VertexId         *d_queue,
    typename KernelPolicy::VertexId         *d_keys_out,
    typename KernelPolicy::Value            *d_values_out,
    typename Problem     ::DataSlice        *d_data_slice,
    typename KernelPolicy::SizeT             input_queue_len,
    typename KernelPolicy::SizeT            *d_output_queue_len,
    typename KernelPolicy::SizeT             partition_size,
    typename KernelPolicy::SizeT             max_vertices,
    typename KernelPolicy::SizeT             max_edges,
    util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
    util::KernelRuntimeStats                 kernel_stats,
    //gunrock::oprtr::advance::TYPE ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    bool                                     input_inverse_graph = false,
    bool                                     output_inverse_graph = false,
    //gunrock::oprtr::advance::REDUCE_TYPE R_TYPE = gunrock::oprtr::advance::EMPTY,
    //gunrock::oprtr::advance::REDUCE_OP R_OP = gunrock::oprtr::advance::NONE,
    typename KernelPolicy::Value            *d_value_to_reduce = NULL,
    typename KernelPolicy::Value            *d_reduce_frontier = NULL)
{
    Dispatch<KernelPolicy, Problem, Functor,
        ADVANCE_TYPE, R_TYPE, R_OP>::RelaxPartitionedEdges2(
        queue_reset,
        queue_index,
        label,
        d_row_offsets,
        d_inverse_row_offsets,
        d_column_indices,
        d_inverse_column_indices,
        d_scanned_edges,
        partition_starts,
        num_partitions,
        d_queue,
        d_keys_out,
        d_values_out,
        d_data_slice,
        input_queue_len,
        d_output_queue_len,
        partition_size,
        max_vertices,
        max_edges,
        work_progress,
        kernel_stats,
        //ADVANCE_TYPE,
        input_inverse_graph,
        output_inverse_graph,
        //R_TYPE,
        //R_OP,
        d_value_to_reduce,
        d_reduce_frontier);
}

/**
 * @brief Kernel entry for relax light edge function
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             label value to use in functor
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_scanned_edges   Device pointer of scanned neighbor list queue of the current frontier
 * @param[in] d_queue           Device pointer of VertexId to the incoming frontier queue
 * @param[out] d_out            Device pointer of VertexId to the outgoing frontier queue
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_queue_len   Length of the incoming frontier queue
 * @param[in] output_queue_len  Length of the outgoing frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the outgoing frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT is set)
 * @param[in] ADVANCE_TYPE      enumerator which shows the advance type: V2V, V2E, E2V, or E2E
 * @param[in] inverse_graph     Whether this iteration's advance operator is in the opposite direction to the previous iteration
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void RelaxLightEdges(
    bool                              queue_reset,
    typename KernelPolicy::VertexId   queue_index,
    typename Functor     ::LabelT     label,
    typename KernelPolicy::SizeT     *d_row_offsets,
    typename KernelPolicy::SizeT     *d_inverse_row_offsets,
    typename KernelPolicy::VertexId  *d_column_indices,
    typename KernelPolicy::VertexId  *d_inverse_column_indices,
    typename KernelPolicy::SizeT     *d_scanned_edges,
    typename KernelPolicy::VertexId  *d_queue,
    typename KernelPolicy::VertexId  *d_keys_out,
    typename KernelPolicy::Value     *d_values_out,
    typename Problem     ::DataSlice *d_data_slice,
    typename KernelPolicy::SizeT      input_queue_len,
    typename KernelPolicy::SizeT     *d_output_queue_len,
    typename KernelPolicy::SizeT      max_vertices,
    typename KernelPolicy::SizeT      max_edges,
    util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
    util::KernelRuntimeStats          kernel_stats,
    //gunrock::oprtr::advance::TYPE ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    bool                              input_inverse_graph = false,
    bool                              output_inverse_graph = false,
    //gunrock::oprtr::advance::REDUCE_TYPE R_TYPE = gunrock::oprtr::advance::EMPTY,
    //gunrock::oprtr::advance::REDUCE_OP R_OP = gunrock::oprtr::advance::NONE,
    typename KernelPolicy::Value     *d_value_to_reduce = NULL,
    typename KernelPolicy::Value     *d_reduce_frontier = NULL)
{
    Dispatch<KernelPolicy, Problem, Functor,
        ADVANCE_TYPE, R_TYPE, R_OP>::RelaxLightEdges(
        queue_reset,
        queue_index,
        label,
        d_row_offsets,
        d_inverse_row_offsets,
        d_column_indices,
        d_inverse_column_indices,
        d_scanned_edges,
        d_queue,
        d_keys_out,
        d_values_out,
        d_data_slice,
        input_queue_len,
        d_output_queue_len,
        max_vertices,
        max_edges,
        work_progress,
        kernel_stats,
        //ADVANCE_TYPE,
        input_inverse_graph,
        output_inverse_graph,
        //R_TYPE,
        //R_OP,
        d_value_to_reduce,
        d_reduce_frontier);
}

/**
 * @brief Kernel entry for computing neighbor list length for each vertex in the current frontier
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offsets     Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_queue           Device pointer of VertexId to the incoming frontier queue
 * @param[out] d_scanned_edges  Device pointer of scanned neighbor list queue of the current frontier
 * @param[in] num_elements      Length of the current frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the outgoing frontier
 * @param[in] ADVANCE_TYPE      Enumerator which shows the advance type: V2V, V2E, E2V, or E2E
 * @param[in] in_inv            Input inverse.
 * @param[in] our_inv           Output inverse.
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void GetEdgeCounts(
    typename KernelPolicy::SizeT    *d_row_offsets,
    typename KernelPolicy::VertexId *d_column_indices,
    typename KernelPolicy::SizeT    *d_column_offsets,
    typename KernelPolicy::VertexId *d_row_indices,
    typename KernelPolicy::VertexId *d_queue,
    typename KernelPolicy::SizeT    *d_scanned_edges,
    typename KernelPolicy::SizeT     num_elements,
    typename KernelPolicy::SizeT     max_vertex,
    typename KernelPolicy::SizeT     max_edge,
    //gunrock::oprtr::advance::TYPE    ADVANCE_TYPE,
    bool                             in_inv,
    bool                             out_inv)
{
    Dispatch<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_TYPE, R_OP>::GetEdgeCounts(
        d_row_offsets,
        d_column_indices,
        d_column_offsets,
        d_row_indices,
        d_queue,
        d_scanned_edges,
        num_elements,
        max_vertex,
        max_edge,
        //ADVANCE_TYPE,
        in_inv,
        out_inv);
}

/*
 * @brief Mark partition size function.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_needles
 * @param[in] split_val
 * @param[in] num_elements Number of elements.
 * @param[in] output_queue_len Output frontier queue length.
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void MarkPartitionSizes(
    typename KernelPolicy::SizeT *d_needles,
    typename KernelPolicy::SizeT  split_val,
    typename KernelPolicy::SizeT  num_elements,
    typename KernelPolicy::SizeT  output_queue_len)
{
    Dispatch<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_TYPE, R_OP>::MarkPartitionSizes(
        d_needles,
        split_val,
        num_elements,
        output_queue_len);
}

}  // edge_map_partitioned
}  // oprtr
}  // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
