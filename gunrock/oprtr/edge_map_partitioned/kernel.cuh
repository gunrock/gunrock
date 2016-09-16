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

#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/advance_base.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned {

/**
* Templated texture reference for visited mask
*/
/*template <typename SizeT>
struct RowOffsetsTex
{
   static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> row_offsets;
};
template <typename SizeT>
texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetsTex<SizeT>::row_offsets;*/



/**
 * Arch dispatch
 */

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

    static __device__ __forceinline__ SizeT GetNeighborListLength(
        SizeT     *&d_row_offsets,
        VertexId  *&d_column_indices,
        VertexId   &vertex_id,
        SizeT      &max_vertex,
        SizeT      &max_edge)
        //gunrock::oprtr::advance::TYPE &ADVANCE_TYPE)
    {
        SizeT first  = /*(d_vertex_id >= max_vertex) ?
            max_edge :*/ //d_row_offsets[d_vertex_id];
            //tex1Dfetch(RowOffsetsTex<SizeT>::row_offsets,  vertex_id);
            _ldg(d_row_offsets + vertex_id);
        SizeT second = /*(d_vertex_id + 1 >= max_vertex) ?
            max_edge :*/ //d_row_offsets[d_vertex_id+1];
            //tex1Dfetch(RowOffsetsTex<SizeT>::row_offsets,  vertex_id + 1);
            _ldg(d_row_offsets + (vertex_id + 1));

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
        bool       in_inv,
        bool       out_inv)
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
                v_id = (d_queue == NULL) ? thread_pos : d_queue[thread_pos];
            } else {
                v_id = (in_inv) ?
                    d_row_indices[thread_pos] : d_column_indices[thread_pos];
            }
        } else v_id = (VertexId) -1;
        if (v_id < 0 || v_id >= max_vertex)
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
        /*if (d_scanned_edges[thread_pos] < 0)
        {
            printf("(%d, %d) : v_id = %d, row_offsets = %d, %d, out_inv = %s\n",
                blockIdx.x, threadIdx.x, v_id,
                tex1Dfetch(RowOffsetsTex<SizeT>::row_offsets,  v_id),
                tex1Dfetch(RowOffsetsTex<SizeT>::row_offsets,  v_id + 1),
                out_inv ? "true" : "false");
        }*/
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
        SizeT                   * output_queue_len,
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
        PrepareQueue(queue_reset, queue_index, input_queue_len, output_queue_len, work_progress);

        __shared__ typename KernelPolicy::SmemStorage smem_storage;

        SizeT block_output_start = (SizeT)blockIdx.x * partition_size;
        if (block_output_start >= output_queue_len[0]) return;
        SizeT block_output_end   = min(
            block_output_start + partition_size, output_queue_len[0]);
        SizeT block_output_size  = block_output_end - block_output_start;
        SizeT block_output_processed = 0;

        SizeT block_input_end    = (blockIdx.x == gridDim.x - 1) ?
            input_queue_len : min(
            partition_starts[blockIdx.x + 1] , input_queue_len);
        if (block_input_end < input_queue_len &&
            block_output_end > (block_input_end > 0 ? d_scanned_edges[block_input_end-1] : 0))
            block_input_end ++;

        SizeT iter_input_start  = partition_starts[blockIdx.x];
        SizeT block_first_v_skip_count =
            (block_output_start != d_scanned_edges[iter_input_start]) ?
                block_output_start -
                    (iter_input_start > 0 ? d_scanned_edges[iter_input_start -1] : 0)
                : 0;
        VertexId* column_indices = (output_inverse_graph)?
            d_inverse_column_indices:
            d_column_indices        ;
        SizeT* row_offsets = (output_inverse_graph) ?
            d_inverse_row_offsets :
            d_row_offsets         ;

        while (block_output_processed < block_output_size &&
            iter_input_start < block_input_end)
        {
            SizeT iter_input_size  = min(
                (SizeT)KernelPolicy::SCRATCH_ELEMENTS-1, block_input_end - iter_input_start);
            SizeT iter_input_end = iter_input_start + iter_input_size;
            SizeT iter_output_end = iter_input_end < input_queue_len ?
                d_scanned_edges[iter_input_end] : output_queue_len[0];
            iter_output_end = min(iter_output_end, block_output_end);
            SizeT iter_output_size = min(
                iter_output_end - block_output_start,
                block_output_size);
            iter_output_size -= block_output_processed;
            SizeT iter_output_end_offset = iter_output_size + block_output_processed;

            VertexId thread_input = iter_input_start + threadIdx.x;
            if (threadIdx.x < KernelPolicy::SCRATCH_ELEMENTS)
            {
                if (thread_input <= block_input_end && thread_input < input_queue_len)
                {
                    VertexId input_item = (d_queue == NULL) ? thread_input : d_queue[thread_input];
                    smem_storage.output_offset[threadIdx.x] =
                        d_scanned_edges [thread_input] - block_output_start;
                    smem_storage.input_queue  [threadIdx.x] = input_item;
                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                    {
                        //smem_storage.vertices [threadIdx.x] = input_item;
                        if (input_item >= 0)
                            smem_storage.row_offset[threadIdx.x]= //row_offsets[input_item];
                                //tex1Dfetch(RowOffsetsTex<SizeT>::row_offsets,  input_item);
                                _ldg(row_offsets + input_item);
                        else smem_storage.row_offset[threadIdx.x] = util::MaxValue<SizeT>();
                    }
                    else if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                    {
                        if (input_inverse_graph) {
                            smem_storage.vertices[threadIdx.x] = d_inverse_column_indices[input_item];
                            smem_storage.row_offset[threadIdx.x] =
                                row_offsets[d_inverse_column_indices[input_item]];
                        } else {
                            smem_storage.vertices[threadIdx.x] = d_column_indices        [input_item];
                            smem_storage.row_offset[threadIdx.x] =
                                row_offsets[d_column_indices[input_item]];
                        }
                    }
                    //smem_storage.row_offset [threadIdx.x] = (output_inverse_graph)?
                    //        d_inverse_row_offsets[smem_storage.vertices[threadIdx.x]] :
                    //        d_row_offsets        [smem_storage.vertices[threadIdx.x]];

                } else {
                    smem_storage.output_offset[threadIdx.x] = util::MaxValue<SizeT>(); //max_edges;
                    smem_storage.vertices     [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.input_queue  [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                    smem_storage.row_offset   [threadIdx.x] = util::MaxValue<SizeT>();
                }
            }
            __syncthreads();

            SizeT    v_index    = 0;
            VertexId v          = 0;
            VertexId input_item = 0;
            SizeT next_v_output_start_offset = 0;
            SizeT v_output_start_offset = 0;
            SizeT row_offset_v = 0;
            for (SizeT thread_output_offset = threadIdx.x + block_output_processed;
                thread_output_offset < iter_output_end_offset;
                thread_output_offset += KernelPolicy::THREADS)
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
                    if (v_index > 0)
                    {
                        block_first_v_skip_count = 0;
                        v_output_start_offset = smem_storage.output_offset[v_index-1];
                    } else v_output_start_offset = block_output_processed;
                }

                SizeT edge_id = row_offset_v + thread_output_offset + block_first_v_skip_count - v_output_start_offset;
                VertexId u = column_indices[edge_id];
                //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                //    u, column_indices + edge_id);
                //u = tex1Dfetch(ColumnIndicesTex<VertexId>::column_indices,
                //    edge_id);
                ProcessNeighbor
                    <KernelPolicy, Problem, Functor,
                    ADVANCE_TYPE, R_TYPE, R_OP>(
                    v, u, d_data_slice, edge_id,
                    iter_input_start + v_index, input_item,
                    block_output_start + thread_output_offset,
                    label, d_keys_out, d_values_out,
                    d_value_to_reduce, d_reduce_frontier);

            } // for
            block_output_processed += iter_output_size;
            iter_input_start += iter_input_size;
            block_first_v_skip_count = 0;

            __syncthreads();
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
        SizeT                   * d_output_queue_length,
        SizeT                    &max_vertices,
        SizeT                    &max_edges,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats &kernel_stats,
        bool                     &input_inverse_graph,
        bool                     &output_inverse_graph,
        Value                   *&d_value_to_reduce,
        Value                   *&d_reduce_frontier)
    {
        PrepareQueue(queue_reset, queue_index, input_queue_length, d_output_queue_length, work_progress);

        __shared__ typename KernelPolicy::SmemStorage smem_storage;
        VertexId input_item      = 0;
        SizeT block_input_start  = (SizeT) blockIdx.x * KernelPolicy::SCRATCH_ELEMENTS;
        SizeT block_input_end    = (block_input_start + KernelPolicy::SCRATCH_ELEMENTS >= input_queue_length) ?
            (input_queue_length - 1) :  (block_input_start + KernelPolicy::SCRATCH_ELEMENTS - 1);
        SizeT thread_input       = block_input_start + threadIdx.x;
        SizeT block_output_start = (block_input_start >= 1) ?
            d_scanned_edges[block_input_start - 1] : 0;
        SizeT block_output_end   = d_scanned_edges[block_input_end];
        SizeT block_output_size  = block_output_end -block_output_start;

        //block_input_end = block_input_end % KernelPolicy::THREADS;
        SizeT* row_offsets = (output_inverse_graph) ?
            d_inverse_row_offsets :
            d_row_offsets         ;
        VertexId* column_indices = (output_inverse_graph)?
            d_inverse_column_indices:
            d_column_indices        ;

        if (threadIdx.x < KernelPolicy::SCRATCH_ELEMENTS)
        {
            if (thread_input <= block_input_end + 1 && thread_input < input_queue_length)
            {
                input_item = (d_queue == NULL) ? thread_input : d_queue[thread_input];
                smem_storage.input_queue  [threadIdx.x] = input_item;
                smem_storage.output_offset[threadIdx.x] = d_scanned_edges[thread_input] - block_output_start;
                if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                    ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                {
                    smem_storage.vertices[threadIdx.x] = input_item;
                    if (input_item >= 0)
                        smem_storage.row_offset[threadIdx.x] = _ldg(row_offsets + input_item);//row_offsets[input_item];
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
            } // end of if thread_input < input_queue_length
            else {
                smem_storage.output_offset[threadIdx.x] = util::MaxValue<SizeT>();//max_edges; // - block_output_start?
                smem_storage.vertices     [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                smem_storage.input_queue  [threadIdx.x] = util::MaxValue<VertexId>();//max_vertices;
                smem_storage.row_offset   [threadIdx.x] = util::MaxValue<SizeT>();
            }
        }

        __syncthreads();
        //SizeT block_output_size = smem_storage.output_offset[block_input_end % KernelPolicy::SCRATCH_ELEMENTS];

        int v_index = 0;
        VertexId v = 0;
        SizeT next_v_output_start_offset = 0;
        SizeT v_output_start_offset = 0;
        SizeT row_offset_v = 0;

        for (SizeT thread_output = threadIdx.x; thread_output - threadIdx.x < block_output_size;
            thread_output += KernelPolicy::THREADS)
        {
            if (thread_output < block_output_size)
            {
                if (thread_output >= next_v_output_start_offset)
                {
                    v_index    = util::BinarySearch<KernelPolicy::SCRATCH_ELEMENTS>(thread_output, smem_storage.output_offset);
                    v          = smem_storage.vertices   [v_index];
                    input_item = smem_storage.input_queue[v_index];
                    v_output_start_offset = (v_index > 0) ? smem_storage.output_offset[v_index-1] : 0;
                    next_v_output_start_offset = (v_index < KernelPolicy::SCRATCH_ELEMENTS) ?
                        smem_storage.output_offset[v_index] : util::MaxValue<SizeT>();
                    row_offset_v = smem_storage.row_offset[v_index];
                }

                SizeT edge_id = row_offset_v - v_output_start_offset + thread_output;
                VertexId u = column_indices [edge_id];
                //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                //    u, column_indices + edge_id);
                ProcessNeighbor<KernelPolicy, Problem, Functor,
                    ADVANCE_TYPE, R_TYPE, R_OP>(
                    v, u, d_data_slice, edge_id,
                    block_input_start + v_index, input_item,
                    block_output_start + thread_output,
                    label, d_keys_out, d_values_out,
                    d_value_to_reduce, d_reduce_frontier);
            }
        } // end of for thread_output
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
