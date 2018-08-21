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

#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/advance_base.cuh>

namespace gunrock {
namespace oprtr {
namespace simplified_advance {

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

    template <typename T>
    static __device__ __forceinline__ SizeT Binary_Search(
        T* data, T item_to_find, SizeT lower_bound, SizeT upper_bound)
    {
        while (lower_bound < upper_bound)
        {
            SizeT mid_point = (lower_bound + upper_bound) >> 1;
            //printf("(%d, %d) looking for %d, current range [%d, %d], mid_point = %d\n",
            //    blockIdx.x, threadIdx.x, item_to_find, lower_bound, upper_bound, data[mid_point]);
            if (_ldg(data + mid_point) < item_to_find)
                lower_bound = mid_point + 1;
            else upper_bound = mid_point;
        }

        SizeT retval = util::InvalidValue<SizeT>();
        if (upper_bound == lower_bound)
        {
            /*if (data[upper_bound] == item_to_find) retval = upper_bound;
            else*/ if (item_to_find < _ldg(data + upper_bound)) retval = upper_bound -1;
            else retval = upper_bound;
        } else retval = util::InvalidValue<SizeT>();

        //printf("(%d, %d) search end, [%d, %d], data[up] = %d, retval = %d\n",
        //    blockIdx.x, threadIdx.x, lower_bound, upper_bound, data[upper_bound], retval);
        return retval;
    }

    static __device__ __forceinline__ void SimpleAdvance(
        bool      &queue_reset,
        VertexId  &queue_index,
        LabelT    &label,
        SizeT     *&d_row_offsets,
        SizeT     *&d_inverse_row_offsets,
        VertexId  *&d_column_indices,
        VertexId  *&d_inverse_column_indices,
        SizeT     *&d_scanned_edges,
        VertexId  *&d_queue, // input  key frontier
        VertexId  *&d_keys_out, // output key frontier
        Value     *&d_values_out,
        DataSlice *&d_data_slice,
        SizeT     &input_queue_length,
        SizeT     *d_output_queue_length,
        SizeT     &max_vertices,
        SizeT     &max_edges,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats     &kernel_stats,
        bool      &input_inverse_graph,
        bool      &output_inverse_graph,
        Value     *&d_value_to_reduce,
        Value     *&d_reduce_frontier)
    {
        //__shared__ typename KernelPolicy::SmemStorage smem;
        //SizeT *output_queue_length;

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            if (queue_reset)
            {
                work_progress.StoreQueueLength(input_queue_length, queue_index);
            } else {
                input_queue_length = work_progress.LoadQueueLength(queue_index);
            }
        }
        //if (threadIdx.x == blockDim.x -1)
        //    output_queue_length = work_progress.GetQueueCounter(queue_index + 1);
        __syncthreads();

        SizeT* row_offsets = (output_inverse_graph) ?
            d_inverse_row_offsets :
            d_row_offsets         ;
        VertexId* column_indices = (output_inverse_graph)?
            d_inverse_column_indices:
            d_column_indices        ;

        SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
        const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
        while (x - threadIdx.x< input_queue_length)
        {
            //bool to_process = true;
            SizeT edge_id = util::InvalidValue<SizeT>();
            VertexId src = util::InvalidValue<VertexId>();
            VertexId des = util::InvalidValue<VertexId>();
            SizeT output_pos = util::InvalidValue<SizeT>();
            if (x < input_queue_length)
            {
                VertexId input_item = (d_queue == NULL) ? NULL : d_queue[x];
                if(input_item){
                    src = input_item;
                    SizeT start_offset = (x > 0) ? d_scanned_edges[x-1] : 0;
                    edge_id =_ldg(row_offsets + input_item) - start_offset;
                    des = _ldg(d_column_indices + edge_id);
                    //src = Binary_Search(d_row_offsets, edge_id, 0, nodes);

                    //printf("(%d, %d) : Edge %d, %d -> %d, input_queue_len = %d\n",
                    //    blockIdx.x, threadIdx.x, edge_id, src, des, input_queue_length);
                    if (Functor::CondEdge(
                        src, des, d_data_slice, edge_id, edge_id,
                        label, x, output_pos))
                    {
                        Functor::ApplyEdge(
                            src, des, d_data_slice, edge_id, edge_id,
                            label, x, output_pos);
                    } else {
                        des = util::InvalidValue<VertexId>();
                    }

                    if(d_keys_out != NULL) {
                        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                            des,
                            d_keys_out + output_pos);
                    }
                }

            }// else to_process = false;

            /*KernelPolicy::BlockScanT::LogicScan(to_process, output_pos, smem.scan_space);
            if (threadIdx.x == blockDim.x -1)
            {
                smem.block_offset = atomicAdd(output_queue_length, output_pos + ((to_process) ? 1 : 0));
            }
            __syncthreads();

            if (to_process)
            {
                output_pos += smem.block_offset;
                if (d_edges_out != NULL)
                    d_edges_out[output_pos] = edge_id;
            }
            __syncthreads();*/
            x += STRIDE;
        }
    } 
};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void SimpleAdvance(
    bool                             &queue_reset,
    typename KernelPolicy::VertexId  &queue_index,
    typename Functor     ::LabelT    &label,
    typename KernelPolicy::SizeT     *&d_row_offsets,
    typename KernelPolicy::SizeT     *&d_inverse_row_offsets,
    typename KernelPolicy::VertexId  *&d_column_indices,
    typename KernelPolicy::VertexId  *&d_inverse_column_indices,
    typename KernelPolicy::SizeT     *&d_scanned_edges,
    typename KernelPolicy::VertexId  *&d_queue, // input  key frontier
    typename KernelPolicy::VertexId  *&d_keys_out, // output key frontier
    typename KernelPolicy::Value     *&d_values_out,
    typename Problem     ::DataSlice *&d_data_slice,
    typename KernelPolicy::SizeT     &input_queue_length,
    typename KernelPolicy::SizeT     *d_output_queue_length,
    typename KernelPolicy::SizeT     &max_vertices,
    typename KernelPolicy::SizeT     &max_edges,
    util::CtaWorkProgress<typename KernelPolicy::SizeT>     &work_progress,
    util::KernelRuntimeStats         &kernel_stats,
    bool                             &input_inverse_graph,
    bool                             &output_inverse_graph,
    typename KernelPolicy::Value     *&d_value_to_reduce,
    typename KernelPolicy::Value     *&d_reduce_frontier)
{
    Dispatch<KernelPolicy, Problem, Functor,
        ADVANCE_TYPE, R_TYPE, R_OP>::SimpleAdvance(
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
        input_queue_length,
        d_output_queue_length,
        max_vertices,
        max_edges,
        work_progress,
        kernel_stats,
        input_inverse_graph,
        output_inverse_graph,
        d_value_to_reduce,
        d_reduce_frontier);
}

} // namespace simplified_advance
} // namespace oprtr
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

