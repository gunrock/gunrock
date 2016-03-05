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
 * @brief simplified filter kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/simplified_filter/cta.cuh>
#include <gunrock/oprtr/simplified_filter/kernel_policy.cuh>
#include <gunrock/oprtr/compacted_cull_filter/kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace simplified_filter {

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    bool     VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef typename Problem::DataSlice     DataSlice;
    typedef typename Functor::LabelT        LabelT;

    static __device__ __forceinline__ void MarkQueue(
        bool                          &queue_reset,
        VertexId                      &queue_index,
        SizeT                         &number_elements,
        VertexId                     *&d_in_queue,
        SizeT                        *&d_markers,
        DataSlice                    *&d_data_slice,
        LabelT                        &label,
        util::CtaWorkProgress<SizeT>  &work_progress,
        util::KernelRuntimeStats      &kernel_stats)
    {
    }

    static __device__ __forceinline__ void AssignQueue (
        bool       &queue_reset,
        VertexId   &queue_index,
        SizeT      &num_vertices,
        VertexId  *&d_out_queue,
        SizeT     *&d_markers,
        DataSlice *&d_data_slice,
        LabelT     &label,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats     &kernel_stats)
    {
    }
};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
struct Dispatch<KernelPolicy, Problem, Functor, true>
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef typename Problem::DataSlice     DataSlice;
    typedef typename Functor::LabelT        LabelT;

    static __device__ __forceinline__ bool MarkVertex(
        //VertexId    s_id,
        VertexId    d_id,
        DataSlice  *d_data_slice)
        //SizeT       node_id,
        //LabelT      label,
        //SizeT       input_pos,
        //SizeT       output_pos,
        //SizeT      *d_markers)
    {
        //if (Functor::CondFilter(s_id, d_id, d_data_slice,
        //    node_id, label, input_pos, output_pos))
        if (!util::isValid(d_id)) return false;
        if (Problem::ENABLE_IDEMPOTENCE && d_data_slice->labels[d_id] != util::MaxValue<LabelT>())
            return false;

        //d_markers[output_pos] = (SizeT)1;
        return true;
    }

    static __device__ __forceinline__ void MarkQueue(
        bool                          &queue_reset,
        VertexId                      &queue_index,
        SizeT                         &num_elements,
        VertexId                     *&d_in_queue,
        SizeT                        *&d_markers,
        DataSlice                    *&d_data_slice,
        LabelT                        &label,
        util::CtaWorkProgress<SizeT>  &work_progress,
        util::KernelRuntimeStats      &kernel_stats)
    {
        if (queue_reset)
            num_elements = work_progress.LoadQueueLength(queue_index);
        SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
        const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
        //__shared__ VertexId outputs[32][64];
        //__shared__ int output_count[32];

        /*if (threadIdx.x == 0)*/ //output_count[threadIdx.x] = 0;
        //__syncthreads();

        while (true)
        {
            if (in_pos >= num_elements) break;
            VertexId v = d_in_queue[in_pos];
            //SizeT output_pos = v;
            if (MarkVertex(
                //util::InvalidValue<VertexId>(), // no pred available
                v,
                d_data_slice))
                //v, // node_id ?
                //label,
                //in_pos,
                //output_pos,
                //d_markers))
            {
                //int target_pos = v & 0x1F;
                //VertexId *target_output = outputs[target_pos];
                //bool done = false;
                //while (!done)
                //{
                //    int count = output_count[target_pos];
                //    target_output[count] = v;
                //    if (target_output[count] == v)
                //        done = true;
                //    output_count[target_pos]++;
                //}
                //output_count++;
                d_markers[v] = (SizeT)1;
            }
            //if (__any(output_count[threadIdx.x] > 32))
            //if (output_count > 1500 - blockDim.x)
            //{
            //    int count = output_count[threadIdx.x];
            //    VertexId *target_output = outputs[threadIdx.x];
            //    for (int i=0; i<count; i++)
            //        d_markers[target_output[i]] = 1;
            //    output_count[threadIdx.x] = 0;
            //}
            //__syncthreads();
            if (in_pos >= num_elements - STRIDE) break;
            in_pos += STRIDE;
        }
        //int count = output_count[threadIdx.x];
        //VertexId *target_output = outputs[threadIdx.x];
        //for (int i=0; i<count; i++)
        //    d_markers[target_output[i]] = 1;
    }

    static __device__ __forceinline__ void AssignVertex(
        VertexId    s_id,
        VertexId    d_id,
        DataSlice  *d_data_slice,
        SizeT       node_id,
        LabelT      label,
        SizeT       input_pos,
        SizeT       output_pos,
        VertexId   *d_output_queue)
    {
        do
        {
            if (Problem::ENABLE_IDEMPOTENCE && d_data_slice->labels[d_id] != util::MaxValue<LabelT>())
            { d_id = util::InvalidValue<VertexId>(); break;}
            if (Functor::CondFilter(s_id, d_id, d_data_slice,
                node_id, label, input_pos, output_pos))
            {
                Functor::ApplyFilter(s_id, d_id, d_data_slice,
                    node_id, label, input_pos, output_pos);
            } else d_id = util::InvalidValue<VertexId>();
        } while (0);
        if (d_output_queue != NULL) d_output_queue[output_pos] = d_id;
    }

    static __device__ __forceinline__ void AssignQueue (
        bool       &queue_reset,
        VertexId   &queue_index,
        SizeT      &num_vertices,
        VertexId  *&d_out_queue,
        SizeT     *&d_markers,
        DataSlice *&d_data_slice,
        LabelT     &label,
        util::CtaWorkProgress<SizeT> &work_progress,
        util::KernelRuntimeStats     &kernel_stats)
    {
        VertexId v = (VertexId)blockIdx.x * blockDim.x + threadIdx.x;
        const VertexId STRIDE = (VertexId)blockDim.x * gridDim.x;
        if (v == 0)
        {
            work_progress.Enqueue(d_markers[num_vertices], queue_index + 1);
        }

        while (true)
        {
            if (v>= num_vertices) break;
            SizeT output_pos = d_markers[v];
            if (d_markers[v+1] != d_markers[v])
            AssignVertex(
                util::InvalidValue<VertexId>(),
                v,
                d_data_slice,
                v,
                label,
                util::InvalidValue<SizeT>(),
                output_pos,
                d_out_queue);
            if (v >= num_vertices - STRIDE) break;
            v += STRIDE;
        }
    }
};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
__launch_bounds__ (KernelPolicy::THREADS/*, KernelPolicy::CTA_OCCUPANCY*/)
  __global__
void MarkQueue(
    bool                             queue_reset,
    typename KernelPolicy::VertexId  queue_index,
    typename KernelPolicy::SizeT     num_elements,
    typename KernelPolicy::VertexId *d_in_queue,
    typename KernelPolicy::SizeT    *d_markers,
    typename Problem::DataSlice     *d_data_slice,
    typename Functor::LabelT         label,
    util::CtaWorkProgress<typename KernelPolicy::SizeT>
                                     work_progress,
    util::KernelRuntimeStats         kernel_stats)
{
    Dispatch<KernelPolicy, Problem, Functor>::MarkQueue (
        queue_reset,
        queue_index,
        num_elements,
        d_in_queue,
        d_markers,
        d_data_slice,
        label,
        work_progress,
        kernel_stats);
}

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
__launch_bounds__ (KernelPolicy::THREADS/*, KernelPolicy::CTA_OCCUPANCY*/)
  __global__
void AssignQueue(
    bool                             queue_reset,
    typename KernelPolicy::VertexId  queue_index,
    typename KernelPolicy::SizeT     num_vertices,
    typename KernelPolicy::VertexId *d_out_queue,
    typename KernelPolicy::SizeT    *d_markers,
    typename Problem::DataSlice     *d_data_slice,
    typename Functor::LabelT         label,
    util::CtaWorkProgress<typename KernelPolicy::SizeT>
                                     work_progress,
    util::KernelRuntimeStats         kernel_stats)
{
    Dispatch<KernelPolicy, Problem, Functor>::AssignQueue(
        queue_reset,
        queue_index,
        num_vertices,
        d_out_queue,
        d_markers,
        d_data_slice,
        label,
        work_progress,
        kernel_stats);
}

template <
    typename _KernelPolicy,
    typename _Problem,
    typename _Functor>
struct KernelParameter
{
    typedef _KernelPolicy KernelPolicy;
    typedef _Problem      Problem;
    typedef _Functor      Functor;
    //typedef Problem::VertexId VertexId;
    //typedef Problem::SizeT    SizeT;
    //typedef Problem::Value    Value;
    //typedef Functor::LabelT   LabelT;

    gunrock::app::EnactorStats<typename KernelPolicy::SizeT>
                                       *enactor_stats;
    gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                       *frontier_attribute;
    typename Functor::LabelT            label;
    typename Problem::DataSlice        *d_data_slice;
    typename KernelPolicy::SizeT       *d_vertex_markers;
    unsigned char                      *d_visited_mask;
    typename KernelPolicy::VertexId    *d_in_key_queue;
    typename KernelPolicy::VertexId    *d_out_key_queue;
    typename KernelPolicy::Value       *d_in_value_queue;
    typename KernelPolicy::Value       *d_out_value_queue;
    //unsigned int                        grid_size, = enactor_stats -> filter_grid_size
    //unsigned int                        block_size, = FilterKernelPolicy::THREADS
    //size_t                              shared_size, = 0
    typename KernelPolicy::SizeT        num_elements;
    typename KernelPolicy::SizeT        num_nodes;
    util::CtaWorkProgress<typename KernelPolicy::SizeT>
                                       *work_progress;
    CudaContext                        *context;
    cudaStream_t                        stream;
    typename KernelPolicy::SizeT        max_in_queue;
    typename KernelPolicy::SizeT        max_out_queue;
    util::KernelRuntimeStats           *kernel_stats;
    bool                                filtering_flag;
    bool                                skip_marking;
};

template <typename Parameter, gunrock::oprtr::filter::MODE FILTER_MODE>
struct LaunchKernel_{
    static cudaError_t Launch(Parameter *parameter)
    {
        extern void UnSupportedFilterMode();
        UnSupportedFilterMode();
        return util::GRError(cudaErrorInvalidDeviceFunction,
            "UnSupportedFilterMode", __FILE__, __LINE__);
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::filter::CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;

    static cudaError_t Launch(Parameter *parameter)
    {
        /*printf("Using original filter, grid_size = %d, block_size = %d, "
            "label = %d, queue_reset = %s, queue_index = %d, "
            "num_elements = %d, in_key = %p, in_value = %p, out_key = %p, "
            "data_slice = %p, visited_mask = %p, max_in_queue = %d, "
            "max_out_queue = %d, flag = %s\n",
            parameter -> enactor_stats -> filter_grid_size,
            Parameter::KernelPolicy::THREADS,
            parameter -> label,
            parameter -> frontier_attribute -> queue_reset ? "true" : "false",
            parameter -> frontier_attribute -> queue_index,
            parameter -> num_elements,
            parameter -> d_in_key_queue,
            parameter -> d_in_value_queue,
            parameter -> d_out_key_queue,
            parameter -> d_data_slice,
            parameter -> d_visited_mask,
            parameter -> max_in_queue,
            parameter -> max_out_queue,
            parameter -> filtering_flag ? "true" : "false");*/
        cudaError_t retval = cudaSuccess;
        gunrock::oprtr::filter::Kernel<
            typename Parameter::KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter -> enactor_stats -> filter_grid_size,
            Parameter::KernelPolicy::THREADS,
            (size_t)0,
            parameter -> stream>>> (
            (VertexId)parameter -> label,
            parameter -> frontier_attribute -> queue_reset,
            (VertexId)parameter -> frontier_attribute -> queue_index,
            parameter -> num_elements,
            parameter -> d_in_key_queue,
            parameter -> d_in_value_queue,
            parameter -> d_out_key_queue,
            parameter -> d_data_slice,
            parameter -> d_visited_mask,
            parameter -> work_progress[0],
            parameter -> max_in_queue,
            parameter -> max_out_queue,
            parameter -> kernel_stats[0],
            parameter -> filtering_flag);
        return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::filter::COMPACTED_CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;

    static cudaError_t Launch(Parameter *parameter)
    {
        /*printf("Using original filter, grid_size = %d, block_size = %d, "
            "label = %d, queue_reset = %s, queue_index = %d, "
            "num_elements = %d, in_key = %p, in_value = %p, out_key = %p, "
            "data_slice = %p, visited_mask = %p, max_in_queue = %d, "
            "max_out_queue = %d, flag = %s\n",
            parameter -> enactor_stats -> filter_grid_size,
            Parameter::KernelPolicy::THREADS,
            parameter -> label,
            parameter -> frontier_attribute -> queue_reset ? "true" : "false",
            parameter -> frontier_attribute -> queue_index,
            parameter -> num_elements,
            parameter -> d_in_key_queue,
            parameter -> d_in_value_queue,
            parameter -> d_out_key_queue,
            parameter -> d_data_slice,
            parameter -> d_visited_mask,
            parameter -> max_in_queue,
            parameter -> max_out_queue,
            parameter -> filtering_flag ? "true" : "false");*/
        cudaError_t retval = cudaSuccess;
        typedef typename gunrock::oprtr::compacted_cull_filter::KernelPolicy<
            typename Parameter::Problem, 350> CCFPolicy;
        gunrock::oprtr::compacted_cull_filter::LaunchKernel<
            CCFPolicy,//typename Parameter::KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter -> enactor_stats -> filter_grid_size,
            CCFPolicy::THREADS,
            (size_t)0,
            parameter -> stream>>> (
            parameter -> label,
            parameter -> frontier_attribute -> queue_reset,
            (VertexId)parameter -> frontier_attribute -> queue_index,
            parameter -> num_elements,
            parameter -> d_in_key_queue,
            parameter -> d_in_value_queue,
            parameter -> d_out_key_queue,
            parameter -> d_data_slice,
            parameter -> d_visited_mask,
            parameter -> work_progress[0],
            parameter -> max_in_queue,
            parameter -> max_out_queue,
            parameter -> kernel_stats[0]);
            //parameter -> filtering_flag);
        return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::filter::SIMPLIFIED>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        if (!parameter -> skip_marking)
        {
            //util::cpu_mt::PrintGPUArray("Input_queue",
            //    parameter -> d_in_key_queue,
            //    parameter -> frontier_attribute -> output_length[0],
            //    -1, -1, -1, parameter -> stream);
            MarkQueue<
                typename Parameter::KernelPolicy,
                typename Parameter::Problem,
                typename Parameter::Functor>
                <<<parameter -> enactor_stats -> filter_grid_size,
                Parameter::KernelPolicy::THREADS,
                (size_t)0,
                parameter -> stream>>> (
                parameter -> frontier_attribute -> queue_reset,
                (VertexId)parameter -> frontier_attribute -> queue_index,
                parameter -> num_elements,
                parameter -> d_in_key_queue,
                parameter -> d_vertex_markers,
                parameter -> d_data_slice,
                parameter -> label,
                parameter -> work_progress[0],
                parameter -> kernel_stats [0]);
        }
        //util::cpu_mt::PrintGPUArray("Markers0",
        //    parameter -> d_vertex_markers,
        //    parameter -> num_nodes + 1,
        //    -1, -1, -1, parameter -> stream);
        Scan<mgpu::MgpuScanTypeExc>(
            parameter -> d_vertex_markers,
            parameter -> num_nodes + 1,
            (SizeT)0,
            mgpu::plus<SizeT>(),
            (SizeT*)NULL,
            (SizeT*)NULL,
            parameter -> d_vertex_markers,
            parameter -> context[0]);
        //util::cpu_mt::PrintGPUArray("Markers1",
        //    parameter -> d_vertex_markers,
        //    parameter -> num_nodes + 1,
        //    -1, -1, -1, parameter -> stream);
        AssignQueue<
            typename Parameter::KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter -> enactor_stats -> filter_grid_size,
            Parameter::KernelPolicy::THREADS,
            (size_t)0,
            parameter -> stream>>> (
            parameter -> frontier_attribute -> queue_reset,
            (VertexId)parameter -> frontier_attribute -> queue_index,
            parameter -> num_nodes,
            parameter -> d_out_key_queue,
            parameter -> d_vertex_markers,
            parameter -> d_data_slice,
            parameter -> label,
            parameter -> work_progress[0],
            parameter -> kernel_stats [0]);
        //cudaStreamSynchronize(parameter -> stream);
        //SizeT output_size;
        //((util::CtaWorkProgressLifetime<SizeT>*)(parameter -> work_progress)) -> GetQueueLength(
        //    parameter -> frontier_attribute -> queue_index + 1, output_size);
        //printf("Output Length = %d\n", output_size);
        //util::cpu_mt::PrintGPUArray("Output_Queue",
        //    parameter -> d_out_key_queue, output_size,
        //    -1, -1, -1, parameter -> stream);
        return retval;
    }
};

template <typename KernelPolicy, typename Problem, typename Functor>
cudaError_t LaunchKernel(
    gunrock::app::EnactorStats<typename KernelPolicy::SizeT>
                                       &enactor_stats,
    gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                       &frontier_attribute,
    typename Functor::LabelT            label,
    typename Problem::DataSlice        *d_data_slice,
    typename KernelPolicy::SizeT       *d_vertex_markers,
    unsigned char                      *d_visited_mask,
    typename KernelPolicy::VertexId    *d_in_key_queue,
    typename KernelPolicy::VertexId    *d_out_key_queue,
    typename KernelPolicy::Value       *d_in_value_queue,
    typename KernelPolicy::Value       *d_out_value_queue,
    //unsigned int                        grid_size, = enactor_stats -> filter_grid_size
    //unsigned int                        block_size, = FilterKernelPolicy::THREADS
    //size_t                              shared_size, = 0
    typename KernelPolicy::SizeT        num_elements,
    typename KernelPolicy::SizeT        num_nodes,
    util::CtaWorkProgress<typename KernelPolicy::SizeT>
                                       &work_progress,
    CudaContext                        &context,
    cudaStream_t                        stream,
    typename KernelPolicy::SizeT        max_in_queue,
    typename KernelPolicy::SizeT        max_out_queue,
    util::KernelRuntimeStats           &kernel_stats,
    bool                                filtering_flag = true,
    bool                                skip_marking = false)
{
    cudaError_t retval = cudaSuccess;
    if (frontier_attribute.queue_reset)
    {
        if (retval = work_progress.Reset_(0,stream))
            return retval;
    }
    typedef KernelParameter<KernelPolicy, Problem, Functor> Parameter;
    Parameter parameter;
    parameter. enactor_stats        = &enactor_stats;
    parameter. frontier_attribute   = &frontier_attribute;
    parameter. label                =  label;
    parameter. d_data_slice         =  d_data_slice;
    parameter. d_vertex_markers     =  d_vertex_markers;
    parameter. d_visited_mask       =  d_visited_mask;
    parameter. d_in_key_queue       =  d_in_key_queue;
    parameter. d_out_key_queue      =  d_out_key_queue;
    parameter. d_in_value_queue     =  d_in_value_queue;
    parameter. d_out_value_queue    =  d_out_value_queue;
    parameter. num_elements         =  num_elements;
    parameter. num_nodes            =  num_nodes;
    parameter. work_progress        = &work_progress;
    parameter. context              = &context;
    parameter. stream               =  stream;
    parameter. max_in_queue         =  max_in_queue;
    parameter. max_out_queue        =  max_out_queue;
    parameter. kernel_stats         = &kernel_stats;
    parameter. filtering_flag       =  filtering_flag;
    parameter. skip_marking         =  skip_marking;

    if (retval = LaunchKernel_<Parameter, KernelPolicy::FILTER_MODE>::Launch(&parameter))
        return retval;

    return retval;
}

} // namespace simplified_filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
