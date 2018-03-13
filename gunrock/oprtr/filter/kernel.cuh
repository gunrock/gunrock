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
 * @brief Filter Kernel Selector
 */
#pragma once

#include <gunrock/app/enactor_types.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/oprtr/cull_filter/kernel.cuh>
#include <gunrock/oprtr/compacted_cull_filter/kernel.cuh>
#include <gunrock/oprtr/simplified_filter/kernel.cuh>
#include <gunrock/oprtr/simplified2_filter/kernel.cuh>
#include <gunrock/oprtr/bypass_filter/kernel.cuh>

namespace gunrock {
namespace oprtr {
namespace filter {

template <
    typename _KernelPolicy,
    typename _Problem,
    typename _Functor>
struct KernelParameter
{
    typedef _KernelPolicy KernelPolicy;
    typedef _Problem      Problem;
    typedef _Functor      Functor;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::Value    Value;
    typedef typename Functor::LabelT   LabelT;
    typedef typename Problem::DataSlice DataSlice;

    gunrock::app::EnactorStats<SizeT>
                     &enactor_stats;
    gunrock::app::FrontierAttribute<SizeT>
                     &frontier_attribute;
    LabelT           &label;
    DataSlice       *&h_data_slice;
    DataSlice       *&d_data_slice;
    SizeT           *&d_vertex_markers;
    unsigned char   *&d_visited_mask;
    VertexId        *&d_in_key_queue;
    VertexId        *&d_out_key_queue;
    Value           *&d_in_value_queue;
    Value           *&d_out_value_queue;
    SizeT            &num_elements;
    SizeT            &num_nodes;
    util::CtaWorkProgressLifetime<SizeT>
                     &work_progress;
    CudaContext      &context;
    cudaStream_t     &stream;
    SizeT            &max_in_queue;
    SizeT            &max_out_queue;
    util::KernelRuntimeStatsLifetime
                     &kernel_stats;
    bool             &filtering_flag;
    bool             &skip_marking;

    KernelParameter(
        gunrock::app::EnactorStats<SizeT>
                         &enactor_stats,
        gunrock::app::FrontierAttribute<SizeT>
                         &frontier_attribute,
        LabelT           &label,
        DataSlice       *&h_data_slice,
        DataSlice       *&d_data_slice,
        SizeT           *&d_vertex_markers,
        unsigned char   *&d_visited_mask,
        VertexId        *&d_in_key_queue,
        VertexId        *&d_out_key_queue,
        Value           *&d_in_value_queue,
        Value           *&d_out_value_queue,
        SizeT            &num_elements,
        SizeT            &num_nodes,
        util::CtaWorkProgressLifetime<SizeT>
                         &work_progress,
        CudaContext      &context,
        cudaStream_t     &stream,
        SizeT            &max_in_queue,
        SizeT            &max_out_queue,
        util::KernelRuntimeStatsLifetime
                         &kernel_stats,
        bool             &filtering_flag,
        bool             &skip_marking) :
        enactor_stats       (enactor_stats      ),
        frontier_attribute  (frontier_attribute ),
        label               (label              ),
        h_data_slice        (h_data_slice       ),
        d_data_slice        (d_data_slice       ),
        d_vertex_markers    (d_vertex_markers   ),
        d_visited_mask      (d_visited_mask     ),
        d_in_key_queue      (d_in_key_queue     ),
        d_out_key_queue     (d_out_key_queue    ),
        d_in_value_queue    (d_in_value_queue   ),
        d_out_value_queue   (d_out_value_queue  ),
        num_elements        (num_elements       ),
        num_nodes           (num_nodes          ),
        work_progress       (work_progress      ),
        context             (context            ),
        stream              (stream             ),
        max_in_queue        (max_in_queue       ),
        max_out_queue       (max_out_queue      ),
        kernel_stats        (kernel_stats       ),
        filtering_flag      (filtering_flag     ),
        skip_marking        (skip_marking       )
    {}
};

template <
    typename Parameter,
    MODE FILTER_MODE>
struct Dispatch{
    static cudaError_t Launch(Parameter &parameter)
    {
        extern void UnSupportedFilterMode();
        UnSupportedFilterMode();
        return util::GRError(cudaErrorInvalidDeviceFunction,
            "UnSupported Filter Mode", __FILE__, __LINE__);
    }
};

template <typename Parameter>
struct Dispatch<Parameter, CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;
    typedef typename Parameter::KernelPolicy::CULL_FILTER KernelPolicy;

    static cudaError_t Launch(Parameter &parameter)
    {
        cudaError_t retval = cudaSuccess;
        gunrock::oprtr::cull_filter::Kernel<
            KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter. enactor_stats.filter_grid_size,
            KernelPolicy::THREADS,
            (size_t)0,
            parameter. stream>>> (
            (VertexId)parameter. label,
            parameter. frontier_attribute. queue_reset,
            (VertexId)parameter. frontier_attribute. queue_index,
            parameter. num_elements,
            parameter. d_in_key_queue,
            parameter. d_in_value_queue,
            parameter. d_out_key_queue,
            parameter. d_data_slice,
            parameter. d_visited_mask,
            parameter. work_progress,
            parameter. max_in_queue,
            parameter. max_out_queue,
            parameter. kernel_stats);
            //parameter. filtering_flag);
        return retval;
    }
};

template <typename Parameter>
struct Dispatch<Parameter, SIMPLIFIED2>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;
    typedef typename Parameter::KernelPolicy::SIMPLIFIED2_FILTER KernelPolicy;

    static cudaError_t Launch(Parameter &parameter)
    {
        cudaError_t retval = cudaSuccess;

        //d_data_slice->valid_in.GetPointer(util::DEVICE)
        //d_data_slice->valid_out.GetPointer(util::DEVICE)
        // launch settings
        int num_block = (parameter.num_elements + KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;

        if (parameter.h_data_slice -> visit_lookup[0].GetPointer(util::DEVICE) == NULL)
        {
            if (retval = parameter.h_data_slice -> visit_lookup[0].
                Allocate(parameter.num_nodes + 1, util::DEVICE))
                return retval;
        } else {
            if (retval = parameter.h_data_slice -> visit_lookup[0].
                EnsureSize(parameter.num_nodes + 1))
                return retval;
        }

        if (parameter.h_data_slice -> valid_in[0].GetPointer(util::DEVICE) == NULL)
        {
            if (retval = parameter.h_data_slice -> valid_in[0].
                Allocate(parameter.num_elements + 1, util::DEVICE))
                return retval;
        } else {
            if (retval = parameter.h_data_slice -> valid_in[0].
                EnsureSize(parameter.num_elements + 1))
                return retval;
        }

        if (parameter.h_data_slice -> valid_out[0].GetPointer(util::DEVICE) == NULL)
        {
            if (retval = parameter.h_data_slice -> valid_out[0].
                Allocate(parameter.num_elements + 1, util::DEVICE))
                return retval;
        } else {
            if (retval = parameter.h_data_slice -> valid_out[0].
                EnsureSize(parameter.num_elements + 1))
                return retval;
        }

        // MarkVisit
        gunrock::oprtr::simplified2_filter::MarkVisit
            <KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<num_block, KernelPolicy::THREADS>>>
            (parameter.d_in_key_queue,
            parameter.h_data_slice->visit_lookup[0].GetPointer(util::DEVICE),
            parameter.num_elements, parameter.max_in_queue);

        // MarkValid
        gunrock::oprtr::simplified2_filter::MarkValid
            <KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<num_block, KernelPolicy::THREADS>>>
            (parameter.d_in_key_queue,
            parameter.h_data_slice->visit_lookup[0].GetPointer(util::DEVICE),
            parameter.h_data_slice->valid_in[0].GetPointer(util::DEVICE),
            parameter.num_elements, parameter.max_in_queue);

        // ExcScan of num_elements+1
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            parameter.h_data_slice->valid_in[0].GetPointer(util::DEVICE),
            parameter.h_data_slice->valid_out[0].GetPointer(util::DEVICE),
            parameter.num_elements+1);
        if (retval = util::GRError(cudaMalloc(
            &d_temp_storage, temp_storage_bytes),
            "cudaMalloc failed", __FILE__, __LINE__))
            return retval;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            parameter.h_data_slice->valid_in[0].GetPointer(util::DEVICE),
            parameter.h_data_slice->valid_out[0].GetPointer(util::DEVICE),
            parameter.num_elements+1);

        // Compact
        gunrock::oprtr::simplified2_filter::Compact
            <KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<num_block, KernelPolicy::THREADS>>>
            (parameter.d_in_key_queue,
            parameter.d_out_key_queue,
            parameter.h_data_slice->valid_out[0].GetPointer(util::DEVICE),
            parameter.label,
            parameter.d_data_slice,
            parameter.num_elements,
            parameter.max_in_queue);

        util::MemsetCopyVectorKernel<<<1,1>>>(
            parameter.work_progress.template GetQueueLengthPointer<unsigned int>(parameter.frontier_attribute.queue_index + 1),
            parameter.h_data_slice -> valid_out[0].GetPointer(util::DEVICE) + parameter.num_elements, 1);

        if (retval = util::GRError(cudaFree(
            d_temp_storage), "cudaFree failed", __FILE__, __LINE__))
            return retval;
        return retval;
    }
};

template <typename Parameter>
struct Dispatch<Parameter, COMPACTED_CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;
    typedef typename Parameter::KernelPolicy::COMPACTED_CULL_FILTER KernelPolicy;

    static cudaError_t Launch(Parameter &parameter)
    {
        cudaError_t retval = cudaSuccess;
        gunrock::oprtr::compacted_cull_filter::LaunchKernel<
            KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter. enactor_stats. filter_grid_size,
            KernelPolicy::THREADS,
            (size_t)0,
            parameter. stream>>> (
            parameter. label,
            parameter. frontier_attribute. queue_reset,
            (VertexId)parameter. frontier_attribute. queue_index,
            parameter. num_elements,
            parameter. d_in_key_queue,
            parameter. d_in_value_queue,
            parameter. d_out_key_queue,
            parameter. d_data_slice,
            parameter. d_visited_mask,
            parameter. work_progress,
            parameter. max_in_queue,
            parameter. max_out_queue,
            parameter. kernel_stats);
            //parameter -> filtering_flag);
        return retval;
    }
};

template <typename Parameter>
struct Dispatch<Parameter, SIMPLIFIED>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;
    typedef typename Parameter::KernelPolicy::SIMPLIFIED_FILTER KernelPolicy;

    static cudaError_t Launch(Parameter &parameter)
    {
        cudaError_t retval = cudaSuccess;
        if (!parameter. skip_marking)
        {
            //util::cpu_mt::PrintGPUArray("Input_queue",
            //    parameter -> d_in_key_queue,
            //    parameter -> frontier_attribute -> output_length[0],
            //    -1, -1, -1, parameter -> stream);
            gunrock::oprtr::simplified_filter::MarkQueue<
                KernelPolicy,
                typename Parameter::Problem,
                typename Parameter::Functor>
                <<<parameter. enactor_stats. filter_grid_size,
                KernelPolicy::THREADS,
                (size_t)0,
                parameter. stream>>> (
                parameter. frontier_attribute. queue_reset,
                (VertexId)parameter. frontier_attribute. queue_index,
                parameter. num_elements,
                parameter. d_in_key_queue,
                parameter. d_vertex_markers,
                parameter. d_data_slice,
                parameter. label,
                parameter. work_progress,
                parameter. kernel_stats );
        }
        //util::cpu_mt::PrintGPUArray("Markers0",
        //    parameter -> d_vertex_markers,
        //    parameter -> num_nodes + 1,
        //    -1, -1, -1, parameter -> stream);
        Scan<mgpu::MgpuScanTypeExc>(
            parameter. d_vertex_markers,
            parameter. num_nodes + 1,
            (SizeT)0,
            mgpu::plus<SizeT>(),
            (SizeT*)NULL,
            (SizeT*)NULL,
            parameter. d_vertex_markers,
            parameter. context);
        //util::cpu_mt::PrintGPUArray("Markers1",
        //    parameter -> d_vertex_markers,
        //    parameter -> num_nodes + 1,
        //    -1, -1, -1, parameter -> stream);
        gunrock::oprtr::simplified_filter::AssignQueue<
            KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<parameter. enactor_stats. filter_grid_size,
            KernelPolicy::THREADS,
            (size_t)0,
            parameter. stream>>> (
            parameter. frontier_attribute. queue_reset,
            (VertexId)parameter. frontier_attribute. queue_index,
            parameter. num_nodes,
            parameter. d_out_key_queue,
            parameter. d_vertex_markers,
            parameter. d_data_slice,
            parameter. label,
            parameter. work_progress,
            parameter. kernel_stats );
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

template <typename Parameter>
struct Dispatch<Parameter, BY_PASS>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value   ;
    typedef typename Parameter::KernelPolicy::BYPASS_FILTER KernelPolicy;

    static cudaError_t Launch(Parameter &parameter)
    {
        cudaError_t retval = cudaSuccess;
        int num_blocks = parameter.num_elements / KernelPolicy::THREADS / 4 + 1;
        if (num_blocks > 480) num_blocks = 480;

        gunrock::oprtr::bypass_filter::Kernel<
            KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor>
            <<<num_blocks, //parameter. enactor_stats.filter_grid_size,
            KernelPolicy::THREADS,
            (size_t)0,
            parameter. stream>>> (
            parameter. label,
            parameter. frontier_attribute. queue_reset,
            (VertexId)parameter. frontier_attribute. queue_index,
            parameter. num_elements,
            parameter. d_in_key_queue,
            //parameter. d_in_value_queue,
            parameter. d_out_key_queue,
            parameter. d_data_slice,
            //parameter. d_visited_mask,
            parameter. work_progress,
            parameter. max_in_queue,
            parameter. max_out_queue,
            parameter. kernel_stats);
            //parameter. filtering_flag);
        return retval;
    }
};

template <typename KernelPolicy, typename Problem, typename Functor>
cudaError_t LaunchKernel(
    gunrock::app::EnactorStats<typename Problem::SizeT>
                                 &enactor_stats,
    gunrock::app::FrontierAttribute<typename Problem::SizeT>
                                 &frontier_attribute,
    typename Functor::LabelT      label,
    typename Problem::DataSlice * h_data_slice,
    typename Problem::DataSlice * d_data_slice,
    typename Problem::SizeT     * d_vertex_markers,
    unsigned char               * d_visited_mask,
    typename Problem::VertexId  * d_in_key_queue,
    typename Problem::VertexId  * d_out_key_queue,
    typename Problem::Value     * d_in_value_queue,
    typename Problem::Value     * d_out_value_queue,
    typename Problem::SizeT       num_elements,
    typename Problem::SizeT       num_nodes,
    util::CtaWorkProgressLifetime<typename Problem::SizeT>
                                 &work_progress,
    CudaContext                  &context,
    cudaStream_t                 &stream,
    typename Problem::SizeT       max_in_queue,
    typename Problem::SizeT       max_out_queue,
    util::KernelRuntimeStatsLifetime
                                 &kernel_stats,
    bool                          filtering_flag = true,
    bool                          skip_marking = false)
{
    cudaError_t retval = cudaSuccess;
    if (frontier_attribute.queue_reset)
    {
        if (retval = work_progress.Reset_(0,stream))
            return retval;
    }
    typedef KernelParameter<KernelPolicy, Problem, Functor> Parameter;
    Parameter parameter(
        enactor_stats       ,
        frontier_attribute  ,
        label               ,
        h_data_slice        ,
        d_data_slice        ,
        d_vertex_markers    ,
        d_visited_mask      ,
        d_in_key_queue      ,
        d_out_key_queue     ,
        d_in_value_queue    ,
        d_out_value_queue   ,
        num_elements        ,
        num_nodes           ,
        work_progress       ,
        context             ,
        stream              ,
        max_in_queue        ,
        max_out_queue       ,
        kernel_stats        ,
        filtering_flag      ,
        skip_marking        );

    if (filtering_flag)
    {
        if (retval = Dispatch<Parameter, KernelPolicy::FILTER_MODE>::Launch(parameter))
        return retval;
    } else {
        if (retval = Dispatch<Parameter, BY_PASS>::Launch(parameter))
        return retval;
    }

    return retval;
}

} // namespace filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
