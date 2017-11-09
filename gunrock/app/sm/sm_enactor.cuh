// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * sm_enactor.cuh
 *
 * @brief Problem enactor for Subgraph Matching
 */

#pragma once

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <gunrock/util/select_utils.cuh>
#include <gunrock/util/segmented_reduce_utils.cuh>
#include <gunrock/util/join.cuh>

#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/app/sm/sm_functor.cuh>

#include <moderngpu.cuh>
#include <cub/cub.cuh>

//using namespace gunrock::app;
using namespace mgpu;
using namespace cub;

namespace gunrock {
namespace app {
namespace sm {

/// Selection functor type
struct GreaterThan
{
    int compare;

    __host__ __device__ __forceinline__
        GreaterThan(int compare) : compare(compare) {}

    __host__ __device__ __forceinline__
        bool operator()(const int &a) const {
            return (a > compare);
        }
};

using namespace mgpu;
using namespace cub;

/**
 * @brief SM enactor class.
 *
 * @tparam _Problem
 * @tparam _INSTRUMWENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <typename _Problem>
class SMEnactor :  public EnactorBase<typename _Problem::SizeT> 
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

    /** 
     * @brief SMEnactor Constructor.
     *
     * @param[in] gpu_idx GPU indices
     */
    SMEnactor(
        int   num_gpus   = 1,  
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = true,
        bool  size_check = true) :
        BaseEnactor(
            EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem (NULL),
        context (NULL) 
    {   
    }   

    /**
    * @brief SMEnactor destructor
    */
    virtual ~SMEnactor()
    {
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitSM(
        ContextPtr *context,
        Problem    *problem,
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = BaseEnactor::Init(
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this -> problem = problem;
        this -> context = context;

        return retval;
    }

    /**
     * @brief Enacts a SM computing on the specified graph.
     *
     * @tparam Advance Kernel policy for forward advance kernel.
     * @tparam Filter Kernel policy for filter kernel.
     * @tparam SMProblem SM Problem type.
     *
     * @param[in] context CudaContext for ModernGPU library
     * @param[in] problem MSTProblem object.
     * @param[in] max_grid_size Max grid size for SM kernel calls.
     *
     * \return cudaError_t object which indicates the success of
     * all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSM()
    {
        // Define functors for primitive
        typedef SMInitFunctor   <VertexId, SizeT, Value, Problem> SMInitFunctor;
        typedef SMFilterFunctor <VertexId, SizeT, Value, Problem> SMFilterFunctor;
        typedef SMDistributeFunctor  <VertexId, SizeT, Value, Problem> SMDistributeFunctor;
        typedef typename SMInitFunctor::LabelT      LabelT;
        typedef util::DoubleBuffer  <VertexId, SizeT, Value> Frontier;
        typedef GraphSlice          <VertexId, SizeT, Value> GraphSliceT;
    
        typedef typename Problem::DataSlice                  DataSlice;

        Problem      *problem            = this -> problem;
        EnactorStats<SizeT> *statistics  = &this->enactor_stats     [0];
        DataSlice    *data_slice         =  problem -> data_slices  [0].GetPointer(util::HOST);
        DataSlice    *d_data_slice       =  problem -> data_slices  [0].GetPointer(util::DEVICE);
        GraphSliceT  *graph_slice        =  problem -> graph_slices [0];
        Frontier     *queue              = &data_slice->frontier_queues[0];
        FrontierAttribute<SizeT>
                     *attributes         = &this->frontier_attribute[0];
        util::CtaWorkProgressLifetime<SizeT>
                     *work_progress      = &this->work_progress     [0];
        cudaStream_t  stream             =  data_slice->streams     [0];
        ContextPtr    context            =  this -> context         [0];
        cudaError_t   retval             = cudaSuccess;
        SizeT        *d_scanned_edges    = NULL;  // Used for LB
        SizeT         nodes              = graph_slice -> nodes;
        SizeT         edges              = graph_slice -> edges;
        bool          debug_info         = 0;   // used for debug purpose

        if (data_slice -> scanned_edges[0].GetSize() == 0)
        {
            if (retval = data_slice -> scanned_edges[0].Allocate(edges, util::DEVICE))
                return retval;
        }
        else if (retval = data_slice -> scanned_edges[0].EnsureSize(edges))
            return retval;
        d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);


        if (debug_info)
        {
            printf("\nBEGIN ITERATION: %lld #NODES: %lld #EDGES: %lld\n",
                statistics->iteration+1,
                (long long)nodes,
                (long long)edges);
            printf(":: initial read in row_offsets ::\n");
            util::DisplayDeviceResults(
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->nodes + 1);
            util::DisplayDeviceResults(
                graph_slice->column_indices.GetPointer(util::DEVICE),
                graph_slice->edges);
        }
        
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = nodes;
        attributes->queue_reset  = true;

        gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, SMInitFunctor>(
                    statistics[0],
                    attributes[0],
                    (VertexId) statistics -> iteration,
                    data_slice,
                    d_data_slice,
                    (SizeT*)NULL, //vertex markers
                    (unsigned char*)NULL, // visited_mask
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    (Value*   )NULL,
                    (Value*   )NULL,
                    graph_slice->edges,
                    graph_slice->nodes,
                    work_progress[0],
                    context[0],
                    stream,
                    queue->keys[attributes->selector  ].GetSize(),
                    queue->keys[attributes->selector^1].GetSize(),
                    statistics -> filter_kernel_stats,
                    false, //By-Pass
                    false //skip_marking
                    );
                    
        if (debug_info)
        {
             if (retval = util::GRError(cudaStreamSynchronize(stream),
                            "SMInit Filter::LaunchKernel failed", __FILE__, __LINE__)) 
                    return retval;
        }
        attributes->selector = attributes->selector^1;
        attributes->queue_index++;
        gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, SMInitFunctor, gunrock::oprtr::advance::V2V>(
                    statistics[0],
                    attributes[0],
                    util::InvalidValue<LabelT>(),
                    data_slice,
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    d_scanned_edges,
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    (Value*   )NULL,
                    (Value*   )NULL,
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,
                    graph_slice->edges,
                    work_progress[0],
                    context[0],
                    stream,
                    false,
                    false,
                    true);

        if (debug_info)
        {
             if (retval = util::GRError(cudaStreamSynchronize(stream),
                            "SMInit Advance::LaunchKernel failed", __FILE__, __LINE__)) 
                    return retval;
        }

        gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, SMFilterFunctor>(
                    statistics[0],
                    attributes[0],
                    (VertexId) statistics -> iteration,
                    data_slice,
                    d_data_slice,
                    (SizeT*)NULL, //vertex markers
                    (unsigned char*)NULL, // visited_mask
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    (Value*   )NULL,
                    (Value*   )NULL,
                    nodes,
                    graph_slice->nodes,
                    work_progress[0],
                    context[0],
                    stream,
                    queue->keys[attributes->selector  ].GetSize(),
                    queue->keys[attributes->selector^1].GetSize(),
                    statistics -> filter_kernel_stats,
                    false, //By-Pass
                    false //skip_marking
                    );

        if (debug_info)
        {
           if (retval = util::GRError(cudaStreamSynchronize(stream),
                        "SMFilterFunctor Filter::LaunchKernel failed", __FILE__, __LINE__)) 
                return retval;
        }
        attributes->selector = attributes->selector^1;
        attributes->queue_index++;
        gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, SMDistributeFunctor>(
                    statistics[0],
                    attributes[0],
                    (VertexId) statistics -> iteration,
                    data_slice,
                    d_data_slice,
                    (SizeT*)NULL, //vertex markers
                    (unsigned char*)NULL, // visited_mask
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    (Value*   )NULL,
                    (Value*   )NULL,
                    nodes,
                    graph_slice->nodes,
                    work_progress[0],
                    context[0],
                    stream,
                    queue->keys[attributes->selector  ].GetSize(),
                    queue->keys[attributes->selector^1].GetSize(),
                    statistics -> filter_kernel_stats,
                    false, //By-Pass
                    false //skip_marking
                    );

        if (debug_info)
        {
             if (retval = util::GRError(cudaStreamSynchronize(stream),
                        "SMDistributeFunctor Filter::LaunchKernel failed", __FILE__, __LINE__)) 
                return retval;
        }

        attributes->selector = attributes->selector^1;
        attributes->queue_length = nodes;

        util::MemsetKernel<<<128,128, 0, stream>>>(
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                0, data_slice->d_src_node_id.GetSize());

        attributes->queue_index++;
        // First iteration of BFS-based joining
        gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, SMDistributeFunctor, gunrock::oprtr::advance::V2V>(
                    statistics[0],
                    attributes[0],
                    util::InvalidValue<LabelT>(),
                    data_slice,
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    d_scanned_edges,
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    (Value*   )NULL,
                    (Value*   )NULL,
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    attributes->queue_length,
	            graph_slice->edges,
                    work_progress[0],
                    context[0],
                    stream,
                    false,
                    false,
                    true);

        if(debug_info) {
            if (retval = util::GRError(cudaStreamSynchronize(stream),
                        "SMDistributeFunctor Adavance::LaunchKernel failed", __FILE__, __LINE__)) 
                return retval;
            printf("After advance:\n");
            util::DisplayDeviceResults(
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                data_slice->d_src_node_id.GetSize());
        }

        GreaterThan select_op(0);
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        // Calculate new src node ids to scanned_edges[0]
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_id.GetPointer(util::DEVICE),
            data_slice -> scanned_edges[0].GetPointer(util::DEVICE), 
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_id.GetPointer(util::DEVICE),
            data_slice -> scanned_edges[0].GetPointer(util::DEVICE), 
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);

        if(debug_info){
            if (retval = util::GRError(cudaStreamSynchronize(stream),
              "cub seg reduce iter 0 cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                return retval;
            printf("src node after seg reduce:\n");
            util::DisplayDeviceResults(
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE),
                graph_slice->nodes+1);
        }

        Scan<mgpu::MgpuScanTypeExc>(
            data_slice -> scanned_edges[0].GetPointer(util::DEVICE), 
            graph_slice->nodes+1,
            (int)0,
            mgpu::plus<int>(),
            (int*)0,
            (int*)0,
            data_slice->d_src_node_id.GetPointer(util::DEVICE),
            context[0]);

        if(debug_info) {
            if (retval = util::GRError(
                    cudaStreamSynchronize(stream),
                   "mgpu scan iter 0 cudaStreamSynchronize failed", __FILE__, __LINE__)) return retval;
            printf("After scan:\n");
            util::DisplayDeviceResults(
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                graph_slice->nodes);
            util::DisplayDeviceResults(
                queue->keys[attributes->selector].GetPointer(util::DEVICE), 
                queue->keys[attributes->selector].GetSize());
            printf("queue keys size:%d, nodes:%d\n", queue->keys[attributes->selector].GetSize(), graph_slice->nodes);
        }

        IntervalExpand(
            graph_slice->edges/2,
            data_slice->d_src_node_id.GetPointer(util::DEVICE), // Expand counts
            queue->keys[attributes->selector].GetPointer(util::DEVICE), //Expand values
            graph_slice->nodes,
            data_slice -> scanned_edges[0].GetPointer(util::DEVICE), //outputs
            context[0]);

        if(debug_info){
            if (retval = util::GRError(
                    cudaStreamSynchronize(stream),
                   "interval expand iter0 cudaStreamSynchronize failed", __FILE__, __LINE__)) return retval;
            printf("new src node ids:\n");
        util::DisplayDeviceResults(
            data_slice -> scanned_edges[0].GetPointer(util::DEVICE), //outputs
            data_slice->d_src_node_id.GetSize());
        }
        //-----------------------------------------------------------------------------------------------------------
        // Compact dest node ids to queue key selector
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        util::MemsetKernel<<<128,128, 0, stream>>>(
                            queue->keys[attributes->selector].GetPointer(util::DEVICE),
                            -1, 
                            queue->keys[attributes->selector].GetSize());
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            data_slice -> num_subs.GetPointer(util::DEVICE),
            queue->keys[attributes->selector^1].GetSize(),
            select_op);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            data_slice -> num_subs.GetPointer(util::DEVICE),
            queue->keys[attributes->selector^1].GetSize(),
            select_op);

        if(debug_info){
            if (retval = util::GRError(cudaStreamSynchronize(stream),
                 "cub select iter 0 cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                return retval;
            printf("number of intermediate results:\n");
            util::DisplayDeviceResults(
                data_slice -> num_subs.GetPointer(util::DEVICE), 
                data_slice -> num_subs.GetSize());

                printf("prev dest node ids:\n");
            util::DisplayDeviceResults(
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE), 
                queue->keys[attributes->selector^1].GetSize());
            printf("new src node ids:\n");
            util::DisplayDeviceResults(
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE), 
                data_slice -> scanned_edges[0].GetSize());
            printf("new dest node ids:\n");
            util::DisplayDeviceResults(
                queue->keys[attributes->selector].GetPointer(util::DEVICE), 
                queue->keys[attributes->selector].GetSize());
        }

        util::WriteToPartial<<<128, 128, 0, stream>>>(
             data_slice -> scanned_edges[0].GetPointer(util::DEVICE), // src_node 
             queue->keys[attributes->selector].GetPointer(util::DEVICE), // dest_node
             data_slice -> d_src_node_id.GetPointer(util::DEVICE),// index for iter>1
             data_slice -> num_subs.GetPointer(util::DEVICE), // # of partial results
             0, // iteration number
             data_slice -> nodes_query, // partial result storing stride
             data_slice -> d_partial.GetPointer(util::DEVICE));
        
        if(debug_info){
            if (retval = util::GRError(cudaStreamSynchronize(stream),
                "Write to partial cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                return retval;
            printf("partial results for iteration 0:\n");
            util::DisplayDeviceResults(
                    data_slice->d_partial.GetPointer(util::DEVICE),
                    data_slice->d_partial.GetSize());
            printf("input size:%d, output size:%d, scanned_edge size:%d\n", queue->keys[attributes->selector  ].GetSize(), queue->keys[attributes->selector^1].GetSize(), data_slice -> scanned_edges[0].GetSize());
        }

        for(SizeT i=1; i<data_slice->nodes_query-1; i++) {
            util::MemsetKernel<<<1,1, 0, stream>>>(
                data_slice -> counter.GetPointer(util::DEVICE),
                i+1, 1);
            util::MemsetKernel<<<128,128, 0, stream>>>(
                    data_slice->d_src_node_id.GetPointer(util::DEVICE),
                    0, data_slice->d_src_node_id.GetSize());
            util::MemsetKernel<<<128,128, 0, stream>>>(
                    data_slice->d_index.GetPointer(util::DEVICE),
                    graph_slice->edges+1, data_slice->d_index.GetSize());
            util::MemsetKernel<<<1,1,0, stream>>>(
                    data_slice->num_subs.GetPointer(util::DEVICE),
                    0, 1);
            attributes->queue_length = edges;
            attributes->queue_index++;
            gunrock::oprtr::advance::LaunchKernel
                        <AdvanceKernelPolicy, Problem, SMDistributeFunctor, gunrock::oprtr::advance::V2V>(
                        statistics[0],
                        attributes[0],
                        util::InvalidValue<LabelT>(),
                        data_slice,
                        d_data_slice,
                        (VertexId*)NULL,
                        (bool*    )NULL,
                        (bool*    )NULL,
                        d_scanned_edges,
                        queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                        NULL,
                        (Value*   )NULL,
                        (Value*   )NULL,
                        graph_slice->row_offsets   .GetPointer(util::DEVICE),
                        graph_slice->column_indices.GetPointer(util::DEVICE),
                        (SizeT*   )NULL,
                        (VertexId*)NULL,
                        attributes->queue_length,
                        0,
                        work_progress[0],
                        context[0],
                        stream,
                        false,
                        false,
                        true);

            if(debug_info) {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                            "SMDistributeFunctor iteration>0 Adavance::LaunchKernel failed", __FILE__, __LINE__)) 
                    return retval;
            }
            d_temp_storage = NULL;
            temp_storage_bytes = 0;

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                data_slice->d_index.GetPointer(util::DEVICE),
                queue->keys[attributes->selector].GetPointer(util::DEVICE), 
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                data_slice->d_index.GetSize());

            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                data_slice->d_index.GetPointer(util::DEVICE),
                queue->keys[attributes->selector].GetPointer(util::DEVICE), 
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                data_slice->d_index.GetSize());

            if(debug_info){
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                  "cub radix sort iter 1 cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                    return retval;
                printf("After sort dest node:\n");
                util::DisplayDeviceResults(
                    queue->keys[attributes->selector].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector].GetSize());
                util::DisplayDeviceResults(
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector^1].GetSize());
            }

            util::MemsetCopyVectorKernel<<<128,128,0,stream>>>(
                data_slice->d_src_node_id.GetPointer(util::DEVICE),
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                data_slice->d_src_node_id.GetSize());

            d_temp_storage = NULL;
            temp_storage_bytes = 0;
            util::MemsetKernel<<<128,128, 0, stream>>>(
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                -1, 
                data_slice->d_src_node_id.GetSize());
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                data_slice -> num_subs.GetPointer(util::DEVICE),
                data_slice->d_src_node_id.GetSize(), 
                select_op);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                data_slice -> num_subs.GetPointer(util::DEVICE),
                data_slice->d_src_node_id.GetSize(), 
                select_op);

            if(debug_info){
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                  "cub select iter >0 cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                    return retval;
                printf("new dest node ids:\n");
                util::DisplayDeviceResults(
                    queue->keys[attributes->selector].GetPointer(util::DEVICE), 
                    queue->keys[attributes->selector].GetSize());

                printf("# of partial results:\n");
                util::DisplayDeviceResults(
                        data_slice->num_subs.GetPointer(util::DEVICE),
                        data_slice->num_subs.GetSize());
            }

            util::WriteToPartial<<<128, 128, 0, stream>>>(
                 (SizeT*)NULL,
                 queue->keys[attributes->selector].GetPointer(util::DEVICE), // dest_node
                 data_slice -> d_src_node_id.GetPointer(util::DEVICE),// index for iter>1
                 data_slice -> num_subs.GetPointer(util::DEVICE), // # of partial results
                 i, // iteration number
                 data_slice -> nodes_query, // partial result storing stride
                 data_slice -> d_partial.GetPointer(util::DEVICE));
            
            if(debug_info) {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                  "Write to partial cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                    return retval;
                printf("partial results before compaction:\n");
                util::DisplayDeviceResults(
                        data_slice->d_partial.GetPointer(util::DEVICE),
                        data_slice->d_partial.GetSize());
            }

            //Compact partial results (filter out those with uncomplete partial results)
            util::MaskOut<<<128,128, 0, stream>>>(i,
                                       data_slice->nodes_query,
                                       data_slice->d_partial.GetSize(),
                                       data_slice->d_partial.GetPointer(util::DEVICE));
            if(debug_info)
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                  "Mask out cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                    return retval;

            util::MemsetKernel<<<128,128, 0, stream>>>(
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE),
                (SizeT)(-1), data_slice -> scanned_edges[0].GetSize());

            GreaterThan select_op1(-1);
            d_temp_storage = NULL;
            temp_storage_bytes = 0;
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                data_slice -> d_partial.GetPointer(util::DEVICE),
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE),
                data_slice -> num_subs.GetPointer(util::DEVICE),
                data_slice -> d_partial.GetSize(),
                select_op1);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                data_slice -> d_partial.GetPointer(util::DEVICE),
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE),
                data_slice -> num_subs.GetPointer(util::DEVICE),
                data_slice -> d_partial.GetSize(),
                select_op1);
            if(debug_info){
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                  "cub select iter >0 cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                    return retval;
                printf("partial results:\n");
                    util::DisplayDeviceResults(
                            data_slice->d_partial.GetPointer(util::DEVICE),
                            data_slice->d_partial.GetSize());
            }

            util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                data_slice->d_partial.GetPointer(util::DEVICE),
                data_slice -> scanned_edges[0].GetPointer(util::DEVICE),
                data_slice->d_partial.GetSize());
        }
        if(debug_info){
            printf("Final result:\n");
            util::DisplayDeviceResults(
                    data_slice -> num_subs.GetPointer(util::DEVICE),
                    1);
            return retval;
        }
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,            // Problem data type
        300,                // CUDA_ARCH
        //INSTRUMENT,         // INSTRUMENT
        0,                  // SATURATION QUIT
        true,               // DEQUEUE_PROBLEM_SIZE
        8,                  // MIN_CTA_OCCUPANCY
        8,                  // LOG_THREADS
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        5,                  // END_BITMASK_CULL
        8>                  // LOG_SCHEDULE_GRANULARITY
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,            // Problem data type
        300,                // CUDA_ARCH
        //INSTRUMENT,         // INSTRUMENT
        8,                  // MIN_CTA_OCCUPANCY
        10,                 // LOG_THREADS
        9,                  // LOG_BLOCKS
        32 * 128,           // LIGHT_EDGE_THRESHOLD
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        32,                 // WARP_GATHER_THRESHOLD
        128 * 4,            // CTA_GATHER_THRESHOLD
        7,                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    AdvanceKernelPolicy;

    /** 
     * @brief Reset enactor
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {   
        return BaseEnactor::Reset();
    } 

    /**
     * @brief Sm Enact initialization.
     *
     * @tparam SMProblem SM Problem type. @see SMProblem
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object which indicates the success of
     * all CUDA function calls.
     */
    cudaError_t Init(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            return InitSM<AdvanceKernelPolicy, FilterKernelPolicy> (
                context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture.\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief Sm Enact kernel entry.
     *
     * @tparam SMProblem SM Problem type. @see SMProblem
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object which indicates the success of
     * all CUDA function calls.
     */
    cudaError_t Enact()
        //ContextPtr  context,
        //Problem* problem,
        //int         max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            return EnactSM<AdvanceKernelPolicy, FilterKernelPolicy> ();
                //context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture.\n");
        return cudaErrorInvalidDeviceFunction;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /** @} */

};

} // namespace sm
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
