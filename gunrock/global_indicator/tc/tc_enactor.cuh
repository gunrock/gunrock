// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * tc_enactor.cuh
 *
 * @brief Problem enactor for Triangle Counting
 */

#pragma once

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/intersection/kernel.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/global_indicator/tc/tc_problem.cuh>
#include <gunrock/global_indicator/tc/tc_functor.cuh>

#include <moderngpu.cuh>
#include <cub/cub.cuh>

#include <fstream>


using namespace gunrock::app;

namespace gunrock {
namespace global_indicator {
namespace tc {

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

using namespace gunrock::app;
using namespace mgpu;
using namespace cub;

/**
 * @brief TC enactor class.
 *
 * @tparam _Problem
 * @tparam _INSTRUMENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
    typename _Problem>
    //bool _INSTRUMENT,
    //bool _DEBUG,
    //bool _SIZE_CHECK >
class TCEnactor :
    public EnactorBase<typename _Problem::SizeT>
{
public:
    typedef _Problem                    Problem;
    typedef typename Problem::SizeT     SizeT;
    typedef typename Problem::VertexId  VertexId;
    typedef typename Problem::Value     Value;
    typedef EnactorBase<SizeT>          BaseEnactor;
    Problem                            *problem;
    ContextPtr                         *context;
    //static const bool INSTRUMENT   =   _INSTRUMENT;
    //static const bool DEBUG        =        _DEBUG;
    //static const bool SIZE_CHECK   =   _SIZE_CHECK;

    /**
     * @brief TCEnactor constructor.
     */
    TCEnactor(
        int num_gpus = 1,
        int *gpu_idx = NULL,
        bool instrument = false,
        bool debug = false,
        bool size_check = true) :
        BaseEnactor(
            EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem(NULL),
        context(NULL)
    {
    }

    /**
     * @brief TCEnactor destructor
     */
    virtual ~TCEnactor()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Release()) return retval;
        return retval;
    }


    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitTC(
        ContextPtr  *context,
        Problem     *problem,
        int          max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this -> problem = problem;
        this -> context = context;
        return retval;
    }

    /**
     * @brief Enacts a TC computing on the specified graph.
     *
     * @tparam Advance Kernel policy for forward advance kernel.
     * @tparam Filter Kernel policy for filter kernel.
     * @tparam Intersection Kernel policy for intersection kernel.
     * @tparam TCProblem TC Problem type.
     *
     * @param[in] context CudaContext for moderngpu library
     * @param[in] problem TCProblem object.
     * @param[in] max_grid_size Max grid size for TC kernel calls.
     *
     * \return cudaError_t object which indicates the success of
     * all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename IntersectionKernelPolicy>
        //typename Problem>
    cudaError_t EnactTC()
        //ContextPtr  context,
        //TCProblem* problem,
        //int         max_grid_size = 0)
    {
        //typedef typename TCProblem::VertexId VertexId;
        //typedef typename TCProblem::SizeT    SizeT;
        //typedef typename TCProblem::Value    Value;

        typedef TCFunctor <VertexId, SizeT, Value, Problem> TCFunctor;

        typedef typename Problem::DataSlice               DataSlice;
        typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
        typedef GraphSlice  <VertexId, SizeT, Value>      GraphSliceT;
        typedef typename TCFunctor::LabelT      LabelT;

        Problem                  *problem       =  this -> problem;
        FrontierAttribute<SizeT> *attributes    = &this -> frontier_attribute[0];
        EnactorStats<SizeT>      *statistics    = &this -> enactor_stats[0];
        GraphSliceT              *graph_slice   = problem -> graph_slices[0];
        DataSlice                *d_data_slice  = problem -> d_data_slices[0];
        DataSlice                *data_slice    = problem -> data_slices[0];
        Frontier                 *queue         = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime<SizeT>
                                 *work_progress = &this -> work_progress[0];
        cudaStream_t              stream        = data_slice->streams[0];
        ContextPtr                context       =  this -> context[0];
        cudaError_t               retval        = cudaSuccess;
        SizeT                    *d_scanned_edges = NULL;  // Used for LB

        // initialization
        //if (retval = EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>::Setup(
        //    problem,
        //    max_grid_size,
        //    AdvanceKernelPolicy::CTA_OCCUPANCY,
        //    FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        if (retval = util::GRError(cudaMalloc(
            (void**)&d_scanned_edges, graph_slice->edges * sizeof(SizeT)),
            "Problem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__))
        {
            return retval;
        }

        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true; 

        // TODO: Add TC algorithm here.
      
        // Prepare src node_ids for edge list
        // TODO: move this to problem. need to send CudaContext to problem too. 

        // 1) Do advance/filter to get rid of neighbors whose nid < sid
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, TCFunctor, gunrock::oprtr::advance::V2V>(
            statistics[0],
            attributes[0],
            util::InvalidValue<LabelT>(),
            data_slice,
            d_data_slice,
            (VertexId*)NULL,
            (bool*)NULL,
            (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*)NULL,
            (VertexId*)NULL,
            graph_slice->nodes,
            graph_slice->edges,
            work_progress[0],
            context[0],
            stream);

        GreaterThan select_op(0);

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), 
            graph_slice->column_indices.GetPointer(util::DEVICE)+graph_slice->edges/2, 
            graph_slice->edges,
            select_op);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), 
            graph_slice->column_indices.GetPointer(util::DEVICE)+graph_slice->edges/2, 
            graph_slice->edges,
            select_op);

        //util::MemsetKernel<<<256, 1024>>>(data_slice->d_edge_list.GetPointer(util::DEVICE), 0, graph_slice->nodes+1);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);

        Scan<mgpu::MgpuScanTypeExc>(
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes+1,
            (int)0,
            mgpu::plus<int>(),
            (int*)0,
            (int*)0,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            context[0]);

        //util::DisplayDeviceResults(graph_slice->row_offsets.GetPointer(util::DEVICE), graph_slice->nodes);

        //util::DisplayDeviceResults(graph_slice->column_indices.GetPointer(util::DEVICE), graph_slice->edges/2);

        IntervalExpand(
            graph_slice->edges/2,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            graph_slice->nodes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            context[0]);

        //util::DisplayDeviceResults(data_slice->d_src_node_ids.GetPointer(util::DEVICE)+graph_slice->edges/2-10, 10);
        //util::DisplayDeviceResults(graph_slice->column_indices.GetPointer(util::DEVICE)+graph_slice->edges/2-10,10);

        /*
        data_slice->d_edge_list.Move(util::DEVICE, util::HOST);
        VertexId *edge_ref = data_slice->d_edge_list.GetPointer(util::HOST);
        std::ofstream e;
        e.open("new.txt");
        for (int i = 0; i < graph_slice->edges/2; ++i) {
            e << edge_ref[i] << std::endl;
        }
        e.close();
        */
      
        /*
        queue->keys[attributes->selector^1].Move(util::DEVICE, util::HOST);
        VertexId *edge = queue->keys[attributes->selector^1].GetPointer(util::HOST);
        data_slice->d_edge_list.Move(util::DEVICE, util::HOST);
        VertexId *edge_ref = data_slice->d_edge_list.GetPointer(util::HOST);
        printf("prepared data.\n");
        std::ofstream e;
        e.open("edges.txt");
        for (int i = 0; i < graph_slice->edges; ++i) {
            e << edge[i] << std::endl;
        }
        e.close();
        e.open("edges_ref.txt");
        for (int i = 0; i < graph_slice->edges; ++i) {
            e << edge_ref[i] << std::endl;
        }
        e.close();
        */

        /*
        data_slice->d_src_node_ids.Move(util::DEVICE,util::HOST);
        VertexId *data = data_slice->d_src_node_ids.GetPointer(util::HOST);
        graph_slice->row_offsets.Move(util::DEVICE, util::HOST);
        SizeT *offsets = graph_slice->row_offsets.GetPointer(util::HOST);
        std::ofstream ref_file;
        std::ofstream offset_file;

        ref_file.open("offsets_ref.txt");

        offset_file.open("offsets.txt");
        for (int i = 0; i <= graph_slice->nodes; ++i)
        {
            ref_file << offsets[i] << std::endl;
        }
        ref_file.close();
        ref_file.open("ref.txt");
        int cur = 0;
        int ref = 0;
        int idx = 0;
        for (int i =0 ; i < graph_slice->edges; ++i) {
            if (i == offsets[idx]) {
                offset_file << offsets[idx] << std::endl;
                ++idx;
                while (i == offsets[idx]) {
                offset_file << offsets[idx] << std::endl;
                ref_file << 0 << std::endl;
                ++idx;
                }
                if (i!=0) {
                    ref_file << cur-ref << std::endl;
                    ref = cur;
                }
            }
            cur += data[i];
        } 
        ref_file << cur-ref << std::endl;
        ref_file.close();
        offset_file.close();
        printf("ref:%d\n", cur);
        */

        //SegReduce
        /*
        SegReduceCsr(
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->edges,
            graph_slice->nodes,
            false,
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            (int)0,
            mgpu::plus<int>(),
            context[0]);
        */

        /*
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1);

        data_slice->d_edge_list.Move(util::DEVICE, util::HOST);
        VertexId *seg = data_slice->d_edge_list.GetPointer(util::HOST);

        std::ofstream seg_file;
        seg_file.open("output.txt");
        int final = 0;
        for (int i =0 ; i < graph_slice->nodes; ++i) {
            seg_file << seg[i] << std::endl;
            final += seg[i];
        } 
        seg_file.close();

        //Scan
        Scan<mgpu::MgpuScanTypeExc>(
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes+1,
            (int)0,
            mgpu::plus<int>(),
            (int*)0,
            (int*)0,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            context[0]);

        IntervalExpand(
            graph_slice->edges/2,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            graph_slice->nodes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            context[0]);
        */

        //if (retval = work_progress->GetQueueLength(++attributes->queue_index, attributes->queue_length, false, stream, true)) return retval;
        //attributes->selector ^= 1;

        //printf("queue length:%d\n", attributes->queue_length);

        //Filter to get edge_list (done)
        //declare edge_list in problem (done)
        //modify intersection operator (done)
        //cubPartition the coarse_count is on device, need to change (done)
        /*
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, TCProblem, TCFunctor>
            <<<statistics->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            statistics->iteration,
            attributes->queue_reset,
            attributes->queue_index,
            attributes->queue_length,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            NULL,
            data_slice->d_edge_list_partitioned.GetPointer(util::DEVICE),
            d_data_slice,
            NULL,
            work_progress[0],
            graph_slice->edges,
            graph_slice->edges,
            statistics->filter_kernel_stats);
        */

        //graph_slice->edges /= 2;

        //GetQueueLength of the new edge_list
        //if (retval = work_progress->GetQueueLength(++attributes->queue_index, attributes->queue_length, false, stream, true)) return retval;

        //printf("queue length:%d\n", attributes->queue_length);

        /*
        data_slice->d_edge_list_partitioned.Move(util::DEVICE, util::HOST);
        VertexId *edge2 = data_slice->d_edge_list_partitioned.GetPointer(util::HOST);

        std::ofstream e;
        e.open("edges_ref.txt");
        for (int i = 0; i < graph_slice->edges; ++i) {
            e << edge2[i] << std::endl;
        }
        e.close();
        */

        /*
        VertexId *srcid = data_slice->d_src_node_ids.GetPointer(util::DEVICE);
        VertexId *dstid = data_slice->d_edge_list_partitioned.GetPointer(util::DEVICE);

        util::DisplayDeviceResults(dstid, 10);
        util::DisplayDeviceResults(dstid+graph_slice->edges-10, 10);
        */

        /*
        util::MemsetMadVectorKernel<<<256, 2014>>>(
            data_slice->d_degrees.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE)+1, -1, graph_slice->nodes);
        */

        // 2) Do intersection using generated edge lists from the previous step.
        //gunrock::oprtr::intersection::LaunchKernel
        //<IntersectionKernelPolicy, TCProblem, TCFunctor>(
        //);

        // Reuse d_scanned_edges
        //SizeT *d_output_counts = d_scanned_edges;
        //util::MemsetKernel<<<256, 1024>>>(d_output_triplets, (SizeT)0, graph_slice->edges);

        // Should make tc_count a member var to TCProblem
        long tc_count = gunrock::oprtr::intersection::LaunchKernel
            <IntersectionKernelPolicy, Problem, TCFunctor>(
            statistics[0],
            attributes[0],
            d_data_slice,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            data_slice->d_degrees.GetPointer(util::DEVICE),
            data_slice->d_edge_tc.GetPointer(util::DEVICE),
            d_output_triplets,
            graph_slice->edges/2,
            graph_slice->nodes,
            graph_slice->edges/2,
            work_progress[0],
            context[0],
            stream);

        //tc_count /= 3;

        printf("tc count:%ld\n", tc_count);

        // end of the TC

        if (d_scanned_edges) cudaFree(d_scanned_edges);
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,            // Problem data type
        300,                // CUDA_ARCH
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
        8,                  // MIN_CTA_OCCUPANCY
        10,                 // LOG_THREADS
        8,                  // LOG_BLOCKS
        32 * 128,           // LIGHT_EDGE_THRESHOLD
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        32,                 // WARP_GATHER_THRESHOLD
        128 * 4,            // CTA_GATHER_THRESHOLD
        7,                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
        AdvanceKernelPolicy; 

    cudaError_t Reset()
    {
        return BaseEnactor::Reset();
    }

    cudaError_t Init(
        ContextPtr *context,
        Problem    *problem,
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
            return InitTC<AdvanceKernelPolicy, FilterKernelPolicy>
                (context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architectures for now
        // TODO: add all the kernel policy settings for all architectures

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief MST Enact kernel entry.
     *
     * @tparam MSTProblem MST Problem type. @see MSTProblem
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object which indicates the success of
     * all CUDA function calls.
     */
    template <int NL_SIZE>
    cudaError_t Enact()
    { 
        typedef gunrock::oprtr::intersection::KernelPolicy<
            Problem,            // Problem data type
            300,                // CUDA_ARCH
            1,                  // MIN_CTA_OCCUPANCY
            10,                 // LOG_THREADS
            8,                  // LOG_BLOCKS
            NL_SIZE>                  // NL_SIZE_THRESHOLD
        IntersectionKernelPolicy;

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
    
            return EnactTC<AdvanceKernelPolicy, FilterKernelPolicy, IntersectionKernelPolicy>();
                //context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architectures for now
        // TODO: add all the kernel policy settings for all architectures

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /** @} */

};

} // namespace tc
} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
