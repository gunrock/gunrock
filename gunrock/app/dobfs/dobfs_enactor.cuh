// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dobfs_enactor.cuh
 *
 * @brief Direction Optimal BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>
#include <gunrock/app/dobfs/dobfs_functor.cuh>


namespace gunrock {
namespace app {
namespace dobfs {

/**
 * @brief DOBFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename _Problem/*, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK*/>
class DOBFSEnactor : public EnactorBase<typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    //static const bool INSTRUMENT = _INSTRUMENT;
    //static const bool DEBUG      = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;
    Problem *problem;
    ContextPtr *context;

    // Members

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for DOBFS kernel call. Must be called prior to each DOBFS search.
     *
     * @param[in] problem DOBFS Problem object which holds the graph data and DOBFS problem data to compute.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    /*cudaError_t Setup(
        Problem *problem)
    {

        return retval;
    }*/

    public:

    /**
     * @brief DOBFSEnactor constructor
     */
    DOBFSEnactor(
        int   num_gpus   = 1,  
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            VERTEX_FRONTIERS, num_gpus, gpu_idx, 
            instrument, debug, size_check),
        problem (NULL),
        context (NULL)
    {
    }

    /**
     * @brief DOBFSEnactor destructor
     */
    virtual ~DOBFSEnactor()
    {
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */
    /** 
     * @brief Reset enactor
     *  
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {
        return BaseEnactor::Reset();
    }

    template<
        typename AdvancekernelPolicy,
        typename FilterkernelPolicy>
    cudaError_t InitDOBFS(
        ContextPtr *context,
        Problem    *problem,
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) 
            return retval;

        this -> problem = problem;
        this -> context = context;

        //graph slice
        GraphSlice<VertexId, SizeT, Value>
            *graph_slice = problem->graph_slices[0];
        typename Problem::DataSlice  
            *data_slice  = problem->data_slices[0];

        if (Problem::ENABLE_IDEMPOTENCE) 
        {
            int bytes = (graph_slice->nodes + 8 - 1) / 8;
            cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();

            gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
            if (retval = util::GRError(cudaBindTexture(
                0,
                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,
                data_slice->d_visited_mask,
                bytes),
                "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__))
                return retval;
        }

        /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
        gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
        if (retval = util::GRError(cudaBindTexture(
            0,
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
            graph_slice->d_column_indices,
            graph_slice->edges * sizeof(VertexId)),
            "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) 
            break;*/
        return retval;
    }


    /**
     * @brief Enacts a direction optimal breadth-first search computing on the specified graph. (now only reverse bfs for testing purpose)
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam BackwardAdvanceKernelPolicy Kernel policy for backward advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam DOBFSProblem BFS Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem DOBFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename BackwardAdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BackwardFilterKernelPolicy>
    cudaError_t EnactDOBFS(
        //ContextPtr     context,
        //Problem       *problem,
        VertexId       src)
        //int            max_grid_size = 0)
    {
        // Functors for reverse BFS
        typedef PrepareUnvisitedQueueFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> UnvisitedQueueFunctor;

        typedef PrepareInputFrontierMapFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> InputFrontierFunctor;

        typedef ReverseBFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> RBFSFunctor;

        typedef SwitchToNormalFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> SwitchFunctor;

        // Functors for BFS
        typedef gunrock::app::bfs::BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> BfsFunctor;

        typedef typename Problem::DataSlice DataSlice;
        typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
        typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;

        Problem      *problem            = this -> problem;
        FrontierAttribute<SizeT>
                     *frontier_attribute = &this->frontier_attribute[0];
        EnactorStats *enactor_stats      = &this->enactor_stats     [0];
        DataSlice    *data_slice         =  problem->data_slices    [0];
        Frontier     *frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime
                     *work_progress      = &this->work_progress     [0];
        cudaStream_t  stream             =  data_slice->streams     [0];
        ContextPtr    context            =  this -> context         [0];
         // Single-gpu graph slice
        GraphSliceT *graph_slice         =  problem -> graph_slices [0];
        DataSlice   *d_data_slice        =  problem -> d_data_slices[0];
        SizeT        num_unvisited_nodes =  graph_slice -> nodes - 1;
        SizeT        current_frontier_size = 1;
        cudaError_t   retval             = cudaSuccess;
        SizeT       *d_scanned_edges     = NULL;

        do {
            // Determine grid size(s)
            if (this -> debug) 
            {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
                fflush(stdout);
            }

            if (retval = data_slice -> scanned_edges[0].EnsureSize(graph_slice->edges))
                return retval;
            d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);

            // Start of Normal BFS
            frontier_attribute->queue_length         = 1;
            frontier_attribute->queue_index          = 0;        // Work queue index
            frontier_attribute->selector             = 0;
            frontier_attribute->queue_reset          = true;

            // Step through BFS iterations
            while (frontier_attribute->queue_length > 0) 
            {
                enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, BfsFunctor>(
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    d_scanned_edges,
                    frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    (Value*   )NULL,
                    frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes, // max_in_queue
                    graph_slice->edges, // max_out_queue
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    false,
                    true);
                if (this -> debug && (retval = util::GRError(
                    cudaThreadSynchronize(), 
                   "Advance::LaunchKernel failed", __FILE__, __LINE__))) 
                    break;

                // Only need to reset queue for once
                frontier_attribute -> queue_reset = false;
                frontier_attribute -> queue_index++;
                frontier_attribute -> selector ^= 1;
                enactor_stats      -> AccumulateEdges(
                    work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(
                        frontier_attribute -> queue_index), stream);

                //if (this -> debug) 
                //{
                //    if (retval = work_progress -> GetQueueLength(
                //        frontier_attribute -> queue_index, 
                //        frontier_attribute -> queue_length)) 
                //        break;
                //    printf(", %lld", (long long) frontier_attribute->queue_length);
                //}
                //if (this -> instrument) 
                //{
                //    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                //        enactor_stats -> advance_grid_size,
                //        enactor_stats -> total_runtimes,
                //        enactor_stats -> total_lifetimes)) 
                //        break;
                //}

                // Check if done
                // if (done[0] == 0) break;
                //if (frontier_attribute->queue_length == 0) break;

                // Filter
                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, BfsFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    (size_t)0, 
                    stream,
                    enactor_stats->iteration + 1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                    frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_pred_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    data_slice->d_visited_mask,
                    work_progress[0],
                    frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),// max_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),// max_out_queue
                    enactor_stats->filter_kernel_stats);

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                if (retval = work_progress -> GetQueueLength(
                    frontier_attribute -> queue_index, 
                    frontier_attribute -> queue_length,
                    false, stream)) 
                    break;

                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                    break;
                if (this -> instrument || this -> debug) 
                {
                    //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
                    if (this -> debug) 
                        printf(", %lld", (long long) frontier_attribute->queue_length);
                    //if (this -> instrument) 
                    //{
                    //    if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                    //                enactor_stats->filter_grid_size,
                    //                enactor_stats->total_runtimes,
                    //                enactor_stats->total_lifetimes)) break;
                    //}
                }

                num_unvisited_nodes -= frontier_attribute->queue_length;
                current_frontier_size = frontier_attribute->queue_length;
                enactor_stats->iteration++;
                if (num_unvisited_nodes < current_frontier_size*problem->alpha) break;

                // Check if done
                if (frontier_attribute->queue_length == 0) break;

                if (this -> debug) 
                    printf("\n%lld", (long long) enactor_stats->iteration);
            }
            if (retval) break;
            if (this -> debug) 
                printf("iter: %lld\n, alpha %f\n", enactor_stats->iteration, problem->alpha);
            // End of Normal BFS

            // Reverse BFS
            if (frontier_attribute->queue_length != 0) 
            {
                if (this -> debug) { printf("in RBFS.\n"); }

                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[0], graph_slice->nodes);
                frontier_attribute->queue_length         = current_frontier_size;
                frontier_attribute->queue_index          = 0;        // Work queue index
                frontier_attribute->selector             = 0;
                frontier_attribute->queue_reset          = true;

                // Prepare unvisited queue
                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, InputFrontierFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    (size_t)0, 
                    stream,
                    (long long)-1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                    (Value*)NULL,
                    frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    (unsigned char*)NULL,
                    work_progress[0],
                    frontier_queue->keys  [frontier_attribute->selector  ].GetSize(), // max_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetSize(), // max_out_queue
                    enactor_stats->filter_kernel_stats);

                //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_prepare_input_frontier::Kernel failed", __FILE__, __LINE__))) break;

                frontier_attribute->queue_length            = graph_slice->nodes;
                frontier_attribute->queue_index             = 0;        // Work queue index
                frontier_attribute->selector                = 0;

                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, UnvisitedQueueFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    (size_t)0,
                    stream,
                    (long long)-1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    data_slice->d_index_queue, // d_in_queue
                    (Value*)NULL,
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    (unsigned char*)NULL,
                    work_progress[0],
                    frontier_queue->keys[frontier_attribute->selector  ].GetSize(), // max_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetSize(), // max_out_queue
                    enactor_stats->filter_kernel_stats);

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                if (retval = work_progress -> GetQueueLength(
                    frontier_attribute -> queue_index, 
                    frontier_attribute -> queue_length,
                    false, stream)) 
                    break;

                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                    break;

                // Now the unvisited queue is frontier_queues.d_keys[0], frontier_map_in and frontier_map_out are both ready too
                // Start Reverse BFS

                SizeT last_queue_length = 0;
                //while (done[0] < 0) {
                while (frontier_attribute->queue_length != 0) 
                {
                    if (last_queue_length == frontier_attribute->queue_length)
                    {
                        //done[0] = 0;
                        frontier_attribute->queue_length = 0;
                        break;
                    }
                    last_queue_length = frontier_attribute->queue_length;

                    enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                    // Edge Map
                    gunrock::oprtr::advance::LaunchKernel
                        <BackwardAdvanceKernelPolicy, Problem, RBFSFunctor>(
                        enactor_stats[0],
                        frontier_attribute[0],
                        d_data_slice,
                        data_slice->d_index_queue,
                        data_slice->d_frontier_map_in,
                        data_slice->d_frontier_map_out,
                        d_scanned_edges,
                        frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), // d_in_queue
                        (VertexId*)NULL,
                        (Value*   )NULL,
                        (Value*   )NULL,
                        (SizeT*   )NULL,
                        (VertexId*)NULL,
                        graph_slice->column_offsets.GetPointer(util::DEVICE),
                        graph_slice->row_indices   .GetPointer(util::DEVICE),
                        graph_slice->nodes, // max_in_queue
                        graph_slice->edges, // max_out_queue
                        work_progress[0],
                        context[0],
                        stream,
                        gunrock::oprtr::advance::V2V,
                        false,
                        false,
                        true);
                    //if (retval = work_progress -> GetQueueLength(
                    //    frontier_attribute -> queue_index, 
                    //    frontier_attribute -> queue_length,
                    //    false, stream)) 
                    //    break;

                    //if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    //    "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                    //    break;

                    //util::DisplayDeviceResults(problem->data_slices[0]->d_frontier_map_out, graph_slice->nodes);
                    //if (this -> debug) 
                    //{
                    //    if (retval = work_progress->GetQueueLength(
                    //        frontier_attribute->queue_index, 
                    //        frontier_attribute->queue_length)) 
                    //            break;
                    //    printf(", %lld", (long long) frontier_attribute->queue_length);
                    //}
                    //if (this -> instrument) 
                    //{
                    //    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                    //        enactor_stats->advance_grid_size,
                    //        enactor_stats->total_runtimes,
                    //        enactor_stats->total_lifetimes)) break;
                    //}

                    // Check if done
                    //if (frontier_attribute->queue_length == 0) break;
                    
                    //enactor_stats -> edges_queued[0] += frontier_attribute -> queue_length;
                    enactor_stats      -> AccumulateEdges(
                        work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(
                        frontier_attribute -> queue_index), stream);
                    // Vertex Map
                    gunrock::oprtr::filter::LaunchKernel
                        <FilterKernelPolicy, Problem, RBFSFunctor>(
                        enactor_stats->filter_grid_size, 
                        FilterKernelPolicy::THREADS,
                        (size_t)0,
                        stream,
                        (long long)-1,
                        frontier_attribute->queue_reset,
                        frontier_attribute->queue_index,
                        frontier_attribute->queue_length,
                        frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                        (Value*)NULL,
                        frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        d_data_slice,
                        (unsigned char*)NULL,
                        work_progress[0],
                        frontier_queue->keys[frontier_attribute->selector  ].GetSize(),// max_in_queue
                        frontier_queue->keys[frontier_attribute->selector^1].GetSize(),// max_out_queue
                        enactor_stats->filter_kernel_stats);

                    frontier_attribute->queue_index++;
                    frontier_attribute->selector ^= 1;
                    enactor_stats->iteration++;

                    if (retval = work_progress -> GetQueueLength(
                        frontier_attribute -> queue_index, 
                        frontier_attribute -> queue_length,
                        false, stream)) 
                        break;

                    if (retval = util::GRError(cudaStreamSynchronize(stream), 
                        "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                        break;

                    //util::DisplayDeviceResults(
                    //    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                    //    frontier_attribute.queue_length);
                    if (this -> instrument || this -> debug) 
                    {
                        //enactor_stats->total_queued[0] += frontier_attribute->queue_length;
                        if (this -> debug) 
                            printf(", %lld", (long long) frontier_attribute->queue_length);
                        //if (this -> instrument) 
                        //{
                        //    if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                        //        enactor_stats->filter_grid_size,
                        //        enactor_stats->total_runtimes,
                        //        enactor_stats->total_lifetimes)) break;
                        //}
                    }
                    if (frontier_attribute->queue_length < graph_slice->nodes/problem->beta) break;

                    // Check if done
                    if (frontier_attribute->queue_length == 0) break;

                    if (this -> debug) 
                        printf("\n%lld", (long long) enactor_stats->iteration);
                }
                if (retval) break;
            } // End of Reverse BFS

            if (this -> debug) 
                printf("iter: %lld\n, beta %f\n", enactor_stats->iteration, problem->beta);

            // Normal BFS
            if (frontier_attribute -> queue_length != 0) 
            {
                if (this -> debug) printf("back to normal BFS.\n");

                //If selector == 1, copy map_in to map_out
                if (frontier_attribute->selector == 1) 
                {
                    if (retval = util::GRError(cudaMemcpyAsync(
                        data_slice->d_frontier_map_out,
                        data_slice->d_frontier_map_in,
                        graph_slice->nodes*sizeof(bool),
                        cudaMemcpyDeviceToDevice, stream),
                        "DOBFS cudaMemcpy frontier_map_in to frontier_map_out failed", 
                        __FILE__, __LINE__)) 
                        break;
                }

                frontier_attribute->queue_length         = graph_slice->nodes;
                //frontier_attribute->queue_length         = 1;
                frontier_attribute->queue_index          = 0;        // Work queue index
                frontier_attribute->selector             = 0;
                frontier_attribute->queue_reset          = true;

                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, SwitchFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    (size_t)0,
                    stream,
                    (long long)-1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    data_slice->d_index_queue,             // d_in_queue
                    (Value*)NULL,
                    frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    (unsigned char*)NULL,
                    work_progress[0],
                    frontier_queue->keys[frontier_attribute->selector  ].GetSize(), // max_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetSize(), // max_out_queue
                    enactor_stats->filter_kernel_stats);

                frontier_attribute->queue_index++;
                if (retval = work_progress -> GetQueueLength(
                    frontier_attribute -> queue_index, 
                    frontier_attribute -> queue_length,
                    false, stream)) 
                    break;
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                    break;

                // Step through BFS iterations
                frontier_attribute->queue_index = 0;

                while (frontier_attribute -> queue_length !=0)
                {
                    enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                    // Edge Map
                    gunrock::oprtr::advance::LaunchKernel
                        <AdvanceKernelPolicy, Problem, BfsFunctor>(
                        enactor_stats[0],
                        frontier_attribute[0],
                        d_data_slice,
                        (VertexId*)NULL,
                        (bool*    )NULL,
                        (bool*    )NULL,
                        d_scanned_edges,
                        frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        (Value*   )NULL,
                        frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        graph_slice->row_offsets   .GetPointer(util::DEVICE),
                        graph_slice->column_indices.GetPointer(util::DEVICE),
                        (SizeT*   )NULL,
                        (VertexId*)NULL,
                        graph_slice->nodes, // max_in_queue
                        graph_slice->edges, // max_out_queue
                        work_progress[0],
                        context[0],
                        stream,
                        gunrock::oprtr::advance::V2V,
                        false,
                        false,
                        true);

                    // Only need to reset queue for once
                    frontier_attribute->queue_reset = false;
                    frontier_attribute->queue_index++;
                    frontier_attribute->selector ^= 1;
                    //if (retval = work_progress -> GetQueueLength(
                    //    frontier_attribute -> queue_index, 
                    //    frontier_attribute -> queue_length,
                    //    false, stream)) 
                    //    break;
                    //if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    //    "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                    //    break;

                    //if (this -> debug) 
                    //{
                        //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    //    printf(", %lld", (long long) frontier_attribute->queue_length);
                    //}

                    //if (this -> instrument)
                    //{
                    //    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                    //        enactor_stats->advance_grid_size,
                    //        enactor_stats->total_runtimes,
                    //        enactor_stats->total_lifetimes)) break;
                    //}

                    // Check if done
                    //if (frontier_attribute -> queue_length == 0) break;
                    //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
                    enactor_stats      -> AccumulateEdges(
                        work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(
                            frontier_attribute -> queue_index), stream);

                    // Vertex Map
                    gunrock::oprtr::filter::LaunchKernel
                        <FilterKernelPolicy, Problem, BfsFunctor>(
                        enactor_stats->filter_grid_size, 
                        FilterKernelPolicy::THREADS,
                        (size_t)0, 
                        stream,
                        enactor_stats->iteration + 1,
                        frontier_attribute->queue_reset,
                        frontier_attribute->queue_index,
                        frontier_attribute->queue_length,
                        frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                        frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_pred_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        d_data_slice,
                        data_slice->d_visited_mask,
                        work_progress[0],
                        frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),// max_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),// max_out_queue
                        enactor_stats->filter_kernel_stats);

                    frontier_attribute->queue_index++;
                    frontier_attribute->selector ^= 1;
                    enactor_stats->iteration++;
                    if (retval = work_progress -> GetQueueLength(
                        frontier_attribute -> queue_index, 
                        frontier_attribute -> queue_length,
                        false, stream)) 
                        break;

                    if (retval = util::GRError(cudaStreamSynchronize(stream), 
                        "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                        break;

                    if (this -> instrument || this -> debug) 
                    {
                        //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
                        if (this -> debug) 
                            printf(", %lld", (long long) frontier_attribute->queue_length);
                        if (this -> instrument) 
                        {
                            if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                                enactor_stats->filter_grid_size,
                                enactor_stats->total_runtimes,
                                enactor_stats->total_lifetimes)) break;
                        }
                    }

                    // Check if done
                    //if (done[0] == 0) break;
                    if (frontier_attribute -> queue_length == 0) break;

                    if (this -> debug) 
                        printf("\n%lld", (long long) enactor_stats->iteration);

                }
                if (retval) break;
            } // end of normal BFS

        } while(0);

        if (this -> debug) 
            printf("\nGPU BFS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */
    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
            FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                  // LOG_THREADS
        8,
        32*128,
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
            AdvanceKernelPolicy;

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
            BackwardFilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_BACKWARD>
            BackwardAdvanceKernelPolicy;

    /**
     * @brief Direction Optimal BFS Enact kernel entry.
     *
     * @tparam DOBFSProblem DOBFS Problem type. @see DOBFSProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to DOBFSProblem object.
     * @param[in] src Source node for DOBFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        ContextPtr  *context,
        Problem     *problem,
        int          max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        {
            if (Problem::ENABLE_IDEMPOTENCE) 
            {
                return InitDOBFS
                    <AdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, max_grid_size);
            } else {
                return InitDOBFS
                    <AdvanceKernelPolicy, BackwardFilterKernelPolicy>(
                    context, problem, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief Direction Optimal BFS Enact kernel entry.
     *
     * @tparam DOBFSProblem DOBFS Problem type. @see DOBFSProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to DOBFSProblem object.
     * @param[in] src Source node for DOBFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        VertexId     src)
    {
        int min_sm_version = -1;
        for (int i=0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        {
            if (Problem::ENABLE_IDEMPOTENCE) 
            {
                return EnactDOBFS
                    <AdvanceKernelPolicy, BackwardAdvanceKernelPolicy, 
                     FilterKernelPolicy, BackwardFilterKernelPolicy> 
                    (src);
            } else {
                return EnactDOBFS
                    <AdvanceKernelPolicy, BackwardAdvanceKernelPolicy, 
                     BackwardFilterKernelPolicy, BackwardFilterKernelPolicy>
                    (src);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace dobfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
