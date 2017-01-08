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
#include <gunrock/util/join.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/app/sm/sm_functor.cuh>


namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief SM enactor class.
 *
 * @tparam _Problem
 * @tparam _INSTRUMWENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
    typename _Problem>
    //bool _INSTRUMENT,
    //bool _DEBUG,
    //bool _SIZE_CHECK >
class SMEnactor :
  public EnactorBase<typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/> 
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    //static const bool INSTRUMENT   =   _INSTRUMENT;
    //static const bool DEBUG        =        _DEBUG;
    //static const bool SIZE_CHECK   =   _SIZE_CHECK;
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
        bool  debug      = false,
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
        //typename SMProblem>
    cudaError_t EnactSM()
        //ContextPtr  context,
        //SMProblem*  problem,
        //int         max_grid_size = 0)
    {
        // Define functors for primitive
        typedef SMInitFunctor       <VertexId, SizeT, Value, Problem> SMInitFunctor;
        //typedef EdgeWeightFunctor<VertexId, SizeT, Value, SMProblem> EdgeWeightFunctor;
        typedef UpdateDegreeFunctor <VertexId, SizeT, Value, Problem> UpdateDegreeFunctor;
        typedef PruneFunctor        <VertexId, SizeT, Value, Problem> PruneFunctor;
        typedef LabelEdgeFunctor    <VertexId, SizeT, Value, Problem> LabelEdgeFunctor;
        typedef CollectFunctor      <VertexId, SizeT, Value, Problem> CollectFunctor;
        typedef typename Problem::DataSlice                  DataSlice;
        typedef util::DoubleBuffer  <VertexId, SizeT, Value> Frontier;
        typedef GraphSlice          <VertexId, SizeT, Value> GraphSliceT;

        Problem      *problem            = this -> problem;
        EnactorStats *statistics         = &this->enactor_stats     [0];
        DataSlice    *data_slice         =  problem -> data_slices  [0].GetPointer(util::HOST);
        DataSlice    *d_data_slice       =  problem -> data_slices  [0].GetPointer(util::DEVICE);
        GraphSliceT  *graph_slice        =  problem -> graph_slices [0];
        Frontier     *queue              = &data_slice->frontier_queues[0];
        FrontierAttribute<SizeT>
                     *attributes         = &this->frontier_attribute[0];
        util::CtaWorkProgressLifetime
                     *work_progress      = &this->work_progress     [0];
        cudaStream_t  stream             =  data_slice->streams     [0];
        ContextPtr    context            =  this -> context         [0];
        cudaError_t   retval             = cudaSuccess;
        SizeT        *d_scanned_edges    = NULL;  // Used for LB
        SizeT         nodes              = graph_slice -> nodes;
        SizeT         edges              = graph_slice -> edges;
        // debug configurations
        //SizeT         num_edges_origin = graph_slice->edges;
        bool          debug_info         = 1;   // used for debug purpose
        //int           tmp_select          = 0; // used for debug purpose
        //int           tmp_length          = 0; // used for debug purpose
        unsigned int   *num_selected     = new unsigned int; // used in cub select

        if (retval = data_slice -> scanned_edges[0].EnsureSize(edges))
            return retval;
        d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);

        if (debug_info)
        {
            printf("\nBEGIN ITERATION: %lld #NODES: %lld #EDGES: %lld\n",
                statistics->iteration+1,
                (long long)nodes,
                (long long)edges);
            printf(":: initial read in row_offsets ::");
            util::DisplayDeviceResults(
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->nodes + 1);
        }

        // Iterate SMInitFunctor and UpdateDegreeFunctor for two iterations
        for(int i=0; i<2; i++)
        {
        ///////////////////////////////////////////////////////////////////////////
            // Initial filtering based on node labels and degrees 
            // And generate candidate sets for query nodes
            attributes->queue_index  = 0;
            attributes->selector     = 0;
            attributes->queue_length = graph_slice->nodes;
            attributes->queue_reset  = true;

            util::MemsetKernel<<<128, 128, 0, stream>>>(
                data_slice -> temp_keys.GetPointer(util::DEVICE),
                (SizeT)0, nodes);

            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, SMInitFunctor>(
                statistics->filter_grid_size,
                FilterKernelPolicy::THREADS,
                (size_t)0, 
                stream,
                statistics->iteration + 1,
                attributes->queue_reset,
                attributes->queue_index,
                attributes->queue_length,
                queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
                (Value* )NULL,
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                d_data_slice,
                (unsigned char*)NULL,
                work_progress[0],
                queue->keys[attributes->selector  ].GetSize(),
                queue->keys[attributes->selector^1].GetSize(),
                statistics->filter_kernel_stats);

            if  (debug_info)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                    "Initial filtering filter::Kernel failed", __FILE__, __LINE__))
                    return retval;
            }

            if (i!=1)
            { //Last round doesn't need the following functor
                ///////////////////////////////////////////////////////////////////////////
                // Update each candidate node's valid degree by checking their neighbors
                attributes->queue_index  ++;
                attributes->selector    ^= 1;
                attributes->queue_length = nodes;
                attributes->queue_reset  = false;

                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, UpdateDegreeFunctor>(
                    statistics[0],
                    attributes[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    d_scanned_edges,  // In order to use the output vertices from previous filter functor
                    queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                    queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
                    gunrock::oprtr::advance::V2V);

                if (debug_info)
                {
                    if (retval = util::GRError(cudaStreamSynchronize(stream),
                        "Update Degree Functor Advance::LaunchKernel failed", __FILE__, __LINE__)) 
                        return retval;
                }
            }
        } // end of for i
            
	    /*mgpu::SegReduceCsr(data_slice->d_c_set, 
            data_slice->temp_keys, 
            data_slice->temp_keys, 
            data_slice->nodes_query * data_slice->nodes_data,
            data_slice->nodes_query,
            false,
            data_slice->temp_keys,
            (int)0,
            mgpu::plus<int>(),
            context);
        */

	    //TODO: Divide the results by hop number of query nodes
        /* util::MemsetDivideVectorKernel<<<128,128,0,stream>>>(
            data_slice -> temp_keys, 
            data_slice -> query_degrees,
            data_slice -> nodes_query);

        enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;
        enactor_stats -> iteration++;
        frontier_attribute->queue_reset = false;
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SMProblem, EdgeWeightFunctor>(
            d_done,
            enactor_stats,
            frontier_attribute,
            data_slice,
            (VertexId*)NULL,
            (bool*)NULL,
            (bool*)NULL,
            d_scanned_edges,
            graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
            graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
            (VertexId*)NULL,
            (VertexId*)NULL,
            graph_slice->d_row_offsets,
            graph_slice->d_column_indices,
            (SizeT*)NULL,
            (VertexId*)NULL,
            graph_slice->frontier_elements[frontier_attribute.selector],
            graph_slice->frontier_elements[frontier_attribute.selector^1],
            this->work_progress,
            context,
            gunrock::oprtr::advance::V2V);

	    //TODO: Potential bitonic sorter under util::sorter
        mgpu::LocalitySortPairs(
            data_slice->edge_weights,
            data_slice->edge_labels, 
            data_slice->edges_query, 
            context);
        */

	    // Run prune functor for several iterations
        for(int i=0; i<2; i++)
        {
            ///////////////////////////////////////////////////////////////////////////
            // Prune out candidates by checking candidate neighbors 
            attributes->queue_index  = 0;
            attributes->selector     = 0;
            attributes->queue_length = edges;
            attributes->queue_reset  = true;

	        gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, PruneFunctor>(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from previous filter functor
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
                gunrock::oprtr::advance::V2V);

	        if (debug_info)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                "Prune Functor Advance::LaunchKernel failed", __FILE__, __LINE__))
                return retval;
            }
	    }

	    util::debug_init<<<128,128, 0, stream>>>(
	        data_slice ->c_set.GetPointer(util::DEVICE),
	        data_slice ->nodes_query,
	        data_slice ->nodes_data);

        // Put the candidate index of each candidate edge into d_temp_keys at its e_id position
        // Fill the non-candidate edge positions with 0 in d_temp_keys
        //////////////////////////////////////////////////////////////////////////////////////
        util::MemsetKernel<<<128,128, 0, stream>>>(
            data_slice -> temp_keys.GetPointer(util::DEVICE),
            (SizeT)0, (nodes < edges) ? edges : nodes);

        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = edges;
        attributes->queue_reset  = true;

	    gunrock::oprtr::advance:: LaunchKernel
            <AdvanceKernelPolicy, Problem, LabelEdgeFunctor>(
            statistics[0],
            attributes[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
            gunrock::oprtr::advance::V2V);

        if (debug_info)
        {
            if (retval = util::GRError(cudaStreamSynchronize(stream),
                "Label Edge Functor Advance::LaunchKernel failed", __FILE__, __LINE__)) 
            return retval;
        }

        util::debug_label<<<128, 128, 0, stream>>>(
            data_slice -> temp_keys.GetPointer(util::DEVICE),
            data_slice -> edges_data);

        // Use d_query_col to store the number of candidate edges for each query edge
        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice -> query_col.GetPointer(util::DEVICE), 
            0, data_slice -> edges_query);

        // collect candidate edges for each query edge
        for(SizeT i=0; i< data_slice -> edges_query; i++)
        {
            // Use d_data_degrees as flags of edge labels, initialized to 0
            util::MemsetKernel<<<128, 128, 0, stream>>>(
                data_slice -> data_degrees.GetPointer(util::DEVICE), 
                (SizeT)0, edges);

            // Mark the candidate edges for each query edge and store in d_data_degrees
            util::Mark<<<128, 128, 0, stream>>>(
                data_slice -> edges_data,
                data_slice -> temp_keys.GetPointer(util::DEVICE),
                data_slice -> data_degrees.GetPointer(util::DEVICE));
        
            Scan<mgpu::MgpuScanTypeInc>(
                data_slice -> data_degrees.GetPointer(util::DEVICE),
                data_slice -> edges_data, 
                (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)NULL, (SizeT*)NULL,
                data_slice -> data_degrees.GetPointer(util::DEVICE), 
                context[0]);

            // update the number of candidate edges for each query edge in d_query_col
            util::Update<<<128, 128, 0, stream>>>(
                i,
                data_slice -> edges_query,
                data_slice -> edges_data,
                data_slice -> data_degrees.GetPointer(util::DEVICE),
                data_slice -> temp_keys.GetPointer(util::DEVICE),
                data_slice -> query_col.GetPointer(util::DEVICE));

            //////////////////////////////////////////////////////////////////////////////////////
            // Collect candidate edges for each query edge using CollectFunctor
            attributes -> queue_reset  = false;
            attributes -> queue_length = edges;
            // attributes -> queue_index++;
            // attributes -> selector ^= 1;

            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, CollectFunctor>(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
                gunrock::oprtr::advance::V2V);

	        if (debug_info)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                    "Collect Functor Advance::LaunchKernel failed", __FILE__, __LINE__))
                return retval;

	            util::debug<<<128,128, 0, stream>>>(
                    i,
                    data_slice -> froms_data    .GetPointer(util::DEVICE), 
                    data_slice -> tos_data      .GetPointer(util::DEVICE), 
                    data_slice -> froms_query   .GetPointer(util::DEVICE), 
                    data_slice -> tos_query     .GetPointer(util::DEVICE), 
                    data_slice -> query_col     .GetPointer(util::DEVICE), 
                    data_slice -> edges_query, 
                    data_slice -> edges_data);

                if (retval = util::GRError(cudaStreamSynchronize(stream),
                    "Debug failed", __FILE__, __LINE__)) 
                    return retval;

	        }
        }  //end of for loop

        //Joining step
	    // Use d_query_row[0] to store the number of matched subgraphs in each join step
	    util::MemsetKernel<<<128,128,0,stream>>>(
	        data_slice -> query_row.GetPointer(util::DEVICE), 
            (SizeT)0, 1);

	    for(SizeT i=0; i< data_slice -> edges_query-1; i++)
        {
            // Use d_data_degrees as flags of success join, initialized to 0
            util::MemsetKernel<<<128,128,0,stream>>>(
                data_slice -> data_degrees.GetPointer(util::DEVICE), 
                (SizeT)0, 3000);

            /////////////////////////////////////////////////////////
            // Kernel Join
            util::Join<<<128,128,0,stream>>>(
                data_slice -> edges_query,
                i,
                data_slice -> query_col     .GetPointer(util::DEVICE),
                data_slice -> query_row     .GetPointer(util::DEVICE),
                data_slice -> data_degrees  .GetPointer(util::DEVICE),
                data_slice -> flag          .GetPointer(util::DEVICE),
                data_slice -> froms_data    .GetPointer(util::DEVICE),
                data_slice -> tos_data      .GetPointer(util::DEVICE),
                data_slice -> froms         .GetPointer(util::DEVICE),
                data_slice -> tos           .GetPointer(util::DEVICE));

       	    if (debug_info)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                    "Join Kernel failed", __FILE__, __LINE__))
                    return retval;
            }

            /*util::debug_0<<<128,128,0,stream>>>(
	            data_slice -> data_degrees.GetPointer(util::DEVICE),
	            data_slice -> edges_data);
            */	
            Scan<mgpu::MgpuScanTypeInc>(
                data_slice -> data_degrees.GetPointer(util::DEVICE),
                3000, 
                (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)NULL, (SizeT*)NULL,
                data_slice -> data_degrees.GetPointer(util::DEVICE), 
                context[0]);

            /*util::MemsetKernel<<<128,128,0,stream>>>(
                data_slice ->c_set.GetPointer(util::DEVICE),0,
			    data_slice ->nodes_query * data_slice ->nodes_data);

	        util::CUBSelect_flagged<SizeT, SizeT>(
                data_slice -> c_set.GetPointer(util::DEVICE),
                1000,
                data_slice -> data_degrees.GetPointer(util::DEVICE), 
                (unsigned int*)data_slice -> query_row.GetPointer(util::DEVICE));

	        util::debug_select<<<128,128,0,stream>>>(
                data_slice -> c_set.GetPointer(util::DEVICE),
		        data_slice -> data_degrees.GetPointer(util::DEVICE));
            */

	        // Collect the valid joined edges to consecutive places	
 	        util::Collect<<<128,128,0,stream>>>(
                data_slice -> edges_query,
	            i,
                data_slice -> data_degrees  .GetPointer(util::DEVICE),
                data_slice -> froms_data    .GetPointer(util::DEVICE),
                data_slice -> tos_data      .GetPointer(util::DEVICE),
                data_slice -> froms         .GetPointer(util::DEVICE),
                data_slice -> tos           .GetPointer(util::DEVICE),
                data_slice -> query_col     .GetPointer(util::DEVICE),
                data_slice -> query_row     .GetPointer(util::DEVICE));
 
       	    if (debug_info)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream),
                    "Collect Kernel failed", __FILE__, __LINE__)) 
                break;
            }
        }

	    /*if(debug_info)
        {
	        util::debug_1<<<128,128,0,stream>>>(
	            data_slice -> froms         .GetPointer(util::DEVICE),
	            data_slice -> tos           .GetPointer(util::DEVICE),
	            data_slice -> data_degrees  .GetPointer(util::DEVICE),
	            data_slice -> query_col     .GetPointer(util::DEVICE),
	            data_slice -> query_row     .GetPointer(util::DEVICE),
	            data_slice -> edges_query);
        }*/

        return retval;
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
