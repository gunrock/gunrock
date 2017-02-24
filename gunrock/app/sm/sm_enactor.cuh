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


//using namespace gunrock::app;
using namespace mgpu;
using namespace cub;

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
    cudaError_t EnactSM()
    {
        // Define functors for primitive
        // Fill d_c_set for each data node
        typedef SMInitFunctor   <VertexId, SizeT, Value, Problem> SMInitFunctor;
        // Select candidate edges
        typedef SMFunctor       <VertexId, SizeT, Value, Problem> SMFunctor;
        
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
            printf(":: initial read in row_offsets ::");
            util::DisplayDeviceResults(
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->nodes + 1);
        }
        
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = nodes;
        attributes->queue_reset  = true;

//     for(int i=0; i<2; i++)  // filter candidate nodes and edges for a few iterations
//     {

        gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, SMInitFunctor, gunrock::oprtr::advance::V2V>(
                    statistics[0],
                    attributes[0],
                    //statistics -> iteration + 1,
                    typename SMInitFunctor::LabelT(),
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


        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = false;

        gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, SMFunctor, gunrock::oprtr::advance::V2V>(
                    statistics[0],
                    attributes[0],
                    statistics -> iteration + 1,
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
                    stream);

        if (debug_info)
        {
           if (retval = util::GRError(cudaStreamSynchronize(stream),
                        "SMFunctor Advance::LaunchKernel failed", __FILE__, __LINE__)) 
                return retval;
        }
        util::GreaterThan select_op(0);

	util::CUBSelect_if(
       	      queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
              graph_slice->column_indices.GetPointer(util::DEVICE), 
	      data_slice ->d_query_row.GetPointer(util::DEVICE), // stores the number of selected edges
              graph_slice->edges,
              select_op);

	util::CUBSegReduce_sum(
        	data_slice->d_in                   .GetPointer(util::DEVICE),
        	queue->values[attributes->selector].GetPointer(util::DEVICE), // output
        	graph_slice->row_offsets	   .GetPointer(util::DEVICE),
        	graph_slice->nodes);

        Scan<mgpu::MgpuScanTypeExc>(
        queue->values[attributes->selector].GetPointer(util::DEVICE),
        graph_slice->nodes+1,
        (int)0,
        mgpu::plus<int>(),
        (int*)0,
        (int*)0,
	graph_slice->row_offsets.GetPointer(util::DEVICE), // Output new row_offsets
        context[0]);

        //util::debugScan<<<128,128, 0, stream>>>(graph_slice->row_offsets.GetPointer(util::DEVICE), graph_slice->nodes+1);
      //}	
	   
	// Convert graph_slice -> row_offsets to edges_list source node list
	// graph_slice -> column_indices is the dest node list itself
        IntervalExpand(
            edges/2, // number of outputs
	    graph_slice->row_offsets.GetPointer(util::DEVICE), // expand counts
            queue->keys[attributes->selector].GetPointer(util::DEVICE), // expand values
            nodes,	// number of inputs
	    queue->values[attributes->selector].GetPointer(util::DEVICE), // output src node list 
            context[0]); 

            // froms stores the flags of candidate edges
	    util::MemsetKernel<<<128,128, 0, stream>>>(
	    	data_slice -> d_in.GetPointer(util::DEVICE),
	    	(Value)0, data_slice -> edges_query * edges/2);
 //           util::debugLabel<<<128,128,0,stream>>>(data_slice -> d_in.GetPointer(util::DEVICE), data_slice -> edges_query * edges/2);
	    // Label candidate edges for each query edge
	    util::Label<<<128,128,0,stream>>>(
	        queue->values[attributes->selector].GetPointer(util::DEVICE), //src node list
	    	data_slice -> d_col_indices .GetPointer(util::DEVICE), // dest node list
	    	data_slice -> froms_query   .GetPointer(util::DEVICE),
	    	data_slice -> tos_query     .GetPointer(util::DEVICE),
	    	data_slice -> d_c_set       .GetPointer(util::DEVICE),
	    	data_slice -> d_in          .GetPointer(util::DEVICE),//label results
		data_slice -> d_query_row   .GetPointer(util::DEVICE), 
	    	data_slice -> edges_data,
	    	data_slice -> edges_query);
            
//           util::debugLabel<<<128,128,0,stream>>>(data_slice -> d_in.GetPointer(util::DEVICE), data_slice -> edges_query * edges/2);

            util::MemsetKernel<<<128, 128, 0, stream>>>(
            	data_slice -> d_query_col.GetPointer(util::DEVICE), 
            	(Value)0, data_slice -> edges_query);


	    util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
		data_slice -> d_c_set.GetPointer(util::DEVICE), data_slice-> edges_query * edges/2);

	    util::CUBSelect_flagged(
		data_slice -> d_c_set       .GetPointer(util::DEVICE),
                data_slice -> d_in          .GetPointer(util::DEVICE),
                data_slice -> d_in	    .GetPointer(util::DEVICE), //store middle results
                data_slice -> d_query_col   .GetPointer(util::DEVICE), // d_query_col[0]
		data_slice -> edges_query * data_slice -> edges_data/2);

	    util::Update<<<128, 128, 0, stream>>>(
                data_slice -> d_in 	    .GetPointer(util::DEVICE), //store middle results
                data_slice -> d_query_col   .GetPointer(util::DEVICE), // d_query_col[0]
		data_slice -> edges_data/2);

	    util::MemsetKernel<<<128,128,0,stream>>>(
	        data_slice -> d_query_row.GetPointer(util::DEVICE), (SizeT)0, 
		data_slice -> nodes_query + 1);

            util::MemsetKernel<<<128,128,0,stream>>>(
                data_slice -> d_c_set.GetPointer(util::DEVICE), 
                (VertexId)0, 
	        data_slice -> edges_data * data_slice -> edges_query /2);

         //   util::DisplayDeviceResults(
         //       data_slice->d_in.GetPointer(util::DEVICE),
	//	data_slice -> edges_query * data_slice -> edges_data/2);

  	    util::Join<<<128, 128, 0, stream>>>(
		    data_slice -> edges_data,
		    data_slice -> edges_query,
		    data_slice -> d_query_col  	       .GetPointer(util::DEVICE),
                    data_slice -> counts               .GetPointer(util::DEVICE),
                    data_slice -> flag                 .GetPointer(util::DEVICE),
	    	    queue->values[attributes->selector].GetPointer(util::DEVICE), //can edge list 
                    data_slice -> d_col_indices        .GetPointer(util::DEVICE),
		    data_slice -> d_in		       .GetPointer(util::DEVICE), // can edge ids
                    data_slice -> d_c_set 	       .GetPointer(util::DEVICE));// output

            util::debug<<<128,128,0,stream>>>(data_slice->counts.GetPointer(util::DEVICE));

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
