// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sm_enactor.cuh
 * @brief Primitive problem enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

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
 * @brief SM Primitive enactor class.
 * @tparam INSTRUMWENT Boolean indicate collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class SMEnactor : public EnactorBase {
 protected:
    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int *done;
    int          *d_done;
    cudaEvent_t  throttle_event;

    /**
     * @brief Prepare the enactor for kernel call.
     * @param[in] problem Problem object holds both graph and primitive data.
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename ProblemData>
    cudaError_t Setup(ProblemData *problem) {
        typedef typename ProblemData::SizeT    SizeT;
        typedef typename ProblemData::VertexId VertexId;

        cudaError_t retval = cudaSuccess;

        // initialize the host-mapped "done"
        if (!done) {
            int flags = cudaHostAllocMapped;

            // allocate pinned memory for done
            if (retval = util::GRError(
                    cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "SMEnactor cudaHostAlloc done failed",
                    __FILE__, __LINE__)) return retval;

            // map done into GPU space
            if (retval = util::GRError(
                    cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "SMEnactor cudaHostGetDevicePointer done failed",
                    __FILE__, __LINE__)) return retval;

            // create throttle event
            if (retval = util::GRError(
                    cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "SMEnactor cudaEventCreateWithFlags throttle_event failed",
                    __FILE__, __LINE__)) return retval;
        }

        done[0] = -1;

        // graph slice
        typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

        do {
            // bind row-offsets and bit-mask texture
            cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(
                    cudaBindTexture(
                        0,
                        oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                        graph_slice->d_row_offsets,
                        (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "SMEnactor cudaBindTexture row_offset_tex_ref failed",
                    __FILE__, __LINE__)) break;
        } while (0);
        return retval;
    }

 public:
    /**
     * @brief SMEnactor constructor
     */
    explicit SMEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG), done(NULL), d_done(NULL) {}

    /**
     * @brief SMEnactor destructor
     */
    virtual ~SMEnactor() {
        if (done) {
            util::GRError(
                cudaFreeHost((void*)done),
                "Enactor cudaFreeHost done failed",
                __FILE__, __LINE__);

            util::GRError(
                cudaEventDestroy(throttle_event),
                "Enactor cudaEventDestroy throttle_event failed",
                __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics the primitive enacted.
     * @param[out] num_iterations Number of iterations (BSP super-steps).
     */
    template <typename VertexId>
    void GetStatistics(VertexId &num_iterations) {
        cudaThreadSynchronize();
        num_iterations = enactor_stats.iteration;
        // TODO: code to extract more statistics if necessary
    }

    /** @} */

    /**
     * @brief Enacts computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam Problem Problem type.
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs
     * @param[in] problem Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SMProblem >
    cudaError_t EnactSM(
        CudaContext &context,
        SMProblem   *problem,
        int         max_grid_size = 0) {
        typedef typename SMProblem::SizeT  SizeT;
	typedef typename SMProblem::Value  Value;
        typedef typename SMProblem::VertexId VertexId;

        typedef SMInitFunctor<VertexId, SizeT, Value, SMProblem> SMInitFunctor;
      //  typedef EdgeWeightFunctor<VertexId, SizeT, Value, SMProblem> EdgeWeightFunctor;
        typedef PruneFunctor<VertexId, SizeT, Value, SMProblem> PruneFunctor;

        cudaError_t retval = cudaSuccess;

	unsigned int *d_scanned_edges = NULL;

        do {
	    //Initialization
	    if (retval = Setup(problem)) break;
	    
	    if (retval = EnactorBase::Setup(max_grid_size,
					    AdvanceKernelPolicy::CTA_OCCUPANCY,
					    FilterKernelPolicy::CTA_OCCUPANCY,
					    AdvanceKernelPolicy::LOAD_BALANCED::BLOCKS)) break;

	    // Single-gpu graph slice
	    typename SMProblem::GraphSlice *graph_slice = problem->graph_slices[0];
	    typename SMProblem::DataSlice *data_slice = problem->d_data_slices[0];

	    if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB)
      	    {
        	if (retval = util::GRError(cudaMalloc((void**)&d_scanned_edges,
         	 graph_slice->edges * sizeof(unsigned int)),
          	"SMProblem cudaMalloc d_scanned_edges failed",
         	 __FILE__, __LINE__)) return retval;
     	    }

	    frontier_attribute.queue_length = data_slice->nodes_data;
	    frontier_attribute.queue_index = 0; // work queue index
	    frontier_attribute.selector = 0;


	    frontier_attribute.queue_reset = false;

	    // Initial filtering based on node labels and degrees
	    gunrock::oprtr::filter::Kernel<FilterKernelPolicy, SMProblem, SMInitFunctor>
		<<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
			0,
			frontier_attribute.queue_reset,
			frontier_attribute.queue_index,
			enactor_stats.num_gpus,
			frontier_attribute.queue_length,
			d_done,
			graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
			NULL,
			NULL,
			data_slice,
			NULL,
			work_progress,
			graph_slice->frontier_elements[frontier_attribute.selector],
			graph_slice->frontier_elements[frontier_attribute.selector^1],
			enactor_stats.filter_kernel_stats,
			false);

	    if(DEBUG && (retval = util::GRError(cudaThreadSynchronize(), 
		"Initial Filtering failed", __FILE__, __LINE__))) break;
	    enactor_stats.iteration++;
	    frontier_attribute.queue_index++;   	   
	    
	    mgpu::SegReduceCsr(data_slice->d_c_set, 
			       data_slice->d_temp_keys, 
			       data_slice->nodes_query * data_slice->nodes_data,
			       data_slice->nodes_query,
			       false,
			       data_slice->d_temp_keys,
			       (int)0,
			       mgpu::plus<int>(),
			       context);

	    //TODO: Potentially divide the results by hop number of query nodes
	    util::MemsetDivideVectorKernel<<<128,128>>>(data_slice->d_temp_keys, 
							data_slice->d_query_degrees,
							data_slice->nodes_query);
/*
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
	    mgpu::LocalitySortPairs(data_slice->d_edge_weights,
				    data_slice->d_edge_labels, data_slice->edges_query, context);

*/
	    mgpu::LocalitySortPairs(data_slice->d_temp_keys,
	   			    data_slice->d_labels,
				    data_slice->nodes_query,
				    context);

		gunrock::oprtr::advance::LaunchKernel
		  <AdvanceKernelPolicy, SMProblem, PruneFunctor>(
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

	   if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          	"advance::Kernel failed", __FILE__, __LINE__))) break;

	     
        } while (0);

        if (DEBUG) {
            printf("\nGPU Primitive Enact Done.\n");
        }

        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Primitive enact kernel entry.
     *
     * @tparam Problem Problem type. @see Problem
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     * @param[in] traversal_mode Traversal Mode for advance operator:
     *            Load-balanced or Dynamic cooperative
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename SMProblem>
    cudaError_t Enact(
        CudaContext &context,
        SMProblem     *problem,
        int         max_grid_size  = 0,
        int         traversal_mode = 0) {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef oprtr::filter::KernelPolicy <
                SMProblem,             // Problem data type
                300,                 // CUDA_ARCH
                INSTRUMENT,          // INSTRUMENT
                0,                   // SATURATION QUIT
                true,                // DEQUEUE_PROBLEM_SIZE
                8,                   // MIN_CTA_OCCUPANCY
                8,                   // LOG_THREADS
                1,                   // LOG_LOAD_VEC_SIZE
                0,                   // LOG_LOADS_PER_TILE
                5,                   // LOG_RAKING_THREADS
                5,                   // END_BITMASK_CULL
                8 >                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

            typedef oprtr::advance::KernelPolicy <
                SMProblem,             // Problem data type
                300,                 // CUDA_ARCH
                INSTRUMENT,          // INSTRUMENT
                1,                   // MIN_CTA_OCCUPANCY
                10,                  // LOG_THREADS
                8,                   // LOG_BLOCKS
                32 * 128,            // LIGHT_EDGE_THRESHOLD (used for LB)
                1,                   // LOG_LOAD_VEC_SIZE
                0,                   // LOG_LOADS_PER_TILE
                5,                   // LOG_RAKING_THREADS
                32,                  // WARP_GATHER_THRESHOLD
                128 * 4,             // CTA_GATHER_THRESHOLD
                7,                   // LOG_SCHEDULE_GRANULARITY
                oprtr::advance::LB >
                AdvanceKernelPolicy;

                return EnactSM<
                    AdvanceKernelPolicy, FilterKernelPolicy, SMProblem>(
                        context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */
};

}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
