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

//#include <thread>
#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
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


template <typename Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK> class Enactor;


/*
 * @brief Iteration structure derived from IterationBase.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct SMIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, true, true, true, false>
{
public:
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;

    typedef SMInitFunctor<VertexId, SizeT, VertexId, Problem> SMInitFunctor;
    //typedef EdgeWeightFunctor<VertexId, SizeT, Value, Problem> EdgeWeightFunctor;
    typedef PruneFunctor<VertexId, SizeT, VertexId, Problem> PruneFunctor;
    typedef CountFunctor<VertexId, SizeT, VertexId, Problem> CountFunctor;

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
	if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter begin",thread_num, enactor_stats->iteration, peer_);
	if(TO_TRACK)
	{
	    printf("%d\t %lld\t %d FullQueue_Core queue_length = %lld\n",
                thread_num, (long long)enactor_stats->iteration, peer_,
                (long long)frontier_attribute -> queue_length);
            fflush(stdout);
	    util::Check_Exist<<<256, 256, 0, stream>>>(
                frontier_attribute -> queue_length,
                data_slice->gpu_idx, 2, enactor_stats -> iteration,
                frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));     
	}
	frontier_attribute->queue_reset = true;
        enactor_stats     ->nodes_queued[0] += frontier_attribute->queue_length;

	///////////////////////////////////////////////////////////////////////////
        // Initial filtering based on node labels and degrees 
        // And generate candidate sets for query nodes

	enactor_stats->iteration=0;
        frontier_attribute->queue_index  = 0;
        frontier_attribute->selector     = 0;
        frontier_attribute->queue_length = graph_slice->nodes;
        frontier_attribute->queue_reset  = true;


        gunrock::oprtr::filter::LaunchKernel
            <FilterKernelPolicy, SMProblem, SMInitFunctor>(
            enactor_stats->filter_grid_size,
            FilterKernelPolicy::THREADS,
            0, stream,
            enactor_stats->iteration + 1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            NULL,
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            NULL,
            work_progress[0],
            frontier_queue->keys[frontier_attribute->selector  ].GetSize(),
            frontier_queue->keys[frontier_attribute->selector^1].GetSize(),
            enactor_stats->filter_kernel_stats);


            if(Enactor::DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                "Initial filtering filter::Kernel failed", __FILE__, __LINE__))) break;
	    if(Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, 
		enactor_stats->iteration);

	    enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;
	    enactor_stats -> iteration++;
	    frontier_attribute->queue_reset = false;
            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;


	 /*   mgpu::SegReduceCsr(data_slice->d_c_set, 
			       data_slice->d_temp_keys, 
			       data_slice->d_temp_keys, 
			       data_slice->nodes_query * data_slice->nodes_data,
			       data_slice->nodes_query,
			       false,
			       data_slice->d_temp_keys,
			       (int)0,
			       mgpu::plus<int>(),
			       context);
	*/
	    //TODO: Divide the results by hop number of query nodes
	   /* util::MemsetDivideVectorKernel<<<128,128>>>(data_slice->d_temp_keys, 
							data_slice->d_query_degrees,
							data_slice->nodes_query);

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
	    /*mgpu::LocalitySortPairs(data_slice->d_temp_keys,
	   			    data_slice->d_query_nodeIDs,
				    data_slice->nodes_query,
				    context);
	   */
/*	    oprtr::advance::LaunchKernel
		  <AdvanceKernelPolicy, Problem, PruneFunctor>(
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
                  graph_slice->d_query_row,
                  graph_slice->d_query_col,
                  (SizeT*)NULL,
                  (VertexId*)NULL,
                  graph_slice->frontier_elements[frontier_attribute.selector],
                  graph_slice->frontier_elements[frontier_attribute.selector^1],
                  this->work_progress,
                  context,
                  gunrock::oprtr::advance::V2V);		
*/

	    ///////////////////////////////////////////////////////////////////////////
            // Prune out candidates by checking candidate neighbours 
            frontier_attribute->queue_index=0;
            frontier_attribute->selector =0;
            frontier_attribute->queue_length = graph_slice->nodes;
            frontier_attribute->queue_reset = true;

            if (retval = work_progress->GetQueueLength(
                frontier_attribute->queue_index,
                frontier_attribute->queue_length)) break;

	    if(Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance begin", thread_num, enactor_stats->iteration, peer_);

           gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SMProblem, PruneFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from previous filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

           if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
                "Prune Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;

	   if(Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration, peer_);
	   frontier_attribute -> queue_reset = false;
	   frontier_attribute -> queue_index++;
	   frontier_attribute -> selector ^= 1;
	   enactor_stats      -> AccumulateEdges(
		work_progress -> GetQueueLengthPointer<unsigned int, SizeT>
		(frontier_attribute->queue_index), stream);

           if (true) {
                if (retval = work_progress->GetQueueLength(
                        frontier_attribute->queue_index,
                        frontier_attribute->queue_length)) break;
                printf("advance queue length: %lld",
                       (long long) frontier_attribute->queue_length);
            }

            // Reset d_temp_keys to 0 and store number of candidate edges for each query edge
            util::MemsetKernel<<<128, 128, 0, stream>>>(
                    data_slice->d_temp_keys.GetPointer(util::DEVICE),
                    0, data_slice->nodes_data);

	    //////////////////////////////////////////////////////////////////////////////////////
            // Count number candidate edges for each query edge using CountFunctor
            //frontier_attribute->queue_index=0;
            //frontier_attribute->selector =0;
            //frontier_attribute->queue_length = graph_slice->nodes;
            //frontier_attribute->queue_reset = true;
	    if(Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance begin", thread_num, enactor_stats->iteration, peer_);

            gunrock::oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, CountFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

           if (Enactor::DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
                "Count Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
	   if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration, peer_);
	   frontier_attribute -> queue_reset = false;
	   frontier_attribute->queue_index++;
	   frontier_attribute->selector ^= 1;
	   if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter begin", thread_num, enactor_stats->iteration, peer_);

            // now the number of candidate edges are stored in d_temp_keys
            // sort the edge order based on number of candidate edges, from fewest to largest
            util::CUBRadixSort<VertexId, SizeT>(
                true, problem->data_slices[0]->edges_query,
                problem->data_slices[0]->d_temp_keys.GetPointer(util::DEVICE),
                problem->data_slices[0]->d_query_edgeId.GetPointer(util::DEVICE));

	    //////////////////////////////////////////////////////////////////////////////////////
            // Collect candidate edges for each query edge using CountFunctor
            //frontier_attribute->queue_index=0;
            //frontier_attribute->selector =0;
            //frontier_attribute->queue_length = graph_slice->nodes;
            //frontier_attribute->queue_reset = true;
	    if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration, peer_);
            gunrock::oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, CollectFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

           if (Enactor::DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
                "Collect Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
	   if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration, peer_);
           frontier_attribute -> queue_reset = false;
           frontier_attribute -> queue_index++;
           frontier_attribute -> selector ^= 1;
           enactor_stats      -> AccumulateEdges(
            work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(frontier_attribute->queue_index), stream);

	   /////////////////////////////////////////////////////////
	   // Kernel Join
           gunrock::util::Join<<<128,128>>>(
                problem->data_slices[0]->edges_query,
                problem->data_slices[0]->d_temp_keys.GetPointer(util::DEVICE),
                problem->data_slices[0]->froms_data.GetPointer(util::DEVICE),
                problem->data_slices[0]->tos_data.GetPointer(util::DEVICE),
                problem->data_slices[0]->flag.GetPointer(util::DEVICE),
                problem->data_slices[0]->froms.GetPointer(util::DEVICE),
                problem->data_slices[0]->tos.GetPointer(util::DEVICE));

            if (d_scanned_edges) cudaFree(d_scanned_edges);
	}
};
/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam SMEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename SMEnactor>
static CUT_THREADPROC SMThread(
    void * thread_data_)
{
    typedef typename SMEnactor::Problem    Problem;
    typedef typename SMEnactor::SizeT      SizeT;
    typedef typename SMEnactor::VertexId   VertexId;
    typedef typename SMEnactor::Value      Value;
    typedef typename Problem::DataSlice    DataSlice;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef SMInitFunctor<VertexId, SizeT, VertexId, Problem> SMFunctor;
    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data->problem;
    SMEnactor    *enactor            =  (SMEnactor*)   thread_data->enactor;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
    Problem::DataSlice *data_slice =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    FrontierAttribute<SizeT>
                 *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

    do {
        printf("CCThread entered\n");fflush(stdout);
        if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
        thread_data->stats = 1;
        while (thread_data->stats !=2) sleep(0);
        thread_data->stats = 3;

        for (int peer_=0; peer_<num_gpus; peer_++)
        {
            frontier_attribute[peer_].queue_index  = 0;
            frontier_attribute[peer_].selector     = 0;
            frontier_attribute[peer_].queue_length = 0;
            frontier_attribute[peer_].queue_reset  = true;
            enactor_stats     [peer_].iteration    = 0;
        }

        gunrock::app::Iteration_Loop
            <1,0, SMEnactor, SMFunctor, SMIteration<AdvanceKernelPolicy, FilterKernelPolicy, SMEnactor> > (thread_data);

        printf("SM_Thread finished\n");fflush(stdout);
    } while (0);
    thread_data->stats = 4;
    CUT_THREADEND;
}

//TODO: preprocess data graph into edge list with edges source id< dest id using IntervalExpand in mgpu
/**
 * @biief SM Primitive enactor class.
 *
 * @tparam _Problem
 * @tparam INSTRUMWENT Boolean indicate collect per-CTA clock-count statistics
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template<
    typename _Problem,
    bool _INSTRUMENT,
    bool _DEBUG,
    bool _SIZE_CHECK>
class SMEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK> 
{
   // Members
   _Problem   *problem;
   ThreadSlice *thread_slices;
   CUTThread   *thread_Ids;

 // Methods
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
   
    /**
     * @brief Primitive Constructor.
     *
     * @param[in] gpu_idx GPU indices
     */
    SMEnactor(int num_gpus = 1, int *gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(
            EDGE_FRONTIERS, num_gpus, gpu_idx) 
    {
	thread_slices = NULL;
 	thread_Ids    = NULL;
 	problem       = NULL;
    }

    /**
     * @brief SMEnactor default Destructor
     */
    ~SMEnactor() 
    {
	cutWaitForThreads(thread_Ids, this->num_gpus);
	delete[] thread_Ids; thread_Ids = NULL;
	delete[] thread_slices; thread_slices = NULL;
	problem = NULL;
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
        num_iterations = this->enactor_stats.iteration;
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
    /*template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SMProblem >
    cudaError_t EnactSM(
        ContextPtr  context,
        SMProblem*    problem,
        int         max_grid_size = 0) {
        typedef typename SMProblem::SizeT  SizeT;
	typedef typename SMProblem::Value  Value;
        typedef typename SMProblem::VertexId VertexId;

        // Define functors used in SMProblem
        typedef SMInitFunctor<VertexId, SizeT, VertexId, SMProblem> SMInitFunctor;
        //typedef EdgeWeightFunctor<VertexId, SizeT, Value, SMProblem> EdgeWeightFunctor;
        typedef PruneFunctor<VertexId, SizeT, VertexId, SMProblem> PruneFunctor;
	typedef CountFunctor<VertexId, SizeT, VertexId, SMProblem> CountFunctor;
	typedef CollectFunctor<VertexId, SizeT, VertexId, SMProblem> CollectFunctor;
	//typedef JoinFunctor<VertexId, SizeT, VertexId, SMProblem> JoinFunctor;

        cudaError_t retval = cudaSuccess;
	SizeT* d_scanned_edges = NULL;  // Used for LB

	FrontierAttribute<SizeT> *frontier_attribute = &this->frontier_attribute[0];
        EnactorStats *enactor_stats = &this->enactor_stats[0];
        typename SMProblem::DataSlice *data_slice = problem->data_slices[0];
	util::DoubleBuffer<SizeT, VertexId, Value>*
            frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime *work_progress = &this->work_progress[0];
        cudaStream_t stream = data_slice->streams[0];

        do {
	    // Lazy Initialization
	    if (retval = Setup(problem)) break;
	    
	    if (retval = EnactorBase<typename _Problem::SizeT,
				     _DEBUG, _SIZE_CHECK>::Setup(
					    problem,
				  	    max_grid_size,
					    AdvanceKernelPolicy::CTA_OCCUPANCY,
					    FilterKernelPolicy::CTA_OCCUPANCY)) break;

	    // Single-gpu graph slice
	    GraphSlice<SizeT, VertexId, Value> *graph_slice = problem->graph_slices[0];
	    typename SMProblem::DataSlice *d_data_slice = problem->d_data_slices[0];

            if (retval = util::GRError(cudaMalloc((void**)&d_scanned_edges,
         	 graph_slice->edges * sizeof(SizeT)),
          	"SMProblem cudaMalloc d_scanned_edges failed",
         	 __FILE__, __LINE__)) return retval;

	    ///////////////////////////////////////////////////////////////////////////
	    // Initial filtering based on node labels and degrees 
	    // And generate candidate sets for query nodes
	    frontier_attribute->queue_index = 0; // work queue index
	    frontier_attribute->selector = 0;
	    frontier_attribute->queue_length = graph_slice->nodes;
	    frontier_attribute->queue_reset = true;

	    oprtr::filter::LaunchKernel
            <FilterKernelPolicy, SMProblem, SMInitFunctor>(
            enactor_stats->filter_grid_size,
            FilterKernelPolicy::THREADS,
            0, stream,
            enactor_stats->iteration + 1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            NULL,
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            NULL,
            work_progress[0],
            frontier_queue->keys[frontier_attribute->selector  ].GetSize(),
            frontier_queue->keys[frontier_attribute->selector^1].GetSize(),
            enactor_stats->filter_kernel_stats);


	    if(DEBUG && (retval = util::GRError(cudaThreadSynchronize(), 
		"Initial filtering filter::Kernel failed", __FILE__, __LINE__))) break;

	    if(frontier_attribute->queue_reset)
		
	    frontier_attribute->queue_index++;
	    frontier_attribute->selector ^= 1;


	    ///////////////////////////////////////////////////////////////////////////
	    // Prune out candidates by checking candidate neighbours 
	    frontier_attribute->queue_index=0;   	   
	    frontier_attribute->selector =0;
	    frontier_attribute->queue_length = graph_slice->nodes;
	    frontier_attribute->queue_reset = true;

            if (retval = work_progress->GetQueueLength(
                frontier_attribute->queue_index,
                frontier_attribute->queue_length)) break;
	    
   	    oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SMProblem, PruneFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from previous filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

	   if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          	"Prune Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;


           if (true) {
                if (retval = work_progress->GetQueueLength(
                        frontier_attribute->queue_index,
                        frontier_attribute->queue_length)) break;
                printf("advance queue length: %lld",
                       (long long) frontier_attribute->queue_length);
            }

	    // Reset d_temp_keys to 0 and store number of candidate edges for each query edge
	    util::MemsetKernel<<<128, 128, 0, stream>>>(
                    data_slice->d_temp_keys.GetPointer(util::DEVICE),
                    0, data_slice->nodes_data);	   
 
	    //////////////////////////////////////////////////////////////////////////////////////
            // Count number candidate edges for each query edge using CountFunctor
	    frontier_attribute->queue_index=0;   	   
	    frontier_attribute->selector =0;
	    frontier_attribute->queue_length = graph_slice->nodes;
	    frontier_attribute->queue_reset = true;
	    
	    oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, CountFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

	   if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          	"Count Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;

	    // now the number of candidate edges are stored in d_temp_keys
	    // sort the edge order based on number of candidate edges, from fewest to largest
	    util::CUBRadixSort<VertexId, SizeT>(
		true, problem->data_slices[0]->edges_query, 
		problem->data_slices[0]->d_temp_keys.GetPointer(util::DEVICE),
	   	problem->data_slices[0]->d_query_edgeId.GetPointer(util::DEVICE));
	    //////////////////////////////////////////////////////////////////////////////////////
            // Collect candidate edges for each query edge using CountFunctor
            frontier_attribute->queue_index=0;
            frontier_attribute->selector =0;
            frontier_attribute->queue_length = graph_slice->nodes;
            frontier_attribute->queue_reset = true;

            oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, CollectFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
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

           if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
                "Collect Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;

	   // Kernel Join
	   util::Join<<<128,128>>>(
		problem->data_slices[0]->edges_query,
	 	problem->data_slices[0]->d_temp_keys.GetPointer(util::DEVICE),
		problem->data_slices[0]->froms_data.GetPointer(util::DEVICE),
		problem->data_slices[0]->tos_data.GetPointer(util::DEVICE),
		problem->data_slices[0]->flag.GetPointer(util::DEVICE),
		problem->data_slices[0]->froms.GetPointer(util::DEVICE),
		problem->data_slices[0]->tos.GetPointer(util::DEVICE));

            if (d_scanned_edges) cudaFree(d_scanned_edges);

        } while (0);

        if (DEBUG) {
            printf("\nGPU Primitive Enact Done.\n");
        }

        return retval;
    }*/

    /**
     *
     * @brief Initialize the problem.
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator
     * @tparam FilterKernelPolicy Kernel policy for filter operator
     *
     * @param[in] context CudaContext pointer for ModernGPU API
     * @param[in] problem Pointer to Problem object
     * @param[in] max_grid_size Maximum grid size for kernel calls
     * @param[in] size_check Whether or not to enable size check
     *
     * \return cudaError_t object Indicates the success of all CUDA calls
     */
    template<
	typename AdvanceKernelPolicy,
	typename FilterKernelPolicy>
    cudaError_t InitSM(
	ContextPtr *context,
	Problem    *problem,
	int 	   max_grid_size = 512,
	bool 	   size_check =true)
    {
	cudaError_t retval = cudaSuccess;
        // Lazy initialization
 	if (retval = EnactorBase<SizeT, DEBUG, SIZE_CHECK> :: Init(
	    problem,
	    max_grid_size,
	    AdvanceKernelPolicy::CTA_OCCUPANCY,
	    FilterKernelPolicy::CTA_OCCUPANCY)) return retval;
	if(DEBUG)
	    printf("SM vertex map occupancy %d, level-grid size %d\n",
		FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size);

	this->problem = problem;
 	thread_slices = new ThreadSlice[this->num_gpus];
	thread_Ids    = new CUTThread[this->num_gpus];

	for(int gpu = 0; gpu<this->num_gpus; gpu++)
	{
	    thread_slices[gpu].thread_num = gpu;
 	    thread_slices[gpu].problem	  = (void*)problem;
	    thread_slices[gpu].enactor    = (void*)this;
 	    thread_slices[gpu].context    = &(context[gpu*this->num_gpus]);
	    thread_slices[gpu].stats      = -1;
	    thread_slices[gpu].thread_Id  = cutStartThread(
		(CUT_THREADROUTINE)&(SMThread<
		AdvanceKernelPolicy, FilterKernelPolicy,
		SMEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
		(void*)&(thread_slices[gpu])) ;
	    thread_Ids[gpu] = thread_slices[gpu].thread_Id;
	}
	return retval;
    }

    /**
     * @brief Reset enactor
     *
     * \retrun cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {
	retrun EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Reset();
    }


    /**
     * @brief Enacts a subgraph matching computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template<
	typename AdvanceKernelPolicy,
	typename FilterKernelPolicy>
    cudaError_t EnactSM()
    {
	cudaError_t retval = cudaSuccess;
	do{
	    for(int gpu=0; gpu<this->num_gpus; gpu++)
	    {
		while(thread_slices[gpu].stats!=1) sleep(0);
		thread_slices[gpu].stats = 2;
	    }
	    for(int gpu=0; gpu<this->num_gpus; gpu++)
		while(thread_slices[gpu].stats!=4) sleep(0);
	    for(int gpu=0; gpu<this->num_gpus; gpu++)
		if(this->enactor_stats[gpu].retval != cudaSuccess)
		{
		    retval = this->enactor_stats[gpu].retval;
		    break;
 		}
	}while(0);
	if(DEBUG) printf("\nGPU SM Done.\n");
	return retval;
    }
   
    /**
     * \addtogroup PublicInterface
     * @{
     */


    typedef gunrock::oprtr::filter::KernelPolicy <
    	Problem,             // Problem data type
        300,                 // CUDA_ARCH
        INSTRUMENT,          // INSTRUMENT
        0,                   // SATURATION QUIT
        true,                // DEQUEUE_PROBLEM_SIZE
        8,                   // MIN_CTA_OCCUPANCY
        7,                   // LOG_THREADS
        1,                   // LOG_LOAD_VEC_SIZE
        0,                   // LOG_LOADS_PER_TILE
        5,                   // LOG_RAKING_THREADS
        0,                   // END_BITMASK_CULL
        8 >                  // LOG_SCHEDULE_GRANULARITY
    FilterPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy <
        Problem,             // Problem data type
        300,                 // CUDA_ARCH
        INSTRUMENT,          // INSTRUMENT
        8,                   // MIN_CTA_OCCUPANCY
        7,                   // LOG_THREADS
        8,                   // LOG_BLOCKS
        32 * 128,            // LIGHT_EDGE_THRESHOLD (used for LB)
        1,                   // LOG_LOAD_VEC_SIZE
        0,                   // LOG_LOADS_PER_TILE
        5,                   // LOG_RAKING_THREADS
        32,                  // WARP_GATHER_THRESHOLD
        128 * 4,             // CTA_GATHER_THRESHOLD
        7,                   // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB >
    AdvancePolicy;

    /**
     * @brief SM enact kernel entry.
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
   // template <typename SMProblem>
    cudaError_t Enact()
    //    ContextPtr  context,
    //    SMProblem*  problem,
    //    int         max_grid_size  = 0) 
    {	
	int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++) {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version) {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300) 
	{
                return EnactSM<
                    AdvancePolicy, FilterPolicy, Problem>();
        //                context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief SM Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] size_check Whether or not to enable size check.
     * 
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
	    ContextPtr *context,
	    Problem    *problem,
	    int        	max_grid_size = 512,
	    bool       	size_check    = true)
   {
  	int min_sm_version = -1;
	for(int gpu=0; gpu<this->num_gpus; gpu++)
	{
	    if (min_sm_version == -1 ||
		this->cuda_props[gpu].device_sm_version < min_sm_version)
		min_sm_version = this->cuda_props[gpu].device_sm_version;
	}

	if(min_sm_version >= 300)
	    return InitSM<AdvancePolicy, FilterPolicy>(
			context, problem, max_grid_size, size_check);
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
