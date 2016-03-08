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
  typename _Problem,
  bool _INSTRUMENT,
  bool _DEBUG,
  bool _SIZE_CHECK >
class SMEnactor :
  public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK> {

  /**
   * @brief Prepare the enactor for SM kernel call.
   * Must be called prior to each SM iteration.
   *
   * @param[in] problem SM Problem object which holds the graph data and
   * SM problem data to compute.
   *
   * \return cudaError_t object which indicates the success of all CUDA calls.
   */
  template <typename ProblemData>
  cudaError_t Setup(ProblemData *problem)
  {
    typedef typename ProblemData::Value    Value;
    typedef typename ProblemData::VertexId VertexId;
    typedef typename ProblemData::SizeT    SizeT;

    cudaError_t retval = cudaSuccess;

    do
    {
      // Graph slice
      // GraphSlice<SizeT, VertexId, Value>*
      //   graph_slice = problem->graph_slices[0];
      // Data slice
      // typename ProblemData::DataSlice*
      //   data_slice = problem->data_slices[0];
    } while (0);

    return retval;
  }

 public:
   typedef _Problem                    SMProblem;
   typedef typename SMProblem::SizeT       SizeT;
   typedef typename SMProblem::VertexId VertexId;
   typedef typename SMProblem::Value       Value;
   static const bool INSTRUMENT   =   _INSTRUMENT;
   static const bool DEBUG        =        _DEBUG;
   static const bool SIZE_CHECK   =   _SIZE_CHECK;
  /**
   * @brief SMEnactor constructor.
   */
  SMEnactor(int *gpu_idx) :
    EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(
      EDGE_FRONTIERS, 1, gpu_idx)
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

  /**
   * @ brief Obtain statistics about the last MST search enacted.
   *
   * @ param[out] total_queued Total queued elements in MST kernel.
   * @ param[out] search_depth Search depth of MST algorithm.
   * @ param[out] avg_duty Average kernel running duty (kernel run time / kernel lifetime).
   * spaces between @ and name are to eliminate doxygen warnings
   */
  /*template <typename VertexId>
  void GetStatistics(
    long long &total_queued,
    VertexId  &search_depth,
    double    &avg_duty)
  {
    cudaDeviceSynchronize();

    total_queued = this->enactor_stats->total_queued[0];
    search_depth = this->enactor_stats->iteration;

    avg_duty = (this->enactor_stats->total_lifetimes >0) ?
      double(this->enactor_stats->total_runtimes) /
        this->enactor_stats->total_lifetimes : 0.0;
  }*/

  /** @} */

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
    typename FilterKernelPolicy,
    typename SMProblem>
  cudaError_t EnactSM(
    ContextPtr  context,
    SMProblem*  problem,
    int         max_grid_size = 0)
  {
    typedef typename SMProblem::VertexId VertexId;
    typedef typename SMProblem::SizeT    SizeT;
    typedef typename SMProblem::Value    Value;

    typedef SMInitFunctor<VertexId, SizeT, VertexId, SMProblem> SMInitFunctor;
    //typedef EdgeWeightFunctor<VertexId, SizeT, Value, SMProblem> EdgeWeightFunctor;
    typedef UpdateDegreeFunctor<VertexId, SizeT, VertexId, SMProblem> UpdateDegreeFunctor;
    typedef PruneFunctor<VertexId, SizeT, VertexId, SMProblem> PruneFunctor;
    typedef LabelEdgeFunctor<VertexId, SizeT, VertexId, SMProblem> LabelEdgeFunctor;
    typedef CollectFunctor<VertexId, SizeT, VertexId, SMProblem> CollectFunctor;

    cudaError_t retval = cudaSuccess;
    SizeT *d_scanned_edges = NULL;  // Used for LB

    FrontierAttribute<SizeT>* attributes = &this->frontier_attribute[0];
    EnactorStats* statistics = &this->enactor_stats[0];
    typename SMProblem::DataSlice* data_slice = problem->data_slices[0];
    util::DoubleBuffer<SizeT, VertexId, Value>*
      queue = &data_slice->frontier_queues[0];
    util::CtaWorkProgressLifetime*
      work_progress = &this->work_progress[0];
    cudaStream_t stream = data_slice->streams[0];

    do
    {
      // initialization
      if (retval = Setup(problem)) break;
      if (retval = EnactorBase<
        typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>::Setup(
          problem,
          max_grid_size,
          AdvanceKernelPolicy::CTA_OCCUPANCY,
          FilterKernelPolicy::CTA_OCCUPANCY)) break;

      // single-GPU graph slice
      GraphSlice<SizeT,VertexId,Value>* graph_slice = problem->graph_slices[0];
      typename SMProblem::DataSlice* d_data_slice = problem->d_data_slices[0];

      if (retval = util::GRError(cudaMalloc(
        (void**)&d_scanned_edges, graph_slice->edges * sizeof(SizeT)),
        "Problem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__))
      {
        return retval;
      }

      // debug configurations
      //SizeT num_edges_origin = graph_slice->edges;
      bool debug_info = 0;   // used for debug purpose
      //int tmp_select  = 0; // used for debug purpose
      //int tmp_length  = 0; // used for debug purpose
      unsigned int *num_selected = new unsigned int; // used in cub select
        if (debug_info)
        {
          printf("\nBEGIN ITERATION: %lld #NODES: %d #EDGES: %d\n",
            statistics->iteration+1,graph_slice->nodes,graph_slice->edges);
        }

        if (debug_info)
        {
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

	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    0, graph_slice->nodes);

	gunrock::oprtr::filter::LaunchKernel
            <FilterKernelPolicy, SMProblem, SMInitFunctor>(
            statistics->filter_grid_size,
            FilterKernelPolicy::THREADS,
            0, stream,
            statistics->iteration + 1,
            attributes->queue_reset,
            attributes->queue_index,
            attributes->queue_length,
            queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
            NULL,
            NULL,//queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            NULL,
            work_progress[0],
            queue->keys[attributes->selector  ].GetSize(),
            queue->keys[attributes->selector^1].GetSize(),
            statistics->filter_kernel_stats);


            if(debug_info && (retval = util::GRError(cudaThreadSynchronize(),
                "Initial filtering filter::Kernel failed", __FILE__, __LINE__))) break;
if(i!=1){ //Last round doesn't need the following functor
        ///////////////////////////////////////////////////////////////////////////
        // Update each candidate node's valid degree by checking their neighbors
        attributes->queue_index  ++;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = false;

	gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SMProblem, UpdateDegreeFunctor>(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from previous filter functor
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
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

	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Update Degree Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
}


	}
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
            mgpu::LocalitySortPairs(data_slice->d_edge_weights,
                                    data_slice->d_edge_labels, data_slice->edges_query, context);

*/

	// Run prune functor for several iterations
        for(int i=0; i<2; i++){
        ///////////////////////////////////////////////////////////////////////////
        // Prune out candidates by checking candidate neighbors 
        attributes->queue_index  =0;
        attributes->selector     =0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset  = true;

	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    0, problem->data_slices[0]->nodes_data * problem->data_slices[0]->nodes_query);

	gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SMProblem, PruneFunctor>(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from previous filter functor
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
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

	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Prune Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
	}
	if(debug_info){
	util::debug_init<<<128,128>>>(
	     problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE),
	     problem->data_slices[0]->nodes_query,
	     problem->data_slices[0]->nodes_data);
	}
	// Put the candidate index of each candidate edge into froms at its e_id position
        // Fill the non-candidate edge positions with 0 in froms
	//////////////////////////////////////////////////////////////////////////////////////
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    0, graph_slice->edges);

        attributes->queue_index  =0;
        attributes->selector   =0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset  = true;

	gunrock::oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, LabelEdgeFunctor>(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
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

	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Label Edge Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
	if(debug_info){
	util::debug_label<<<128,128>>>(
		problem->data_slices[0]->froms.GetPointer(util::DEVICE),
		problem->data_slices[0]->edges_data);
	}
	// Use d_query_col to store the number of candidate edges for each query edge
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE), 0, 
		(problem->data_slices[0]->edges_query));

	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->froms_data.GetPointer(util::DEVICE), 0, 
		(problem->data_slices[0]->edges_data/2 * problem->data_slices[0]->edges_query));
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->tos_data.GetPointer(util::DEVICE), 0, 
		(problem->data_slices[0]->edges_query * problem->data_slices[0]->edges_data/2));

	// collect candidate edges for each query edge
	for(SizeT i=0; i<problem->data_slices[0]->edges_query; i++){

	// Use d_data_degrees as flags of edge labels, initialized to 0
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE), 0, graph_slice->edges);


	// Mark the candidate edges for each query edge and store in d_data_degrees
	util::Mark<<<128,128>>>(
	    problem->data_slices[0]->edges_data,
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE));

	Scan<mgpu::MgpuScanTypeInc>(
	    (int*)problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    problem->data_slices[0]->edges_data, (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
	    (int*)problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE), context[0]);
	// update the number of candidate edges for each query edge in d_query_col
	util::Update<<<128,128>>>(
	    i, problem->data_slices[0]->edges_query,
	    problem->data_slices[0]->edges_data,
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE));

	//////////////////////////////////////////////////////////////////////////////////////
        // Collect candidate edges for each query edge using CollectFunctor
	attributes -> queue_reset = false;
        attributes->queue_length = graph_slice->edges;
        attributes -> queue_index=0;
        attributes -> selector =0;

        gunrock::oprtr::advance:: LaunchKernel<AdvanceKernelPolicy, SMProblem, CollectFunctor>
	(
                statistics[0],
                attributes[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,  // In order to use the output vertices from prevs filter functor 
                queue->keys[attributes->selector].GetPointer(util::DEVICE),
                queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
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

	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Collect Functor Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
	if(debug_info){
	util::debug<<<128,128>>>(
	    i,
	    problem->data_slices[0]->froms_data.GetPointer(util::DEVICE), 
	    problem->data_slices[0]->tos_data.GetPointer(util::DEVICE), 
	    problem->data_slices[0]->froms_query.GetPointer(util::DEVICE), 
	    problem->data_slices[0]->tos_query.GetPointer(util::DEVICE), 
	    problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE), 
	    problem->data_slices[0]->edges_query, problem->data_slices[0]->edges_data);

	}

	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Debug failed", __FILE__, __LINE__))) break;
	}  //end of for loop


//Joining step
	// Use d_query_row[0] to store the number of matched subgraphs in each join step
	// Use d_query_row[1] to store the number of matched subgraphs in previous iteration
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE), 0, 
	    problem->data_slices[0]->nodes_query+1);

	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE), -1, 
	    problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data * 
	    problem->data_slices[0]->edges_query);

	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->tos.GetPointer(util::DEVICE), -1, 
	    problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data * 
	    problem->data_slices[0]->edges_query);

	for(int i=0; i<problem->data_slices[0]->edges_query-1; i++){
	// Use d_c_set as flags of success join, initialized to 0
	util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE), false, 
	    problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data*
	    problem->data_slices[0]->edges_data/2);
	/////////////////////////////////////////////////////////
        // Kernel Join
        gunrock::util::Join<<<128,128>>>(
                problem->data_slices[0]->edges_query,
	    	i,
                problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE),
                problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE),
                problem->data_slices[0]->flag.GetPointer(util::DEVICE),
                problem->data_slices[0]->froms_data.GetPointer(util::DEVICE),
                problem->data_slices[0]->tos_data.GetPointer(util::DEVICE),
                problem->data_slices[0]->froms.GetPointer(util::DEVICE),
                problem->data_slices[0]->tos.GetPointer(util::DEVICE));


       	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Join Kernel failed", __FILE__, __LINE__))) break;
	if(debug_info==1 && i==0)
util::debug_0<<<128,128>>>(
	problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE),
        problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE),
	problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE),
        problem->data_slices[0]->froms_data.GetPointer(util::DEVICE),
        problem->data_slices[0]->tos_data.GetPointer(util::DEVICE),
	problem->data_slices[0]->edges_query,
	i);
	
/*	Scan<mgpu::MgpuScanTypeInc>(
	    (int*)problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    5000, 
	    (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
	    (int*)problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE), context[0]);
*/

//if(util::GRError(
//	(retval = cudaMalloc(&d_in,problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data*problem->data_slices[0]->edges_data/2*sizeof(bool))),
//	"CUBSelect malloc d_in failed",
//	__FILE__, __LINE__))  return retval;

        util::MemsetKernel<<<128,128>>>(
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE), 0, 
	    problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data);
	util::MemsetIdxKernel<<<128,128>>>(
	    problem->data_slices[0]->d_in.GetPointer(util::DEVICE),
		problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data
		*problem->data_slices[0]->edges_data/2);
	util::CUBSelect_flagged<VertexId, SizeT, SizeT, bool>(
		problem->data_slices[0]->d_in.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE),
		problem->data_slices[0]->nodes_data*problem->data_slices[0]->nodes_data
		*problem->data_slices[0]->edges_data/2);
	//if(debug_info){ 
	util::debug_select<<<128,128>>>(i,
		problem->data_slices[0]->d_in.GetPointer(util::DEVICE),
		problem->data_slices[0]->nodes_data,
		problem->data_slices[0]->edges_data,
		problem->data_slices[0]->d_c_set.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
		problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE));  
//	}
//if(util::GRError(
//	(retval = cudaFree(d_in)),
//	"CUBSelect free d_in failed",
//	__FILE__, __LINE__)) return retval;

	// Collect the valid joined edges to consecutive places	
 	util::Collect<<<128,128>>>(
            problem->data_slices[0]->edges_query,
	    i,
            problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    problem->data_slices[0]->froms_data.GetPointer(util::DEVICE),
	    problem->data_slices[0]->tos_data.GetPointer(util::DEVICE),
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    problem->data_slices[0]->tos.GetPointer(util::DEVICE),
            problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE),
            problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE));
 
       	if (debug_info && (retval = util::GRError(cudaDeviceSynchronize(),
                "Collect Kernel failed", __FILE__, __LINE__))) break;
	if(debug_info)
	util::debug_1<<<128,128>>>(
	    problem->data_slices[0]->froms.GetPointer(util::DEVICE),
	    problem->data_slices[0]->tos.GetPointer(util::DEVICE),
	    problem->data_slices[0]->d_data_degrees.GetPointer(util::DEVICE),
	    problem->data_slices[0]->d_query_col.GetPointer(util::DEVICE),
	    problem->data_slices[0]->d_query_row.GetPointer(util::DEVICE),
	    problem->data_slices[0]->edges_query);
        }


      if (d_scanned_edges) cudaFree(d_scanned_edges);
      if (retval) 
	break;

    } while(0);
    return retval;
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
  template <typename SMProblem>
  cudaError_t Enact(
    ContextPtr  context,
    SMProblem* problem,
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
      typedef gunrock::oprtr::filter::KernelPolicy<
        SMProblem,          // Problem data type
        300,                // CUDA_ARCH
        INSTRUMENT,         // INSTRUMENT
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
        SMProblem,          // Problem data type
        300,                // CUDA_ARCH
        INSTRUMENT,         // INSTRUMENT
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
      return EnactSM<AdvanceKernelPolicy, FilterKernelPolicy,
        SMProblem>(context, problem, max_grid_size);
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

} // namespace sm
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
