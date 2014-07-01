// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mst_enactor.cuh
 *
 * @brief MST Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/mark_segment.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

#include <moderngpu.cuh>
#include <limits>
#include <thrust/sort.h>

namespace gunrock {
namespace app {
namespace mst {

using namespace mgpu;

/**
 * @brief MST problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not
 * to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class MSTEnactor : public EnactorBase
{
  // Members
protected:

  /**
   * CTA duty kernel stats
   */

  unsigned long long total_runtimes;  // Total working time by each CTA
  unsigned long long total_lifetimes; // Total life time of each CTA
  unsigned long long total_queued;

  /**
   * A pinned, mapped word that the traversal kernels will signal when done
   */
  int	         *vertex_flag;
  volatile int *done;
  int          *d_done;
  cudaEvent_t  throttle_event;

// Methods
protected:

  /**
   * @brief Prepare the enactor for MST kernel call.
   * Must be called prior to each MST iteration.
   *
   * @param[in] problem MST Problem object which holds the graph data and MST problem data to compute.
   * @param[in] edge_map_grid_size CTA occupancy for advance kernel call.
   * @param[in] filter_grid_size CTA occupancy for filter kernel call.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  template <typename ProblemData>
  cudaError_t Setup(ProblemData *problem)
  {
    typedef typename ProblemData::SizeT    SizeT;
    typedef typename ProblemData::VertexId VertexId;

    cudaError_t retval = cudaSuccess;

    do 
    {
      //initialize the host-mapped "done"
      if (!done)
      {
	      int flags = cudaHostAllocMapped;

      	// Allocate pinned memory for done
      	if (retval = util::GRError(cudaHostAlloc(
          (void**)&done, sizeof(int) * 1, flags),
      		"MSTEnactor cudaHostAlloc done failed",
          __FILE__, __LINE__)) break;

      	// Map done into GPU space
      	if (retval = util::GRError(cudaHostGetDevicePointer(
          (void**)&d_done, (void*) done, 0),
      		"MSTEnactor cudaHostGetDevicePointer done failed",
          __FILE__, __LINE__)) break;

      	// Create throttle event
      	if (retval = util::GRError(cudaEventCreateWithFlags(
          &throttle_event, cudaEventDisableTiming),
      		"MSTEnactor cudaEventCreateWithFlags throttle_event failed",
          __FILE__, __LINE__)) break;
	    }

      // graph slice
      // typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

      /*   // moved to the begining of each iteration
      // Bind row-offsets and bitmask texture
      cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
      if (retval = util::GRError(cudaBindTexture(
    	  0,
    	  gunrock::oprtr::advance::RowOffsetTex<SizeT>::ref,
    	  graph_slice->d_row_offsets,
    	  row_offsets_desc,
    	  (graph_slice->nodes + 1) * sizeof(SizeT)),
    	  "MSTEnactor cudaBindTexture row_offset_tex_ref failed",
        __FILE__, __LINE__)) break;

      cudaChannelFormatDesc column_indices_desc = cudaCreateChannelDesc<VertexId>();
      if (retval = util::GRError(cudaBindTexture(
    	  0,
    	  gunrock::oprtr::advance::ColumnIndicesTex<SizeT>::ref,
    	  graph_slice->d_column_indices,
    	  column_indices_desc,
    	  graph_slice->edges * sizeof(VertexId)),
    	  "MSTEnactor cudaBindTexture column_indices_tex_ref failed",
        __FILE__, __LINE__)) break;
      */
    } while (0);

    return retval;
  }

public:

  /**
   * @brief MSTEnactor constructor
   */
  MSTEnactor(bool DEBUG = false) :
    EnactorBase(EDGE_FRONTIERS, DEBUG),
    vertex_flag(NULL),
    done(NULL),
    d_done(NULL)
  {
    vertex_flag = new int;
    vertex_flag[0] = 0;
  }

  /**
   * @brief MSTEnactor destructor
   */
  virtual ~MSTEnactor()
  {
    if (vertex_flag) delete vertex_flag;
    if (done)
    {
      util::GRError(cudaFreeHost((void*)done),
		    "MSTEnactor cudaFreeHost done failed",
        __FILE__, __LINE__);

      util::GRError(cudaEventDestroy(throttle_event),
		    "MSTEnactor cudaEventDestroy throttle_event failed",
        __FILE__, __LINE__);
    }
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Obtain statistics about the last MST search enacted.
   *
   * @param[out] total_queued Total queued elements in MST kernel.
   * @param[out] search_depth Search depth of MST algorithm.
   * @param[out] avg_duty Average kernel running duty
   * (kernel run time/kernel lifetime).
   */
  template <typename VertexId>
  void GetStatistics(
    long long &total_queued,
		VertexId  &search_depth,
		double    &avg_duty)
  {
    cudaThreadSynchronize();

    total_queued = enactor_stats.total_queued;
    search_depth = enactor_stats.iteration;

    avg_duty = (enactor_stats.total_lifetimes > 0) ?
      double(enactor_stats.total_runtimes) / enactor_stats.total_lifetimes : 0.0;
  }

  /** @} */

  /**
   * @brief Enacts a MST computing on the specified graph.
   *
   * @tparam Advance Kernel policy for forward advance.
   * @tparam Filter Kernel policy for vertex mapping.
   * @tparam MSTProblem MST Problem type.
   *
   * @param[in] cuda moderngpu context.
   * @param[in] problem MSTProblem object.
   * @param[in] max_grid_size Max grid size for MST kernel calls.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename MSTProblem>
  cudaError_t EnactMST(
    CudaContext &context,
		MSTProblem  *problem,
		int         max_grid_size = 0)
  {
    typedef typename MSTProblem::SizeT    SizeT;
    typedef typename MSTProblem::VertexId VertexId;

    typedef SuccFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> SuccFunctor;

    typedef RmCycFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> RmCycFunctor;

    typedef PtrJumpFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> PtrJumpFunctor;

    typedef EdgeRmFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> EdgeRmFunctor;

    typedef RmFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> RmFunctor;

    typedef VtxLenFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> VtxLenFunctor;

    typedef RowOffsetsFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> RowOFunctor;

    typedef EdgeLenFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> EdgeLenFunctor;

    typedef EdgeOffsetsFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> EdgeOffsetsFunctor;

    typedef SuEdgeRmFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> SuEdgeRmFunctor;

    typedef OrFunctor<
      VertexId,
      SizeT,
      VertexId,
      MSTProblem> OrFunctor;

    cudaError_t retval = cudaSuccess;

    unsigned int *d_scanned_edges = NULL;
    do 
    {
      if (DEBUG)
	    {
	      printf("Iteration, Edge map queue, Vertex map queue\n");
	      printf("0\n");
      }

      // initialization
      if (retval = Setup(problem)) break;
      if (retval = EnactorBase::Setup(
        problem,
        max_grid_size,
        AdvanceKernelPolicy::CTA_OCCUPANCY,
        FilterKernelPolicy::CTA_OCCUPANCY)) break;

      // single-gpu graph slice
      typename MSTProblem::GraphSlice *graph_slice = problem->graph_slices[0];
      typename MSTProblem::DataSlice  *data_slice  = problem->d_data_slices[0];

      if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB)
      {
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_scanned_edges,
          graph_slice->edges * sizeof(unsigned int)),
          "MSTProblem cudaMalloc d_scanned_edges failed",
          __FILE__, __LINE__)) return retval;
      }

      fflush(stdout);

      // keep record of original number of edges in graph
      int  num_edges_origin = graph_slice->edges;
      int  loop_limit = 0;
      bool debug_info = 0;

      // recursive Loop for MST implementations
      while (graph_slice->nodes > 1)
      {
	      printf("\nBEGIN: ITERATION:%lld #NODES:%d #EDGES:%d\n",
	        enactor_stats.iteration, graph_slice->nodes, graph_slice->edges);

	      // bind row_offsets and bitmask texture
      	cudaChannelFormatDesc	row_offsets_desc = cudaCreateChannelDesc<SizeT>();
	      if (retval = util::GRError(cudaBindTexture(
      	  0,
      	  gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
      	  graph_slice->d_row_offsets,
      	  row_offsets_desc,
      	  (graph_slice->nodes + 1) * sizeof(SizeT)),
      	  "MSTEnactor cudaBindTexture row_offset_tex_ref failed",
          __FILE__, __LINE__)) break;

        printf("----> finished binding row_offsets: new row_offsets. \n");

        /* // vertex mapping replaced with mark_segment kernel
      	// generate flag array using SuccFunctor - vertex mapping
      	frontier_attribute.queue_length = graph_slice->nodes;
        frontier_attribute.queue_index  = 0; // Work queue index
        frontier_attribute.selector     = 0;
        frontier_attribute.queue_reset  = true;

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, SuccFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    enactor_stats.iteration+1,		   // current graph traversal iteration
      	    frontier_attribute.queue_reset,	 // reset queue counter
      	    frontier_attribute.queue_index,	 // current frontier queue counter index
      	    enactor_stats.num_gpus,          // number of gpu(s)
      	    frontier_attribute.queue_length, // number of element(s)
      	    NULL,		                         // d_done
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],   // d_in_queue
      	    NULL, 	    	                   // d_pred_in_queue
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1], // d_out_queue
      	    data_slice,		                   // problem data
      	    NULL, 		                       // visited mask
      	    work_progress,	                 // work progress
      	    graph_slice->frontier_elements[frontier_attribute.selector],   // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1], // max_out_queue
      	    enactor_stats.filter_kernel_stats); // kernel stats

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"Filter::Kernel failed", __FILE__, __LINE__))) break;
        */
        
        printf("MARKING THE MST EDGES ...\n");

        // generate flag array using mark_segment kernel
        util::markSegmentFromOffsets<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
          graph_slice->d_row_offsets,
          graph_slice->nodes);

        printf("----> finished marking segmentation from offsets: flag array. \n");

      	// generate keys array using mgpu::scan
      	Scan<MgpuScanTypeInc>(
          (int*)problem->data_slices[0]->d_flag_array, graph_slice->edges,
      		(int)0, mgpu::plus<int>(), (int*)0, (int*)0,
      		(int*)problem->data_slices[0]->d_keys_array, context);

        printf("----> finished segmented minimum inclusive scan: keys array. \n"); 

      	// Select minimum edge_values and keys using mgpu::ReduceByKey
      	int numSegments;
      	ReduceByKey(
          problem->data_slices[0]->d_keys_array, 
          problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges,
          std::numeric_limits<int>::max(), 
          mgpu::minimum<int>(),
      		mgpu::equal_to<int>(),
          problem->data_slices[0]->d_reduced_keys,
      		problem->data_slices[0]->d_reduced_vals, 
          &numSegments, (int*)0, context);

      	printf("----> finished segmented reduction: reduced keys, edge values. \n");

      	if (debug_info)
      	{
      	  printf(":: origin flag array ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_flag_array, num_edges_origin);
      	  printf(":: origin keys array ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, num_edges_origin);
      	  printf(":: origin edge_values ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, num_edges_origin);
      	  printf(":: reduced keys array ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_reduced_keys, graph_slice->nodes);
      	  printf(":: reduced edge_values ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_reduced_vals, graph_slice->nodes);
        }

        // reset d_temp_storage used for atomicCAS to avoid multuple selections
      	util::MemsetKernel<<<128, 128>>>(
          problem->data_slices[0]->d_temp_storage, -1, graph_slice->nodes);

      	// Generate Successor Array using SuccFunctor - edge mapping
      	// Successor Array holds the outgoing v for each u
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset  = true;

      	gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, MSTProblem, SuccFunctor>(
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
          gunrock::oprtr::advance::V2E);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

        printf("----> finished advance kernel (edge mapping): successors array. \n");

      	if (debug_info)
      	{
      	  printf(":: successor array ::\n");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_successors, graph_slice->nodes);
      	  printf(":: mst_output boolean array ::\n");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_mst_output, graph_slice->edges);
      	}
        
      	// remove cycles using RmCycFuntor - Advance Edge Mapping, S(S(u)) = u
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset  = true;

      	gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, MSTProblem, RmCycFunctor>(
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
          gunrock::oprtr::advance::V2E);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

        printf("----> finished removing cycles from successors: new successors array. \n");

      	if (debug_info)
      	{
      	  printf(":: remove cycles from successors ::\n");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_successors, graph_slice->nodes);
      	  printf(":: mst output boolean array ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_mst_output, graph_slice->edges);
      	}

        printf("GRAPH CONSTRUCTION ...\n");
      	/*
      	// Pointer Jumpping to get representative array
      	util::MemsetCopyVectorKernel<<<128,128>>>(
          problem->data_slices[0]->d_representatives,
      		problem->data_slices[0]->d_successors,
      		graph_slice->nodes);
        */
      	// Using Vertex Mapping: PtrJumpFunctor
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset  = true;

      	vertex_flag[0] = 0;
      	while (!vertex_flag[0])
      	{
      	  vertex_flag[0] = 1;
      	  if (retval = util::GRError(cudaMemcpy(
            problem->data_slices[0]->d_vertex_flag,
      			vertex_flag,
      			sizeof(int),
      			cudaMemcpyHostToDevice),
      			"MSTProblem cudaMemcpy vertex_flag to d_vertex_flag failed",
      			__FILE__, __LINE__)) return retval;

    	    gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, PtrJumpFunctor>
    	      <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
    	        0,
          		frontier_attribute.queue_reset,
          		frontier_attribute.queue_index,
          		1,
          		frontier_attribute.queue_length,
          		NULL,						  
          		graph_slice->frontier_queues.d_keys[frontier_attribute.selector],    
          		NULL,
          		graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
          		data_slice,
          		NULL,
          		work_progress,
          		graph_slice->frontier_elements[frontier_attribute.selector], 
          		graph_slice->frontier_elements[frontier_attribute.selector^1],
          		enactor_stats.filter_kernel_stats);

    	    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
    				"filter::Kernel First Pointer Jumping Round failed",
    				__FILE__, __LINE__))) break;

    	    if (frontier_attribute.queue_reset) frontier_attribute.queue_reset = false;
          frontier_attribute.queue_index++;
          frontier_attribute.selector^=1;

    	    if (retval = util::GRError(cudaMemcpy(
            vertex_flag,
    				problem->data_slices[0]->d_vertex_flag,
    				sizeof(int),
    				cudaMemcpyDeviceToHost),
    				"MSTProblem cudaMemcpy d_vertex_flag to vertex_flag failed",
    				__FILE__, __LINE__)) return retval;

    	    if (vertex_flag[0]) break; // check if done
      	}

        printf("----> finished pointer doubling operation: representatives.\n");
      	
        if (debug_info)
      	{
      	  printf(":: Pointer Jumping: successors (now contains representatives) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_successors, graph_slice->nodes);
      	}

        /*
      	// Assigning Ids to supervertices stored in d_superVertex
      	// copy representatives to d_superVertex
      	util::MemsetCopyVectorKernel<<<128,128>>>(
          problem->data_slices[0]->d_superVertex,
      		problem->data_slices[0]->d_representatives,
      		graph_slice->nodes);

      	// Fill in the d_nodes : 0, 1, 2, ... , nodes
      	util::MemsetIdxKernel<<<128, 128>>>(
          problem->data_slices[0]->d_nodes,
          graph_slice->nodes);

      	// Mergesort pairs.
      	MergesortPairs(
          problem->data_slices[0]->d_superVertex,
      		problem->data_slices[0]->d_nodes,
      		graph_slice->nodes, mgpu::less<int>(), context);

      	// Scan of the flag assigns new supervertex ids stored in d_Cflag
      	util::markSegment<<<128, 128>>>(
          problem->data_slices[0]->d_Cflag,
      		problem->data_slices[0]->d_superVertex,
      		graph_slice->nodes);

      	if (debug_info)
      	{
      	  printf(":: Super-Vertices Ids (d_superVertex) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_superVertex, graph_slice->nodes);
      	  printf(":: Sorted nodes according to supervertices (d_nodes) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_nodes, graph_slice->nodes);
      	  printf(":: New Super-vertex Ids (C Flag) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_Cflag, graph_slice->nodes);
      	}

      	// Calculate the length of New Vertex List - VLENFunctor vertex mapping
      	// d_row_offsets[node] = (d_Cflag[node] == 0) ? -1 : 1
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset = true;

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, VtxLenFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,              // Current graph traversal iteration
      	    queue_reset,    // reset queue counter
      	    frontier_attribute.queue_index,    // Current frontier queue counter index
      	    1,              // Number of gpu(s)
      	    frontier_attribute.queue_length,   // Number of element(s)
      	    NULL,           // d_done
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
      	    NULL,           // d_pred_in_queue
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,     // Problem
      	    NULL,           // visited mask
      	    work_progress,  // work progress
      	    graph_slice->frontier_elements[frontier_attribute.selector],       // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],     // max_out_queue
      	    enactor_stats.filter_kernel_stats);                 // kernel stats

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

      	// Segmented reduction: generate C keys array using mgpu::scan
      	Scan<MgpuScanTypeInc>(
          (int*)problem->data_slices[0]->d_Cflag, graph_slice->nodes,
      		(int)0, mgpu::plus<int>(), (int*)0, (int*)0,
      		(int*)problem->data_slices[0]->d_Ckeys, context);

      	if (debug_info)
      	{
      	  printf(":: Generated Ckeys Array using Cflag (d_Ckeys) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_Ckeys, graph_slice->nodes);
      	}

      	// Remove edges using advance to mark -1 in Edges List - EdgeRmFunctor advance
      	// d_edges, d_edge_vals, d_keys_array, d_flag_array, d_eId, d_edgeFlag = d_representatives[s_id == d_id] ? -1 : old value
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset = true;

      	gunrock::oprtr::advance::Kernel<AdvanceKernelPolicy, MSTProblem, EdgeRmFunctor>
      	  <<<edge_map_grid_size, AdvanceKernelPolicy::THREADS>>>(
            frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    enactor_stats.iteration,
      	    frontier_attribute.queue_length,
      	    d_done,
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],          // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],        // d_out_queue
      	    graph_slice->d_column_indices,
      	    data_slice,
      	    this->work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],               // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],             // max_out_queue
      	    this->edge_map_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Mergesort pairs
      	MergesortPairs(
          problem->data_slices[0]->d_nodes,
      		problem->data_slices[0]->d_Ckeys,
      		graph_slice->nodes, mgpu::less<int>(), context);

      	if (debug_info)
      	{
      	  printf(":: Edge Removal (edges) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: Edge Removal (keys) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, graph_slice->edges);
      	  printf(":: Edge Removal (edge_values) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: Edge Removal (d_eId) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	  printf(":: Edge Removal (d_edgeFlag) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edgeFlag, graph_slice->edges);
      	  printf(":: Sorted Ckeys matching nodes ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_Ckeys, graph_slice->nodes);
      	  printf(":: Ordered Edge List ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	}

      	// Generate new edge list, keys array and edge_values list
      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edges,
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,	// d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edges
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edges,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_keys_array,
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to keys
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_keys_array,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edge_vals
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edge_vals,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,       // d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edges
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_eId,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edgeFlag,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,       // d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edgeFlag
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edgeFlag,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	// Generate Flag array with new length
      	// Remove "-1" items in edge list using RmFunctor - vertex mapping
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_flag_array,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,       // d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_flag_array
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	printf("----> 3.first time reduce edge length, before: Q: %d \n", frontier_attribute.queue_index);

      	// Update Length of edges in graph_slice
      	frontier_attribute.queue_index++;
      	if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, queue_length)) break;
      	graph_slice->edges = queue_length;
      	printf("----> 3.5 after update length: Q: %d, L: %d\n", frontier_attribute.queue_index, queue_length);

      	// TODO: Disordered on Midas. Sort to make sure correctness
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);
      	MergesortPairs(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges, mgpu::less<int>(), context);
      	MergesortPairs(
          problem->data_slices[0]->d_keys_array,
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges, mgpu::less<int>(), context);

      	if (debug_info)
      	{
      	  printf(":: d_edges after first reduction ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: d_keys_array after first reduction ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, graph_slice->edges);
      	  printf(":: d_edge_vals after first reduction ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: d_eId after first reduction ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	}

      	// Finding representatives for keys and edges using EdgeRmFunctor - Vertex Mapping
      	// problem->d_keys_array[node] = problem->d_Ckeys[problem->d_keys_array[node]];
      	// problem->d_edges[node] = problem->d_Ckeys[problem->d_edges[node]];
      	// Removing edges inside each super-vertex
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, EdgeRmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,              // Current graph traversal iteration
      	    frontier_attribute.queue_reset,    // reset queue counter
      	    frontier_attribute.queue_index,    // Current frontier queue counter index
      	    1,              // Number of gpu(s)
      	    frontier_attribute.queue_length,   // Number of element(s)
      	    NULL,           // d_done
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,           // d_pred_in_queue
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,     // Problem
      	    NULL,           // visited mask
      	    work_progress,  // work progress
      	    graph_slice->frontier_elements[frontier_attribute.selector],       // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],     // max_out_queue
      	    enactor_stats.filter_kernel_stats);                 // kernel stats

      	// Used for sorting edge_values: copy d_keys_array -> d_flag_array
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);

      	// Used for sorting eId	: copy d_keys_array -> d_temp_storage
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);

      	// Sort edges by keys
      	MergesortPairs(
          problem->data_slices[0]->d_keys_array,
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges,
      		mgpu::less<int>(), context);

      	// Sort edge_values by keys
      	MergesortPairs(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges,
      		mgpu::less<int>(), context);

      	// Sort eId by keys
      	MergesortPairs(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges,
      		mgpu::less<int>(), context);

      	if (debug_info)
      	{
      	  printf(":: Finding representatives (Sorted by keys Edges) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: Finding representatives (Sorted by keys Weights) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: Finding representatives (Sorted by keys eId) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	}

      	printf("----> 4.got new keys, edges, edge_values \n");

      	// Calculate the length of New RmFunctor - Vertex List
      	// d_row_offset already marked "-1" using VtxLenFunctor
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->nodes;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_row_offsets,
      		graph_slice->nodes);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,						 	//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      	// d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    	// d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_row_offsets
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_row_offsets,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->nodes);

      	printf("----> 5.update vertex list length, before:");
      	printf(" Q: %d, L: %d\n", frontier_attribute.queue_index, queue_length);

      	// Update vertex length of graph_slice
      	frontier_attribute.queue_index++;
      	if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, queue_length)) break;
      	graph_slice->nodes = queue_length;

      	printf("----> 6.update #nodes (If only ONE node left, break), Q: %d, L: %d\n",
      	  frontier_attribute.queue_index, graph_slice->nodes);

      	// Break the Loop if only 1 super_vertex left. -- moved to the end
      	if (graph_slice->nodes == 1) break;

      	// Assign row_offsets to vertex list
      	// Generate flag array for next iteration using markSegment kernel
      	util::markSegment<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);

      	//printf(":: Mark segment to generate Flag Array for next iteration ::");
      	//util::DisplayDeviceResults(problem->data_slices[0]->d_flag_array, graph_slice->edges);

      	// Generate row_offsets for next iteration using RowOFunctor - vertex mapping
      	// if d_flag_array[node] == 1:
      	// d_row_offsets[d_key[node]] == node
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RowOFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	//printf(":: row_offsets ::");
      	//util::DisplayDeviceResults(problem->data_slices[0]->d_row_offsets, graph_slice->nodes);

      	// Removing Duplicated Edges Between Supervertices
      	// Segmented Sort Edges, Weights and eId
      	// Copy d_edges to d_temp_storage to use for second sort
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges);

      	SegSortPairsFromIndices(
          problem->data_slices[0]->d_edges,
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges,
      		problem->data_slices[0]->d_row_offsets,
      		graph_slice->nodes, context);

      	SegSortPairsFromIndices(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges,
      		problem->data_slices[0]->d_row_offsets,
      		graph_slice->nodes, context);

      	if (debug_info)
      	{
      	  printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_edges) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_edge_vals) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_eId) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	}

      	// Generate new edge flag array using markSegment kernel
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edgeFlag,
      		problem->data_slices[0]->d_flag_array,
      		graph_slice->edges);

      	util::markSegment<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges);

      	// Segmented reduction: generate keys array using mgpu::scan
      	Scan<MgpuScanTypeInc>(
          (int*)problem->data_slices[0]->d_flag_array, graph_slice->edges,
      		(int)0, mgpu::plus<int>(), (int*)0, (int*)0,
      		(int*)problem->data_slices[0]->d_edgeKeys, context);

      	// Calculate edge_flag array using OrFunctor - vertex mapping
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, OrFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,              // Current graph traversal iteration
      	    frontier_attribute.queue_reset,    // reset queue counter
      	    frontier_attribute.queue_index,    // Current frontier queue counter index
      	    1,              // Number of gpu(s)
      	    frontier_attribute.queue_length,   // Number of element(s)
      	    NULL,           // d_done
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,           // d_pred_in_queue
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,     // Problem
      	    NULL,           // visited mask
      	    work_progress,  // work progress
      	    graph_slice->frontier_elements[frontier_attribute.selector],       // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],     // max_out_queue
      	    enactor_stats.filter_kernel_stats);                 // kernel stats

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

      	if (debug_info)
      	{
      	  printf(":: Edge Flag ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edgeFlag, graph_slice->edges);
      	  printf(":: Edge Keys ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edgeKeys, graph_slice->edges);
      	  printf(":: Old Keys ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, graph_slice->edges);
      	}

      	printf("----> 7.got edge flag, edge keys\n");

      	// Calculate the length of New Edge offset array
      	frontier_attribute.queue_index  = 0;
      	frontier_attribute.selector     = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, EdgeLenFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,              // Current graph traversal iteration
      	    frontier_attribute.queue_reset,    // reset queue counter
      	    frontier_attribute.queue_index,    // Current frontier queue counter index
      	    1,              // Number of gpu(s)
      	    frontier_attribute.queue_length,   // Number of element(s)
      	    NULL,           // d_done
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,           // d_pred_in_queue
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,     // Problem
      	    NULL,           // visited mask
      	    work_progress,  // work progress
      	    graph_slice->frontier_elements[frontier_attribute.selector],       // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],     // max_out_queue
      	    enactor_stats.filter_kernel_stats);                 // kernel stats

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

      	// Reduce Length using Rmfunctor
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_reset = true;

      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edge_offsets,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,							//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],	// d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    	// d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           	// max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         	// max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      	  "advance::Kernel failed", __FILE__, __LINE__))) break;

      	printf("----> 8.reduce length of edge offsets \n");
      	// Get edge offset length
      	// int edge_offsets_length;
      	// frontier_attribute.queue_index++;
      	// if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, queue_length)) break;
      	// edge_offsets_length = queue_length;
      	// printf("\n edge offsets length = %ld \n", queue_length);

      	// Generate New Edge offsets using EdgeOFunctor - vertex mapping
      	// if d_flag_array[node] == 1: d_edge_offsets[d_edgeKeys[node]] = node
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, EdgeOFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// printf(":: Edge_offsets ::");
      	// util::DisplayDeviceResults(problem->data_slices[0]->d_edge_offsets, edge_offsets_length);

      	printf("----> 9.got edge offsets\n");

      	// Segment Sort edge_values and eId using edge_offsets
      	SegSortPairsFromIndices(
          problem->data_slices[0]->d_edge_vals,
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges,
      		problem->data_slices[0]->d_edge_offsets,
      		graph_slice->nodes, context);

      	if (debug_info)
      	{
      	  printf(":: SegmentedSort using edge_offsets (d_edges) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: SegmentedSort using edge_offsets (d_edge_vals) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: SegmentedSort using edge_offsets (d_eId) ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	}

      	// Mark -1 to Edges needed to be removed using advance
      	// d_edges, d_edge_vals, d_keys_array, d_eId = -1 if (d_flag_array[node] == 0)
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, SuEdgeRmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL, //d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

      	if (debug_info)
        {
        	// mark -1 check
        	printf(":: -1 edges for current iteration ::");
        	util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
        	printf(":: -1 keys for current iteration ::");
        	util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, graph_slice->edges);
        	printf(":: -1 edge_values for current iteration ::");
        	util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
        	printf(":: -1 d_eId for current iteration ::");
        	util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
        }

      	// Reduce new edges, edge_values, eId, keys using Rmfunctor
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,							//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      	// d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edges
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edges,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;
      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_edge_vals
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_edge_vals,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;
      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
            0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// Copy back to d_keys_array
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_keys_array,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RmFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],      // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],    // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;
      	// Copy back to d_eId
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_eId,
      		graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
      		graph_slice->edges);

      	printf("----> 10.2ed update edge list length, before: Q: %d, L: %d \n", frontier_attribute.queue_index, queue_length);

      	// Update edge length of graph_slice, final edge length for next iteration
      	frontier_attribute.queue_index++;
      	if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, queue_length)) break;
      	graph_slice->edges = queue_length;
      	printf("----> 11.updated edges length, Q: %d, L: %d \n", frontier_attribute.queue_index, graph_slice->edges);

      	// TODO: Disordered on Midas
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_keys_array,
          graph_slice->edges);
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_keys_array,
          graph_slice->edges);

      	// TODO Disordered on Midas, Sort is a temp solution to ensure correctness
      	MergesortPairs(
          problem->data_slices[0]->d_temp_storage,
      		problem->data_slices[0]->d_eId,
      		graph_slice->edges,
      		mgpu::less<int>(), context);
      	MergesortPairs(
          problem->data_slices[0]->d_keys_array,
      		problem->data_slices[0]->d_edge_vals,
      		graph_slice->edges,
      		mgpu::less<int>(), context);
      	MergesortPairs(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges,
      		mgpu::less<int>(), context);

      	// copy back d_edges back to original column indices in graph_slice
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->d_column_indices,
      		problem->data_slices[0]->d_edges,
      		graph_slice->edges);

      	if (debug_info)
      	{
      	  printf(":: Final edges for current iteration ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
      	  printf(":: check graph_slice d_column_indices ::");
      	  util::DisplayDeviceResults(graph_slice->d_column_indices, graph_slice->edges);
      	  printf(":: Final keys for current iteration ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_keys_array, graph_slice->edges);
      	  printf(":: Final edge_values for current iteration ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_edge_vals, graph_slice->edges);
      	  printf(":: Final d_eId for current iteration ::");
      	  util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
      	}

      	// Finding final flag array for next iteration
      	// Generate edge flag array for next iteration using markSegment kernel
      	util::markSegment<<<128, 128>>>(
          problem->data_slices[0]->d_flag_array,
      		problem->data_slices[0]->d_keys_array,
      		graph_slice->edges);

      	//printf(":: Final d_flag_array for current iteration ::");
      	//util::DisplayDeviceResults(problem->data_slices[0]->d_flag_array, graph_slice->edges);

      	printf("----> 12.finish final edges, keys, edge_values and eId \n");

      	// Generate row_offsets for next iteration
      	frontier_attribute.queue_index = 0;
      	frontier_attribute.selector = 0;
      	frontier_attribute.queue_length = graph_slice->edges;
      	frontier_attribute.queue_reset = true;

      	// Fill in frontier queue
      	util::MemsetIdxKernel<<<128, 128>>>(
          graph_slice->frontier_queues.d_values[frontier_attribute.selector],
          graph_slice->edges);

      	gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MSTProblem, RowOFunctor>
      	  <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
      	    0,
      	    frontier_attribute.queue_reset,
      	    frontier_attribute.queue_index,
      	    1,
      	    frontier_attribute.queue_length,
      	    NULL,						//d_done,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector],    // d_in_queue
      	    NULL,
      	    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],  // d_out_queue
      	    data_slice,
      	    NULL,
      	    work_progress,
      	    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
      	    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
      	    enactor_stats.filter_kernel_stats);

      	if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
      		"advance::Kernel failed", __FILE__, __LINE__))) break;

      	// copy back to graph_slice d_row_offsets
      	util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->d_row_offsets,
      		problem->data_slices[0]->d_row_offsets,
      		graph_slice->nodes);

      	// Break the Loop if only 1 super_vertex left.
      	// if (graph_slice->nodes == 1) break;

      	printf("END OF ITERATION:%lld #NODES LEFT: %lld #EDGES LEFT: %lld\n",
      	  enactor_stats.iteration, graph_slice->nodes, graph_slice->edges);

      	// final selected edges for current iteration
      	// printf("\n Selected Edges for current iteration \n");
      	// util::DisplayDeviceResults(problem->data_slices[0]->d_mst_output, num_edges_origin);
*/
      	// final number of selected edges
      	int tmp_length = Reduce(
          problem->data_slices[0]->d_mst_output,
          num_edges_origin,
          context);
      	printf(" Number of selected edges so far - %d\n", tmp_length);

      	enactor_stats.iteration++;

      	if (INSTRUMENT || DEBUG)
      	  {
      	    if (retval = work_progress.GetQueueLength(
              frontier_attribute.queue_index, 
              frontier_attribute.queue_length)) break;
      	    enactor_stats.total_queued += frontier_attribute.queue_length;
      	    if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
      	    if (INSTRUMENT)
      	    {
      		    if (retval = enactor_stats.filter_kernel_stats.Accumulate(
                enactor_stats.filter_grid_size,
                enactor_stats.total_runtimes,
                enactor_stats.total_lifetimes)) break;
      	    }
      	    if (done[0] == 0) break; // check if done
      	    if (DEBUG) printf("\n %lld \n", (long long) enactor_stats.iteration);
      	  }

      	// Test
      	loop_limit++;
      	if (loop_limit >= 1) break;
      } // Recursive Loop
      /*
      // final number of selected edges
      int num_edges_select = Reduce(
        problem->data_slices[0]->d_mst_output, 
        num_edges_origin, context);
      printf("----> Number of edges selected - %d\n", num_edges_select);

      // mgpu reduce to calculate total edge_values
      // int total_weights_gpu = Reduce(
        problem->data_slices[0]->d_oriWeights, 
        num_edges_origin, context);
      // printf(" total edge_values gpu = %d\n", total_weights_gpu);
      */
      if (retval) break;

      // Check if any of the frontiers overflowed due to redundant expansion
      bool overflowed = false;
      if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
      if (overflowed)
      {
        retval = util::GRError(cudaErrorInvalidConfiguration,
      	  "Frontier queue overflow. Please increase queue-sizing factor.",
      	  __FILE__, __LINE__);
      	break;
      }

    }while(0);

  printf("\n GPU Minimum Spanning Tree Computation Complete. \n");

  return retval;
}

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief MST Enact kernel entry.
   *
   * @tparam MSTProblem MST Problem type. @see MSTProblem
   *
   * @param[in] moderngpu CudaContext context.
   * @param[in] problem Pointer to MSTProblem object.
   * @param[in] max_grid_size Max grid size for MST kernel calls.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  template <typename MSTProblem>
  cudaError_t Enact(
    CudaContext &context,
		MSTProblem  *problem,
		int         max_grid_size = 0)
  {
    if (this->cuda_props.device_sm_version >= 300)
    {
	     typedef gunrock::oprtr::filter::KernelPolicy<
    	  MSTProblem,         // Problem data type
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
    	  MSTProblem,         // Problem data type
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
        gunrock::oprtr::advance::LB>
    	  AdvanceKernelPolicy;

    	return EnactMST<
        AdvanceKernelPolicy,
        FilterKernelPolicy,
        MSTProblem>(context, problem, max_grid_size);
    }

    //to reduce compile time, get rid of other architecture for now
    //TODO: add all the kernelpolicy settings for all archs

    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
  }

  /** @} */

};

} // namespace mst
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: