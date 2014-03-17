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

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

#include <moderngpu.cuh>
#include <limits> // Used in Reduce By Key
#include <thrust/sort.h>

namespace gunrock {
namespace app {
namespace mst {

using namespace mgpu;

/**
 * @brief MST problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class MSTEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    int 		*vertex_flag;
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the MST search
     */
    long long	iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for MST kernel call. Must be called prior to each MST iteration.
     *
     * @param[in] problem MST Problem object which holds the graph data and MST problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {
            //initialize the host-mapped "done"
            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "MSTEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "MSTEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "MSTEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            //initialize runtime stats
            if (retval = edge_map_kernel_stats.Setup(edge_map_grid_size)) break;
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           =  0;
            total_runtimes      =  0;
            total_lifetimes     =  0;
            total_queued        =  0;
            done[0]             = -1;

            // graph slice
            // typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
	    
	    /*
            // Bind row-offsets and bitmask texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "MSTEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "MSTEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief MSTEnactor constructor
     */
    MSTEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
	vertex_flag(NULL), // TODO
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
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "MSTEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "MSTEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last MST search enacted.
     *
     * @param[out] total_queued Total queued elements in MST kernel running.
     * @param[out] search_depth Search depth of MST algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = this->iteration;

        avg_duty = (total_lifetimes > 0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam MSTProblem MST Problem type.
     *
     * @param[in] problem MSTProblem object.
     * @param[in] max_grid_size Max grid size for MST kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename MSTProblem>
    cudaError_t EnactMST(
    CudaContext                         &context,
    MSTProblem                          *problem,
    int                                 max_grid_size = 0)
    {
        typedef typename MSTProblem::SizeT      SizeT;
        typedef typename MSTProblem::VertexId   VertexId;

        typedef FLAGFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> FlagFunctor;
	
	typedef RCFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> RcFunctor;

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

	typedef RMFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> RmFunctor;

	typedef VLENFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> VlenFunctor;

	typedef RowOFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> RowOFunctor;

	typedef ELENFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> ElenFunctor;
	
	typedef EdgeOFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> EdgeOFunctor;

	typedef SuEdgeRmFunctor<
            VertexId,
            SizeT,
            VertexId,
            MSTProblem> SuEdgeRmFunctor;
	
	typedef ORFunctor<
	    VertexId,
            SizeT,
            VertexId,
            MSTProblem> OrFunctor;	

	cudaError_t retval = cudaSuccess;

        do {
	 
	
		/* Determine grid size(s) */
		int edge_map_occupancy = EdgeMapPolicy::CTA_OCCUPANCY;
		int edge_map_grid_size = MaxGridSize(edge_map_occupancy, max_grid_size);

		int vertex_map_occupancy = VertexMapPolicy::CTA_OCCUPANCY;
            	int vertex_map_grid_size = MaxGridSize(vertex_map_occupancy, max_grid_size);	
            	
		if (DEBUG) {
                	printf("MST edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                	printf("MST vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                	printf("Iteration, Edge map queue, Vertex map queue\n");
                	printf("0");
            	}

		/* Lazy initialization */
		if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

		/* Single-gpu graph slice */
            	typename MSTProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            	typename MSTProblem::DataSlice *data_slice = problem->d_data_slices[0];
	
		fflush(stdout);
                
		// Used for diaplaying d_selector 
		int originalEdgeLen;
		originalEdgeLen = graph_slice->edges;
		
		/* Step through MST iterations */
		// Recursive Loop
		// int tempIter = 0;
		while (graph_slice->nodes > 1) {
		
			// Bind row-offsets and bitmask texture
			cudaChannelFormatDesc	row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::GRError(cudaBindTexture(
				0,
				gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
				graph_slice->d_row_offsets,
				row_offsets_desc,
				(graph_slice->nodes + 1) * sizeof(SizeT)),
				"MSTEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
			
			// Used for mapping operators 
			SizeT 		queue_length;
			VertexId 	queue_index;	// Work queue index
			int 		selector;
			SizeT 		num_elements;
			bool 		queue_reset;
		
				
			// printf("\n:: Read In Row offsets ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_row_offsets, graph_slice->nodes);
			// printf(":: Previous Selector Boolean Array ::");
                        // util::DisplayDeviceResults(problem->data_slices[0]->d_selector, originalEdgeLen);		
			

			/* Vertex Map to mark segmentations */
			queue_index = 0;
                        selector    = 0;
                        num_elements = graph_slice->nodes;
                        queue_reset = true;
	
			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, FlagFunctor>
			<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,		// Current graph traversal iteration 
				queue_reset,	// reset queue counter 
				queue_index,	// Current frontier queue counter index
				1,		// Number of gpu(s) 
				num_elements,	// Number of element(s) 
				NULL,		// d_done 
				graph_slice->frontier_queues.d_keys[selector],      // d_in_queue 
				NULL, 	    	// d_pred_in_queue 
				graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
				data_slice,	// Problem 
				NULL, 		// visited mask 
				work_progress,	// work progress 
				graph_slice->frontier_elements[selector],       // max_in_queue 
				graph_slice->frontier_elements[selector^1],	// max_out_queue 
				this->vertex_map_kernel_stats);			// kernel stats 
			
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), 
				"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// Segmented reduction: generate keys array using mgpu::scan 
			Scan<MgpuScanTypeInc>((int*)problem->data_slices[0]->d_flag, graph_slice->edges, (int)0, 
				mgpu::plus<int>(), (int*)0, (int*)0, (int*)problem->data_slices[0]->d_keys, context);
			
			// Reduce By Keys using mgpu::ReduceByKey
			int numSegments;
			ReduceByKey(problem->data_slices[0]->d_keys, problem->data_slices[0]->d_weights, 
				graph_slice->edges,std::numeric_limits<int>::max(), mgpu::minimum<int>(), 
				mgpu::equal_to<int>(), problem->data_slices[0]->d_reducedKeys,
				problem->data_slices[0]->d_reducedWeights, &numSegments, (int*)0, context);	
			
			/*	
			printf(":: Flag Array ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_flag, graph_slice->edges); 
			printf(":: Keys array ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
			printf(":: Edge List ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			printf(":: Edge Weights ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);			
			*/

			/*
			printf(":: Reduced keys ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_reducedKeys, graph_slice->nodes);
			printf(":: Reduced weights ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_reducedWeights, graph_slice->nodes);
			*/
			
			/* Generate Successor Array using Edge Mapping */
			queue_index = 0;
			selector    = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;
			
			gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, MSTProblem, FlagFunctor>
			<<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
				queue_reset,
				queue_index,
				1,
				iteration,
				num_elements,
				d_done,
				graph_slice->frontier_queues.d_keys[selector],		// d_in_queue
				NULL,
				graph_slice->frontier_queues.d_keys[selector^1],	// d_out_queue
				graph_slice->d_column_indices,
				data_slice,
				this->work_progress,
				graph_slice->frontier_elements[selector],		// max_in_queue
				graph_slice->frontier_elements[selector^1],		// max_out_queue
				this->edge_map_kernel_stats);	
				
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), 
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// printf(":: Successor Array ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_successor, graph_slice->nodes);

			/* Remove Cycles using Vertex Mapping */
			queue_index  = 0;   
			selector     = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;
			
			gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, MSTProblem, RcFunctor>
			<<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
				queue_reset,
				queue_index,
				1,
				iteration,
				num_elements,
				d_done,
				graph_slice->frontier_queues.d_keys[selector],          // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_keys[selector^1],        // d_out_queue
				graph_slice->d_column_indices,
				data_slice,
				this->work_progress,
				graph_slice->frontier_elements[selector],               // max_in_queue
				graph_slice->frontier_elements[selector^1],             // max_out_queue
				this->edge_map_kernel_stats);

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// printf(":: Remove Cycles ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_successor, graph_slice->nodes);
			// printf(":: Selector Boolean Array ::");
                        // util::DisplayDeviceResults(problem->data_slices[0]->d_selector, originalEdgeLen);

			/* Point Jumpping to get d_represent array */	
			util::MemsetCopyVectorKernel<<<128,128>>>(problem->data_slices[0]->d_represent,
				problem->data_slices[0]->d_successor, graph_slice->nodes);

			// Using Vertex Mapping: PtrJumpFunctor
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;
			
			vertex_flag[0] = 0;
			while (!vertex_flag[0]) {
				vertex_flag[0] = 1;
				if (retval = util::GRError(cudaMemcpy(
					problem->data_slices[0]->d_vertex_flag,
					vertex_flag,
					sizeof(int),
					cudaMemcpyHostToDevice),
					"MSTProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;	
				gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, PtrJumpFunctor>
					<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				    	0,
				    	queue_reset,
				   	queue_index,
				    	1,
				    	num_elements,
				    	NULL,//d_done,
				    	graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
				    	NULL,
				    	graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
				    	data_slice,
				    	NULL,
				    	work_progress,
				    	graph_slice->frontier_elements[selector],           // max_in_queue
				    	graph_slice->frontier_elements[selector^1],         // max_out_queue
				    	this->vertex_map_kernel_stats);

				if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), 
					"vertex_map::Kernel First Pointer Jumping Round failed", __FILE__, __LINE__))) break;
				// Only reset once
				if (queue_reset) queue_reset = false;
				
				if (retval = util::GRError(cudaMemcpy(
					vertex_flag,
					problem->data_slices[0]->d_vertex_flag,
					sizeof(int),
					cudaMemcpyDeviceToHost),
					"MSTProblem cudaMemcpy d_vertex_flag to vertex_flag failed", __FILE__, __LINE__)) return retval;
				// Check if done
				if (vertex_flag[0]) break;
			}
			
			// printf(":: Pointer Jumping (d_represent) ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_represent, graph_slice->nodes);
			
			/* Assigning ids to supervertices */
			util::MemsetCopyVectorKernel<<<128,128>>>(problem->data_slices[0]->d_superVertex,
				problem->data_slices[0]->d_represent, graph_slice->nodes);
			
			// Fill in the d_nodes : 0, 1, 2, ... , nodes 
			util::MemsetIdxKernel<<<128, 128>>>(problem->data_slices[0]->d_nodes, graph_slice->nodes);
	
			// Mergesort pairs.
			MergesortPairs(problem->data_slices[0]->d_superVertex, 
				problem->data_slices[0]->d_nodes, graph_slice->nodes, mgpu::less<int>(), context);
			
			/*
			printf(":: Super-Vertices Ids ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_superVertex, graph_slice->nodes);
			printf(":: Sorted nodes according to supervertices (d_nodes) ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_nodes, graph_slice->nodes);		
			*/

			// Scan of the flag assigns new supervertex ids
			util::markSegment<<<128, 128>>>(problem->data_slices[0]->d_Cflag, 
				problem->data_slices[0]->d_superVertex, graph_slice->nodes);
			
			// printf(":: New Super-vertex Ids (C Flag) ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_Cflag, graph_slice->nodes);	
			
			/* Calculate New the length of New Vertex List */
			queue_index  = 0;
			selector     = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;
			
			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, VlenFunctor>
			<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,              // Current graph traversal iteration
				queue_reset,    // reset queue counter
				queue_index,    // Current frontier queue counter index
				1,              // Number of gpu(s)
				num_elements,   // Number of element(s)
				NULL,           // d_done
				graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
				NULL,           // d_pred_in_queue
				graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
				data_slice,     // Problem
				NULL,           // visited mask
				work_progress,  // work progress
				graph_slice->frontier_elements[selector],       // max_in_queue
				graph_slice->frontier_elements[selector^1],     // max_out_queue
				this->vertex_map_kernel_stats);                 // kernel stats

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// Segmented reduction: generate keys array using mgpu::scan 
			Scan<MgpuScanTypeInc>((int*)problem->data_slices[0]->d_Cflag, graph_slice->nodes, (int)0, 
				mgpu::plus<int>(), (int*)0, (int*)0, (int*)problem->data_slices[0]->d_Ckeys, context);
			
			// printf(":: C Keys Array ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_Ckeys, graph_slice->nodes);
	
			/* Edge removal using edge mapping */
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;

			gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, MSTProblem, EdgeRmFunctor>
			<<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
				queue_reset,
				queue_index,
				1,
				iteration,
				num_elements,
				d_done,
				graph_slice->frontier_queues.d_keys[selector],          // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_keys[selector^1],        // d_out_queue
				graph_slice->d_column_indices,
				data_slice,
				this->work_progress,
				graph_slice->frontier_elements[selector],               // max_in_queue
				graph_slice->frontier_elements[selector^1],             // max_out_queue
				this->edge_map_kernel_stats);
			
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			/*
			// Debug Mark -1 means need to be removed
			printf(":: Edge Removal (edges) ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			printf(":: Edge Removal (keys) ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
			printf(":: Edge Removal (weights) ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);
			printf(":: Edge Removal (d_eId) ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
			printf(":: Edge Removal (d_edgeFlag) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edgeFlag, graph_slice->edges);
			*/

			// Mergesort pairs.
			MergesortPairs(problem->data_slices[0]->d_nodes,
				problem->data_slices[0]->d_Ckeys, graph_slice->nodes, mgpu::less<int>(), context);
			
			/*
			printf(":: Sorted Ckeys matching nodes ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_Ckeys, graph_slice->nodes);			
			printf(":: Ordered Edge List ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			*/

			/* Generate new edge list, keys array and weights list using vertex mapping */	
			// Edge mapping to select edges
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;
			
			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], 
				problem->data_slices[0]->d_edges, graph_slice->edges);
			
			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,
				queue_reset,
				queue_index,
				1,
				num_elements,
				NULL,	// d_done,
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,
				NULL,
				work_progress,
				graph_slice->frontier_elements[selector],           // max_in_queue
				graph_slice->frontier_elements[selector^1],         // max_out_queue
				this->vertex_map_kernel_stats);
			
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_edges
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_edges,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);
			
			// Edge mapping to select keys
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;
			
			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
				problem->data_slices[0]->d_keys, graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,
				queue_reset,
				queue_index,
				1,
				num_elements,
				NULL,//d_done,
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,
				NULL,
				work_progress,
				graph_slice->frontier_elements[selector],           // max_in_queue
				graph_slice->frontier_elements[selector^1],         // max_out_queue
				this->vertex_map_kernel_stats);
			
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to keys
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keys,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);

			// Edge mapping to select weights
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;
			
			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
				problem->data_slices[0]->d_weights, graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,
				queue_reset,
				queue_index,
				1,
				num_elements,
				NULL,//d_done,
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,
				NULL,
				work_progress,
				graph_slice->frontier_elements[selector],           // max_in_queue
				graph_slice->frontier_elements[selector^1],         // max_out_queue
				this->vertex_map_kernel_stats);
		
			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_weights	
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_weights,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);
			
			// Edge mapping to select eId
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;

			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
				problem->data_slices[0]->d_eId, graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,
				queue_reset,
				queue_index,
				1,
				num_elements,
				NULL,       // d_done,
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,
				NULL,
				work_progress,
				graph_slice->frontier_elements[selector],           // max_in_queue
				graph_slice->frontier_elements[selector^1],         // max_out_queue
				this->vertex_map_kernel_stats);

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// Copy back to d_edges
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_eId,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);	
			
			// Edge mapping to reduce the lenght of edgeFlag
                        queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;

                        // Fill in frontier queue
                        util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_edgeFlag, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,       // d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

                        // Copy back to d_edgeFlag
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_edgeFlag,
                                graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);
			

			// Generate Flag array with new length
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;

			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
				problem->data_slices[0]->d_flag, graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				    0,
				    queue_reset,
				    queue_index,
				    1,
				    num_elements,
				    NULL,       // d_done,
				    graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				    NULL,
				    graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				    data_slice,
				    NULL,
				    work_progress,
				    graph_slice->frontier_elements[selector],           // max_in_queue
				    graph_slice->frontier_elements[selector^1],         // max_out_queue
				    this->vertex_map_kernel_stats);

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_flag
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_flag,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);
				
			// Update length of edges in graph_slice 
			queue_index++;	
			if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
			graph_slice->edges = queue_length;
			
			// TODO: Disordered on Midas. Sort to make sure correctness 
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keysCopy,
				problem->data_slices[0]->d_keys, graph_slice->edges);
			MergesortPairs(problem->data_slices[0]->d_keysCopy,
				problem->data_slices[0]->d_eId, graph_slice->edges, mgpu::less<int>(), context);
			MergesortPairs(problem->data_slices[0]->d_keys,
				problem->data_slices[0]->d_weights, graph_slice->edges, mgpu::less<int>(), context);

			// TODO: Disordered on Midas
			// util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			// util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
			// util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);	
			// util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);

			/* Finding representative for keys and edges using Vertex Mapping */
			// Removing edges inside each super-vertex
			queue_index  = 0;
			selector     = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;
			
			// Fill in frontier queue
			util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);
			
			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, EdgeRmFunctor>
			<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,              // Current graph traversal iteration
				queue_reset,    // reset queue counter
				queue_index,    // Current frontier queue counter index
				1,              // Number of gpu(s)
				num_elements,   // Number of element(s)
				NULL,           // d_done
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,           // d_pred_in_queue
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,     // Problem
				NULL,           // visited mask
				work_progress,  // work progress
				graph_slice->frontier_elements[selector],       // max_in_queue
				graph_slice->frontier_elements[selector^1],     // max_out_queue
				this->vertex_map_kernel_stats);                 // kernel stats
			
			// Used for sorting weights 	
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_flag,
				problem->data_slices[0]->d_keys, graph_slice->edges);
			
			// Used for sorting eId
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keysCopy,
				problem->data_slices[0]->d_keys, graph_slice->edges);
			
			// Mergesort pairs. Sort edges by keys
			MergesortPairs(problem->data_slices[0]->d_keys,
				problem->data_slices[0]->d_edges, graph_slice->edges, mgpu::less<int>(), context);
			
			// printf(":: Finding representative: NEW Sorted by keys Edges ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			 
			// Sort weights by keys	
			MergesortPairs(problem->data_slices[0]->d_flag,
				problem->data_slices[0]->d_weights, graph_slice->edges, mgpu::less<int>(), context);
			
			// printf(":: Finding representative: NEW Sorted by keys Weights ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);	
	
			// Sort eId by keys
			MergesortPairs(problem->data_slices[0]->d_keysCopy,
				problem->data_slices[0]->d_eId, graph_slice->edges, mgpu::less<int>(), context);
			
			// printf(":: Finding representative: NEW Sorted by keys eId ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);

			/* Generate the length of NEW Vertex List */	
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->nodes;
			queue_reset = true;

			// Fill in frontier queue
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
				problem->data_slices[0]->d_row_offsets, graph_slice->nodes);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
			<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				0,
				queue_reset,
				queue_index,
				1,
				num_elements,
				NULL,//d_done,
				graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				NULL,
				graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				data_slice,
				NULL,
				work_progress,
				graph_slice->frontier_elements[selector],           // max_in_queue
				graph_slice->frontier_elements[selector^1],         // max_out_queue
				this->vertex_map_kernel_stats);

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_row_offsets
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_row_offsets,
				graph_slice->frontier_queues.d_values[selector^1], graph_slice->nodes);
			
			/* Update length of graph_slice */
			queue_index++;
			if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
			graph_slice->nodes = queue_length;
				
			// Break the Loop if only ONE super-vertex left.
			if (graph_slice->nodes == 1) break;

			// Assign row_offsets to vertex list 
			// Generate new flag array using markSegment kernel 
			util::markSegment<<<128, 128>>>(problem->data_slices[0]->d_flag,
				problem->data_slices[0]->d_keys, graph_slice->edges);
			
			// printf(":: markSegment using d_keys => Flag array ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_flag, graph_slice->edges);

			// Generate New Row_Offsets 
			queue_index = 0;
			selector = 0;
			num_elements = graph_slice->edges;
			queue_reset = true;

			// Fill in frontier queue
			util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RowOFunctor>
				<<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
				    0,
				    queue_reset,
				    queue_index,
				    1,
				    num_elements,
				    NULL,//d_done,
				    graph_slice->frontier_queues.d_values[selector],      // d_in_queue
				    NULL,
				    graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
				    data_slice,
				    NULL,
				    work_progress,
				    graph_slice->frontier_elements[selector],           // max_in_queue
				    graph_slice->frontier_elements[selector^1],         // max_out_queue
				    this->vertex_map_kernel_stats);

			if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
				"edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// printf(":: Row_offsets Functor: NEW row_offsets ::");
			// util::DisplayDeviceResults(problem->data_slices[0]->d_row_offsets, graph_slice->nodes);
			
			/* Removing Duplicated Edges Between Supervertices */	
			// Segmented Sort Edges, Weights and eId
			// Copy d_edges to d_keysCopy to use for second sort
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keysCopy,
                                problem->data_slices[0]->d_edges, graph_slice->edges);

			SegSortPairsFromIndices(problem->data_slices[0]->d_edges, problem->data_slices[0]->d_weights, 
				graph_slice->edges, problem->data_slices[0]->d_row_offsets, graph_slice->nodes, context);
						
			SegSortPairsFromIndices(problem->data_slices[0]->d_keysCopy, problem->data_slices[0]->d_eId,
                                graph_slice->edges, problem->data_slices[0]->d_row_offsets, graph_slice->nodes, context);
			
			/*	
			printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_edges) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
			printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_weights) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);
			printf(":: Removing Duplicated Edges Between Supervertices After SegmentedSort (d_eId) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
			*/	
	
			// Generate new edge flag array using markSegment kernel
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_edgeFlag,
                                problem->data_slices[0]->d_flag, graph_slice->edges);
			util::markSegment<<<128, 128>>>(problem->data_slices[0]->d_flag,
                                problem->data_slices[0]->d_edges, graph_slice->edges);
						
			// Segmented reduction: generate keys array using mgpu::scan 
                        Scan<MgpuScanTypeInc>((int*)problem->data_slices[0]->d_flag, graph_slice->edges, (int)0, 
				mgpu::plus<int>(), (int*)0, (int*)0, (int*)problem->data_slices[0]->d_edgeKeys, context);	

			queue_index  = 0;
                        selector     = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;

                        util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, OrFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,              // Current graph traversal iteration
                                queue_reset,    // reset queue counter
                                queue_index,    // Current frontier queue counter index
                                1,              // Number of gpu(s)
                                num_elements,   // Number of element(s)
                                NULL,           // d_done
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,           // d_pred_in_queue
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,     // Problem
                                NULL,           // visited mask
                                work_progress,  // work progress
                                graph_slice->frontier_elements[selector],       // max_in_queue
                                graph_slice->frontier_elements[selector^1],     // max_out_queue
                                this->vertex_map_kernel_stats);                 // kernel stats

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;		
			
			/*	
			printf(":: Edge Flag ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edgeFlag, graph_slice->edges);
			printf(":: Edge Keys ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edgeKeys, graph_slice->edges);
			printf(":: Old Keys ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
			*/

			// Calculate the length of New Edge offset array
			queue_index  = 0;
                        selector     = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;
			
			util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, ElenFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,              // Current graph traversal iteration
                                queue_reset,    // reset queue counter
                                queue_index,    // Current frontier queue counter index
                                1,              // Number of gpu(s)
                                num_elements,   // Number of element(s)
                                NULL,           // d_done
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,           // d_pred_in_queue
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,     // Problem
                                NULL,           // visited mask
                                work_progress,  // work progress
                                graph_slice->frontier_elements[selector],       // max_in_queue
                                graph_slice->frontier_elements[selector^1],     // max_out_queue
                                this->vertex_map_kernel_stats);                 // kernel stats

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// Reduce Length using Rmfunctor
			queue_index = 0;
                        selector = 0;
                        queue_reset = true;
			
			util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_edge_offsets, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,							//d_done,
                                graph_slice->frontier_queues.d_values[selector],	// d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    	// d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           	// max_in_queue
                                graph_slice->frontier_elements[selector^1],         	// max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

			// Get edge offset length
			int edgeOffsetsLen;
			queue_index++;
			if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                        edgeOffsetsLen = queue_length;
			
			// Generate New Edge offsets 
                        queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;

                        // Fill in frontier queue
                        util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, EdgeOFunctor>
                                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

                        // printf(":: Edge_offsets ::");
                        // util::DisplayDeviceResults(problem->data_slices[0]->d_edge_offsets, edgeOffsetsLen);
			
			// Segment Sort weights and eId using edge_offsets
			SegSortPairsFromIndices(problem->data_slices[0]->d_weights, problem->data_slices[0]->d_eId,
                                graph_slice->edges, problem->data_slices[0]->d_edge_offsets, graph_slice->nodes, context);
		
			/*
			printf(":: SegmentedSort using edge_offsets (d_edges) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);	
			printf(":: SegmentedSort using edge_offsets (d_weights) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);
                        printf(":: SegmentedSort using edge_offsets (d_eId) ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);	
			*/

			// Mark -1 to Edges needed to be removed using edge mapping
                        queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;
			
			util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

			gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, SuEdgeRmFunctor>
                                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// mark -1 check 
			/*
			printf(":: -1 edges for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
                        printf(":: -1 keys for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
                        printf(":: -1 weights for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);
                        printf(":: -1 d_eId for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
			*/

			// Reduce new edges, weights, eId, keys using Rmfunctor
                        queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
			queue_reset = true;
			// Fill in frontier queue
                        util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_edges, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,							//d_done,
                                graph_slice->frontier_queues.d_values[selector],      	// d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_edges
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_edges,
                                graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);

			queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;
                        // Fill in frontier queue
                        util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_weights, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_weights
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_weights,
                                graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);			

			queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;
                        // Fill in frontier queue
                        util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_keys, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			
			// Copy back to d_keys
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keys,
                                graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);
			
			queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;
                        // Fill in frontier queue
                        util::MemsetCopyVectorKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector],
                                problem->data_slices[0]->d_eId, graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RmFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
			// Copy back to d_eId
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_eId,
                                graph_slice->frontier_queues.d_values[selector^1], graph_slice->edges);


			// Update length of graph_slice 
                        queue_index++;
                        if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                        graph_slice->edges = queue_length;
			
			// Sort eId to make sure correctness
                        util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_keysCopy,
                                problem->data_slices[0]->d_keys, graph_slice->edges);
			util::MemsetCopyVectorKernel<<<128, 128>>>(problem->data_slices[0]->d_flag,
                                problem->data_slices[0]->d_keys, graph_slice->edges);

			// TODO Disordered in Midas, Sort is a temp solution to ensure correctness
                        MergesortPairs(problem->data_slices[0]->d_keysCopy,
                                problem->data_slices[0]->d_eId, graph_slice->edges, mgpu::less<int>(), context);
                        MergesortPairs(problem->data_slices[0]->d_keys,
                                problem->data_slices[0]->d_weights, graph_slice->edges, mgpu::less<int>(), context);
			MergesortPairs(problem->data_slices[0]->d_flag,
                                problem->data_slices[0]->d_edges, graph_slice->edges, mgpu::less<int>(), context);

			/*
			printf(":: Final edges for current iteration ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_edges, graph_slice->edges);
                        printf(":: Final keys for current iteration ::");
			util::DisplayDeviceResults(problem->data_slices[0]->d_keys, graph_slice->edges);
                        printf(":: Final weights for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_weights, graph_slice->edges);
                        printf(":: Final d_eId for current iteration ::");
                        util::DisplayDeviceResults(problem->data_slices[0]->d_eId, graph_slice->edges);
			*/

			// Finding final Flag array for next iteration
			// Generate new edge flag array using markSegment kernel
                        util::markSegment<<<128, 128>>>(problem->data_slices[0]->d_flag,
                                problem->data_slices[0]->d_keys, graph_slice->edges);

			// printf(":: Final d_flag for current iteration ::");
                        // util::DisplayDeviceResults(problem->data_slices[0]->d_flag, graph_slice->edges);
			
			/* Generate New row_offsets */
                        queue_index = 0;
                        selector = 0;
                        num_elements = graph_slice->edges;
                        queue_reset = true;

                        // Fill in frontier queue
                        util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->edges);

                        gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, MSTProblem, RowOFunctor>
                                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                    0,
                                    queue_reset,
                                    queue_index,
                                    1,
                                    num_elements,
                                    NULL,						//d_done,
                                    graph_slice->frontier_queues.d_values[selector],    // d_in_queue
                                    NULL,
                                    graph_slice->frontier_queues.d_values[selector^1],  // d_out_queue
                                    data_slice,
                                    NULL,
                                    work_progress,
                                    graph_slice->frontier_elements[selector],           // max_in_queue
                                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                                    this->vertex_map_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                                "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

                        // printf(":: Final row_offsets for current iteration ::");
                        // util::DisplayDeviceResults(problem->data_slices[0]->d_row_offsets, graph_slice->nodes);
			
			printf("\n Iteration No.%d", iteration);
			printf("\n Number of Nodes Left : %d \n Number of Edges Left : %d \n ", graph_slice->nodes, graph_slice->edges);
			
			iteration++;
			
			if (INSTRUMENT || DEBUG) {
			    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
			    total_queued += queue_length;
			    if (DEBUG) printf(", %lld", (long long) queue_length);
			    if (INSTRUMENT) {
				if (retval = vertex_map_kernel_stats.Accumulate(
				    vertex_map_grid_size,
				    total_runtimes,
				    total_lifetimes)) break;
			    }
			    if (done[0] == 0) break; // check if done
			    if (DEBUG) printf("\n %lld \n", (long long) iteration);
			}
			
			// Test
			// tempIter++;
			// if (tempIter == 30) break;
            	} // Recursive Loop
	
			

		if (retval) break;

            	/* Check if any of the frontiers overflowed due to redundant expansion */
            	bool overflowed = false;
            	if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            	if (overflowed) {
                	retval = util::GRError(cudaErrorInvalidConfiguration,
			"Frontier queue overflow. Please increase queue-sizing factor.", __FILE__, __LINE__);
                	break;
            	}
	
	}while(0);


        if (DEBUG) printf("\n GPU MST Done.\n");
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
     * @param[in] problem Pointer to MSTProblem object.
     * @param[in] src Source node for MST.
     * @param[in] max_grid_size Max grid size for MST kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename MSTProblem>
    cudaError_t Enact(
        CudaContext                     &context,
        MSTProblem                      *problem,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
            	MSTProblem,                         // Problem data type
            	300,                                // CUDA_ARCH
            	INSTRUMENT,                         // INSTRUMENT
            	0,                                  // SATURATION QUIT
            	true,                               // DEQUEUE_PROBLEM_SIZE
            	8,                                  // MIN_CTA_OCCUPANCY
            	8,                                  // LOG_THREADS
            	1,                                  // LOG_LOAD_VEC_SIZE
            	0,                                  // LOG_LOADS_PER_TILE
            	5,                                  // LOG_RAKING_THREADS
            	5,                                  // END_BITMASK_CULL
            	8>                                  // LOG_SCHEDULE_GRANULARITY
                	VertexMapPolicy;

            typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                MSTProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                0,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            	    // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                   	EdgeMapPolicy;

            return EnactMST<EdgeMapPolicy, VertexMapPolicy, MSTProblem>(
                    context, problem, max_grid_size);
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
