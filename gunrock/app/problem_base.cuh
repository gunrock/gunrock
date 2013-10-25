// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base struct for all the application types
 */

#pragma once

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>

#include <vector>

namespace gunrock {
namespace app {

/**
 * Enumeration of global frontier queue configurations
 */

enum FrontierType {
    VERTEX_FRONTIERS,		// O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,			// O(m) ping-pong global edge frontiers
	MIXED_FRONTIERS 		// O(n) global vertex frontier, O(m) global edge frontier
};


/**
 * Type of problem
 */
template <
	typename 	_VertexId,						                        // Type of signed integer to use as vertex id (e.g., uint32)
	typename 	_SizeT,							                        // Type of unsigned integer to use for array indexing (e.g., uint32)
    util::io::ld::CacheModifier _QUEUE_READ_MODIFIER,					// Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer, where util::io::ld::cg is req'd for fused-iteration implementations incorporating software global barriers.
	util::io::ld::CacheModifier _COLUMN_READ_MODIFIER,					// Load instruction cache-modifier for reading CSR column-indices
    util::io::ld::CacheModifier _EDGE_VALUES_READ_MODIFIER,             // Load instruction cache-modifier for reading edge values
	util::io::ld::CacheModifier _ROW_OFFSET_ALIGNED_READ_MODIFIER,		// Load instruction cache-modifier for reading CSR row-offsets (when 8-byte aligned)
	util::io::ld::CacheModifier _ROW_OFFSET_UNALIGNED_READ_MODIFIER,	// Load instruction cache-modifier for reading CSR row-offsets (when 4-byte aligned)
	util::io::st::CacheModifier _QUEUE_WRITE_MODIFIER>					// Store instruction cache-modifier for writing outgoign frontier vertex-ids. Valid on SM2.0 or newer, where util::io::st::cg is req'd for fused-iteration implementations incorporating software global barriers.

struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;

    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER 					= _QUEUE_READ_MODIFIER;
	static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER 					= _COLUMN_READ_MODIFIER;
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = _EDGE_VALUES_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER 		= _ROW_OFFSET_ALIGNED_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER 	= _ROW_OFFSET_UNALIGNED_READ_MODIFIER;
	static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER 					= _QUEUE_WRITE_MODIFIER;

	/**
	 * Graph Slice which contains common graph structural data
	 * and input/output queue
	 */
	struct GraphSlice
	{
	    //Slice Index
	    int             index;

	    SizeT           *d_row_offsets;             // CSR format row offset on device memory
	    VertexId        *d_column_indices;          // CSR format column indices on device memory

	    //Frontier queues. Used to track working frontier.
	    util::DoubleBuffer<VertexId, VertexId>      frontier_queues;
	    SizeT                                       frontier_elements[2];

	    //Number of nodes and edges in slice
	    VertexId        nodes;
	    SizeT           edges;

	    //CUDA stream to use for processing the slice
	    cudaStream_t    stream;

	    /**
	     * Constructor
	     */
	    GraphSlice(int index, cudaStream_t stream) :
	        index(index),
	        d_row_offsets(NULL),
	        d_column_indices(NULL),
	        nodes(0),
	        edges(0),
	        stream(stream)
	    {
	        // Initialize double buffer frontier queue lengths
	        for (int i = 0; i < 2; ++i)
	        {
	            frontier_elements[i] = 0;
	        }
	    }

	    /**
	     * Destructor
	     */
	    virtual ~GraphSlice()
	    {
	        // Set device (use slice index)
	        util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

	        // Free pointers
	        if (d_row_offsets)      util::GRError(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
	        if (d_column_indices)   util::GRError(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
	        for (int i = 0; i < 2; ++i) {
	            if (frontier_queues.d_keys[i])      util::GRError(cudaFree(frontier_queues.d_keys[i]), "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__);
	            if (frontier_queues.d_values[i])    util::GRError(cudaFree(frontier_queues.d_values[i]), "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__);
	        }

	        // Destroy stream
	        if (stream) {
	            util::GRError(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
	        }
	    }
	};

	// Members

	// Number of GPUs to be sliced over
	int                 num_gpus;

	// Size of the graph
	SizeT               nodes;
	SizeT               edges;

	// Set of graph slices (one for each GPU)
	GraphSlice**        graph_slices;

	// Methods
	
	/**
	 * Constructor
	 */
	ProblemBase() :
	    num_gpus(0),
	    nodes(0),
	    edges(0)
	    {}
	
	/**
	 * Destructor
	 */
	virtual ~ProblemBase()
	{
	    // Cleanup graph slices on the heap
	    for (int i = 0; i < num_gpus; ++i)
	    {
	        delete graph_slices[i];
	    }
	    delete[] graph_slices;
	}

	/**
	 * Returns index of the gpu that owns the neighbor list of
	 * the specified vertex
	 */
	template <typename VertexId>
	int GpuIndex(VertexId vertex)
	{
	    if (num_gpus <= 1) {
	        
	        // Special case for only one GPU, which may be set as with
	        // an ordinal other than 0.
	        return graph_slices[0]->index;
	    } else {
	        return vertex % num_gpus;
	    }
	}

	/**
	 * Returns the row within a gpu's GraphSlice row_offsets vector
	 * for the specified vertex
	 */
	template <typename VertexId>
	VertexId GraphSliceRow(VertexId vertex)
	{
	    return vertex / num_gpus;
	}

	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t Init(
	    bool        stream_from_host,       // Only meaningful for single-GPU
	    SizeT       nodes,
	    SizeT       edges,
	    SizeT       *h_row_offsets,
	    VertexId    *h_column_indices,
	    int         num_gpus)
	{
	    cudaError_t retval      = cudaSuccess;
	    this->nodes             = nodes;
	    this->edges             = edges;
	    this->num_gpus          = num_gpus;

	    do {
	        graph_slices = new GraphSlice*[num_gpus];
	        if (num_gpus <= 1) {

	            // Create a single graph slice for the currently-set gpu
	            int gpu;
	            if (retval = util::GRError(cudaGetDevice(&gpu), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
	            graph_slices[0] = new GraphSlice(gpu, 0);
	            graph_slices[0]->nodes = nodes;
	            graph_slices[0]->edges = edges;

	            if (stream_from_host) {

	                // Map the pinned graph pointers into device pointers
	                if (retval = util::GRError(cudaHostGetDevicePointer(
	                                (void **)&graph_slices[0]->d_row_offsets,
	                                (void *) h_row_offsets, 0),
	                            "ProblemBase cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaHostGetDevicePointer(
	                                (void **)&graph_slices[0]->d_column_indices,
	                                (void *) h_column_indices, 0),
	                            "ProblemBase cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;
	            } else {

	                // Allocate and initialize d_row_offsets
	                if (retval = util::GRError(cudaMalloc(
	                    (void**)&graph_slices[0]->d_row_offsets,
	                    (graph_slices[0]->nodes+1) * sizeof(SizeT)),
	                    "ProblemBase cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

	                if (retval = util::GRError(cudaMemcpy(
	                    graph_slices[0]->d_row_offsets,
	                    h_row_offsets,
	                    (graph_slices[0]->nodes+1) * sizeof(SizeT),
	                    cudaMemcpyHostToDevice),
	                    "ProblemBase cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
                    
                    // Allocate and initialize d_column_indices
	                if (retval = util::GRError(cudaMalloc(
	                    (void**)&graph_slices[0]->d_column_indices,
	                    graph_slices[0]->edges * sizeof(VertexId)),
	                    "ProblemBase cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

	                if (retval = util::GRError(cudaMemcpy(
	                    graph_slices[0]->d_column_indices,
	                    h_column_indices,
	                    graph_slices[0]->edges * sizeof(VertexId),
	                    cudaMemcpyHostToDevice),
	                    "ProblemBase cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;



	            } //end if(stream_from_host)
	        } else {
	            //TODO: multiple GPU graph slices
	        }//end if(num_gpu<=1)
	    } while (0);

	    return retval;
	}

	/**
	 * Performs any initialization work needed for ProblemBase. Must be called
	 * prior to each search
	 */
	cudaError_t Reset(
	    FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
	    double queue_sizing)            // Size scaling factor for work queue allocation
	    {
	        cudaError_t retval = cudaSuccess;

	        for (int gpu = 0; gpu < num_gpus; ++gpu) {

	            // Set device
	            if (retval = util::GRError(cudaSetDevice(graph_slices[gpu]->index),
	                        "ProblemBase cudaSetDevice failed", __FILE__, __LINE__)) return retval;


			    //
			    // Allocate frontier queues if necessary
			    //

			    // Determine frontier queue sizes
			    SizeT new_frontier_elements[2] = {0,0};

			    switch (frontier_type) {
			        case VERTEX_FRONTIERS :
				        // O(n) ping-pong global vertex frontiers
				        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
				        new_frontier_elements[1] = new_frontier_elements[0];
				        break;

			        case EDGE_FRONTIERS :
				        // O(m) ping-pong global edge frontiers
				        new_frontier_elements[0] = double(graph_slices[gpu]->edges) * queue_sizing;
				        new_frontier_elements[1] = new_frontier_elements[0];
				        break;

			        case MIXED_FRONTIERS :
				        // O(n) global vertex frontier, O(m) global edge frontier
				        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
				        new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
				        break;

			        }

			    // Iterate through global frontier queue setups
			    for (int i = 0; i < 2; i++) {

				    // Allocate frontier queue if not big enough
				    if (graph_slices[gpu]->frontier_elements[i] < new_frontier_elements[i]) {

					    // Free if previously allocated
					    if (graph_slices[gpu]->frontier_queues.d_keys[i]) {
						    if (retval = util::GRError(cudaFree(
							                graph_slices[gpu]->frontier_queues.d_keys[i]),
								        "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
					    }

					    graph_slices[gpu]->frontier_elements[i] = new_frontier_elements[i];

					    if (retval = util::GRError(cudaMalloc(
						                (void**) &graph_slices[gpu]->frontier_queues.d_keys[i],
						                graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
							        "ProblemBase cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
				    }
			    }
	        }
            
	        return retval;
	    }
};

} // namespace app
} // namespace gunrock

