// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mst_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace mst {

/**
 * @brief PageRank Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing MST value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 *
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value,
	bool        _USE_DOUBLE_BUFFER>
struct MSTProblem : ProblemBase<_VertexId, _SizeT, _USE_DOUBLE_BUFFER> // USE_DOUBLE_BUFFER = false
{

    typedef _VertexId	VertexId;
    typedef _SizeT		SizeT;
    typedef _Value		Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
    * @brief Data slice structure which contains MST problem specific data.
    */
    struct DataSlice
    {
        // device storage arrays
        SizeT		*d_labels;
        SizeT   	*d_edges;		/* Edge list */
        Value   	*d_weights;	   	/* Store weights for each edge */
        Value       *d_oriWeights;  /* Original weight list used for total weight calculate */
        Value       *d_reducedWeights;	/* Store minimum weights for each vertex */
        SizeT		*d_flag;		/* Flag array to mark segments */
        SizeT		*d_keys;		/* Used for segmented reduction */	
        SizeT       *d_keysCopy;            /* Used for pair soring (temp) */
        VertexId	*d_reducedKeys;		/* Used for segmented reduction */
        VertexId	*d_successor;		/* Used for storing dest.ID that have min_weights */ 
        VertexId    *d_represent;           /* Used for storing represetatives for each successor */
        VertexId    *d_superVertex;         /* Used for storing supervertex in order */
        VertexId	*d_row_offsets;		/* Used for accessing row_offsets */
        VertexId	*d_edgeId;		/* Used for storing vid of edges have min_weights */
        int         *d_vertex_flag;         /* Finish flag for per-vertex kernels in MST algorithm */
        VertexId	*d_nodes;		/* Used for nodes vid */
        VertexId	*d_Cflag;		/* Used for a scan of the flag assigns new supervertex Ids */
        VertexId	*d_Ckeys;		/* Used for storing new keys array */
        SizeT		*d_eId;			/* Used for keeping current iteration edge Ids to select edges */
		int		    *d_selector;		/* Used for recording selected edges for MST */
        VertexId	*d_edge_offsets;	/* Used for removing edges between supervertices */		
        SizeT 		*d_edgeFlag;		/* Used for removing edges between supervertices */
        SizeT		*d_edgeKeys;		/* Used for removing edges between supervertices */	
	};
	
    // Members
   
    // Number of GPUs to be sliced over
    int                 num_gpus;
	
	// Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Selector, which d_rank array stores the final page rank?
    SizeT               selector;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;

    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    int                 *gpu_idx;

    // Methods

	/**
	 * @brief MSTProblem default constructor
	 */

	MSTProblem():
	nodes(0),
	edges(0),
	num_gpus(0) {}

	/**
	 * @brief MSTProblem constructor
	 *
	 * @param[in] stream_from_host Whether to stream data from host.
	 * @param[in] graph Reference to the CSR graph object we process on.
	 * @param[in] num_gpus Number of the GPUs used.
	 */
	MSTProblem(
        bool        stream_from_host,       // Only meaningful for single-GPU
		const       Csr<VertexId, Value, SizeT> &graph,
		int         num_gpus) :
	    num_gpus(num_gpus)
	{
	    Init(
		stream_from_host,
		graph,
		num_gpus);
	}

	/**
	 * @brief MSTProblem default destructor
	 */
	~MSTProblem()
	{
	    for (int i = 0; i < num_gpus; ++i)
	    {
		if (util::GRError(cudaSetDevice(gpu_idx[i]),
		    	"~MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
		
		if (data_slices[i]->d_edges)            
            util::GRError(cudaFree(data_slices[i]->d_edges),
                "GpuSlice cudaFree d_edges failed", __FILE__, __LINE__);
		if (data_slices[i]->d_weights)	        
            util::GRError(cudaFree(data_slices[i]->d_weights), 
		        "GpuSlice cudaFree d_weights failed", __FILE__, __LINE__);
	 	if (data_slices[i]->d_oriWeights)
            util::GRError(cudaFree(data_slices[i]->d_oriWeights),
                "GpuSlice cudaFree d_oriWeights failed", __FILE__, __LINE__);
        if (data_slices[i]->d_reducedWeights)  	
            util::GRError(cudaFree(data_slices[i]->d_reducedWeights),
                "GpuSlice cudaFree d_reducedWeights failed", __FILE__, __LINE__);
		if (data_slices[i]->d_flag)		        
            util::GRError(cudaFree(data_slices[i]->d_flag),
                "GpuSlice cudaFree d_flag failed", __FILE__, __LINE__);
		if (data_slices[i]->d_keys)		        
            util::GRError(cudaFree(data_slices[i]->d_keys),
                "GpuSlice cudaFree d_keys failed", __FILE__, __LINE__);
		if (data_slices[i]->d_keysCopy)         
            util::GRError(cudaFree(data_slices[i]->d_keysCopy),
                "GpuSlice cudaFree d_keysCopy failed", __FILE__, __LINE__);
		if (data_slices[i]->d_reducedKeys)	    
            util::GRError(cudaFree(data_slices[i]->d_reducedKeys),
                "GpuSlice cudaFree d_reducedKeys failed", __FILE__, __LINE__);
		if (data_slices[i]->d_successor)	    
            util::GRError(cudaFree(data_slices[i]->d_successor),
                "GpuSlice cudaFree d_successor failed", __FILE__, __LINE__);	
		if (data_slices[i]->d_represent)	    
            util::GRError(cudaFree(data_slices[i]->d_represent),
                "GpuSlice cudaFree d_represent failed", __FILE__, __LINE__);
		if (data_slices[i]->d_superVertex)  	
            util::GRError(cudaFree(data_slices[i]->d_superVertex),
			    "GpuSlice cudaFree d_superVertex failed", __FILE__, __LINE__);
		if (data_slices[i]->d_row_offsets)	    
            util::GRError(cudaFree(data_slices[i]->d_row_offsets),
                "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
		if (data_slices[i]->d_edgeId)		
            util::GRError(cudaFree(data_slices[i]->d_edgeId),
                "GpuSlice cudaFree d_edgeId failed", __FILE__, __LINE__);
		if (data_slices[i]->d_vertex_flag)	
            util::GRError(cudaFree(data_slices[i]->d_vertex_flag),
			    "GpuSlice cudaFree d_vertex_flag failed", __FILE__, __LINE__);
		if (data_slices[i]->d_nodes)      	
            util::GRError(cudaFree(data_slices[i]->d_nodes),
                "GpuSlice cudaFree d_nodes failed", __FILE__, __LINE__);
		if (data_slices[i]->d_Cflag)            
            util::GRError(cudaFree(data_slices[i]->d_Cflag),
                "GpuSlice cudaFree d_Cflag failed", __FILE__, __LINE__);
		if (data_slices[i]->d_Ckeys)            
            util::GRError(cudaFree(data_slices[i]->d_Ckeys),
                "GpuSlice cudaFree d_Ckeys failed", __FILE__, __LINE__);	
		if (data_slices[i]->d_eId)             	
            util::GRError(cudaFree(data_slices[i]->d_eId),
                "GpuSlice cudaFree d_eId failed", __FILE__, __LINE__);
		if (data_slices[i]->d_selector)         
            util::GRError(cudaFree(data_slices[i]->d_selector),
                "GpuSlice cudaFree d_selector failed", __FILE__, __LINE__);
		if (data_slices[i]->d_edge_offsets) 	
            util::GRError(cudaFree(data_slices[i]->d_edge_offsets),
                "GpuSlice cudaFree d_edge_offsets failed", __FILE__, __LINE__);
		if (data_slices[i]->d_edgeFlag)         
            util::GRError(cudaFree(data_slices[i]->d_edgeFlag),
                "GpuSlice cudaFree d_edgeFlag failed", __FILE__, __LINE__);
		if (data_slices[i]->d_edgeKeys)     	
            util::GRError(cudaFree(data_slices[i]->d_edgeKeys),
                "GpuSlice cudaFree d_edgeKeys failed", __FILE__, __LINE__);

		if (d_data_slices[i])
            util::GRError(cudaFree(d_data_slices[i]), 
			    "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
	    }
	    if (d_data_slices)  delete[] d_data_slices;
	    if (data_slices) delete[] data_slices;
	}

	/**
	 * \addtogroup PublicInterface
	 * @{
	 */

	/**
	 * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
	 *
	 * @param[out] h_rank host-side vector to store page rank values.
	 *
	 *\return cudaError_t object which indicates the success of all CUDA function calls.
	 */
	//TODO: write extract function
	cudaError_t Extract(SizeT *h_flag)
	{
	    cudaError_t retval = cudaSuccess;

	    do {
		if (num_gpus == 1) {

		    // Set device
		    if (util::GRError(cudaSetDevice(gpu_idx[0]),
				"MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

		    if (retval = util::GRError(cudaMemcpy(
			    h_flag,
				data_slices[0]->d_flag,
				sizeof(SizeT) * edges,
				cudaMemcpyDeviceToHost),
			    "MSTProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
		} else {
		    // TODO: multi-GPU extract result
		} //end if (data_slices.size() ==1)
	    } while(0);

	    return cudaSuccess;
	}

	/**
	 * @brief MSTProblem initialization
	 *
	 * @param[in] stream_from_host Whether to stream data from host.
	 * @param[in] graph Reference to the CSR graph object we process on. @see Csr
	 * @param[in] _num_gpus Number of the GPUs used.
	 *
	 * \return cudaError_t object which indicates the success of all CUDA function calls.
	 */
	cudaError_t Init(
		bool    stream_from_host,       // Only meaningful for single-GPU
		const   Csr<VertexId, Value, SizeT> &graph,
		int     _num_gpus)
	{
	    num_gpus = _num_gpus;
	    nodes = graph.nodes;
	    edges = graph.edges;
	    VertexId *h_row_offsets = graph.row_offsets;
	    VertexId *h_column_indices = graph.column_indices;
		ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>::Init(stream_from_host,
			nodes,
			edges,
			h_row_offsets,
			h_column_indices,
			NULL,
			NULL,
			num_gpus);

	    // No data in DataSlice needs to be copied from host

	    /**
	     * Allocate output labels/preds
	     */
	    cudaError_t retval = cudaSuccess;
	    data_slices = new DataSlice*[num_gpus];
	    d_data_slices = new DataSlice*[num_gpus];

	    do {
		if (num_gpus <= 1) 
        {
			gpu_idx = (int*)malloc(sizeof(int));
		    	// Create a single data slice for the currently-set gpu
		    	int gpu;
		    	if (retval = util::GRError(cudaGetDevice(&gpu), 
				    "MSTProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
		    	gpu_idx[0] = gpu;

		    	data_slices[0] = new DataSlice;
		    	if (retval = util::GRError(cudaMalloc(
				    (void**)&d_data_slices[0],
				    sizeof(DataSlice)),
				    "MSTProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

		    	// Create SoA on device
		    	SizeT	*d_edges;        // edges list
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_edges,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_edges Failed", __FILE__, __LINE__)) return retval;
			    if (retval = util::GRError(cudaMemcpy(
                    d_edges,
                    graph.column_indices,
				    edges * sizeof(SizeT),
                    cudaMemcpyHostToDevice),
                    "ProblemBase cudaMemcpy d_edges failed", __FILE__, __LINE__)) return retval;			
			    data_slices[0]->d_edges = d_edges;

			    Value	*d_weights;
		    	if (retval = util::GRError(cudaMalloc(
			    	(void**)&d_weights,
			    	edges * sizeof(Value)),
				    "MSTProblem cudaMalloc d_weights failed", __FILE__, __LINE__)) return retval;
			    if (retval = util::GRError(cudaMemcpy(
                    d_weights,
                    graph.edge_values,
                    edges * sizeof(Value),
                    cudaMemcpyHostToDevice),
                    "ProblemBase cudaMemcpy d_weights failed", __FILE__, __LINE__)) return retval;
			    data_slices[0]->d_weights = d_weights;

                Value   *d_oriWeights;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_oriWeights,
                    edges * sizeof(Value)),
                    "MSTProblem cudaMalloc d_oriWeights failed", __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMemcpy(
                    d_oriWeights,
                    graph.edge_values,
                    edges * sizeof(Value),
                    cudaMemcpyHostToDevice),
                    "ProblemBase cudaMemcpy d_oriWeights failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_oriWeights = d_oriWeights;

			    Value	*d_reducedWeights;	// Same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_reducedWeights,
                    nodes * sizeof(Value)),
                    "MSTProblem cudaMalloc d_reducedWeights failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_reducedWeights = d_reducedWeights;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_reducedWeights, 0, nodes);

                SizeT	*d_flag;	// Flag array has the same size as #edges 
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_flag,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_flag Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_flag = d_flag;		
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_flag, 0, edges);	

                SizeT   *d_keys;        // Keys array has the same size as #edges 
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_keys,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_keys Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_keys = d_keys;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_keys, 0, edges);

			    SizeT   *d_keysCopy;        // Keys array has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_keysCopy,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_keysCopy Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_keysCopy = d_keysCopy;
		        util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_keysCopy, 0, edges);

		    	VertexId	*d_reducedKeys;   // Reduced Keys array has the same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_reducedKeys,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_reducedKeys Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_reducedKeys = d_reducedKeys;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_reducedKeys, 0, nodes);
		
                VertexId	*d_successor;	// Successor array has the same size as #nodes 
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_successor,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_successor Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_successor = d_successor;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_successor, 0, nodes);	     	
			
			    VertexId        *d_represent;   // Successor array has the same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_represent,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_represent Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_represent = d_represent;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_represent, 0, nodes);				
	
			    VertexId        *d_superVertex;   // Supervertex array has the same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_superVertex,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_superVertex Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_superVertex = d_superVertex;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_superVertex, 0, nodes);

			    VertexId        *d_row_offsets; // Row offsets has the same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_row_offsets,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_row_offsets Failed", __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMemcpy(
                    d_row_offsets,
                    graph.row_offsets,
                    nodes * sizeof(VertexId),
                    cudaMemcpyHostToDevice),
                    "ProblemBase cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) return retval;
		    	data_slices[0]->d_row_offsets = d_row_offsets;

			    VertexId        *d_edgeId;   // EdgeID array has the same size as #nodes
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_edgeId,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_edgeId Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_edgeId = d_edgeId;
		 	    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_edgeId, 0, nodes);

			    int *d_vertex_flag;
                if (retval = util::GRError(cudaMalloc(
                     (void**)&d_vertex_flag,
                     sizeof(int)),
                     "MSTProblem cudaMalloc d_vertex_flag failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_vertex_flag = d_vertex_flag;

			    VertexId	*d_nodes;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_nodes,
                    nodes * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_nodes failed", __FILE__, __LINE__)) return retval;
			    data_slices[0]->d_nodes = d_nodes;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_nodes, 0, nodes);			
			
			    SizeT           *d_Cflag;        // CFlag array has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_Cflag,
                    nodes * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_Cflag Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_Cflag = d_Cflag;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_Cflag, 0, nodes);			

			    SizeT           *d_Ckeys;        // CFlag array has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_Ckeys,
                    nodes * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_Ckeys Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_Ckeys = d_Ckeys;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_Ckeys, 0, nodes);

			    SizeT           *d_eId;        // has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_eId,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_eId Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_eId = d_eId;
			    util::MemsetIdxKernel<<<128, 128>>>(data_slices[0]->d_eId, edges);

			    int   *d_selector;        // has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_selector,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_selector Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_selector = d_selector;
			    util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_selector, 0, edges);
				
			    VertexId        *d_edge_offsets;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_edge_offsets,
                    edges * sizeof(VertexId)),
                    "MSTProblem cudaMalloc d_edge_offsets failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_edge_offsets = d_edge_offsets;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_edge_offsets, 0, edges);

			    SizeT           *d_edgeFlag;        // has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_edgeFlag,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_edgeFlag Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_edgeFlag = d_edgeFlag;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_edgeFlag, 0, edges);

			    SizeT           *d_edgeKeys;        // has the same size as #edges
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_edgeKeys,
                    edges * sizeof(SizeT)),
                    "MSTProblem cudaMalloc d_edgeKeys Failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_edgeKeys = d_edgeKeys;
                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_edgeKeys, 0, edges);

			    data_slices[0]->d_labels = NULL;
		}
		//TODO: add multi-GPU allocation code
	    } while (0);

	    return retval;
	}

	/**
	 *  @brief Performs any initialization work needed for MST problem type. Must be called prior to each MST iteration.
	 *
	 *  @param[in] src Source node for one MST computing pass.
	 *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
	 * 
	 *  \return cudaError_t object which indicates the success of all CUDA function calls.
	 */
	cudaError_t Reset(
		FrontierType frontier_type)             // The frontier type (i.e., edge/vertex/mixed)
	{
	    typedef ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER> BaseProblem;
	    //load ProblemBase Reset
	    BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0

	    cudaError_t retval = cudaSuccess;

	    for (int gpu = 0; gpu < num_gpus; ++gpu) {
		// Set device
		if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
	        "MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

		// Allocate output if necessary
		if (!data_slices[gpu]->d_edges) 
        {
            SizeT	*d_edges;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_edges,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_edges failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_edges = d_edges;
		}

		if (!data_slices[gpu]->d_weights) 
        {
		    Value    *d_weights;
		    if (retval = util::GRError(cudaMalloc(
			    (void**)&d_weights,
			    edges * sizeof(Value)),
		    "MSTProblem cudaMalloc d_weights failed", __FILE__, __LINE__)) return retval;
	    	data_slices[gpu]->d_weights = d_weights;
		} 
	
        if (!data_slices[gpu]->d_oriWeights)
        {
            Value    *d_oriWeights;
            if (retval = util::GRError(cudaMalloc(
            (void**)&d_oriWeights,
            edges * sizeof(Value)),
            "MSTProblem cudaMalloc d_oriWeights failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_oriWeights = d_oriWeights;
        }

		if(!data_slices[gpu]->d_reducedWeights) 
        {
            Value    *d_reducedWeights;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_reducedWeights,
                nodes * sizeof(Value)),
                "MSTProblem cudaMalloc d_reducedWeights failed", __FILE__, __LINE__)) return retval;
		    data_slices[gpu]->d_reducedWeights = d_reducedWeights;
        }

		if (!data_slices[gpu]->d_flag) 
        {
            SizeT	*d_flag;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_flag,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_flag Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_flag = d_flag;	
		}
		
		if (!data_slices[gpu]->d_keys) 
        {
            SizeT   *d_keys;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_keys,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_keys Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_keys = d_keys;
        }
	
		if (!data_slices[gpu]->d_keysCopy) 
        {
            SizeT   *d_keysCopy;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_keysCopy,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_keysCopy Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_keysCopy = d_keysCopy;
        }

		if (!data_slices[gpu]->d_successor) 
        {
            VertexId	*d_successor;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_successor,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_successor Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_successor = d_successor;
		}
	
		if (!data_slices[gpu]->d_represent) 
        {
            VertexId        *d_represent;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_represent,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_represent Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_represent = d_represent;
        }
	
		if (!data_slices[gpu]->d_superVertex) 
        {
            VertexId        *d_superVertex;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_superVertex,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_superVertex Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_superVertex = d_superVertex;
        }

		if (!data_slices[gpu]->d_reducedKeys) 
        {
            VertexId        *d_reducedKeys;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_reducedKeys,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_reducedKeys Failed", __FILE__, __LINE__)) return retval;
		    data_slices[gpu]->d_reducedKeys = d_reducedKeys;
        }
		
		if (!data_slices[gpu]->d_row_offsets) 
        {
            VertexId     	*d_row_offsets;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_row_offsets,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_row_offsets Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_row_offsets = d_row_offsets;
        }
		
		if (!data_slices[gpu]->d_edgeId) 
        {
            VertexId        *d_edgeId;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_edgeId,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_edgeId Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_edgeId = d_edgeId;
        }
		
		int *vertex_flag = new int;
		// Allocate vertex_flag if necessary
        if (!data_slices[gpu]->d_vertex_flag) 
        {
            int    *d_vertex_flag;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_vertex_flag,
                sizeof(int)),
                "MSTProblem cudaMalloc d_vertex_flag failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_vertex_flag = d_vertex_flag;
        }
        vertex_flag[0] = 1;
        if (retval = util::GRError(cudaMemcpy(
            data_slices[gpu]->d_vertex_flag,
            vertex_flag,
            sizeof(int),
            cudaMemcpyHostToDevice),
            "MSTProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;
        delete vertex_flag;
		
		if (!data_slices[gpu]->d_nodes) 
        {
            VertexId        *d_nodes;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_nodes,
                nodes * sizeof(VertexId)),
                "MSTProblem cudaMalloc d_nodes Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_nodes = d_nodes;
       }
	   
	   if (!data_slices[gpu]->d_Cflag) 
       {
            SizeT   *d_Cflag;
            if(retval = util::GRError(cudaMalloc(
                (void**)&d_Cflag,
                nodes * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_Cflag Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_Cflag = d_Cflag;
       }
	   
	   if (!data_slices[gpu]->d_Ckeys) 
       {
            SizeT   *d_Ckeys;
            if(retval = util::GRError(cudaMalloc(
                (void**)&d_Ckeys,
                nodes * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_Ckeys Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_Ckeys = d_Ckeys;
       }

	   if (!data_slices[gpu]->d_eId) 
       {
            SizeT   *d_eId;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_eId,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_eId Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_eId = d_eId;
       }	   
	   
	   if (!data_slices[gpu]->d_selector) 
       {
            int   *d_selector;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_selector,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_selector Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_selector = d_selector;
       }	   

	   if (!data_slices[gpu]->d_edge_offsets) 
       {
            VertexId   *d_edge_offsets;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_edge_offsets,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_edge_offsets Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_edge_offsets = d_edge_offsets;
       }
	
	   if (!data_slices[gpu]->d_edgeFlag) 
       {
            SizeT   *d_edgeFlag;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_edgeFlag,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_edgeFlag Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_edgeFlag = d_edgeFlag;
       }

	   if (!data_slices[gpu]->d_edgeKeys) 
       {
            SizeT   *d_edgeKeys;
            if (retval = util::GRError(cudaMalloc(
                (void**)&d_edgeKeys,
                edges * sizeof(SizeT)),
                "MSTProblem cudaMalloc d_edgeKeys Failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->d_edgeKeys = d_edgeKeys;
        }

		data_slices[gpu]->d_labels = NULL;

		if (retval = util::GRError(cudaMemcpy(
            d_data_slices[gpu],
            data_slices[gpu],
            sizeof(DataSlice),
            cudaMemcpyHostToDevice),
			"MSTProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

	    }

	    // Fillin the initial input_queue for MST problem, this needs to be modified
	    // in multi-GPU scene

	    // Put every vertex in frontier queue
	    util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);
	    return retval;
	}

	/** @} */

};

} //namespace mst
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
