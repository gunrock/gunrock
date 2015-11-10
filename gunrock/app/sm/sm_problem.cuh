// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sm_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief Problem structure stores device-side vectors
 * @tparam _VertexId Type use as vertex id (e.g., uint32)
 * @tparam _SizeT    Type use for array indexing. (e.g., uint32)
 * @tparam _Value    Type use for computed value.
 */
template<typename _VertexId, typename _SizeT, typename _Value>
struct SMProblem : ProblemBase<_VertexId, _SizeT, false> {
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT;
    typedef _Value    Value;

    static const bool MARK_PREDECESSORS  = false;
    static const bool ENABLE_IDEMPOTENCE = false;

    /**
     * @brief Data slice structure which contains problem specific data.
     */
    struct DataSlice {
        // device storage arrays
        VertexId *d_query_labels;  /** < Used for input query graph labels */
	VertexId *d_data_labels;   /** < Used for input data graph labels */
	VertexId *d_labels;        /** < Used for input query node indices */    
	VertexId *d_edge_labels;   /** < Used for input query edge indices */
	VertexId *d_row_offsets;   /** < Used for query row offsets     */
	VertexId *d_column_indices;/** < Used for query column indices  */ 
	Value    *d_edge_weights;  /** < Used for storing edge weights    */
	SizeT    *d_data_degrees;  /** < Used for input data graph degrees */
	SizeT 	 *d_query_degrees; /** < Used for input query graph degrees */
	SizeT 	 *d_temp_keys;     /** < Used for candidate matrix row keys */
	SizeT    nodes_data;       /** < Used for number of data nodes  */
	SizeT	 nodes_query;      /** < Used for number of query nodes */
	bool     *d_c_set;         /** < Used for candidate set  */
        
    };

    // Number of GPUs to be sliced over
    int       num_gpus;

    // Size of the query graph
    SizeT     nodes_query;
    SizeT     edges_query;
    
    // Size of the data graph
    SizeT     nodes_data;
    SizeT     edges_data;
    unsigned int num_matches;

    // Set of data  slices (one for each GPU)
    DataSlice **data_slices;

    // putting structure on device while keeping the SoA structure
    DataSlice **d_data_slices;

    // device index for each data slice
    int       *gpu_idx;

    /**
     * @brief Default constructor
     */
    SMProblem(): nodes_query(0), nodes_data(0), edges_query(0), edges_data(0), num_gpus(0),num_matches(0) {}

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    SMProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph_query,
                  const Csr<VertexId, Value, SizeT> &graph_data,
		  VertexId *h_query_labels,
	 	  VertexId *h_data_labels,
		  VertexId *h_index,
		  VertexId *h_edge_index,
		  bool* h_c_set,
                  int   num_gpus) :
        num_gpus(num_gpus) {
	Init(stream_from_host, graph_query, graph_data, h_query_labels, h_data_labels, h_index, h_edge_index, num_gpus);
    }

    /**
     * @brief Default destructor
     */
    ~SMProblem() {
        for (int i = 0; i < num_gpus; ++i) {
            if (util::GRError(
                    cudaSetDevice(gpu_idx[i]),
                    "~Problem cudaSetDevice failed",
                    __FILE__, __LINE__)) break;
            if (data_slices[i]->d_query_labels) {
                util::GRError(cudaFree(data_slices[i]->d_query_labels),
                              "GpuSlice cudaFree d_query_labels failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_data_labels) {
                util::GRError(cudaFree(data_slices[i]->d_data_labels),
                              "GpuSlice cudaFree d_data_labels failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_labels) {
                util::GRError(cudaFree(data_slices[i]->d_labels),
                              "GpuSlice cudaFree d_labels failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_edge_labels) {
                util::GRError(cudaFree(data_slices[i]->d_edge_labels),
                              "GpuSlice cudaFree d_edge_labels failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_edge_weights) {
                util::GRError(cudaFree(data_slices[i]->d_edge_weights),
                              "GpuSlice cudaFree d_edge_weights failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_row_offsets) {
                util::GRError(cudaFree(data_slices[i]->d_row_offsets),
                              "GpuSlice cudaFree d_row_offsets failed",
                              __FILE__, __LINE__);
            }
            if (data_slices[i]->d_column_indices) {
                util::GRError(cudaFree(data_slices[i]->d_column_indices),
                              "GpuSlice cudaFree d_column_indices failed",
                              __FILE__, __LINE__);
            }
	    if (data_slices[i]->d_c_set) {
		util::GRError(cudaFree(data_slices[i]->d_c_set), 
			      "GpuSlice cudaFree d_c_set fail",
			      __FILE__, __LINE__);
	    }
	    if (data_slices[i]->d_query_degrees) {
		util::GRError(cudaFree(data_slices[i]->d_query_degrees), 
			      "GpuSlice cudaFree d_query_degrees fail",
			      __FILE__, __LINE__);
	    }
	    if (data_slices[i]->d_data_degrees) {
		util::GRError(cudaFree(data_slices[i]->d_data_degrees), 
			      "GpuSlice cudaFree d_data_degrees fail",
			      __FILE__, __LINE__);
	    }
	    if (data_slices[i]->d_temp_keys) {
		util::GRError(cudaFree(data_slices[i]->d_temp_keys), 
			      "GpuSlice cudaFree d_temp_keys fail",
			      __FILE__, __LINE__);
	    }
            if (d_data_slices[i]) {
                util::GRError(cudaFree(d_data_slices[i]),
                              "GpuSlice cudaFree data_slices failed",
                              __FILE__, __LINE__);
            }
        }
        if (d_data_slices) delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy results computed on the GPU back to host-side vectors.
     * @param[out] h_labels
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(bool* h_c_set) {
        cudaError_t retval = cudaSuccess;

        do {
	    // Set device
            if (num_gpus == 1) {
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                                  "Problem cudaSetDevice failed",
                                  __FILE__, __LINE__)) break;

                if (retval = util::GRError(
                        cudaMemcpy(h_c_set,
                                   data_slices[0]->d_c_set,
                                   sizeof(bool) * nodes_query * nodes_data,
                                   cudaMemcpyDeviceToHost),
                        "Problem cudaMemcpy d_c_set failed",
                        __FILE__, __LINE__)) break;

                // TODO: code to extract other results here

            } else {
                // multi-GPU extension code
            }
        } while (0);

        return retval;
    }

    /**
     * @brief Problem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Init(
        bool  stream_from_host,  // only meaningful for single-GPU
        Csr<VertexId, Value, SizeT> &graph_query,
        const Csr<VertexId, Value, SizeT> &graph_data,
	VertexId *h_query_labels,
	VertexId *h_data_labels,
 	VertexId *h_index,
	VertexId *h_edge_index,
        int   _num_gpus) {
        num_gpus = _num_gpus;
        nodes_query = graph_query.nodes;
        edges_query = graph_query.edges;
        nodes_data  = graph_data.nodes;
        edges_data  = graph_data.edges;
        VertexId *h_row_offsets = graph_data.row_offsets;
        VertexId *h_column_indices = graph_data.column_indices;
	VertexId *h_query_rowoffsets = graph_query.row_offsets;
	VertexId *h_query_columnindices = graph_query.column_indices;
	SizeT *h_query_degrees = (SizeT*) malloc(sizeof(SizeT) * graph_query.nodes);
 	graph_query.GetNodeDegree(h_query_degrees);
	SizeT *h_temp_keys = (SizeT*) malloc(sizeof(SizeT) * graph_query.nodes);
	for(int i=0; i<graph_query.nodes; i++) h_temp_keys[i] = i*graph_data.nodes;

        ProblemBase<_VertexId, _SizeT, false>::Init(
            stream_from_host,
            nodes_data,  
            edges_data, 
            h_row_offsets,
            h_column_indices,
            NULL,
            NULL,
            num_gpus);

	
        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];
        
	//copy query graph labels and data graph labels from input
	
        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));

                // create a single data slice for the currently-set GPU
                int gpu;
                if (retval = util::GRError(
                        cudaGetDevice(&gpu),
                        "Problem cudaGetDevice failed",
                        __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_slices[0],
                                   sizeof(DataSlice)),
                        "Problem cudaMalloc d_data_slices failed",
                        __FILE__, __LINE__)) return retval;

                // create SoA on device
                VertexId *d_query_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_query_labels,
                                   nodes_query * sizeof(VertexId)),
                        "Problem cudaMalloc d_query_labels failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_query_labels, h_query_labels,
				   nodes_query * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_query_labels failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_query_labels = d_query_labels;

                VertexId *d_data_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_labels,
                                   nodes_data * sizeof(VertexId)),
                        "Problem cudaMalloc d_data_labels failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_data_labels, h_data_labels,
				   nodes_data * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_data_labels failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_data_labels = d_data_labels;

                VertexId *d_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_labels,
                                   nodes_query * sizeof(VertexId)),
                        "Problem cudaMalloc d_labels failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_labels, h_index,
				   nodes_query * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_labels failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_labels = d_labels;

                VertexId *d_edge_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_edge_labels,
                                   edges_query * sizeof(VertexId)),
                        "Problem cudaMalloc d_edge_labels failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_edge_labels, h_edge_index,
				   edges_query * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_edge_labels failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_edge_labels = d_edge_labels;

                VertexId *d_row_offsets;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_row_offsets,
                                   (nodes_query+1) * sizeof(VertexId)),
                        "Problem cudaMalloc d_row_offsets failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_row_offsets, h_query_rowoffsets,
				   (nodes_query+1) * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_row_offsets failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_row_offsets = d_row_offsets;


                VertexId *d_column_indices;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_column_indices,
                                   edges_query * sizeof(VertexId)),
                        "Problem cudaMalloc d_column_indices failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_column_indices, h_query_columnindices,
				   edges_query * sizeof(VertexId),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_column_indices failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_column_indices = d_column_indices;

                Value *d_edge_weights;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_edge_weights,
                                   edges_data * sizeof(Value)),
                        "Problem cudaMalloc d_edge_weights failed",
                        __FILE__, __LINE__)) return retval;
		data_slices[0]->d_edge_weights = d_edge_weights;

                bool *d_c_set;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_c_set,
                                   nodes_query * nodes_data * sizeof(bool)),
                        "Problem cudaMalloc d_c_set failed",
                        __FILE__, __LINE__)) return retval;

	        util::MemsetKernel<<<128, 128>>>(d_c_set, (bool)0,
					     nodes_query * nodes_data);
                data_slices[0]->d_c_set = d_c_set;

                SizeT *d_data_degrees;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_degrees,
                                   nodes_data * sizeof(SizeT)),
                        "Problem cudaMalloc d_data_degrees failed",
                        __FILE__, __LINE__)) return retval;

                data_slices[0]->d_data_degrees = d_data_degrees;


                SizeT *d_query_degrees;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_query_degrees,
                                   nodes_query * sizeof(SizeT)),
                        "Problem cudaMalloc d_query_degrees failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_query_degrees, h_query_degrees,
				   nodes_query * sizeof(SizeT),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_query_degrees failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_query_degrees = d_query_degrees;
                
		SizeT *d_temp_keys;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_temp_keys,
                                   nodes_query * sizeof(SizeT)),
                        "Problem cudaMalloc d_temp_keys failed",
                        __FILE__, __LINE__)) return retval;

		if (retval = util::GRError(
			cudaMemcpy(d_temp_keys, h_temp_keys,
				   nodes_query * sizeof(SizeT),
				   cudaMemcpyHostToDevice),
			"Problem cudaMemcpy d_temp_keys failed",
			__FILE__, __LINE__)) return retval;

                data_slices[0]->d_temp_keys = d_temp_keys;

		data_slices[0]->nodes_data = nodes_data;
		data_slices[0]->nodes_query = nodes_query;

            }
            // add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for primitive
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation
     *  \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Reset(
        FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
        double queue_sizing) {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.

        typedef ProblemBase<_VertexId, _SizeT, false> BaseProblem;

        // load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(gpu_idx[gpu]),
                    "Problem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

            // allocate output labels if necessary
	    util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_c_set, (bool)0,
					     nodes_query * nodes_data);

            // TODO: code to for other allocations here

            if (retval = util::GRError(
                    cudaMemcpy(d_data_slices[gpu],
                               data_slices[gpu],
                               sizeof(DataSlice),
                               cudaMemcpyHostToDevice),
                    "Problem cudaMemcpy data_slices to d_data_slices failed",
                    __FILE__, __LINE__)) return retval;
        }

        // TODO: fill in the initial input_queue for problem
        // e.g., put every vertex in frontier queue
        util::MemsetIdxKernel<<<128, 128>>>(
            BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes_data);

	util::MemsetMadVectorKernel<<<128, 128>>>(
	    data_slices[0]->d_data_degrees,
	    BaseProblem::graph_slices[0]->d_row_offsets,
	    &BaseProblem::graph_slices[0]->d_row_offsets[1], -1, nodes_data);

        return retval;
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
