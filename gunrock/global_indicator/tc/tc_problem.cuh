// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file tc_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

using namespace gunrock::app;

namespace gunrock {
namespace global_indicator {
namespace tc {

/**
 * @brief Problem structure stores device-side vectors
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value    Type of float or double to use for computing value.
 */
template<typename VertexId, 
	 typename SizeT, 
	 typename Value>
struct TCProblem : ProblemBase<VertexId, SizeT, Value,
	false,
	false,
	false,
	false,	// _ENABLE_BACKWARD
	false,	// _KEEP_ORDER
	false>  // _KEEP_NODE_NUM
{

    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
     
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>{
        // device storage arrays
	util::Array1D<SizeT, VertexId> d_src_node_ids;  // Used for ...

    util::Array1D<SizeT, VertexId> labels; // does not used in MST
	util::Array1D<SizeT, VertexId> d_dst_node_ids;  // Used for ...
    util::Array1D<SizeT, VertexId> d_edge_list;
    util::Array1D<SizeT, VertexId> d_edge_list_partitioned;
    util::Array1D<SizeT, SizeT> d_degrees; // Used for store node degree
	util::Array1D<SizeT, SizeT> d_flags;         /** < Used for candidate set boolean matrix */

	/*
         * @brief Default constructor
         */
        DataSlice()
        {
	    labels		.SetName("labels");
	    d_src_node_ids	.SetName("d_src_node_ids");
	    d_dst_node_ids	.SetName("d_dst_node_ids");
	    d_edge_list	    .SetName("d_edge_list");
	    d_edge_list_partitioned	    .SetName("d_edge_list_partitioned");
        d_degrees. SetName("d_degrees");
	    d_flags		    .SetName("d_flags");
	}
	 /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            d_src_node_ids.Release();
            d_dst_node_ids.Release();
            d_edge_list.Release();
            d_edge_list_partitioned.Release();
            d_degrees.Release();
            labels.Release();
            d_flags.Release();
	}
        
    }; // DataSlice

    // Members

    // Number of GPUs to be sliced over
    int num_gpus;

    // Size of the graph
    SizeT nodes;
    SizeT edges;

    // Set of data slices (one for each GPU)
    DataSlice **data_slices;

    DataSlice **d_data_slices;

    // device index for each data slice
    int       *gpu_idx;

    /**
     * @brief Default constructor
     */
    TCProblem(): nodes(0), edges(0), num_gpus(0) {}

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph_data  Reference to the data  CSR graph object we process on.
     */
    TCProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph,
                  int   num_gpus) : num_gpus(num_gpus) {
        Init(stream_from_host, graph, num_gpus);
    }

    /**
     * @brief Default destructor
     */
    ~TCProblem() {
        for (int i = 0; i < num_gpus; ++i) {
            if (util::GRError(
                        cudaSetDevice(gpu_idx[i]),
                        "~Problem cudaSetDevice failed",
                        __FILE__, __LINE__)) break;

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
    cudaError_t Extrac(void) {
        cudaError_t retval = cudaSuccess;

        do {
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
        bool  			     stream_from_host,  // only meaningful for single-GPU
        Csr<VertexId, Value, SizeT>& graph,
        int   			     _num_gpus,
	cudaStream_t* 		     streams = NULL) {
        num_gpus = _num_gpus;
        nodes  = graph.nodes;
        edges  = graph.edges;

        ProblemBase<
	VertexId, SizeT, Value,
		false,
		false,
		false,
		false, // _ENABLE_BACKWARD
		false, //_KEEP_ORDER
		false >::Init(stream_from_host,
		              &graph,  
            		      NULL,
            		      num_gpus,
			      NULL,
			      "random");

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];
        if (streams == NULL) {
            streams = new cudaStream_t[num_gpus];
            streams[0] = 0;
        }

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

		data_slices[0][0].streams.SetPointer(streams, 1);
                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph,// Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices

        // create SoA on device
		if(retval = data_slices[gpu]->d_src_node_ids.Allocate(edges, util::DEVICE))  return retval; 
		if(retval = data_slices[gpu]->d_dst_node_ids.Allocate(edges, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->d_edge_list.Allocate(edges, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->d_edge_list_partitioned.Allocate(edges, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->d_degrees.Allocate(nodes, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->d_flags.Allocate(edges, util::DEVICE))  return retval;
		if(retval = data_slices[gpu]->labels.Allocate(nodes, util::DEVICE))  return retval;

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


        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(gpu_idx[gpu]),
                    "TCProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

	    data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing);

            // allocate output labels if necessary
        if (data_slices[gpu]->d_src_node_ids.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_src_node_ids.Allocate(edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_dst_node_ids.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_dst_node_ids.Allocate(edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_edge_list.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_edge_list.Allocate(edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_edge_list_partitioned.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_edge_list_partitioned.Allocate(edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_degrees.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_degrees.Allocate(nodes, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_flags.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_flags.Allocate(edges, util::DEVICE)) 
                return retval;
if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->labels.Allocate(nodes, util::DEVICE)) 
                return retval;
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
        util::MemsetIdxKernel<<<128, 128>>>(
                data_slices[0]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);

        util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[0]->d_degrees.GetPointer(util::DEVICE),
                this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE),
                this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE)+1, -1, nodes);

        return retval;
    }

    /** @} */
};

}  // namespace tc
}  // namespace global_indicator
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
