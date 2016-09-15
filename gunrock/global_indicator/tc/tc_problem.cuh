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

template <
    typename VertexId,
    typename SizeT,
    typename Value>
    //bool _MARK_PREDECESSORS,
    //bool _ENABLE_IDEMPOTENCE>
    //bool _USE_DOUBLE_BUFFER>
struct TCProblem : ProblemBase <
    VertexId, SizeT, Value,
    true, //_MARK_PREDECESSORS,
    false> //_ENABLE_IDEMPOTENCE>
    //_USE_DOUBLE_BUFFER,
    //false,                // _ENABLE_BACKWARD
    //false,                // _KEEP_ORDER
    //false >               // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS  = true;
    static const bool ENABLE_IDEMPOTENCE = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 2;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 2;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    bool use_double_buffer;
    typedef unsigned char MaskT;
    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
     
    struct DataSlice : BaseDataSlice {
        // device storage arrays
	util::Array1D<SizeT, VertexId> d_src_node_ids;  // Used for ...
	util::Array1D<SizeT, SizeT> d_edge_tc;  // Used for ...

    util::Array1D<SizeT, VertexId> labels; // does not used in MST
    util::Array1D<SizeT, VertexId> d_edge_list;
    util::Array1D<SizeT, SizeT> d_degrees; // Used for store node degree

	/*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
	    labels		.SetName("labels");
	    d_src_node_ids	.SetName("d_src_node_ids");
	    d_edge_tc	    .SetName("d_edge_tc");
	    d_edge_list	    .SetName("d_edge_list");
        d_degrees. SetName("d_degrees");
	}
	 /*
         * @brief Default destructor
         */
        virtual ~DataSlice()
        {
            Release();
        }

        cudaError_t Release()
        {
            cudaError_t retval = cudaSuccess;
            if (retval = util::SetDevice(this->gpu_idx)) return retval;
            if (retval = BaseDataSlice::Release())  return retval;
            if (retval = d_src_node_ids.Release()) return retval;
            if (retval = d_edge_tc.Release()) return retval;
            if (retval = d_edge_list.Release()) return retval;
            if (retval = d_degrees.Release()) return retval;
            if (retval = labels.Release()) return retval;
            return retval;
        }
        
    }; // DataSlice

    // Members


    // Set of data slices (one for each GPU)
    DataSlice **data_slices;

    DataSlice **d_data_slices;

    /**
     * @brief Default constructor
     */
    TCProblem(bool use_double_buffer) :
    BaseProblem(use_double_buffer,
    false,
    false,
    false),
    data_slices (NULL),
    d_data_slices(NULL)
    {
    }

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph_data  Reference to the data  CSR graph object we process on.
     */
    /*TCProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph,
                  int   num_gpus) : num_gpus(num_gpus) {
        Init(stream_from_host, graph, num_gpus);
    }*/

    /**
     * @brief Default destructor
     */
    virtual ~TCProblem() {
        Release();
    }
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices==NULL) return retval;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            if (retval = util::SetDevice(this->gpu_idx[i])) return retval;
            if (retval = data_slices[i]->Release()) return retval;
        }
        delete[] data_slices;data_slices=NULL;
        if (retval = BaseProblem::Release()) return retval;
        return retval;
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
    cudaError_t Extract(VertexId *source_ids, VertexId *dest_ids, SizeT *edge_tc) {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {
                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                this->graph_slices[0]->edges /= 2;
                data_slices[0]->d_src_node_ids.SetPointer(source_ids);
                if (retval = data_slices[0]->d_src_node_ids.Move(util::DEVICE,util::HOST, this->graph_slices[0]->edges, 0)) return retval;
            
                this->graph_slices[0]->column_indices.SetPointer(dest_ids);
                if (retval = this->graph_slices[0]->column_indices.Move(util::DEVICE,util::HOST, this->graph_slices[0]->edges, 0)) return retval;

                data_slices[0]->d_edge_tc.SetPointer(edge_tc);
                if (retval = data_slices[0]->d_edge_tc.Move(util::DEVICE,util::HOST, this->graph_slices[0]->edges, 0)) return retval;

            } else {
            // does not support multi-GPU yet
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
        bool  			     stream_from_host,  // only meaningful for single-GPU
        Csr<VertexId, Value, SizeT>* graph,
        Csr<VertexId, Value, SizeT>* inv_graph = NULL,
        int   			     num_gpus = 1,
        int*                 gpu_idx = NULL,
        std::string          partition_method = "random",
	cudaStream_t* 		     streams = NULL,
        float                queue_sizing = 2.0f,
        float                in_sizing = 1.0f,
        float                partition_factor = -1.0f,
        int                  partition_seed = -1) {

        BaseProblem::Init(
            stream_from_host,
            graph,
            inv_graph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);
	

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];

	//copy query graph labels and data graph labels from input
	
        do {
            if (num_gpus <= 1) {

                // create a single data slice for the currently-set GPU
                int gpu = 0;
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_slices[gpu],
                                   sizeof(DataSlice)),
                        "Problem cudaMalloc d_data_slices failed",
                        __FILE__, __LINE__)) return retval;

		data_slices[gpu][0].streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);
                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    this->use_double_buffer,
                    //0,           // Number of vertex associate
                    //0,           // Number of value associate
                    graph,// Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices

        // create SoA on device
		if(retval = data_slices[gpu]->d_src_node_ids.Allocate(this->edges, util::DEVICE))  return retval; 
		if(retval = data_slices[gpu]->d_edge_tc.Allocate(this->edges, util::DEVICE))  return retval; 
		if(retval = data_slices[gpu]->d_edge_list.Allocate(this->edges, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->d_degrees.Allocate(this->nodes, util::DEVICE))   return retval;
		if(retval = data_slices[gpu]->labels.Allocate(this->nodes, util::DEVICE))  return retval;

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
        double queue_sizing,
        double queue_sizing1 = -1.0) {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.


        cudaError_t retval = cudaSuccess;

        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(this->gpu_idx[gpu]),
                    "TCProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

	    data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                this->use_double_buffer,
                queue_sizing, queue_sizing1);

        if (retval = data_slices[gpu]->frontier_queues[0].keys[0].EnsureSize(
            this->nodes, util::DEVICE));
        if (retval = data_slices[gpu]->frontier_queues[0].keys[1].EnsureSize(
            this->edges, util::DEVICE));

            // allocate output labels if necessary
        if (data_slices[gpu]->d_src_node_ids.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_src_node_ids.Allocate(this->edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_edge_tc.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_edge_tc.Allocate(this->edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_edge_list.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_edge_list.Allocate(this->edges, util::DEVICE)) 
                return retval;
        if (data_slices[gpu]->d_degrees.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->d_degrees.Allocate(this->nodes, util::DEVICE)) 
                return retval;
if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) 
            if (retval = data_slices[gpu]->labels.Allocate(this->nodes, util::DEVICE)) 
                return retval;
            // TODO: code to for other allocations here  

           // TODO: fill in the initial input_queue for problem
        util::MemsetIdxKernel<<<256, 1024>>>(
                data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), this->nodes);

        util::MemsetKernel<<<256, 1024>>>(data_slices[0]->d_edge_tc.GetPointer(util::DEVICE), (SizeT)0, this->edges);

        util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[0]->d_degrees.GetPointer(util::DEVICE),
                this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE),
                this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE)+1, -1, this->nodes);

        if (retval = util::GRError(
                    cudaMemcpy(d_data_slices[gpu],
                               data_slices[gpu],
                               sizeof(DataSlice),
                               cudaMemcpyHostToDevice),
                    "Problem cudaMemcpy data_slices to d_data_slices failed",
                    __FILE__, __LINE__)) return retval;
            }

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
