// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * top_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace top {

/**
 * @brief PageRank Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId    Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT       Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value       Type of float or double to use for computing TOP value.
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value>
struct TOPProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{
    typedef _VertexId   VertexId;
	typedef _SizeT	    SizeT;
	typedef _Value      Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains TOP problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        SizeT       *d_labels;
        VertexId    *d_node_id;
        Value       *d_degrees;

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
     * @brief TOPProblem default constructor
     */

    TOPProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief TOPProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    TOPProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, Value, SizeT> &graph,
               int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            num_gpus);
    }

    /**
     * @brief TOPProblem default destructor
     */
    ~TOPProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~TOPProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

            if (data_slices[i]->d_degrees)  util::GRError(cudaFree(data_slices[i]->d_degrees), 
                "GpuSlice cudaFree d_degrees failed", __FILE__, __LINE__);

            if (d_data_slices[i])   util::GRError(cudaFree(d_data_slices[i]), 
                "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices) delete[] d_data_slices;
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
    cudaError_t Extract(Value *h_rank)
    {
        /*cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "TOPProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_rank,
                                data_slices[0]->d_rank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "TOPProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);*/

        return cudaSuccess;
    }

    /**
     * @brief TOPProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            const Csr<VertexId, Value, SizeT> &graph,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;
        ProblemBase<VertexId, SizeT, false>::Init(stream_from_host,
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
                    "TOPProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_data_slices[0],
                    sizeof(DataSlice)),
                    "TOPProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                VertexId    *d_node_id;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_degrees,
                    nodes * sizeof(Value)),
                    "TOPProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;

                Value    *d_degrees;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_degrees,
                    nodes * sizeof(Value)),
                    "TOPProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;

                data_slices[0]->d_degrees = NULL;
                data_slices[0]->d_labels  = NULL;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for TOP problem type. Must be called prior to each TOP iteration.
     *
     *  @param[in] src Source node for one TOP computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type)             // The frontier type (i.e., edge/vertex/mixed)
    {
        typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) 
        {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                "TOPProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output if necessary
            if (!data_slices[gpu]->d_node_id) {
                VertexId    *d_node_id;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_node_id,
                    nodes * sizeof(Value)),
                    "TOPProblem cudaMalloc d_node_id failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_node_id = d_node_id;
            }

            if (!data_slices[gpu]->d_degrees) {
                Value    *d_degrees;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_degrees,
                    nodes * sizeof(Value)),
                    "TOPProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_degrees = d_degrees;
            }

            data_slices[gpu]->d_labels = NULL;

            if (retval = util::GRError(cudaMemcpy(
                d_data_slices[gpu],
                data_slices[gpu],
                sizeof(DataSlice),
                cudaMemcpyHostToDevice),
                "TOPProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;
        }

        
        // Fillin the initial input_queue for TOP problem, this needs to be modified
        // in multi-GPU scene

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        // set track node ids
        util::MemsetIdxKernel<<<128, 128>>>(data_slices[0]->d_node_id, nodes);

        // count number of degrees for each node
        util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[0]->d_degrees, 
            BaseProblem::graph_slices[0]->d_row_offsets, 
            &BaseProblem::graph_slices[0]->d_row_offsets[1], -1, nodes);
        return retval;
    }

    /** @} */

};

} //namespace top
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
