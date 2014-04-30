// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * salsa_problem.cuh
 *
 * @brief GPU Storage management Structure for SALSA(Stochastic Approach for Link-Structure Analysis) Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace salsa {

/**
 * @brief SALSA Problem structure stores device-side vectors for doing SALSA Algorithm on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing SALSA rank value.
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value>
struct SALSAProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{

    typedef _VertexId 			VertexId;
	typedef _SizeT			    SizeT;
	typedef _Value              Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains SALSA problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        Value   *d_hrank_curr;          /**< Used for ping-pong hub rank value */
        Value   *d_arank_curr;          /**< Used for ping-pong authority rank value */
        Value   *d_hrank_next;          /**< Used for ping-pong page rank value */       
        Value   *d_arank_next;          /**< Used for ping-pong page rank value */
        VertexId   *d_in_degrees;          /**< Used for keeping in-degree for each vertex */
        VertexId   *d_out_degrees;         /**< Used for keeping out-degree for each vertex */
        SizeT   *d_labels;
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
     * @brief SALSAProblem default constructor
     */

    SALSAProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief SALSAProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    SALSAProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, Value, SizeT> &graph,
               const Csr<VertexId, Value, SizeT> &inv_graph,
               int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            inv_graph,
            num_gpus);
    }

    /**
     * @brief SALSAProblem default destructor
     */
    ~SALSAProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_hrank_curr)      util::GRError(cudaFree(data_slices[i]->d_hrank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_arank_curr)      util::GRError(cudaFree(data_slices[i]->d_arank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_hrank_next)      util::GRError(cudaFree(data_slices[i]->d_hrank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_arank_next)      util::GRError(cudaFree(data_slices[i]->d_arank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_in_degrees)      util::GRError(cudaFree(data_slices[i]->d_in_degrees), "GpuSlice cudaFree d_in_degrees failed", __FILE__, __LINE__);
            if (data_slices[i]->d_out_degrees)      util::GRError(cudaFree(data_slices[i]->d_out_degrees), "GpuSlice cudaFree d_out_degrees failed", __FILE__, __LINE__);
            if (d_data_slices[i])                 util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
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
     * @param[out] h_hrank host-side vector to store hub rank values.
     *
     * @param[out] h_arank host-side vector to store authority rank values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_hrank, Value *h_arank)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_hrank,
                                data_slices[0]->d_hrank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "SALSAProblem cudaMemcpy d_hranks failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_arank,
                                data_slices[0]->d_arank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "SALSAProblem cudaMemcpy d_aranks failed", __FILE__, __LINE__)) break;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief SALSAProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] hub_graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] auth_graph Reference to the CSC graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            const Csr<VertexId, Value, SizeT> &hub_graph,
            const Csr<VertexId, Value, SizeT> &auth_graph,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = hub_graph.nodes;
        edges = hub_graph.edges;
        SizeT *h_row_offsets = hub_graph.row_offsets;
        VertexId *h_column_indices = hub_graph.column_indices;
        SizeT *h_col_offsets = auth_graph.row_offsets;
        VertexId *h_row_indices = auth_graph.column_indices;
            ProblemBase<VertexId, SizeT, false>::Init(stream_from_host,
                    nodes,
                    edges,
                    h_row_offsets,
                    h_column_indices,
                    h_col_offsets,
                    h_row_indices,
                    num_gpus);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];

        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "SALSAProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "SALSAProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                Value    *d_hrank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_hrank1,
                        nodes * sizeof(Value)),
                    "SALSAProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_hrank_curr = d_hrank1;

                Value    *d_arank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_arank1,
                        nodes * sizeof(Value)),
                    "SALSAProblem cudaMalloc d_arank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_arank_curr = d_arank1;

                Value    *d_hrank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_hrank2,
                        nodes * sizeof(Value)),
                    "SALSAProblem cudaMalloc d_hrank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_hrank_next = d_hrank2;

                Value    *d_arank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_arank2,
                        nodes * sizeof(Value)),
                    "SALSAProblem cudaMalloc d_arank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_arank_next = d_arank2;

                VertexId   *d_in_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_in_degrees,
                        nodes * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_in_degrees = d_in_degrees;

                VertexId   *d_out_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_out_degrees,
                        nodes * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_out_degrees = d_out_degrees;
 
                data_slices[0]->d_labels = NULL;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for PR problem type. Must be called prior to each PR iteration.
     *
     *  @param[in] src Source node for one PR computing pass.
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

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output page ranks if necessary
            if (!data_slices[gpu]->d_hrank_curr) {
                Value    *d_hrank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_hrank1,
                                nodes * sizeof(Value)),
                            "SALSAProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_hrank_curr = d_hrank1;
            }

            if (!data_slices[gpu]->d_arank_curr) {
                Value    *d_arank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_arank1,
                                nodes * sizeof(Value)),
                            "SALSAProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_arank_curr = d_arank1;
            }

            if (!data_slices[gpu]->d_hrank_next) {
                Value    *d_hrank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_hrank2,
                                nodes * sizeof(Value)),
                            "SALSAProblem cudaMalloc d_hrank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_hrank_next = d_hrank2;
            } 

            if (!data_slices[gpu]->d_arank_next) {
                Value    *d_arank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_arank2,
                                nodes * sizeof(Value)),
                            "SALSAProblem cudaMalloc d_arank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_arank_next = d_arank2;
            }

            // Allocate d_degrees if necessary
            if (!data_slices[gpu]->d_in_degrees) {
                VertexId    *d_in_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_in_degrees,
                                nodes * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_in_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_in_degrees = d_in_degrees;
            }

            // Allocate d_degrees if necessary
            if (!data_slices[gpu]->d_out_degrees) {
                VertexId    *d_out_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_out_degrees,
                                nodes * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_out_degrees = d_out_degrees;
            }

            data_slices[gpu]->d_labels = NULL;

            // Initial rank_curr = 0 
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_hrank_curr, (Value)1.0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_arank_curr, (Value)1.0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_hrank_next, (Value)0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_arank_next, (Value)0, nodes);

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_out_degrees, 0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_in_degrees, 0, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[gpu]->d_out_degrees, BaseProblem::graph_slices[gpu]->d_row_offsets, &BaseProblem::graph_slices[gpu]->d_row_offsets[1], -1, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[gpu]->d_in_degrees, BaseProblem::graph_slices[gpu]->d_column_offsets, &BaseProblem::graph_slices[gpu]->d_column_offsets[1], -1, nodes);
            
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "SALSAProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }
        
        // Fillin the initial input_queue for SALSA problem, this needs to be modified
        // in multi-GPU scene

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        return retval;
    }

    /** @} */

};

} //namespace salsa
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
