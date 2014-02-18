// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief PageRank Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing PR value.
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value>
struct PRProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{

    typedef _VertexId 			VertexId;
	typedef _SizeT			    SizeT;
	typedef _Value              Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains PR problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        Value   *d_rank_curr;           /**< Used for ping-pong page rank value */
        Value   *d_rank_next;           /**< Used for ping-pong page rank value */       
        SizeT   *d_degrees;             /**< Used for keeping out-degree for each vertex */
        Value   *d_threshold;               /**< Used for recording accumulated error */
        Value   *d_delta;
        SizeT   *d_nodes;
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
     * @brief PRProblem default constructor
     */

    PRProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief PRProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    PRProblem(bool        stream_from_host,       // Only meaningful for single-GPU
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
     * @brief PRProblem default destructor
     */
    ~PRProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~PRProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_rank_curr)      util::GRError(cudaFree(data_slices[i]->d_rank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_rank_next)      util::GRError(cudaFree(data_slices[i]->d_rank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_degrees)      util::GRError(cudaFree(data_slices[i]->d_degrees), "GpuSlice cudaFree d_degrees failed", __FILE__, __LINE__);
            if (data_slices[i]->d_threshold)      util::GRError(cudaFree(data_slices[i]->d_threshold), "GpuSlice cudaFree d_threshold failed", __FILE__, __LINE__);
            if (data_slices[i]->d_delta)      util::GRError(cudaFree(data_slices[i]->d_delta), "GpuSlice cudaFree d_delta failed", __FILE__, __LINE__);
            if (data_slices[i]->d_nodes)      util::GRError(cudaFree(data_slices[i]->d_nodes), "GpuSlice cudaFree d_nodes failed", __FILE__, __LINE__);
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
     * @param[out] h_rank host-side vector to store page rank values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_rank)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "PRProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_rank,
                                data_slices[0]->d_rank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "PRProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief PRProblem initialization
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
                if (retval = util::GRError(cudaGetDevice(&gpu), "PRProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "PRProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                Value    *d_rank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank1,
                        nodes * sizeof(Value)),
                    "PRProblem cudaMalloc d_rank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_rank_curr = d_rank1;

                Value    *d_rank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank2,
                        nodes * sizeof(Value)),
                    "PRProblem cudaMalloc d_rank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_rank_next = d_rank2;
 
                VertexId   *d_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_degrees,
                        nodes * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_degrees = d_degrees;

                Value   *d_threshold;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_threshold,
                        1 * sizeof(Value)),
                    "PRProblem cudaMalloc d_threshold failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_threshold = d_threshold;

                Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_delta,
                        1 * sizeof(Value)),
                    "PRProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_delta = d_delta;

                SizeT    *d_nodes;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_nodes,
                        1 * sizeof(SizeT)),
                    "PRProblem cudaMalloc d_nodes failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_nodes = d_nodes;

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
            Value    delta,
            Value    threshold,
            FrontierType frontier_type)             // The frontier type (i.e., edge/vertex/mixed)
    {
        typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output page ranks if necessary
            if (!data_slices[gpu]->d_rank_curr) {
                Value    *d_rank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank1,
                                nodes * sizeof(Value)),
                            "PRProblem cudaMalloc d_rank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_rank_curr = d_rank1;
            }

            if (!data_slices[gpu]->d_rank_next) {
                Value    *d_rank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank2,
                                nodes * sizeof(Value)),
                            "PRProblem cudaMalloc d_rank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_rank_next = d_rank2;
            } 

            if (!data_slices[gpu]->d_delta) {
                Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_delta,
                                1 * sizeof(Value)),
                            "PRProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_delta = d_delta;
            }

            if (!data_slices[gpu]->d_threshold) {
                Value    *d_threshold;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_threshold,
                                1 * sizeof(Value)),
                            "PRProblem cudaMalloc d_threshold failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_threshold = d_threshold;
            }

            if (!data_slices[gpu]->d_nodes) {
                SizeT    *d_nodes;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_nodes,
                                1 * sizeof(SizeT)),
                            "PRProblem cudaMalloc d_nodes failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_nodes = d_nodes;
            }

            // Allocate d_degrees if necessary
            if (!data_slices[gpu]->d_degrees) {
                VertexId    *d_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_degrees,
                                nodes * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_degrees = d_degrees;
            }

            data_slices[gpu]->d_labels = NULL;

            // Initial rank_next = 0 
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_curr,
                (Value)1.0/nodes, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_next, (Value)0.0, nodes);

            // Compute degrees
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_degrees, 0, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[gpu]->d_degrees, BaseProblem::graph_slices[0]->d_row_offsets, &BaseProblem::graph_slices[0]->d_row_offsets[1], -1, nodes);

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_delta,
                            (Value*)&delta,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                        "PRProblem cudaMemcpy d_delta failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_threshold,
                            (Value*)&threshold,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                        "PRProblem cudaMemcpy d_threshold failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_nodes,
                            (SizeT*)&nodes,
                            sizeof(SizeT),
                            cudaMemcpyHostToDevice),
                        "PRProblem cudaMemcpy d_nodes failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "PRProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }

        
        // Fillin the initial input_queue for PR problem, this needs to be modified
        // in multi-GPU scene

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        return retval;
    }

    /** @} */

};

} //namespace pr
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
