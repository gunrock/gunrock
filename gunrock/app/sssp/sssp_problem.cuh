// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Single-Source Shortest Path Problem structure stores device-side vectors for doing SSSP computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,
    typename    _Value,
    bool        _MARK_PREDECESSORS>
struct SSSPProblem : ProblemBase<_VertexId, _SizeT, false>
{

    typedef _VertexId       VertexId;
    typedef _SizeT          SizeT;
    typedef _Value          Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;
    static const bool MARK_PATHS            = _MARK_PREDECESSORS;

    //Helper structures

    /**
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        Value               *d_labels;              /**< Used for source distance */
        Value               *d_weights;             /**< Used for storing edge weights */
        VertexId            *d_preds;               /**< Used for storing the actual shortest path */
        VertexId            *d_visit_lookup;        /**< Used for check duplicate */
        float               *d_delta;
    };

    // Members
    
    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    int                 *gpu_idx;

    // Methods

    /**
     * @brief SSSPProblem default constructor
     */

    SSSPProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief SSSPProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    SSSPProblem(bool        stream_from_host,       // Only meaningful for single-GPU
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
     * @brief SSSPProblem default destructor
     */
    ~SSSPProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~SSSPProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels)      util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (data_slices[i]->d_weights)      util::GRError(cudaFree(data_slices[i]->d_weights), "GpuSlice cudaFree d_weights failed", __FILE__, __LINE__);
            if (data_slices[i]->d_delta)      util::GRError(cudaFree(data_slices[i]->d_delta), "GpuSlice cudaFree d_delta failed", __FILE__, __LINE__);
            if (data_slices[i]->d_preds)      util::GRError(cudaFree(data_slices[i]->d_preds), "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
            if (data_slices[i]->d_visit_lookup)      util::GRError(cudaFree(data_slices[i]->d_visit_lookup), "GpuSlice cudaFree d_visit_lookup failed", __FILE__, __LINE__);
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
     * @brief Copy result labels computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store computed node predecessors (used for extracting the actual shortest path).
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "SSSPProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_labels,
                                data_slices[0]->d_labels,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "SSSPProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
                if (MARK_PATHS) {
                    if (retval = util::GRError(cudaMemcpy(
                                    h_preds,
                                    data_slices[0]->d_preds,
                                    sizeof(VertexId) * nodes,
                                    cudaMemcpyDeviceToHost),
                                "SSSPProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                }
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief SSSPProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     * @param[in] delta_factor Parameter for delta-stepping SSSP
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            const Csr<VertexId, Value, SizeT> &graph,
            int         _num_gpus,
            int         delta_factor = 16)
    {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        SizeT *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;
            ProblemBase<_VertexId, _SizeT, false>::Init(stream_from_host,
                    nodes,
                    edges,
                    h_row_offsets,
                    h_column_indices,
                    NULL,
                    NULL,
                    num_gpus);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];

        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "SSSPProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "SSSPProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                Value    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_labels,
                        nodes * sizeof(Value)),
                    "SSSPProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_labels;

                Value    *d_weights;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_weights,
                        edges * sizeof(Value)),
                    "SSSPProblem cudaMalloc d_weights failed", __FILE__, __LINE__)) return retval;

                float    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_delta,
                        1 * sizeof(float)),
                    "SSSPProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;

                VertexId    *d_preds = NULL;
                if (MARK_PATHS) {
                    if (retval = util::GRError(cudaMalloc(
                                    (void**)&d_preds,
                                    nodes * sizeof(VertexId)),
                                "SSSPProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                }
                data_slices[0]->d_preds = d_preds;

                VertexId    *d_visit_lookup;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_visit_lookup,
                                nodes * sizeof(VertexId)),
                            "SSSPProblem cudaMalloc d_visit_lookup failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_visit_lookup = d_visit_lookup;
    
                if (retval = util::GRError(cudaMemcpy(
                        d_weights,
                        graph.edge_values,
                        edges * sizeof(Value),
                        cudaMemcpyHostToDevice),
                        "ProblemBase cudaMemcpy d_weights failed", __FILE__, __LINE__)) return retval;

                data_slices[0]->d_weights = d_weights;

                float delta = EstimatedDelta(graph)*delta_factor;
                printf("estimated delta:%5f\n", delta);

                if (retval = util::GRError(cudaMemcpy(
                            d_delta,
                            (float*)&delta,
                            sizeof(float),
                            cudaMemcpyHostToDevice),
                        "SSSPProblem cudaMemcpy d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_delta = d_delta;
            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for SSSP problem type. Must be called prior to each SSSP run.
     *
     *  @param[in] src Source node for one SSSP computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId    src,
            FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing)                    // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
    {
        typedef ProblemBase<_VertexId, _SizeT, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output labels if necessary
            if (!data_slices[gpu]->d_labels) {
                Value    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_labels,
                                nodes * sizeof(Value)),
                            "SSSPProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, UINT_MAX, nodes);

            if (!data_slices[gpu]->d_preds && MARK_PATHS) {
                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_preds,
                        nodes * sizeof(VertexId)),
                    "SSSPProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_preds = d_preds;
            }

            if (!data_slices[gpu]->d_visit_lookup) {
                VertexId    *d_visit_lookup;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_visit_lookup,
                        nodes * sizeof(VertexId)),
                    "SSSPProblem cudaMalloc d_visit_lookup failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_visit_lookup = d_visit_lookup;
            }

            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "SSSPProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;


        }

        
        // Fillin the initial input_queue for SSSP problem, this needs to be modified
        // in multi-GPU scene
        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[0]->frontier_queues.d_keys[0],
                        &src,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_labels+src,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;


        if (MARK_PATHS) util::MemsetIdxKernel<<<128, 128>>>(data_slices[0]->d_preds, nodes);
        util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_visit_lookup, -1, nodes);


        return retval;
    }

    float EstimatedDelta(const Csr<VertexId, Value, SizeT> &graph) {
        double  avgV = graph.average_edge_value;
        int     avgD = graph.average_degree;
        return avgV * 32 / avgD;
    }

    /** @} */

};

} //namespace sssp
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
