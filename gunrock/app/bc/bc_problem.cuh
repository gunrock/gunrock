// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_problem.cuh
 *
 * @brief GPU Storage management Structure for BC Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Betweenness Centrality Problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    _VertexId,                      
    typename    _SizeT,                        
    typename    _Value,                       
    bool        _USE_DOUBLE_BUFFER>
struct BCProblem : ProblemBase<_VertexId, _SizeT,
                                _USE_DOUBLE_BUFFER>
{
    typedef _VertexId       VertexId;
    typedef _SizeT          SizeT;
    typedef _Value          Value;
    
    //Helper structures
    
    /** 
     * @brief Data slice structure which contains BC problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_labels;              // Used for source distance
        VertexId        *d_preds;               // Used for predecessor
        Value           *d_bc_values;           // Used to store final BC values for each node
        Value           *d_sigmas;              // Accumulated sigma values for each node
        Value           *d_deltas;              // Accumulated delta values for each node
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
     * @brief BCProblem default constructor
     */

    BCProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief BCProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] nodes Number of nodes in the CSR graph.
     * @param[in] edges Number of edges in the CSR graph.
     * @param[in] h_row_offsets Host-side row offsets array.
     * @param[in] h_column_indices Host-side column indices array.
     * @param[in] num_gpus Number of the GPUs used.
     */
    BCProblem(bool        stream_from_host,       // Only meaningful for single-GPU
            SizeT       nodes,
            SizeT       edges,
            SizeT       *h_row_offsets,
            VertexId    *h_column_indices,
            int         num_gpus) :
        nodes(nodes),
        edges(edges),
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            nodes,
            edges,
            h_row_offsets,
            h_column_indices,
            num_gpus);
    }

    /**
     * @brief BCProblem default destructor
     */
    ~BCProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~BCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels)      util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (data_slices[i]->d_preds)      util::GRError(cudaFree(data_slices[i]->d_preds), "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
            if (data_slices[i]->d_bc_values)      util::GRError(cudaFree(data_slices[i]->d_bc_values), "GpuSlice cudaFree d_bc_values failed", __FILE__, __LINE__);
            if (data_slices[i]->d_sigmas)      util::GRError(cudaFree(data_slices[i]->d_sigmas), "GpuSlice cudaFree d_sigmas failed", __FILE__, __LINE__);
            if (data_slices[i]->d_deltas)      util::GRError(cudaFree(data_slices[i]->d_deltas), "GpuSlice cudaFree d_deltas failed", __FILE__, __LINE__);
            if (d_data_slices[i])                 util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices)  delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * @brief Extract into a single host vector the BC results disseminated across all GPUs.
     *
     * @param[out] h_sigmas host-side vector to store computed sigma values. (Meaningful only in single-pass BC)
     * @param[out] h_bc_values host-side vector to store BC_values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_sigmas, Value *h_bc_values)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "BCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_bc_values,
                                data_slices[0]->d_bc_values,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "BCProblem cudaMemcpy d_bc_values failed", __FILE__, __LINE__)) break;

                if (h_sigmas) {
                    if (retval = util::GRError(cudaMemcpy(
                                    h_sigmas,
                                    data_slices[0]->d_sigmas,
                                    sizeof(Value) * nodes,
                                    cudaMemcpyDeviceToHost),
                                "BCProblem cudaMemcpy d_sigmas failed", __FILE__, __LINE__)) break;
                }
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief BCProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] _nodes Number of nodes in the CSR graph.
     * @param[in] _edges Number of edges in the CSR graph.
     * @param[in] h_row_offsets Host-side row offsets array.
     * @param[in] h_column_indices Host-side column indices array.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            SizeT       _nodes,
            SizeT       _edges,
            SizeT       *h_row_offsets,
            VertexId    *h_column_indices,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = _nodes;
        edges = _edges;
        ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER>::Init(stream_from_host,
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
                if (retval = util::GRError(cudaGetDevice(&gpu), "BCProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "BCProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                VertexId    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_labels,
                        nodes * sizeof(VertexId)),
                    "BCProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_labels;

                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_preds,
                        nodes * sizeof(VertexId)),
                    "BCProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_preds = d_preds;
 
                Value   *d_bc_values;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_bc_values,
                        nodes * sizeof(Value)),
                    "BCProblem cudaMalloc d_bc_values failed", __FILE__, __LINE__)) return retval;

                data_slices[0]->d_bc_values = d_bc_values;

                util::MemsetKernel<<<128, 128>>>(data_slices[0]->d_bc_values, (Value)0.0f, nodes);


                Value   *d_sigmas;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_sigmas,
                        nodes * sizeof(Value)),
                    "BCProblem cudaMalloc d_sigmas failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_sigmas = d_sigmas;
                
                Value   *d_deltas;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_deltas,
                        nodes * sizeof(Value)),
                    "BCProblem cudaMalloc d_deltas failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_deltas = d_deltas;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BC problem type. Must be called prior to each BC run.
     *
     *  @param[in] src Source node for one BC computing pass. If equals to -1 then compute BC value for each node.
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
        typedef ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        // Reset all data but d_bc_values (Because we need to accumulate it)
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output labels if necessary
            if (!data_slices[gpu]->d_labels) {
                VertexId    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_labels,
                                nodes * sizeof(VertexId)),
                            "BCProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, -1, nodes);

            // Allocate preds if necessary
            if (!data_slices[gpu]->d_preds) {
                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_preds,
                                nodes * sizeof(VertexId)),
                            "BCProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_preds = d_preds;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_preds, -2, nodes);

            // Allocate bc_values if necessary
            if (!data_slices[gpu]->d_bc_values) {
                Value    *d_bc_values;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_bc_values,
                                nodes * sizeof(Value)),
                            "BCProblem cudaMalloc d_bc_values failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_bc_values = d_bc_values;
            }

            // Allocate deltas if necessary
            if (!data_slices[gpu]->d_deltas) {
                Value    *d_deltas;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_deltas,
                                nodes * sizeof(Value)),
                            "BCProblem cudaMalloc d_deltas failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_deltas = d_deltas;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_deltas, (Value)0.0f, nodes);

            // Allocate deltas if necessary
            if (!data_slices[gpu]->d_sigmas) {
                Value    *d_sigmas;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_sigmas,
                                nodes * sizeof(Value)),
                            "BCProblem cudaMalloc d_sigmas failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_sigmas = d_sigmas;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_sigmas, (Value)0.0f, nodes);


            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "BCProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;
        }
        
        // Fillin the initial input_queue for BC problem, this needs to be modified
        // in multi-GPU scene
        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[0]->frontier_queues.d_keys[0],
                        &src,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_labels+src,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy labels failed", __FILE__, __LINE__)) return retval;
        VertexId src_pred = -1; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_preds+src,
                        &src_pred,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy preds failed", __FILE__, __LINE__)) return retval;
        Value src_sigma = 1.0f;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_sigmas+src,
                        &src_sigma,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy sigmas failed", __FILE__, __LINE__)) return retval;

        return retval;
    }

};

} //namespace bc
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
