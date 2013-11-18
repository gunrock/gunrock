// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dobfs_problem.cuh
 *
 * @brief GPU Storage management Structure for Direction Optimal BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace dobfs {

/**
 * @brief Direction Optimal Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value,                          
    bool        _MARK_PREDECESSORS,             
    bool        _USE_DOUBLE_BUFFER>
struct DOBFSProblem : ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    

    //Helper structures

    /**
     * @brief Data slice structure which contains DOBFS problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_labels;              /**< Used for source distance */
        VertexId        *d_preds;               /**< Used for predecessor */
        SizeT           *d_col_offsets;         /**< CSC format column offset on device memory */
        VertexId        *d_row_indices;         /**< CSC format row indices on device memory */
        bool            *d_frontier_map_in;     /**< Input frontier bitmap */
        bool            *d_frontier_map_out;    /**< Output frontier bitmap */

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
     * @brief DOBFSProblem default constructor
     */

    DOBFSProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief DOBFSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] inv_graph Reference to the inverse (CSC) graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    DOBFSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
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
     * @brief DOBFSProblem default destructor
     */
    ~DOBFSProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~DOBFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels)       util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (data_slices[i]->d_preds)        util::GRError(cudaFree(data_slices[i]->d_preds), "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
            if (data_slices[i]->d_col_offsets)  util::GRError(cudaFree(data_slices[i]->d_col_offsets), "GpuSlice cudaFree d_col_offsets failed", __FILE__, __LINE__);
            if (data_slices[i]->d_row_indices)  util::GRError(cudaFree(data_slices[i]->d_row_indices), "GpuSlice cudaFree d_row_indices failed", __FILE__, __LINE__);
            if (data_slices[i]->d_frontier_map_in)  util::GRError(cudaFree(data_slices[i]->d_frontier_map_in), "GpuSlice cudaFree d_frontier_map_in failed", __FILE__, __LINE__);
            if (data_slices[i]->d_frontier_map_out)  util::GRError(cudaFree(data_slices[i]->d_frontier_map_out), "GpuSlice cudaFree d_frontier_map_out failed", __FILE__, __LINE__);
            if (d_data_slices[i])               util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
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
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store predecessor vertex ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "DOBFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_labels,
                                data_slices[0]->d_labels,
                                sizeof(VertexId) * nodes,
                                cudaMemcpyDeviceToHost),
                            "DOBFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

                if (_MARK_PREDECESSORS) {
                    if (retval = util::GRError(cudaMemcpy(
                                    h_preds,
                                    data_slices[0]->d_preds,
                                    sizeof(VertexId) * nodes,
                                    cudaMemcpyDeviceToHost),
                                "DOBFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                }

            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief DOBFSProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] inv_graph Reference to the inverse (CSC) graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            const Csr<VertexId, Value, SizeT> &graph,
            const Csr<VertexId, Value, SizeT> &inv_graph,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;
        VertexId *h_col_offsets = inv_graph.row_offsets;
        VertexId *h_row_indices = inv_graph.column_indices;
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
                if (retval = util::GRError(cudaGetDevice(&gpu), "DOBFSProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "DOBFSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                VertexId    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_labels,
                        nodes * sizeof(VertexId)),
                    "DOBFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_labels;
 
                VertexId   *d_preds;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_preds,
                        nodes * sizeof(VertexId)),
                    "DOBFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_preds = d_preds;

                SizeT    *d_col_offsets;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_col_offsets,
                                nodes * sizeof(SizeT)),
                            "DOBFSProblem cudaMalloc d_col_offsets failed", __FILE__, __LINE__)) return retval;

                if (retval = util::GRError(cudaMemcpy(
                        d_col_offsets,
                        h_col_offsets,
                        (nodes+1) * sizeof(SizeT),
                        cudaMemcpyHostToDevice),
                        "ProblemBase cudaMemcpy d_col_offsets failed", __FILE__, __LINE__)) break;

                data_slices[gpu]->d_col_offsets = d_col_offsets;

                VertexId    *d_row_indices;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_row_indices,
                                nodes * sizeof(VertexId)),
                            "DOBFSProblem cudaMalloc d_row_indices failed", __FILE__, __LINE__)) return retval;

                if (retval = util::GRError(cudaMemcpy(
                        d_row_indices,
                        h_row_indices,
                        edges * sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                        "ProblemBase cudaMemcpy d_row_indices failed", __FILE__, __LINE__)) break;
                data_slices[gpu]->d_row_indices = d_row_indices;

                bool    *d_frontier_map_in;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_frontier_map_in,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_frontier_map_in failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_frontier_map_in = d_frontier_map_in;

                bool    *d_frontier_map_out;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_frontier_map_out,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_frontier_map_out failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_frontier_map_out = d_frontier_map_out;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for DOBFS problem type. Must be called prior to each DOBFS run.
     *
     *  @param[in] src Source node for one BFS computing pass.
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
                            "DOBFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, -1, nodes);

            // Allocate preds if necessary
            if (!data_slices[gpu]->d_preds) {
                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_preds,
                                nodes * sizeof(VertexId)),
                            "DOBFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_preds = d_preds;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_preds, -2, nodes);

            // Allocate input/output frontier map if necessary
            if (!data_slices[gpu]->d_frontier_map_in) {
                bool    *d_frontier_map_in;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_frontier_map_in,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_frontier_map_in failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_frontier_map_in = d_frontier_map_in;
            }

            if (!data_slices[gpu]->d_frontier_map_out) {
                bool    *d_frontier_map_out;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_frontier_map_out,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_frontier_map_out failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_frontier_map_out = d_frontier_map_out;
            }


            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_frontier_map_in, false, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_frontier_map_out, false, nodes);
                
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "DOBFSProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;
        }

        
        // Fillin the initial input_queue for BFS problem, this needs to be modified
        // in multi-GPU scene
        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[0]->frontier_queues.d_keys[0],
                        &src,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "DOBFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_labels+src,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "DOBFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_pred = -1; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_preds+src,
                        &src_pred,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "DOBFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;

        return retval;
    }

    /** @} */

};

} //namespace dobfs
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
