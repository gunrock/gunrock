// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_problem.cuh
 *
 * @brief GPU Storage management Structure for BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE  Boolean type parameter which defines whether to enable idempotence operation for graph traverse.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value,                          
    bool        _MARK_PREDECESSORS,             
    bool        _ENABLE_IDEMPOTENCE,
    bool        _USE_DOUBLE_BUFFER>
struct BFSProblem : ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    //Helper structures

    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_labels;              /**< Used for source distance */
        VertexId        *d_preds;               /**< Used for predecessor */
        unsigned char   *d_visited_mask;        /**< Used for bitmask for visited nodes */
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
     * @brief BFSProblem default constructor
     */

    BFSProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief BFSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    BFSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
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
     * @brief BFSProblem default destructor
     */
    ~BFSProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels)      util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (data_slices[i]->d_preds)      util::GRError(cudaFree(data_slices[i]->d_preds), "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
            if (data_slices[i]->d_visited_mask)      util::GRError(cudaFree(data_slices[i]->d_visited_mask), "GpuSlice cudaFree d_visited_mask failed", __FILE__, __LINE__);
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
                            "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_labels,
                                data_slices[0]->d_labels,
                                sizeof(VertexId) * nodes,
                                cudaMemcpyDeviceToHost),
                            "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

                if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE) {
                    if (retval = util::GRError(cudaMemcpy(
                                    h_preds,
                                    data_slices[0]->d_preds,
                                    sizeof(VertexId) * nodes,
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                }

            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief BFSProblem initialization
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
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "BFSProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "BFSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                VertexId    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_labels,
                        nodes * sizeof(VertexId)),
                    "BFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_labels;
 
                VertexId   *d_preds = NULL;
                if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE) {
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_preds,
                        nodes * sizeof(VertexId)),
                    "BFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                }
                data_slices[0]->d_preds = d_preds;

                unsigned char *d_visited_mask = NULL;
                int visited_mask_bytes  = ((nodes * sizeof(unsigned char))+7)/8;
                if (_ENABLE_IDEMPOTENCE) {
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_visited_mask,
                        visited_mask_bytes),
                    "BFSProblem cudaMalloc d_visited_mask failed", __FILE__, __LINE__)) return retval;
                }
                data_slices[0]->d_visited_mask = d_visited_mask;
            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BFS problem type. Must be called prior to each BFS run.
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
        typedef ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER> BaseProblem;
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
                            "BFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, -1, nodes);

            // Allocate preds if necessary
            if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE && !data_slices[gpu]->d_preds) {
                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_preds,
                                nodes * sizeof(VertexId)),
                            "BFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_preds = d_preds;
            }

            if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)
                util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_preds, -2, nodes);

            if (_ENABLE_IDEMPOTENCE) {
                int visited_mask_bytes  = ((nodes * sizeof(unsigned char))+7)/8;
                int visited_mask_elements = visited_mask_bytes * sizeof(unsigned char);
                util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_visited_mask, (unsigned char)0, visited_mask_elements);
            }
                
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }

        
        // Fillin the initial input_queue for BFS problem, this needs to be modified
        // in multi-GPU scene
        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[0]->frontier_queues.d_keys[0],
                        &src,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_labels+src,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE) {
            VertexId src_pred = -1; 
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[0]->d_preds+src,
                            &src_pred,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        }

        return retval;
    }

    /** @} */

};

} //namespace bfs
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
