// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mis_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

#include <algorithm>
#include <vector>

namespace gunrock {
namespace app {
namespace mis {

/**
 * @brief Maximal Independent Set Problem structure stores device-side vectors for doing MIS on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of unsigned integer to use for store independent set ID.
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value>
struct MISProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{

    typedef _VertexId 			VertexId;
	typedef _SizeT			    SizeT;
	typedef _Value              Value;

    static const bool MARK_PREDECESSORS     = false;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains MIS problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        Value   *d_labels;  /**< for MinMax method, d_labels store generated random numbers, for hash method, d_labels stores the hash_func(node_id)*/
        Value   *d_mis_ids; /**< Store the MIS IDs (or you can imagine it as graph coloring*/
        Value   *d_values_to_reduce; /**< Store values to reduce, length=|E|*/
        Value   *d_reduced_values; /**< Store reduced values, length=|V|*/
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
     * @brief MISProblem default constructor
     */

    MISProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief MISProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    MISProblem(bool        stream_from_host,       // Only meaningful for single-GPU
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
     * @brief MISProblem default destructor
     */
    ~MISProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~MISProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels)      util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (data_slices[i]->d_mis_ids)      util::GRError(cudaFree(data_slices[i]->d_mis_ids), "GpuSlice cudaFree d_mis_ids failed", __FILE__, __LINE__);
            if (data_slices[i]->d_values_to_reduce)      util::GRError(cudaFree(data_slices[i]->d_values_to_reduce), "GpuSlice cudaFree d_value_to_reduce failed", __FILE__, __LINE__);
            if (data_slices[i]->d_reduced_values)      util::GRError(cudaFree(data_slices[i]->d_reduced_values), "GpuSlice cudaFree d_reduced_values failed", __FILE__, __LINE__);
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
     * @param[out] h_node_id host-side vector to store node Vertex ID.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_mis_ids)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "MISProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_mis_ids,
                                data_slices[0]->d_mis_ids,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "MISProblem cudaMemcpy d_mis_ids failed", __FILE__, __LINE__)) break;

            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief MISProblem initialization
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
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "MISProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "MISProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                Value *d_random_labels;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_random_labels,
                        nodes * sizeof(Value)),
                    "MISProblem cudaMalloc d_random_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_random_labels;

                Value    *d_mis_ids;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_mis_ids,
                        nodes * sizeof(Value)),
                    "MISProblem cudaMalloc d_mis_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_mis_ids = d_mis_ids;

                Value *d_values_to_reduce;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_values_to_reduce,
                        edges * sizeof(Value)),
                    "MISProblem cudaMalloc d_values_to_reduce failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_values_to_reduce = d_values_to_reduce;

                Value *d_reduced_values;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_reduced_values,
                        nodes * sizeof(Value)),
                    "MISProblem cudaMalloc d_reduced_values failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_reduced_values = d_reduced_values;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for MIS problem type. Must be called prior to each MIS iteration.
     *
     *  @param[in] src Source node for one MIS computing pass.
     *  @param[in] delta Tuning parameter for switching to backward BFS
     *  @param[in] threshold Threshold for remove node from MIS computation process.
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
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output page ranks if necessary
            if (!data_slices[gpu]->d_labels) {
                Value *d_labels;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_labels,
                                nodes * sizeof(Value)),
                            "MISProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            if (!data_slices[gpu]->d_mis_ids) {
                Value    *d_mis_ids;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_mis_ids,
                                nodes * sizeof(Value)),
                            "MISProblem cudaMalloc d_mis_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_mis_ids = d_mis_ids;
            }

            if (!data_slices[gpu]->d_values_to_reduce) {
                Value *d_vtr;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_vtr,
                                edges * sizeof(Value)),
                            "MISProblem cudaMalloc d_values_to_reduce failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_values_to_reduce = d_vtr;
            }

            if (!data_slices[gpu]->d_reduced_values) {
                Value *d_rv;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rv,
                                nodes * sizeof(Value)),
                            "MISProblem cudaMalloc d_reduced_values failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_reduced_values = d_rv;
            }

            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                            "MISProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;


            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_mis_ids, -1, nodes);
        }

        // Fillin the initial input_queue for MIS problem, this needs to be modified
        // in multi-GPU scene
        std::vector<Value> rand_vec;

        // Use random_shuffle now, limited to less than 4 billion nodes though.
        // TODO: when use CUDA 7, switch to shuffle()
        for (int i = 0; i < nodes; ++i) {
            rand_vec.push_back(i);
        }
        std::random_shuffle(rand_vec.begin(), rand_vec.end());

        if (retval = util::GRError(cudaMemcpy(
                                data_slices[0]->d_labels,
                                &rand_vec[0],
                                sizeof(Value) * nodes,
                                cudaMemcpyHostToDevice),
                            "MISProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) return retval;

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        return retval;
    }

    /** @} */

};

} //namespace mis
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
