// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sample_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace sample {

/**
 * @brief Problem structure stores device-side vectors
 * @tparam _VertexId Type use as vertex id (e.g., uint32)
 * @tparam _SizeT    Type use for array indexing. (e.g., uint32)
 * @tparam _Value    Type use for computed value.
 */
template<typename _VertexId, typename _SizeT, typename _Value>
struct SampleProblem : ProblemBase<_VertexId, _SizeT, false> {
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT;
    typedef _Value    Value;

    static const bool MARK_PREDECESSORS  = false;
    static const bool ENABLE_IDEMPOTENCE = false;

    /**
     * @brief Data slice structure which contains problem specific data.
     */
    struct DataSlice {
        // device storage arrays
        VertexId *d_labels;  // used for ...

        // TODO: other primitive-specific device arrays here
    };

    int       num_gpus;
    SizeT     nodes;
    SizeT     edges;

    // data slices (one for each GPU)
    DataSlice **data_slices;

    // putting structure on device while keeping the SoA structure
    DataSlice **d_data_slices;

    // device index for each data slice
    int       *gpu_idx;

    /**
     * @brief Default constructor
     */
    SampleProblem(): nodes(0), edges(0), num_gpus(0) {}

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    SampleProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph,
                  int   num_gpus) :
        num_gpus(num_gpus) {
        Init(stream_from_host, graph, num_gpus);
    }

    /**
     * @brief Default destructor
     */
    ~SampleProblem() {
        for (int i = 0; i < num_gpus; ++i) {
            if (util::GRError(
                    cudaSetDevice(gpu_idx[i]),
                    "~Problem cudaSetDevice failed",
                    __FILE__, __LINE__)) break;
            if (data_slices[i]->d_labels) {
                util::GRError(cudaFree(data_slices[i]->d_labels),
                              "GpuSlice cudaFree d_labels failed",
                              __FILE__, __LINE__);
            }

            // TODO: code to clean up primitive-specific device arrays here

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
    cudaError_t Extract(VertexId *h_labels) {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                                  "Problem cudaSetDevice failed",
                                  __FILE__, __LINE__)) break;

                if (retval = util::GRError(
                        cudaMemcpy(h_labels,
                                   data_slices[0]->d_labels,
                                   sizeof(VertexId) * nodes,
                                   cudaMemcpyDeviceToHost),
                        "Problem cudaMemcpy d_labels failed",
                        __FILE__, __LINE__)) break;

                // TODO: code to extract other results here

            } else {
                // multi-GPU extension code
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
        bool  stream_from_host,  // only meaningful for single-GPU
        const Csr<VertexId, Value, SizeT> &graph,
        int   _num_gpus) {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;

        ProblemBase<_VertexId, _SizeT, false>::Init(
            stream_from_host,
            nodes,
            edges,
            h_row_offsets,
            h_column_indices,
            NULL,
            NULL,
            num_gpus);

        // no data in DataSlice needs to be copied from host

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];

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

                // create SoA on device
                VertexId *d_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_labels,
                                   nodes * sizeof(VertexId)),
                        "Problem cudaMalloc d_labels failed",
                        __FILE__, __LINE__)) return retval;
                data_slices[0]->d_labels = d_labels;

                // TODO: code to initialize other device arrays here
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

        typedef ProblemBase<_VertexId, _SizeT, false> BaseProblem;

        // load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(gpu_idx[gpu]),
                    "Problem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

            // allocate output labels if necessary
            if (!data_slices[gpu]->d_labels) {
                VertexId *d_labels;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_labels, nodes * sizeof(VertexId)),
                        "Problem cudaMalloc d_labels failed",
                        __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            util::MemsetKernel<<< 128, 128>>>(
                data_slices[gpu]->d_labels, -1, nodes);

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
        // e.g., put every vertex in frontier queue
        util::MemsetIdxKernel<<<128, 128>>>(
            BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        return retval;
    }

    /** @} */
};

}  // namespace sample
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
