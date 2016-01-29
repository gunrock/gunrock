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
 *
 * @tparam VertexId            Type of signed integer to use as vertex IDs.
 * @tparam SizeT               Type of int / uint to use for array indexing.
 * @tparam Value               Type of float or double to use for attributes.
 * @tparam _MARK_PREDECESSORS  Whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE Whether to enable idempotent operation.
 * @tparam _USE_DOUBLE_BUFFER  Whether to use double buffer.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool _MARK_PREDECESSORS,
    bool _ENABLE_IDEMPOTENCE,
    bool _USE_DOUBLE_BUFFER >
struct SampleProblem : ProblemBase<VertexId, SizeT, Value,
    _MARK_PREDECESSORS,
    _ENABLE_IDEMPOTENCE,
    _USE_DOUBLE_BUFFER,
    false,   // _ENABLE_BACKWARD
    false,   // _KEEP_ORDER
    false> { // _KEEP_NODE_NUM
    static const bool MARK_PREDECESSORS  =  _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;

    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value> {
        // device storage arrays

        // TODO(developer): other primitive-specific device arrays here
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
     *
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
     *
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
                                 cudaMemcpy(
                                     h_labels,
                                     data_slices[0]->labels.GetPointer(util::DEVICE),
                                     sizeof(VertexId) * nodes,
                                     cudaMemcpyDeviceToHost),
                                 "Problem cudaMemcpy d_labels failed",
                                 __FILE__, __LINE__)) break;

                // TODO(developer): code to extract other results here

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
     * @param[in] streams CUDA streams
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Init(
        bool                         stream_from_host,
        Csr<VertexId, Value, SizeT>& graph,
        int                          _num_gpus,
        cudaStream_t*                streams = NULL) {
        num_gpus = _num_gpus;
        nodes    = graph.nodes;
        edges    = graph.edges;

        ProblemBase <
        VertexId, SizeT, Value,
                  _MARK_PREDECESSORS,
                  _ENABLE_IDEMPOTENCE,
                  _USE_DOUBLE_BUFFER,
                  false, // _ENABLE_BACKWARD
                  false, // _KEEP_ORDER
                  false >::Init(stream_from_host,
                                &graph,
                                NULL,
                                num_gpus,
                                NULL,
                                "random");

        // no data in DataSlice needs to be copied from host

        //
        // Allocate output labels.
        //
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];
        if (streams == NULL) {
            streams = new cudaStream_t[num_gpus];
            streams[0] = 0;
        }

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

                data_slices[0][0].streams.SetPointer(streams, 1);
                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph,      // Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices

                // create SoA on device
                if (retval = data_slices[0]->labels.Allocate(nodes, util::DEVICE)) {
                    return retval;
                }

                // TODO(developer): code to initialize other device arrays here
            }
            // add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for primitive.
     *
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed).
     *  @param[in] queue_sizing Size scaling factor for work queue allocation.
     *  \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Reset(
        FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
        double queue_sizing) {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                             cudaSetDevice(gpu_idx[gpu]),
                             "Problem cudaSetDevice failed",
                             __FILE__, __LINE__)) return retval;

            data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing);

            // allocate output labels if necessary
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) {
                if (retval = data_slices[gpu]->labels.Allocate(nodes, util::DEVICE)) {
                    return retval;
                }
            }

            util::MemsetKernel <<< 128, 128>>>(
                data_slices[gpu]->labels.GetPointer(util::DEVICE),
                _ENABLE_IDEMPOTENCE ? -1 : (util::MaxValue<Value>() - 1), nodes);

            // TODO(developer): code to for other allocations here

            if (retval = util::GRError(
                             cudaMemcpy(d_data_slices[gpu],
                                        data_slices[gpu],
                                        sizeof(DataSlice),
                                        cudaMemcpyHostToDevice),
                             "Problem cudaMemcpy data_slices to d_data_slices failed",
                             __FILE__, __LINE__)) return retval;
        }

        // TODO(developer): fill in the initial input_queue if necessary

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
