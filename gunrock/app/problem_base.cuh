// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base struct for all the application types
 */

#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>

#include <vector>

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */

enum FrontierType {
    VERTEX_FRONTIERS,       // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,         // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS         // O(n) global vertex frontier, O(m) global edge frontier
};


/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    bool        _USE_DOUBLE_BUFFER>

struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;

    /**
     * Load instruction cache-modifier const defines.
     */

    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER                    = util::io::ld::cg;             // Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
    static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER                   = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR column-indices.
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = util::io::ld::NONE;           // Load instruction cache-modifier for reading edge values.
    static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER       = util::io::ld::cg;             // Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
    static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER     = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
    static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER                   = util::io::st::cg;             // Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer

    /**
     * @brief Graph slice structure which contains common graph structural data and input/output queue.
     */
    struct GraphSlice
    {
        //Slice Index
        int             index;

        SizeT           *d_row_offsets;             // CSR format row offset on device memory
        VertexId        *d_column_indices;          // CSR format column indices on device memory
        SizeT           *d_column_offsets;          // CSR format column offset on device memory
        VertexId        *d_row_indices;             // CSR format row indices on device memory

        //Frontier queues. Used to track working frontier.
        util::DoubleBuffer<VertexId, VertexId>      frontier_queues;
        SizeT                                       frontier_elements[2];

        //Number of nodes and edges in slice
        VertexId        nodes;
        SizeT           edges;

        //CUDA stream to use for processing the slice
        cudaStream_t    stream;

        /**
         * @brief GraphSlice Constructor
         *
         * @param[in] index GPU index, reserved for multi-GPU use in future.
         * @param[in] stream CUDA Stream we use to allocate storage for this graph slice.
         */
        GraphSlice(int index, cudaStream_t stream) :
            index(index),
            d_row_offsets(NULL),
            d_column_indices(NULL),
            d_column_offsets(NULL),
            d_row_indices(NULL),
            nodes(0),
            edges(0),
            stream(stream)
        {
            // Initialize double buffer frontier queue lengths
            for (int i = 0; i < 2; ++i)
            {
                frontier_elements[i] = 0;
            }
        }

        /**
         * @brief GraphSlice Destructor to free all device memories.
         */
        virtual ~GraphSlice()
        {
            // Set device (use slice index)
            util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

            // Free pointers
            if (d_row_offsets)      util::GRError(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
            if (d_column_indices)   util::GRError(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
            if (d_column_offsets)   util::GRError(cudaFree(d_column_offsets), "GpuSlice cudaFree d_column_offsets failed", __FILE__, __LINE__);
            if (d_row_indices)      util::GRError(cudaFree(d_row_indices), "GpuSlice cudaFree d_row_indices failed", __FILE__, __LINE__);
            for (int i = 0; i < 2; ++i) {
                if (frontier_queues.d_keys[i])      util::GRError(cudaFree(frontier_queues.d_keys[i]), "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__);
                if (frontier_queues.d_values[i])    util::GRError(cudaFree(frontier_queues.d_values[i]), "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__);
            }

            // Destroy stream
            if (stream) {
                util::GRError(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
            }
        }
    };

    // Members

    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Set of graph slices (one for each GPU)
    GraphSlice**        graph_slices;

    // Methods
    
    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase() :
        num_gpus(0),
        nodes(0),
        edges(0)
        {}
    
    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        // Cleanup graph slices on the heap
        for (int i = 0; i < num_gpus; ++i)
        {
            delete graph_slices[i];
        }
        delete[] graph_slices;
    }

    /**
     * @brief Get the GPU index for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Index of the gpu that owns the neighbor list of the specified vertex
     */
    template <typename VertexId>
    int GpuIndex(VertexId vertex)
    {
        if (num_gpus <= 1) {
            
            // Special case for only one GPU, which may be set as with
            // an ordinal other than 0.
            return graph_slices[0]->index;
        } else {
            return vertex % num_gpus;
        }
    }

    /**
     * @brief Get the row offset for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Row offset of the specified vertex. If a single GPU is used,
     * this will be the same as the vertex id.
     */
    template <typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
        return vertex / num_gpus;
    }

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] nodes Number of nodes in the CSR graph.
     * @param[in] edges Number of edges in the CSR graph.
     * @param[in] h_row_offsets Host-side row offsets array.
     * @param[in] h_column_indices Host-side column indices array.
     * @param[in] h_column_offsets Host-side column offsets array.
     * @param[in] h_row_indices Host-side row indices array.
     * @param[in] num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        SizeT       nodes,
        SizeT       edges,
        SizeT       *h_row_offsets,
        VertexId    *h_column_indices,
        SizeT       *h_column_offsets = NULL,
        VertexId    *h_row_indices = NULL,
        int         num_gpus = 1)
    {
        cudaError_t retval      = cudaSuccess;
        this->nodes             = nodes;
        this->edges             = edges;
        this->num_gpus          = num_gpus;

        do {
            graph_slices = new GraphSlice*[num_gpus];
            if (num_gpus <= 1) {

                // Create a single graph slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
                graph_slices[0] = new GraphSlice(gpu, 0);
                graph_slices[0]->nodes = nodes;
                graph_slices[0]->edges = edges;

                if (stream_from_host) {

                    // Map the pinned graph pointers into device pointers
                    if (retval = util::GRError(cudaHostGetDevicePointer(
                                    (void **)&graph_slices[0]->d_row_offsets,
                                    (void *) h_row_offsets, 0),
                                "ProblemBase cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaHostGetDevicePointer(
                                    (void **)&graph_slices[0]->d_column_indices,
                                    (void *) h_column_indices, 0),
                                "ProblemBase cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;
                    if (h_column_offsets != NULL) {
                        if (retval = util::GRError(cudaHostGetDevicePointer(
                                        (void **)&graph_slices[0]->d_column_offsets,
                                        (void *) h_column_offsets, 0),
                                    "ProblemBase cudaHostGetDevicePointer d_column_offsets failed", __FILE__, __LINE__)) break;
                    }

                    if (h_row_indices != NULL) {
                        if (retval = util::GRError(cudaHostGetDevicePointer(
                                        (void **)&graph_slices[0]->d_row_indices,
                                        (void *) h_row_indices, 0),
                                    "ProblemBase cudaHostGetDevicePointer d_row_indices failed", __FILE__, __LINE__)) break;
                    }
                } else {

                    // Allocate and initialize d_row_offsets
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&graph_slices[0]->d_row_offsets,
                        (graph_slices[0]->nodes+1) * sizeof(SizeT)),
                        "ProblemBase cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                        graph_slices[0]->d_row_offsets,
                        h_row_offsets,
                        (graph_slices[0]->nodes+1) * sizeof(SizeT),
                        cudaMemcpyHostToDevice),
                        "ProblemBase cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
                    
                    // Allocate and initialize d_column_indices
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&graph_slices[0]->d_column_indices,
                        graph_slices[0]->edges * sizeof(VertexId)),
                        "ProblemBase cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                        graph_slices[0]->d_column_indices,
                        h_column_indices,
                        graph_slices[0]->edges * sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                        "ProblemBase cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

                    if (h_column_offsets != NULL) {
                        // Allocate and initialize d_column_offsets
                        if (retval = util::GRError(cudaMalloc(
                                        (void**)&graph_slices[0]->d_column_offsets,
                                        (graph_slices[0]->nodes+1) * sizeof(SizeT)),
                                    "ProblemBase cudaMalloc d_column_offsets failed", __FILE__, __LINE__)) break;

                        if (retval = util::GRError(cudaMemcpy(
                                        graph_slices[0]->d_column_offsets,
                                        h_column_offsets,
                                        (graph_slices[0]->nodes+1) * sizeof(SizeT),
                                        cudaMemcpyHostToDevice),
                                    "ProblemBase cudaMemcpy d_column_offsets failed", __FILE__, __LINE__)) break;
                    }

                    if (h_row_indices != NULL) {
                        // Allocate and initialize d_row_indices
                        if (retval = util::GRError(cudaMalloc(
                                        (void**)&graph_slices[0]->d_row_indices,
                                        graph_slices[0]->edges * sizeof(VertexId)),
                                    "ProblemBase cudaMalloc d_row_indices failed", __FILE__, __LINE__)) break;

                        if (retval = util::GRError(cudaMemcpy(
                                        graph_slices[0]->d_row_indices,
                                        h_row_indices,
                                        graph_slices[0]->edges * sizeof(VertexId),
                                        cudaMemcpyHostToDevice),
                                    "ProblemBase cudaMemcpy d_row_indices failed", __FILE__, __LINE__)) break;
                    }

                } //end if(stream_from_host)
            } else {
                //TODO: multiple GPU graph slices
            }//end if(num_gpu<=1)
        } while (0);

        return retval;
    }

    /**
     * @brief Performs any initialization work needed for ProblemBase. Must be called prior to each search
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] queue_sizing Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing = 2.0)            // Size scaling factor for work queue allocation
        {
            cudaError_t retval = cudaSuccess;

            for (int gpu = 0; gpu < num_gpus; ++gpu) {

                // Set device
                if (retval = util::GRError(cudaSetDevice(graph_slices[gpu]->index),
                            "ProblemBase cudaSetDevice failed", __FILE__, __LINE__)) return retval;

                //
                // Allocate frontier queues if necessary
                //

                // Determine frontier queue sizes
                SizeT new_frontier_elements[2] = {0,0};

                switch (frontier_type) {
                    case VERTEX_FRONTIERS :
                        // O(n) ping-pong global vertex frontiers
                        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case EDGE_FRONTIERS :
                        // O(m) ping-pong global edge frontiers
                        new_frontier_elements[0] = double(graph_slices[gpu]->edges > graph_slices[gpu]->nodes ? graph_slices[gpu]->edges : graph_slices[gpu]->nodes) * queue_sizing;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case MIXED_FRONTIERS :
                        // O(n) global vertex frontier, O(m) global edge frontier
                        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
                        new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
                        break;

                    }

                // Iterate through global frontier queue setups
                for (int i = 0; i < 2; i++) {

                    // Allocate frontier queue if not big enough
                    if (graph_slices[gpu]->frontier_elements[i] < new_frontier_elements[i]) {

                        // Free if previously allocated
                        if (graph_slices[gpu]->frontier_queues.d_keys[i]) {
                            if (retval = util::GRError(cudaFree(
                                            graph_slices[gpu]->frontier_queues.d_keys[i]),
                                        "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                        }

                        // Free if previously allocated
                        if (_USE_DOUBLE_BUFFER) {
                            if (graph_slices[gpu]->frontier_queues.d_values[i]) {
                                if (retval = util::GRError(cudaFree(
                                                graph_slices[gpu]->frontier_queues.d_values[i]),
                                            "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                            }
                        }

                        graph_slices[gpu]->frontier_elements[i] = new_frontier_elements[i];

                        if (retval = util::GRError(cudaMalloc(
                                        (void**) &graph_slices[gpu]->frontier_queues.d_keys[i],
                                        graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
                                    "ProblemBase cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                        if (_USE_DOUBLE_BUFFER) {
                            if (retval = util::GRError(cudaMalloc(
                                            (void**) &graph_slices[gpu]->frontier_queues.d_values[i],
                                            graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
                                        "ProblemBase cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                        }
                    }
                }
            }
            
            return retval;
        }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
