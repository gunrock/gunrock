// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_problem.cuh
 *
 * @brief GPU Storage management Structure for CC Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace cc {

/**
 * @brief Connected Component Problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    VertexId,                       // Type of signed integer to use as vertex id (e.g., uint32)
    typename    SizeT,                          // Type of unsigned integer to use for array indexing (e.g., uint32)
    typename    Value,                          // Type of edge value (e.g., float)
    bool        _USE_DOUBLE_BUFFER>
struct CCProblem : ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER>
{
    //Helper structures

    /** 
     * @brief Data slice structure which contains CC problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_component_ids;               // Used for component id
        int             *d_masks;                       // Size equals to node number, show if a node is the root
        bool            *d_marks;                       // Size equals to edge number, show if two vertices belong to the same component
        VertexId        *d_froms;                        // Size equals to edge number, from vertex of one edge
        VertexId        *d_tos;                          // Size equals to edge number, to vertex of one edge
        int             *d_vertex_flag;
        int             *d_edge_flag;
    };

    // Members
    
    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;
    unsigned int        num_components;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    int                 *gpu_idx;

    // Methods

    /**
     * @brief CCProblem default constructor
     */

    CCProblem():
    nodes(0),
    edges(0),
    num_gpus(0),
    num_components(0) {}

    CCProblem(bool      stream_from_host,       // Only meaningful for single-GPU
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
     * @brief CCProblem default destructor
     */
    ~CCProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~CCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_component_ids)    util::GRError(cudaFree(data_slices[i]->d_component_ids), "GpuSlice cudaFree d_component_ids failed", __FILE__, __LINE__);
            if (data_slices[i]->d_froms)    util::GRError(cudaFree(data_slices[i]->d_froms), "GpuSlice cudaFree d_froms failed", __FILE__, __LINE__);
            if (data_slices[i]->d_tos)    util::GRError(cudaFree(data_slices[i]->d_tos), "GpuSlice cudaFree d_tos failed", __FILE__, __LINE__);
            if (data_slices[i]->d_marks)            util::GRError(cudaFree(data_slices[i]->d_marks), "GpuSlice cudaFree d_marks failed", __FILE__, __LINE__);
            if (data_slices[i]->d_masks)            util::GRError(cudaFree(data_slices[i]->d_masks), "GpuSlice cudaFree d_masks failed", __FILE__, __LINE__);
            if (data_slices[i]->d_vertex_flag)            util::GRError(cudaFree(data_slices[i]->d_vertex_flag), "GpuSlice cudaFree d_vertex_flag failed", __FILE__, __LINE__);
            if (data_slices[i]->d_edge_flag)            util::GRError(cudaFree(data_slices[i]->d_edge_flag), "GpuSlice cudaFree d_edge_flag failed", __FILE__, __LINE__);
            if (d_data_slices[i])                   util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices)  delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * @brief Extract into a single host vector the CC results disseminated across all GPUs.
     *
     * @param[out] h_component_ids host-side vector to store computed component ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_component_ids)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "CCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_component_ids,
                                data_slices[0]->d_component_ids,
                                sizeof(VertexId) * nodes,
                                cudaMemcpyDeviceToHost),
                            "CCProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1) 
            for (int i = 0; i < nodes; ++i)
            {
                if (h_component_ids[i] == i)
                {
                   ++num_components;
                }
            }

        } while(0);

        return retval;
    }

    /**
     * @brief Compute histogram for component ids.
     *
     * @param[in] h_component_ids host-side vector stores  component ids.
     * @param[out] h_roots host-side vector to store root node id for each component.
     * @param[out] h_histograms host-side vector to store histograms.
     *
     */
    void ComputeCCHistogram(VertexId *h_component_ids, VertexId *h_roots, unsigned int *h_histograms)
    {
            //Get roots for each component and the total number of component.
            num_components = 0;
            for (int i = 0; i < nodes; ++i)
            {
                if (h_component_ids[i] == i)
                {
                   h_roots[num_components] = i;
                   h_histograms[num_components] = 0;
                   ++num_components;
                }
            }

            for (int i = 0; i < nodes; ++i)
            {
                for (int j = 0; j < num_components; ++j)
                {
                    if (h_component_ids[i] == h_roots[j])
                    {
                        ++h_histograms[j];
                        break;
                    }
                }
            }
    }

    /**
     * @brief CCProblem initialization
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
            const Csr<VertexId, Value, SizeT> &graph,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;
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
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];

    
        cudaError_t retval = cudaSuccess;
        
        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "CCProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;

                // Construct coo from/to edge list from row_offsets and column_indices
                VertexId *froms = new VertexId[edges];
                VertexId *tos = new VertexId[edges];
                for (int i = 0; i < nodes; ++i)
                {
                    for (int j = h_row_offsets[i]; j < h_row_offsets[i+1]; ++j)
                    {
                        froms[j] = i;
                        tos[j] = h_column_indices[j];
                    }
                }

                VertexId    *d_froms;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_froms,
                                edges * sizeof(VertexId)),
                            "CCProblem cudaMalloc d_froms failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_froms = d_froms;

                VertexId    *d_tos;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_tos,
                                edges * sizeof(VertexId)),
                            "CCProblem cudaMalloc d_tos failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_tos = d_tos;

                if (retval = util::GRError(cudaMemcpy(
                                d_froms,
                                froms,
                                edges * sizeof(VertexId),
                                cudaMemcpyHostToDevice),
                            "CCProblem cudaMemcpy froms to d_froms failed", __FILE__, __LINE__)) return retval;

                if (retval = util::GRError(cudaMemcpy(
                                d_tos,
                                tos,
                                edges * sizeof(VertexId),
                                cudaMemcpyHostToDevice),
                            "CCProblem cudaMemcpy tos to d_tos failed", __FILE__, __LINE__)) return retval; 

                if (froms) delete[] froms;
                if (tos) delete[] tos;

                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "CCProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                VertexId    *d_component_ids;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_component_ids,
                        nodes * sizeof(VertexId)),
                    "CCProblem cudaMalloc d_component_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_component_ids = d_component_ids;
 
                int   *d_masks;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_masks,
                        nodes * sizeof(int)),
                    "CCProblem cudaMalloc d_masks failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_masks = d_masks;

                bool   *d_marks;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_marks,
                        edges * sizeof(bool)),
                    "CCProblem cudaMalloc d_marks failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_marks = d_marks;

                int    *d_vertex_flag;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_vertex_flag,
                        sizeof(int)),
                    "CCProblem cudaMalloc d_vertex_flag failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_vertex_flag = d_vertex_flag;

                int    *d_edge_flag;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_edge_flag,
                        sizeof(int)),
                    "CCProblem cudaMalloc d_edge_flag failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_edge_flag = d_edge_flag;
            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for CC problem type. Must be called prior to each BC run.
     *
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
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
                        "CCProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output component_ids if necessary
            if (!data_slices[gpu]->d_component_ids) {
                VertexId    *d_component_ids;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_component_ids,
                                nodes * sizeof(VertexId)),
                            "CCProblem cudaMalloc d_component_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_component_ids = d_component_ids;
            }

            util::MemsetIdxKernel<<<128, 128>>>(data_slices[gpu]->d_component_ids, nodes);

            // Allocate marks if necessary
            if (!data_slices[gpu]->d_marks) {
                bool    *d_marks;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_marks,
                                edges * sizeof(bool)),
                            "CCProblem cudaMalloc d_marks failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_marks = d_marks;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_marks, false, edges);

            // Allocate masks if necessary
            if (!data_slices[gpu]->d_masks) {
                int    *d_masks;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_masks,
                                nodes * sizeof(int)),
                            "CCProblem cudaMalloc d_masks failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_masks = d_masks;
            }
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_masks, 0, nodes);
            int *vertex_flag = new int;
            int *edge_flag = new int;
            // Allocate vertex_flag if necessary
            if (!data_slices[gpu]->d_vertex_flag) {
                int    *d_vertex_flag;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_vertex_flag,
                                sizeof(int)),
                            "CCProblem cudaMalloc d_vertex_flag failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_vertex_flag = d_vertex_flag;
            }
            vertex_flag[0] = 1;
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_vertex_flag,
                            vertex_flag,
                            sizeof(int),
                            cudaMemcpyHostToDevice),
                        "CCProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;
            delete vertex_flag;

            // Allocate edge_flag if necessary
            if (!data_slices[gpu]->d_edge_flag) {
                int    *d_edge_flag;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_edge_flag,
                                sizeof(int)),
                            "CCProblem cudaMalloc d_edge_flag failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_edge_flag = d_edge_flag;
            }

            edge_flag[0] = 1;
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_edge_flag,
                            edge_flag,
                            sizeof(int),
                            cudaMemcpyHostToDevice),
                        "CCProblem cudaMemcpy edge_flag to d_edge_flag failed", __FILE__, __LINE__)) return retval;
            delete edge_flag;
                
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "CCProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;
        }

        // Initialize edge frontier_queue
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], edges);

        // Initialize vertex frontier queue
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_values[0], nodes);
        
        return retval;
    }

};

} //namespace cc
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
