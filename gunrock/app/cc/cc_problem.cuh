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

template <
    typename    VertexId,                       // Type of signed integer to use as vertex id (e.g., uint32)
    typename    SizeT,                          // Type of unsigned integer to use for array indexing (e.g., uint32)
    typename    Value,                          // Type of edge value (e.g., float)
    bool        _USE_DOUBLE_BUFFER>
struct CCProblem : ProblemBase<VertexId, SizeT,
                                _USE_DOUBLE_BUFFER>
{
    //Helper structures
/** * Data slice per GPU
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_component_ids;               // Used for component id
        int             *d_masks;                       // Size equals to node number, show if a node is the root
        bool            *d_marks;                       // Size equals to edge number, show if two vertices belong to the same component
        VertexId        *d_froms;                        // Size equals to edge number, from vertex of one edge
        VertexId        *d_tos;                          // Size equals to edge number, to vertex of one edge
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
     * Constructor
     */

    CCProblem():
    nodes(0),
    edges(0),
    num_gpus(0),
    num_components(0) {}

    CCProblem(bool      stream_from_host,       // Only meaningful for single-GPU
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
     * Destructor
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
            if (d_data_slices[i])                   util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices)  delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * Extract into a single host vector the CC results disseminated across
     * all GPUs
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

    // Compute size and root of each component
    void ComputeDetails(VertexId *h_component_ids, VertexId *h_roots, unsigned int *h_histograms)
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

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     * Performs any initialization work needed for CC problem type. Must be called
     * prior to each CC run
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

