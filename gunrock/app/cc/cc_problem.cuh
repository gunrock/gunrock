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
 * @brief Connected Component Problem structure stores device-side vectors for doing connected component computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        _USE_DOUBLE_BUFFER>
struct CCProblem : ProblemBase<VertexId, SizeT, Value,
    false, // _MARK_PREDECESSORS
    false, // _ENABLE_IDEMPOTENCE,
    _USE_DOUBLE_BUFFER, 
    false, // _EnABLE_BACKWARD
    false, // _KEEP_ORDER
    true>  // _KEEP_NODE_NUM
{
    //Helper structures

    /** 
     * @brief Data slice structure which contains CC problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, VertexId> component_ids; /**< Used for component id */
        util::Array1D<SizeT, VertexId> old_c_ids;
        util::Array1D<SizeT, SizeT   > CID_markers;
        util::Array1D<SizeT, int     > masks;         /**< Size equals to node number, show if a node is the root */
        util::Array1D<SizeT, bool    > marks;         /**< Size equals to edge number, show if two vertices belong to the same component */
        util::Array1D<SizeT, VertexId> froms;         /**< Size equals to edge number, from vertex of one edge */
        util::Array1D<SizeT, VertexId> tos;           /**< Size equals to edge number, to vertex of one edge */
        util::Array1D<SizeT, int     > vertex_flag;   /**< Finish flag for per-vertex kernels in CC algorithm */
        util::Array1D<SizeT, int     > edge_flag;     /**< Finish flag for per-edge kernels in CC algorithm */
        util::Array1D<SizeT, VertexId> labels;
        util::Array1D<SizeT, VertexId> preds;
        util::Array1D<SizeT, VertexId> temp_preds;
        int turn;
        //DataSlice *d_pointer;
        bool has_change;
        //util::CtaWorkProgressLifetime *work_progress;

        DataSlice()
        {
            component_ids.SetName("component_ids");
            old_c_ids    .SetName("old_c_ids"    );
            CID_markers  .SetName("CID_markers"  );
            masks        .SetName("masks"        );
            marks        .SetName("marks"        );
            froms        .SetName("froms"        );
            tos          .SetName("tos"          );
            vertex_flag  .SetName("vertex_flag"  );
            edge_flag    .SetName("edge_flag"    );
            turn          = 0;
            //d_pointer     = NULL;
            //work_progress = NULL;
            has_change    = true;
            //labels       .SetName("labels"       );
        }

        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            component_ids.Release();
            old_c_ids    .Release();
            CID_markers  .Release();
            masks        .Release();
            marks        .Release();
            froms        .Release();
            tos          .Release();
            vertex_flag  .Release();
            edge_flag    .Release();
            //d_pointer     = NULL;
            //work_progress = NULL;
            //labels       .Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            VertexId *original_vertex,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {   
            cudaError_t retval = cudaSuccess;
            SizeT       nodes  = graph->nodes;
            SizeT       edges  = graph->edges;

            if (num_gpus>1) for (int gpu=0; gpu<num_gpus; gpu++)
            {
                num_in_nodes [gpu] = nodes;
                num_out_nodes[gpu] = gpu==1?nodes:0;
            }

            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;
            for (int peer_ = 2; peer_ < num_gpus; peer_++)
            {
                this->keys_out [peer_].SetPointer(this->keys_out[1].GetPointer(util::DEVICE), this->keys_out[1].GetSize(), util::DEVICE);
                this->keys_outs[peer_] = this->keys_out[1].GetPointer(util::DEVICE);
                this->vertex_associate_out[peer_][0].SetPointer(this->vertex_associate_out[1][0].GetPointer(util::DEVICE), this->vertex_associate_out[1][0].GetSize(), util::DEVICE);
                this->vertex_associate_outs[peer_][0] = this->vertex_associate_out[1][0].GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE)) return retval;
            }

            //printf("@ gpu %d: nodes = %d, edges = %d\n", gpu_idx, nodes, edges);
            // Create a single data slice for the currently-set gpu
            if (retval = froms .Allocate(edges, util::HOST | util::DEVICE)) return retval;
            if (retval = tos   .Allocate(edges, util::DEVICE)) return retval;
            if (retval = tos   .SetPointer(graph->column_indices)) return retval;
            // Construct coo from/to edge list from row_offsets and column_indices
            for (int node=0; node<graph->nodes; node++)
            {
                //if (node == 131070 || node == 131071) 
                //    printf("node %d @ gpu %d : %d -> %d\n", node, gpu_idx, graph->row_offsets[node], graph->row_offsets[node+1]);
                int start_edge = graph->row_offsets[node], end_edge = graph->row_offsets[node+1];
                for (int edge = start_edge; edge < end_edge; ++edge)
                {
                    froms[edge] = node;
                    //tos  [edge] = graph->column_indices[edge];
                    //if (froms[edge]==131070 || froms[edge]==131071 || tos[edge]==131070 || tos[edge]==131071)
                    //    printf("edge %d @ gpu %d : %d -> %d\n", edge, gpu_idx, froms[edge], tos[edge]); 
                }
            }
            if (retval = froms.Move(util::HOST, util::DEVICE)) return retval;
            if (retval = tos  .Move(util::HOST, util::DEVICE)) return retval;
            if (retval = froms.Release(util::HOST)) return retval;
            if (retval = tos  .Release(util::HOST)) return retval;

            // Create SoA on device
            if (retval = component_ids.Allocate(nodes  , util::DEVICE)) return retval;
            if (retval = old_c_ids    .Allocate(nodes  , util::DEVICE)) return retval;
            if (retval = CID_markers  .Allocate(nodes+1, util::DEVICE)) return retval;
            if (retval = masks        .Allocate(nodes  , util::DEVICE)) return retval;
            if (retval = marks        .Allocate(edges  , util::DEVICE)) return retval;
            if (retval = vertex_flag  .Allocate(1, util::HOST | util::DEVICE)) return retval;
            if (retval = edge_flag    .Allocate(1, util::HOST | util::DEVICE)) return retval;

            if (retval = this->frontier_queues[0].keys  [0].Allocate(edges+2, util::DEVICE)) return retval;
            if (retval = this->frontier_queues[0].keys  [1].Allocate(edges+2, util::DEVICE)) return retval;
            if (retval = this->frontier_queues[0].values[0].Allocate(nodes+2, util::DEVICE)) return retval;
            if (retval = this->frontier_queues[0].values[1].Allocate(nodes+2, util::DEVICE)) return retval;
            if (num_gpus > 1) {
                this->frontier_queues[num_gpus].keys  [0].SetPointer(this->frontier_queues[0].keys  [0].GetPointer(util::DEVICE), edges+2, util::DEVICE);
                this->frontier_queues[num_gpus].keys  [1].SetPointer(this->frontier_queues[0].keys  [1].GetPointer(util::DEVICE), edges+2, util::DEVICE);
                this->frontier_queues[num_gpus].values[0].SetPointer(this->frontier_queues[0].values[0].GetPointer(util::DEVICE), nodes+2, util::DEVICE);
                this->frontier_queues[num_gpus].values[1].SetPointer(this->frontier_queues[0].values[1].GetPointer(util::DEVICE), nodes+2, util::DEVICE);
            }
            return retval;
        }
    };

    // Members
    unsigned int        num_components;

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;
   
    // Methods

    /**
     * @brief CCProblem default constructor
     */
    CCProblem()
    {
        num_components = 0;
        data_slices    = NULL;
    }

    /**
     * @brief CCProblem default destructor
     */
    ~CCProblem()
    {
        if (data_slices == NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices;data_slices=NULL;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result component ids computed on the GPU back to a host-side vector.
     *
     * @param[out] h_component_ids host-side vector to store computed component ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_component_ids)
    {
        cudaError_t retval = cudaSuccess;
        int *marker=new int[this->nodes];
        memset(marker, 0, sizeof(int) * this->nodes);

        do {
            if (this->num_gpus == 1) {
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
                data_slices[0]->component_ids.SetPointer(h_component_ids);
                if (retval = data_slices[0]->component_ids.Move(util::DEVICE, util::HOST)) return retval;
                num_components=0;
                for (int node=0; node<this->nodes; node++)
                if (marker[h_component_ids[node]] == 0) 
                {
                    num_components++;
                    //printf("%d\t ",node);
                    marker[h_component_ids[node]]=1;
                }

            } else {
                VertexId **th_component_ids = new VertexId*[this->num_gpus];
                for (int gpu=0; gpu< this->num_gpus; gpu++)
                {
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                    if (retval = data_slices[gpu]->component_ids.Move(util::DEVICE, util::HOST)) return retval;
                    th_component_ids[gpu] = data_slices[gpu]->component_ids.GetPointer(util::HOST);
                }
                
                num_components=0;
                for (VertexId node=0; node<this->nodes; node++)
                {
                    h_component_ids[node]=th_component_ids[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                    if (marker[h_component_ids[node]] == 0) 
                    {
                        num_components++;
                        //printf("%d ",node);
                        marker[h_component_ids[node]]=1;
                    }
                }
            } //end if
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
        //Get roots for each component and the total number of component
        //VertexId *min_nodes = new VertexId[this->nodes];
        VertexId *counter   = new VertexId[this->nodes];
        for (int i = 0; i < this->nodes; i++)
        {
            //min_nodes[i] = this->nodes;
            counter  [i] = 0;
        }
        //for (int i = 0; i < this->nodes; i++)
        //    if (min_nodes[h_component_ids[i]] > i) min_nodes[h_component_ids[i]] = i;
        num_components = 0;
        for (int i = 0; i < this->nodes; i++)
        {
            if (counter[h_component_ids[i]]==0)
            {
                //h_histograms[num_components] = counter[h_component_ids[i]];
                h_roots[num_components] = i;
                ++num_components;
                //printf("%d\t", i);
            }
            counter[h_component_ids[i]]++;
        }
        for (int i = 0; i < num_components; i++)
            h_histograms[i] = counter[h_component_ids[h_roots[i]]];
        /*for (int i = 0; i < this->nodes; ++i)
        {
            if (h_component_ids[i] == i)
            {
               h_roots[num_components] = i;
               h_histograms[num_components] = counter[h_component_ids[i]];
               ++num_components;
            }
        }*/

        /*for (int i = 0; i < this->nodes; ++i)
        {
            for (int j = 0; j < num_components; ++j)
            {
                if (h_component_ids[i] == h_roots[j])
                {
                    ++h_histograms[j];
                    break;
                }
            }
        }*/
        //delete[] min_nodes; min_nodes = NULL;
        delete[] counter  ; counter   = NULL;
    }

    /**
     * @brief CCProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool          stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> 
                         *graph,
            Csr<VertexId, Value, SizeT>
                         *inversgraph      = NULL,
            int           num_gpus         = 1,
            int          *gpu_idx          = NULL,
            std::string   partition_method = "random",
            cudaStream_t *streams          = NULL,
            float         queue_sizing     = 2.0f,
            float         in_sizing        = 1.0f,
            float         partition_factor = -1.0f,
            int           partition_seed   = -1)
    {
        ProblemBase<VertexId, SizeT, Value, false, false, _USE_DOUBLE_BUFFER, false, false, true>::Init(
            stream_from_host,
            graph,
            inversgraph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);

        // No data in DataSlice needs to be copied from host
         
        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
        
        do {
            for (int gpu=0; gpu<this->num_gpus; gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* data_slice_ = data_slices[gpu].GetPointer(util::HOST);
                //data_slice_->d_pointer = data_slices[gpu].GetPointer(util::DEVICE);
                data_slice_->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);
                if (retval = data_slice_->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    this->num_gpus>1? 1:0,
                    0,
                    &(this->sub_graphs[gpu]),
                    this->num_gpus>1? this->graph_slices[gpu]->in_counter .GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->original_vertex.GetPointer(util::HOST) : NULL,
                    queue_sizing,
                    in_sizing)) return retval;
            }
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for CC problem type. Must be called prior to each CC run.
     *
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,   // The frontier type (i.e., edge/vertex/mixed)

        double       queue_sizing)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            SizeT nodes = this->sub_graphs[gpu].nodes;
            SizeT edges = this->sub_graphs[gpu].edges;
            DataSlice *data_slice_ = data_slices[gpu].GetPointer(util::HOST);
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            //if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, _USE_DOUBLE_BUFFER)) return retval;
            if (retval = data_slice_->frontier_queues[0].keys  [0].EnsureSize(edges+2)) return retval;
            if (retval = data_slice_->frontier_queues[0].keys  [1].EnsureSize(edges+2)) return retval;
            if (retval = data_slice_->frontier_queues[0].values[0].EnsureSize(nodes+2)) return retval;
            if (retval = data_slice_->frontier_queues[0].values[1].EnsureSize(nodes+2)) return retval;

            // Allocate output component_ids if necessary
            util::MemsetIdxKernel<<<128, 128>>>(data_slice_->component_ids .GetPointer(util::DEVICE), nodes);

            // Allocate marks if necessary
            util::MemsetKernel   <<<128, 128>>>(data_slice_->marks         .GetPointer(util::DEVICE), false, edges);

            // Allocate masks if necessary
            util::MemsetKernel    <<<128, 128>>>(data_slice_->masks        .GetPointer(util::DEVICE),     0, nodes);

            // Allocate vertex_flag if necessary
            data_slice_->vertex_flag[0]=1;
            if (retval = data_slice_->vertex_flag.Move(util::HOST, util::DEVICE)) return retval;

            // Allocate edge_flag if necessary
            data_slice_->vertex_flag[0]=1;
            if (retval = data_slice_->edge_flag  .Move(util::HOST, util::DEVICE)) return retval;

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;  
 
            // Initialize edge frontier_queue
            util::MemsetIdxKernel<<<128, 128>>>(data_slice_->frontier_queues[0].keys  [0].GetPointer(util::DEVICE), edges);

            // Initialize vertex frontier queue
            util::MemsetIdxKernel<<<128, 128>>>(data_slice_->frontier_queues[0].values[0].GetPointer(util::DEVICE), nodes);
        }
       
        return retval;
    }

    /** @} */
};

} //namespace cc
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
