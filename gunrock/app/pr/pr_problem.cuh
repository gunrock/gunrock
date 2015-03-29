// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief PageRank Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing PR value.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value>
struct PRProblem : ProblemBase<VertexId, SizeT, Value,
    true,  // _MARK_PREDECESSORS
    false, // _ENABLE_IDEMPOTENCE
    false, // _USE_DOUBLE_BUFFER
    false, // _ENABLE_BACKWARD
    false, // _KEEP_ORDER
    true> // _KEEP_NODE_NUM
{
    //static const bool MARK_PREDECESSORS     = true;
    //static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains PR problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > rank_curr;           /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, Value   > rank_next;           /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, SizeT   > degrees;             /**< Used for keeping out-degree for each vertex */
        util::Array1D<SizeT, SizeT   > degrees_pong;
        util::Array1D<SizeT, SizeT   > labels;
        util::Array1D<SizeT, VertexId> node_ids;
        util::Array1D<SizeT, SizeT   > markers;
        util::Array1D<SizeT, VertexId> *temp_keys_out;
        Value    threshold;               /**< Used for recording accumulated error */
        Value    delta;
        VertexId src_node;
        bool     to_continue;
        SizeT    local_nodes;
        SizeT    edge_map_queue_len;
        SizeT    max_iter;
        SizeT    PR_queue_length;
        int      PR_queue_selector;
        bool     final_event_set;

        DataSlice()
        {
            rank_curr   .SetName("rank_curr"   );
            rank_next   .SetName("rank_next"   );
            degrees     .SetName("degrees"     );
            degrees_pong.SetName("degrees_pong");
            labels      .SetName("labels"      );
            node_ids    .SetName("node_ids"    );
            markers     .SetName("markers"     );
            temp_keys_out = NULL;
            //threshold   .SetName("threshold"   );
            //delta       .SetName("delta"       );
            //src_node    .SetName("src_node"    );
            threshold   = 0;
            delta       = 0;
            src_node    = -1;
            to_continue = true;
            local_nodes = 0;
            edge_map_queue_len = 0;
            max_iter    = 0;
            final_event_set = false;
            PR_queue_length = 0;
            PR_queue_selector = 0;
        }

        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            rank_curr   .Release();
            rank_next   .Release();
            degrees     .Release();
            degrees_pong.Release();
            labels      .Release();
            node_ids    .Release();
            markers     .Release();
            delete[] temp_keys_out; temp_keys_out = NULL;
            //threshold   .Release();
            //delta       .Release();
            //src_node    .Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT       nodes  = graph->nodes;
            //SizeT       edges  = graph->edges;
            //SizeT *_num_in_nodes  = num_in_nodes;
            //SizeT *_num_out_nodes = num_out_nodes;
            
            /*if (num_gpus > 1)
            {
                //_num_in_nodes = new  SizeT[num_gpus+1];
                //_num_out_nodes = new SizeT[num_gpus+1];
                for (int gpu=0; gpu<=num_gpus; gpu++)
                {
                    num_in_nodes [gpu] = nodes;
                    num_out_nodes[gpu] = gpu==1 ? nodes : 0;
                }
            }*/

            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            temp_keys_out = new util::Array1D<SizeT, VertexId>[num_gpus];
            if (num_gpus > 1)
            {
                printf("Allocating keys_out[0] %d\n", local_nodes);fflush(stdout);
                if (retval = this->keys_out[0].Allocate(local_nodes, util::DEVICE)) return retval;
                this->keys_outs[0] = this->keys_out[0].GetPointer(util::DEVICE);
                for (int peer_ = 0; peer_ < num_gpus; peer_++)
                {
                    if (retval = this->keys_marker[peer_].EnsureSize(nodes)) return retval;
                    this->keys_markers[peer_] = this->keys_marker[peer_].GetPointer(util::DEVICE);
                }
                this->keys_markers.Move(util::HOST, util::DEVICE);
            }
            for (int peer_ = 2; peer_ < num_gpus; peer_++)
            {
                temp_keys_out  [peer_] = this->keys_out[peer_];
                this->keys_out [peer_] = this->keys_out[1];
                //this->keys_out [peer_].SetPointer(this->keys_out[1].GetPointer(util::DEVICE), this->keys_out[1].GetSize(), util::DEVICE);
                this->keys_outs[peer_] = this->keys_out[1].GetPointer(util::DEVICE);
                
                //this->value__associate_out[peer_][0].SetPointer(this->value__associate_out[1][0].GetPointer(util::DEVICE), this->value__associate_out[1][0].GetSize(), util::DEVICE);
                //this->value__associate_outs[peer_][0] = this->value__associate_out[1][0].GetPointer(util::DEVICE);
                //if (retval = this->value__associate_outs[peer_].Move(util::HOST, util::DEVICE)) return retval;
            }
            this->keys_outs.Move(util::HOST, util::DEVICE);
    
            // Create SoA on device
            if (retval = rank_curr   .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = rank_next   .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = degrees     .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = degrees_pong.Allocate(nodes+1, util::DEVICE)) return retval;
            if (retval = node_ids    .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = markers     .Allocate(nodes, util::DEVICE)) return retval;
            //if (retval = delta       .Allocate(1    , util::DEVICE | util::HOST)) return retval;
            //if (retval = threshold   .Allocate(1    , util::DEVICE | util::HOST)) return retval;
            //if (retval = src_node    .Allocate(1    , util::DEVICE | util::HOST)) return retval;
            return retval;
       }
    };

    // Members
    
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;
   
    // Methods

    /**
     * @brief PRProblem default constructor
     */
    PRProblem():
    data_slices(NULL) {}

    /**
     * @brief PRProblem default destructor
     */
    ~PRProblem()
    {
        if (data_slices == NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices; data_slices=NULL;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_rank host-side vector to store page rank values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_rank, VertexId *h_node_id)
    {
        cudaError_t retval = cudaSuccess;

        do {
            //if (this->num_gpus == 1) {
                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
                data_slices[0]->rank_curr.SetPointer(h_rank);
                if (retval = data_slices[0]->rank_curr.Move(util::DEVICE, util::HOST)) return retval;
                data_slices[0]->node_ids .SetPointer(h_node_id);
                if (retval = data_slices[0]->node_ids .Move(util::DEVICE, util::HOST)) return retval;
            /*} else {
                Value    **th_rank    = new Value*   [this->num_gpus];
                VertexId **th_node_id = new VertexId*[this->num_gpus];
                for (int gpu=0; gpu<this->num_gpus; gpu++)
                {
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                    if (retval = data_slices[gpu]->rank_curr.Move(util::DEVICE, util::HOST)) return retval;
                    if (retval = data_slices[gpu]->node_ids .Move(util::DEVICE, util::HOST)) return retval;
                    th_rank   [gpu] = data_slices[gpu]->rank_curr.GetPointer(util::HOST);
                    th_node_id[gpu] = data_slices[gpu]->node_ids .GetPointer(util::HOST);
                }
                for (VertexId node = 0; node<this->nodes; node++)
                {
                    int gpu = this->partition_tables[0][node];
                    VertexId node_ = this->convertion_tables[0][node];
                    h_rank   [node] = th_rank   [gpu][node_];
                    h_node_id[node] = th_node_id[gpu][node_];
                }
                delete[] th_rank   ; th_rank    = NULL;
                delete[] th_node_id; th_node_id = NULL;
            } //end if (this->num_gpus ==1)*/
        } while(0);

        return retval;
    }

    /**
     * @brief PRProblem initialization
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
        ProblemBase<VertexId, SizeT, Value, true, false, false, false, false, true> :: Init(
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
        SizeT *local_nodes = new SizeT[this->num_gpus];

        if (this->num_gpus > 1)
        {
            for (int gpu=0; gpu<this->num_gpus; gpu++)
                local_nodes[gpu] = 0;
            for (SizeT v=0; v<graph->nodes; v++)
                local_nodes[this->partition_tables[0][v]] ++;
            for (int gpu=0; gpu<this->num_gpus; gpu++)
            for (int peer=0; peer<this->num_gpus; peer++)
            {
                int peer_;
                if (gpu == peer) peer_ = 0;
                else peer_ = gpu<peer? peer: peer+1;
                SizeT max_nodes = local_nodes[gpu] > local_nodes[peer]? local_nodes[gpu] : local_nodes[peer];
                this->graph_slices[gpu]->in_counter[peer_] = max_nodes;
                this->graph_slices[gpu]->out_counter[peer_] = max_nodes;
            }
        }

        do {
            for (int gpu=0; gpu<this->num_gpus; gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* data_slice_ = data_slices[gpu].GetPointer(util::HOST);
                data_slice_->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);
                if (this->num_gpus > 1) data_slice_->local_nodes = local_nodes[gpu];
                if (retval = data_slice_->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    this->num_gpus>1? 0 : 0,
                    this->num_gpus>1? 1 : 0,
                    &(this->sub_graphs[gpu]),
                    this->num_gpus>1? this->graph_slices[gpu]->in_counter .GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                    queue_sizing,
                    in_sizing)) return retval;
            }
        } while (0);

        delete[] local_nodes; local_nodes = NULL;
        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for PR problem type. Must be called prior to each PR iteration.
     *
     *  @param[in] src Source node for one PR computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId src,
            Value    delta,
            Value    threshold,
            SizeT    max_iter,
            FrontierType frontier_type, // The frontier type (i.e., edge/vertex/mixed)
            double   queue_sizing)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            SizeT nodes = this->sub_graphs[gpu].nodes;
            //SizeT edges = this->sub_graphs[gpu].edges;

            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output page ranks if necessary
            if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, false)) return retval;
            if (data_slices[gpu]->rank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->rank_curr.Allocate(nodes, util::DEVICE)) return retval;
            
            if (data_slices[gpu]->rank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->rank_next.Allocate(nodes, util::DEVICE)) return retval;

            if (data_slices[gpu]->node_ids .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->node_ids .Allocate(nodes, util::DEVICE)) return retval;

            // Allocate degrees if necessary
            if (data_slices[gpu]->degrees  .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->degrees  .Allocate(nodes, util::DEVICE)) return retval;
            // Allocate degrees_pong if necessary
            if (data_slices[gpu]->degrees_pong.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->degrees_pong.Allocate(nodes+1, util::DEVICE)) return retval; 

            // Initial rank_next = 0 
            //util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_curr,
            //    (Value)1.0/nodes, nodes);
            util::MemsetKernel          <<<128, 128>>>(
                data_slices[gpu]->rank_next.GetPointer(util::DEVICE), (Value)0.0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->rank_curr.GetPointer(util::DEVICE), (Value)(1.0 - delta), nodes);
            
            // Compute degrees
            util::MemsetKernel          <<<128, 128>>>(
                data_slices[gpu]->degrees  .GetPointer(util::DEVICE), 0, nodes);
            
            if (this->num_gpus == 1)
            {
                util::MemsetMadVectorKernel <<<128, 128>>>(
                    data_slices[gpu]->degrees.GetPointer(util::DEVICE), 
                    this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE), 
                    this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE)+1, -1, nodes);
                data_slices[gpu]->local_nodes = nodes;

                if (retval = data_slices[gpu]->frontier_queues[0].keys[0].EnsureSize(nodes)) return retval;
                util::MemsetIdxKernel<<<128, 128>>>(data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);
            } else {
                data_slices[gpu]->degrees_pong.SetPointer(this->org_graph->row_offsets, nodes+1, util::HOST);
                data_slices[gpu]->degrees_pong.Move(util::HOST, util::DEVICE);
                util::MemsetMadVectorKernel <<<128, 128>>>(
                    data_slices[gpu]->degrees.GetPointer(util::DEVICE),
                    data_slices[gpu]->degrees_pong.GetPointer(util::DEVICE),
                    data_slices[gpu]->degrees_pong.GetPointer(util::DEVICE)+1, -1, nodes);
                data_slices[gpu]->degrees_pong.UnSetPointer(util::HOST);

                //util::cpu_mt::PrintCPUArray<SizeT, int>("partition_table", this->graph_slices[gpu]->partition_table.GetPointer(util::HOST), nodes, gpu);
                if (retval = data_slices[gpu]->frontier_queues[0].keys[0].EnsureSize(data_slices[gpu]->local_nodes)) return retval; 
                VertexId *temp_keys = new VertexId[data_slices[gpu]->frontier_queues[0].keys[0].GetSize()];
                SizeT counter = 0;
                for (VertexId v=0; v<nodes; v++)
                if (this->graph_slices[gpu]->partition_table[v] == 0)
                {
                    temp_keys[counter] = v;
                    counter++;
                }
                data_slices[gpu]->local_nodes = counter;
                
                //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("temp_keys", temp_keys, counter, gpu);
                data_slices[gpu]->frontier_queues[0].keys[0].SetPointer(temp_keys);
                data_slices[gpu]->frontier_queues[0].keys[0].Move(util::HOST, util::DEVICE);
                data_slices[gpu]->frontier_queues[0].keys[0].UnSetPointer(util::HOST);
                delete[] temp_keys;temp_keys=NULL; 
                //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys", data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), counter, gpu);
            }
            util::MemsetCopyVectorKernel<<<128, 128>>>(
                data_slices[gpu]->degrees_pong.GetPointer(util::DEVICE), 
                data_slices[gpu]->degrees     .GetPointer(util::DEVICE), nodes);
            util::MemsetIdxKernel       <<<128, 128>>>(
                data_slices[gpu]->node_ids    .GetPointer(util::DEVICE), nodes);

            //printf("delta = %d threshold = %f src_node = %d\n", delta, threshold, src);
            data_slices[gpu]->delta       = delta    ; //data_slices[gpu]->delta    .Move(util::HOST, util::DEVICE);
            data_slices[gpu]->threshold   = threshold; //data_slices[gpu]->threshold.Move(util::HOST, util::DEVICE);
            data_slices[gpu]->src_node    = src      ; //data_slices[gpu]->src_node .Move(util::HOST, util::DEVICE);
            //util::cpu_mt::PrintGPUArray("delta"    , data_slices[gpu]->delta    .GetPointer(util::DEVICE), 1);
            //util::cpu_mt::PrintGPUArray("threshold", data_slices[gpu]->threshold.GetPointer(util::DEVICE), 1);
            //util::cpu_mt::PrintGPUArray("src_node" , data_slices[gpu]->src_node .GetPointer(util::DEVICE), 1);
            data_slices[gpu]->to_continue = true;
            data_slices[gpu]->max_iter    = max_iter;
            data_slices[gpu]->final_event_set = false;
            data_slices[gpu]->PR_queue_length = 1;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        return retval;
    }

    /** @} */

};

} //namespace pr
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
