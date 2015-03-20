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
    false> // _KEEP_NODE_NUM
{
    //static const bool MARK_PREDECESSORS     = true;
    //static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains PR problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > rank_curr;           /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, Value   > rank_next;           /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, SizeT   > degrees;             /**< Used for keeping out-degree for each vertex */
        util::Array1D<SizeT, SizeT   > degrees_pong;
        util::Array1D<SizeT, SizeT   > labels;
        util::Array1D<SizeT, VertexId> node_ids;
        Value    threshold;               /**< Used for recording accumulated error */
        Value    delta;
        VertexId src_node;

        ~DataSlice()
        {
            if (util::SetDevice(gpu_idx[i])) return;
            rank_curr   .Release();
            rank_next   .Release();
            degrees     .Release();
            degrees_pong.Release();
            labels      .Release();
            node_ids    .Release();
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
            SizeT       edges  = graph->edges;

            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            // Create SoA on device
            if (retval = rank_curr  .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = rank_next  .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = degrees    .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = degree_pong.Allocate(nodes, util::DEVICE)) return retval;
            if (retval = node_ids   .Allocate(nodes, util::DEVICE)) return retval;
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
        for (int i = 0; i < num_gpus; ++i)
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
            if (this->num_gpus == 1) {
                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
                data_slices[0]->rank_curr.SetPointer(h_rank);
                if (retval = data_slices[0]->rank_curr.Move(util::DEVICE, util::HOST)) return retval;
                data_slices[0]->node_ids .SetPointer(h_node_id);
                if (retval = data_slices[0]->node_ids .Move(util::DEVICE, util::HOST)) return retval;
            } else {
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
            } //end if (this->num_gpus ==1)
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
        ProblemBase<VertexId, SizeT, Value, false, false, false, false, false, false> :: Init(
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
                data_slices[gpu].Setname("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* data_slice_ = data_slices[gpu].GetPointer(util::HOST);
                if (retval = data_slice_->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    this->num_gpus>1? 1 : 0,
                    this->num_gpus>1? 1 : 0,
                    &(this->sub_graphs[gpu]),
                    this->num_gpus>1? this->graph_slices[gpu]->in_counter .GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                    queue_sizing,
                    in_sizing)) return retval;
            }
        } while (0);

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
            FrontierType frontier_type, // The frontier type (i.e., edge/vertex/mixed)
            double   queue_sizing)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            SizeT nodes = this->sub_graphs[gpu].nodes;
            SizeT edges = this->sub_graphs[gpu].edges;

            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output page ranks if necessary
            if (retval = data_slice[gpu]->Reset(frontier_type, this->slices[gpu], queue_sizing, USE_DOUBLE_BUFFER)) return retval;
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
                if (retval = data_slices[gpu]->degrees_pong.Allocate(nodes, util::DEVICE)) return retval; 

            // Initial rank_next = 0 
            //util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_curr,
            //    (Value)1.0/nodes, nodes);
            util::MemsetKernel          <<<128, 128>>>(
                data_slices[gpu]->rank_next.GetPointer(util::DEVICE), (Value)0.0, nodes);
            
            // Compute degrees
            util::MemsetKernel          <<<128, 128>>>(
                data_slices[gpu]->degrees  .GetPointer(util::DEVICE), 0, nodes);
            util::MemsetMadVectorKernel <<<128, 128>>>(
                data_slices[gpu]->degrees.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE)+1, -1, nodes);
            util::MemsetCopyVectorKernel<<<128, 128>>>(
                data_slices[gpu]->degrees_pong.GetPointer(util::DEVICE), 
                data_slices[gpu]->degrees     .GetPointer(util::DEVICE), nodes);
            util::MemsetIdxKernel       <<<128, 128>>>(
                data_slices[gpu]->node_ids    .GetPointer(util::DEVICE), nodes);

            data_slices[gpu]->delta = delta;
            data_slices[gpu]->threshold = threshold;
            data_slices[gpu]->src_node = src;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;

            // Put every vertex in there
            util::MemsetIdxKernel<<<128, 128>>>(data_slices[gpu]->frontier_queues.keys[0].GetPointer(util::DEVICE), nodes);
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
