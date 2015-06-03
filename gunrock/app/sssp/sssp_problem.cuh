// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Single-Source Shortest Path Problem structure stores device-side vectors for doing SSSP computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 */
template <
    typename    VertexId,                       
    typename    SizeT,
    typename    Value,
    bool        _MARK_PATHS>
struct SSSPProblem : ProblemBase<VertexId, SizeT, Value, 
    true, //MARK_PREDECESSORS
    false, //ENABLE_IDEMPOTENCE
    false, //USE_DOUBLE_BUFFER
    false, //ENABLE_BACKWARD
    false, //KEEP_ORDER
    false> //KEEP_NODE_NUM
{
    static const bool MARK_PATHS            = _MARK_PATHS;

    //Helper structures

    /**
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, Value       >    labels     ;     /**< Used for source distance */
        util::Array1D<SizeT, Value       >    weights    ;     /**< Used for storing edge weights */
        util::Array1D<SizeT, VertexId    >    visit_lookup;    /**< Used for check duplicate */
        util::Array1D<SizeT, float       >    delta;
        util::Array1D<SizeT, int         >    sssp_marker;

        DataSlice()
        {
            labels          .SetName("labels"          );  
            weights         .SetName("weights"         );
            visit_lookup    .SetName("visit_lookup"    );
            delta           .SetName("delta"           );
            sssp_marker     .SetName("sssp_marker"     );
        }

        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            weights       .Release();
            visit_lookup  .Release();
            delta         .Release();
            sssp_marker   .Release();
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
            int   delta_factor = 16,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {
            cudaError_t retval  = cudaSuccess;
            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            if (retval = labels      .Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = weights     .Allocate(graph->edges,util::DEVICE)) return retval;
            if (retval = delta       .Allocate(1           ,util::DEVICE)) return retval;
            if (retval = visit_lookup.Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = sssp_marker .Allocate(graph->nodes,util::DEVICE)) return retval;
 
            weights.SetPointer(graph->edge_values, graph->edges, util::HOST);
            if (retval = weights.Move(util::HOST, util::DEVICE)) return retval;
            
            float _delta = EstimatedDelta(graph)*delta_factor;
            printf("estimated delta:%5f\n", _delta);
            delta.SetPointer(&_delta, util::HOST);
            if (retval = delta.Move(util::HOST, util::DEVICE)) return retval;

            if (MARK_PATHS)
            {
                if (retval = this->preds.Allocate(graph->nodes,util::DEVICE)) return retval;
                if (retval = this->temp_preds.Allocate(graph->nodes, util::DEVICE)) return retval;
            } else {
                if (retval = this->preds.Release()) return retval;
                if (retval = this->temp_preds.Release()) return retval;
            }

            if (num_gpus >1)
            {
                this->value__associate_orgs[0] = labels.GetPointer(util::DEVICE);
                if (MARK_PATHS)
                    this->vertex_associate_orgs[0] = this->preds.GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                //if (retval = temp_marker.Allocate(graph->nodes, util::DEVICE)) return retval;
            }

            //util::cpu_mt::PrintMessage("DataSlice Init() end.");
            return retval;
        } // Init
        
        float EstimatedDelta(const Csr<VertexId, Value, SizeT> &graph) {
            double  avgV = graph.average_edge_value;
            int     avgD = graph.average_degree;
            return avgV * 32 / avgD;
        }

        cudaError_t Reset(
            FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
            GraphSlice<SizeT, VertexId, Value>  *graph_slice,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {   
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice -> nodes;
            SizeT edges = graph_slice -> edges;
            SizeT new_frontier_elements[2] = {0,0};
            if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

            for (int peer=0; peer<this->num_gpus; peer++)
                this->out_length[peer] = 1;

            if (this->num_gpus>1)
                util::cpu_mt::PrintCPUArray<int, SizeT>("in_counter", graph_slice->in_counter.GetPointer(util::HOST), this->num_gpus+1, this->gpu_idx);

            for (int peer=0;peer<(this->num_gpus > 1 ? this->num_gpus+1 : 1);peer++)
            for (int i=0; i < 2; i++)
            {
                double queue_sizing_ = i==0?queue_sizing : queue_sizing1;
                switch (frontier_type) {
                    case VERTEX_FRONTIERS :
                        // O(n) ping-pong global vertex frontiers
                        new_frontier_elements[0] = double(this->num_gpus>1? graph_slice->in_counter[peer]:nodes) * queue_sizing_ +2;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case EDGE_FRONTIERS :
                        // O(m) ping-pong global edge frontiers
                        new_frontier_elements[0] = double(edges) * queue_sizing_ +2;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case MIXED_FRONTIERS :
                        // O(n) global vertex frontier, O(m) global edge frontier
                        new_frontier_elements[0] = double(this->num_gpus>1? graph_slice->in_counter[peer]:nodes) * queue_sizing_ +2;
                        new_frontier_elements[1] = double(edges) * queue_sizing_ +2;
                        break;
                }

                // Iterate through global frontier queue setups
                //for (int i = 0; i < 2; i++) 
                {
                    if (peer == this->num_gpus && i == 1) continue;
                    if (new_frontier_elements[i] > edges + 2 && queue_sizing_ > 10) new_frontier_elements[i] = edges + 2;
                    //if (peer == this->num_gpus && new_frontier_elements[i] > nodes * this->num_gpus) new_frontier_elements[i] = nodes * this->num_gpus;
                    if (this->frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]) {

                        // Free if previously allocated
                        if (retval = this->frontier_queues[peer].keys[i].Release()) return retval;

                        // Free if previously allocated
                        if (false) {
                            if (retval = this->frontier_queues[peer].values[i].Release()) return retval;
                        }
                        //frontier_elements[peer][i] = new_frontier_elements[i];

                        if (retval = this->frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                        if (false) {
                            if (retval = this->frontier_queues[peer].values[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                        }
                    } //end if
                } // end for i<2

                if (peer == this->num_gpus || i == 1)
                {
                    continue;
                }
                //if (peer == num_gpu) continue;
                SizeT max_elements = new_frontier_elements[0];
                if (new_frontier_elements[1] > max_elements) max_elements=new_frontier_elements[1];
                if (max_elements > nodes) max_elements = nodes;
                if (this->scanned_edges[peer].GetSize() < max_elements)
                {
                    if (retval = this->scanned_edges[peer].Release()) return retval;
                    if (retval = this->scanned_edges[peer].Allocate(max_elements, util::DEVICE)) return retval;
                }
            }
           
            // Allocate output labels if necessary
            if (this->labels      .GetPointer(util::DEVICE) == NULL)
                if (retval = this->labels      .Allocate(nodes, util::DEVICE)) return retval;

            if (this->preds       .GetPointer(util::DEVICE) == NULL && MARK_PATHS)
                if (retval = this->preds       .Allocate(nodes, util::DEVICE)) return retval;

            if (this->visit_lookup.GetPointer(util::DEVICE) == NULL)
                if (retval = this->visit_lookup.Allocate(nodes, util::DEVICE)) return retval;
            
            if (MARK_PATHS) util::MemsetIdxKernel<<<128, 128>>>(this->preds.GetPointer(util::DEVICE), nodes);
            util::MemsetKernel<<<128, 128>>>(this->labels      .GetPointer(util::DEVICE), util::MaxValue<Value>(), nodes);
            util::MemsetKernel<<<128, 128>>>(this->visit_lookup.GetPointer(util::DEVICE), -1, nodes);
            util::MemsetKernel<<<128, 128>>>(sssp_marker.GetPointer(util::DEVICE), (int)0, nodes);
            return retval;
        }
    }; // DataSlice

    // Members   
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>          *data_slices;
   
    // Methods

    /**
     * @brief SSSPProblem default constructor
     */

    SSSPProblem()
    {
        data_slices = NULL;
    }

    /**
     * @brief SSSPProblem default destructor
     */
    ~SSSPProblem()
    {
        if (data_slices==NULL) return;
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
     * @brief Copy result labels computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store computed node predecessors (used for extracting the actual shortest path).
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {

                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                data_slices[0]->labels.SetPointer(h_labels);
                if (retval = data_slices[0]->labels.Move(util::DEVICE,util::HOST)) return retval;

                if (MARK_PATHS) {
                    data_slices[0]->preds.SetPointer(h_preds);
                    if (retval = data_slices[0]->preds.Move(util::DEVICE,util::HOST)) return retval;
                }   

            } else {
                Value    **th_labels=new Value*[this->num_gpus];
                VertexId **th_preds =new VertexId*[this->num_gpus];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {   
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                    if (retval = data_slices[gpu]->labels.Move(util::DEVICE,util::HOST)) return retval;
                    th_labels[gpu]=data_slices[gpu]->labels.GetPointer(util::HOST);
                    if (MARK_PATHS) {
                        if (retval = data_slices[gpu]->preds.Move(util::DEVICE,util::HOST)) return retval;
                        th_preds[gpu]=data_slices[gpu]->preds.GetPointer(util::HOST);
                    }   
                } //end for(gpu)

                for (VertexId node=0;node<this->nodes;node++)
                if (this-> partition_tables[0][node]>=0 && this-> partition_tables[0][node]<this->num_gpus &&
                    this->convertion_tables[0][node]>=0 && this->convertion_tables[0][node]<data_slices[this->partition_tables[0][node]]->labels.GetSize())
                    h_labels[node]=th_labels[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                else {
                    printf("OutOfBound: node = %d, partition = %d, convertion = %d\n",
                           node, this->partition_tables[0][node], this->convertion_tables[0][node]); 
                           //data_slices[this->partition_tables[0][node]]->labels.GetSize());
                    fflush(stdout);
                }   

               if (MARK_PATHS)
                    for (VertexId node=0;node<this->nodes;node++)
                        h_preds[node]=th_preds[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {   
                    if (retval = data_slices[gpu]->labels.Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->preds.Release(util::HOST)) return retval;
                }   
                delete[] th_labels;th_labels=NULL;
                delete[] th_preds ;th_preds =NULL;
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief SSSPProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     * @param[in] delta_factor Parameter for delta-stepping SSSP
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool          stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> *graph,
            Csr<VertexId, Value, SizeT> *inversgraph = NULL,
            int           num_gpus = 1,
            int*          gpu_idx  = NULL,
            std::string   partition_method = "random",
            cudaStream_t* streams = NULL,
            int           delta_factor = 16,
            float         queue_sizing = 2.0,
            float         in_sizing = 1.0,
            float         partition_factor = -1.0,
            int           partition_seed   = -1)
    {
        ProblemBase<VertexId, SizeT, Value, true, false, false, false, false, false>::Init(
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

        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
                _data_slice->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);

                _data_slice->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    this->num_gpus > 1? (MARK_PATHS ? 1 : 0) : 0,
                    this->num_gpus > 1? 1 : 0,
                    &(this->sub_graphs[gpu]),
                    this->num_gpus > 1? this->graph_slices[gpu]->in_counter.GetPointer(util::HOST) : NULL,
                    this->num_gpus > 1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST): NULL,
                    this->num_gpus > 1? this->graph_slices[gpu]->original_vertex.GetPointer(util::HOST) : NULL,
                    delta_factor,
                    queue_sizing,
                    in_sizing);
            } // end for (gpu)
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for SSSP problem type. Must be called prior to each SSSP run.
     *
     *  @param[in] src Source node for one SSSP computing pass.
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

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing)) return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fillin the initial input_queue for SSSP problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1)
        {
            gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables [0][src];
            tsrc= this->convertion_tables[0][src];
        }
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        Value src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,
                        &src_label,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        if (MARK_PATHS)
        {
            VertexId src_pred = -1;
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu]->preds.GetPointer(util::DEVICE)+tsrc,
                &src_pred,
                sizeof(Value),
                cudaMemcpyHostToDevice),
                "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        }
        return retval;
    }

    /** @} */

};

} //namespace sssp
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
