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
struct SSSPProblem : ProblemBase<VertexId, SizeT, Value, false>
{
    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;
    static const bool MARK_PATHS            = _MARK_PATHS;

    //Helper structures

    /**
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value       >    labels;          /**< Used for source distance */
        util::Array1D<SizeT, Value       >    weights;         /**< Used for storing edge weights */
        util::Array1D<SizeT, VertexId    >    preds;           /**< Used for storing the actual shortest path */
        //util::Array1D<SizeT, VertexId    >    visit_lookup;    /**< Used for check duplicate */
        //util::Array1D<SizeT, float       >    delta;
        int                                   num_associate,gpu_idx;
        util::Array1D<SizeT, VertexId    >    *associate_in[2];
        util::Array1D<SizeT, VertexId*   >    associate_ins[2];
        util::Array1D<SizeT, VertexId    >    *associate_out;
        util::Array1D<SizeT, VertexId*   >    associate_outs;
        util::Array1D<SizeT, VertexId*   >    associate_orgs;
        util::Array1D<SizeT, SizeT       >    out_length    ;   
        util::Array1D<SizeT, SizeT       >    in_length[2]  ;
        util::Array1D<SizeT, VertexId    >    keys_in  [2]  ;

        DataSlice()
        {
            num_associate   = 0;
            gpu_idx         = 0;
            associate_in[0] = NULL;
            associate_in[1] = NULL;
            associate_out   = NULL;
            labels          .SetName("labels"          );  
            preds           .SetName("preds"           );  
            weights         .SetName("weights"         );
            //visit_lookup    .SetName("visit_lookup"    );
            //delta           .SetName("delta"           );
            associate_ins[0].SetName("associate_ins[0]");
            associate_ins[1].SetName("associate_ins[1]");
            associate_outs  .SetName("associate_outs"  );  
            associate_orgs  .SetName("associate_orgs"  );  
            out_length      .SetName("out_length"      );  
            in_length    [0].SetName("in_length[0]"    );  
            in_length    [1].SetName("in_length[1]"    );  
            keys_in      [0].SetName("keys_in[0]"      );  
            keys_in      [1].SetName("keys_in[1]"      );  
        }

        ~DataSlice()
        {
            util::cpu_mt::PrintMessage("~DataSlice() begin.");
            if (util::SetDevice(gpu_idx)) return;
            labels        .Release();
            preds         .Release();
            weights       .Release();
            //visit_lookup  .Release();
            //delta         .Release();
            keys_in    [0].Release();
            keys_in    [1].Release();
            //visited_mask  .Release();
            in_length  [0].Release();
            in_length  [1].Release();
            out_length    .Release();
            associate_orgs.Release();

            if (associate_in[0] != NULL)
            {
                for (int i=0;i<num_associate;i++)
                {
                    associate_in[0][i].Release();
                    associate_in[1][i].Release();
                }
                delete[] associate_in[0];
                delete[] associate_in[1];
                associate_in[0]=NULL;
                associate_in[1]=NULL;
                associate_ins[0].Release();
                associate_ins[1].Release();
            }

            if (associate_out != NULL)
            {
                for (int i=0;i<num_associate;i++)
                    associate_out[i].Release();
                delete[] associate_out;
                associate_out=NULL;
                associate_outs.Release();
            }
            util::cpu_mt::PrintMessage("~DataSlice() end.");
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT num_in_nodes,
            SizeT num_out_nodes)
        {
            cudaError_t retval  = cudaSuccess;
            this->gpu_idx       = gpu_idx;
            this->num_associate = num_associate;
            if (retval = util::SetDevice(gpu_idx))  return retval;
            // Create SoA on device
            //printf("#edges=%d, edge_values=%p\n",graph->edges,graph->edge_values);
            //util::cpu_mt::PrintCPUArray<SizeT,Value>("weight",graph->edge_values,graph->edges);
            if (retval = labels .Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = weights.Allocate(graph->edges,util::DEVICE)) return retval;
            weights.SetPointer(graph->edge_values, graph->edges, util::HOST);
            if (retval = weights.Move(util::HOST, util::DEVICE)) return retval;
            //printf("on cpu=%p, on gpu=%p\n",weights.GetPointer(util::HOST),weights.GetPointer(util::DEVICE));
            
            /*if (retval = delta.Allocate(1,util::DEVICE)) return retval;
            float _delta = EstimatedDelta(graph)*2;
            printf("estimated delta:%5f\n", _delta);
            delta.SetPointer(&_delta, util::HOST);
            if (retval = delta.Move(util::HOST, util::DEVICE)) return retval;
            if (retval = visit_loopup.Allocate(graph->nodes, util::DEVICE)) return retval;
            */

            if (MARK_PATHS)
            {
                if (retval = preds.Allocate(graph->nodes,util::DEVICE)) return retval;
            }

            if (num_associate != 0)
            {
                if (retval = associate_orgs.Allocate(num_associate, util::HOST | util::DEVICE)) return retval;
                associate_orgs[0]=labels.GetPointer(util::DEVICE);
                if (MARK_PATHS)
                     associate_orgs[1]=preds.GetPointer(util::DEVICE);
                if (retval = associate_orgs.Move(util::HOST,util::DEVICE)) return retval;
            }

            if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = out_length.Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
            
            // Create incoming buffer on device
            if (num_in_nodes > 0)
            for (int t=0;t<2;t++) {
                associate_in[t]=new util::Array1D<SizeT,VertexId>[num_associate];
                associate_ins[t].SetName("associate_ins");
                if (retval = associate_ins[t].Allocate(num_associate, util::DEVICE | util::HOST)) return retval;
                for (int i=0;i<num_associate;i++)
                {
                    associate_in[t][i].SetName("associate_ins[]");
                    if (retval = associate_in[t][i].Allocate(num_in_nodes,util::DEVICE)) return retval;
                    associate_ins[t][i]=associate_in[t][i].GetPointer(util::DEVICE);
                }
                if (retval = associate_ins[t].Move(util::HOST,util::DEVICE)) return retval;
                if (retval = keys_in[t].Allocate(num_in_nodes,util::DEVICE)) return retval;
            }
            
            // Create outgoing buffer on device
            if (num_out_nodes > 0)
            {
                associate_out=new util::Array1D<SizeT,VertexId>[num_associate];
                associate_outs.SetName("associate_outs");
                if (retval = associate_outs.Allocate(num_associate, util::HOST | util::DEVICE)) return retval;
                for (int i=0;i<num_associate;i++)
                {
                    associate_out[i].SetName("associate_out[]");
                    if (retval = associate_out[i].Allocate(num_out_nodes, util::DEVICE)) return retval;
                    associate_outs[i]=associate_out[i].GetPointer(util::DEVICE);
                }
                if (retval = associate_outs.Move(util::HOST, util::DEVICE)) return retval;
            }

            return retval;
        } // Init

    }; // DataSlice

    // Members
    
    // Number of GPUs to be sliced over
    //int                 num_gpus;

    // Size of the graph
    //SizeT               nodes;
    //SizeT               edges;

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>          *data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    //DataSlice           **d_data_slices;

    // Device indices for each data slice
    //int                 *gpu_idx;

    // Methods

    /**
     * @brief SSSPProblem default constructor
     */

    SSSPProblem()
    {
        data_slices = NULL;
    }

    /**
     * @brief SSSPProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    /*SSSPProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, unsigned int, SizeT> &graph,
               int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            num_gpus);
    }*/

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
                VertexId **th_labels=new VertexId*[this->num_gpus];
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
                    h_labels[node]=th_labels[this->partition_tables[0][node]][this->convertion_tables[0][node]];
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
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool          stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> &graph,
            Csr<VertexId, Value, SizeT> *inversgraph = NULL,
            int           num_gpus = 1,
            int*          gpu_idx  = NULL,
            std::string   partition_method = "random",
            float         queue_sizing = 2.0)
    {
        //num_gpus = _num_gpus;
        //nodes = graph.nodes;
        //edges = graph.edges;
        //VertexId *h_row_offsets = graph.row_offsets;
        //VertexId *h_column_indices = graph.column_indices;
        ProblemBase<VertexId, SizeT, Value, false>::Init(
            stream_from_host,
            &graph,
            inversgraph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        //data_slices = new DataSlice*[num_gpus];
        //d_data_slices = new DataSlice*[num_gpus];
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);

                if (this->num_gpus > 1)
                {
                    if (MARK_PATHS)
                        _data_slice->Init(
                            this->num_gpus,
                            this->gpu_idx[gpu],
                            2,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus] 
                                - this->graph_slices[gpu]->out_offset[1]);
                    else _data_slice->Init(
                            this->num_gpus,
                            this->gpu_idx[gpu],
                            1,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus] 
                                - this->graph_slices[gpu]->out_offset[1]);
                } else { _data_slice->Init(
                            this->num_gpus,
                            this->gpu_idx[gpu],
                            0,
                            &(this->sub_graphs[gpu]),
                            0,
                            0);
                }
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
        typedef ProblemBase<VertexId, SizeT, Value, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output labels if necessary
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->labels.Allocate(this->sub_graphs[gpu].nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->labels.GetPointer(util::DEVICE), INT_MAX, this->sub_graphs[gpu].nodes);

            if (data_slices[gpu]->preds.GetPointer(util::DEVICE) == NULL && MARK_PATHS)
                if (retval = data_slices[gpu]->preds.Allocate(this->sub_graphs[gpu].nodes, util::DEVICE)) return retval;

            //if (data_slices[gpu]->visit_loopup.GetPointer(util::DEVICE) == NULL)
            //    if (retval = data_slices[gpu]->visit_loopup.Allocate(this->sub_graph[gpu].nodes, util::DEVICE)) return retval;
            
            if (MARK_PATHS) util::MemsetIdxKernel<<<128, 128>>>(data_slices[gpu]->preds.GetPointer(util::DEVICE), this->sub_graphs[gpu].nodes);
            //util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->visit_lookup.GetPointer(util::DEVICE), -1, this->sub_graph[gpu].nodes);

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
                        BaseProblem::graph_slices[gpu]->frontier_queues.keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;

        return retval;
    }

    float EstimatedDelta(const Csr<VertexId, unsigned int, SizeT> &graph) {
        double  avgV = graph.average_edge_value;
        int     avgD = graph.average_degree;
        return avgV * 32 / avgD;
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
