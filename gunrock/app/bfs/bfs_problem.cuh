// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_problem.cuh
 *
 * @brief GPU Storage management Structure for BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE  Boolean type parameter which defines whether to enable idempotence operation for graph traverse.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value,                          
    bool        _MARK_PREDECESSORS,             
    bool        _ENABLE_IDEMPOTENCE,
    bool        _USE_DOUBLE_BUFFER>
struct BFSProblem : ProblemBase<VertexId, SizeT, Value,
                                _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    //Helper structures

    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, VertexId      > labels        ;   
        util::Array1D<SizeT, VertexId      > preds         ;   
        util::Array1D<SizeT, unsigned char > visited_mask  ;
        util::Array1D<SizeT, unsigned char > temp_marker   ;
        util::Array1D<SizeT, VertexId      > temp_preds    ;
        util::Array1D<SizeT, unsigned int  > scanned_edges ;
        util::scan::MultiScan<VertexId, SizeT, true, 256, 8>*
                                             Scaner;
        /*int             num_associate,gpu_idx;
        util::Array1D<SizeT,VertexId > *associate_in[2];
        util::Array1D<SizeT,VertexId*> associate_ins[2];
        util::Array1D<SizeT,VertexId > *associate_out;
        util::Array1D<SizeT,VertexId*> associate_outs;
        util::Array1D<SizeT,VertexId*> associate_orgs;
        util::Array1D<SizeT,SizeT    > out_length    ;   
        util::Array1D<SizeT,SizeT    > in_length[2]  ;
        util::Array1D<SizeT,VertexId > keys_in  [2]  ;*/

        DataSlice()
        {   
            /*util::cpu_mt::PrintMessage("DataSlice() begin.");
            num_associate   = 0;
            gpu_idx         = 0;
            associate_in[0] = NULL;
            associate_in[1] = NULL;
            associate_out   = NULL;*/
            labels          .SetName("labels"          );  
            preds           .SetName("preds"           );  
            visited_mask    .SetName("visited_mask"    );
            scanned_edges   .SetName("scanned_edges"   );
            temp_preds      .SetName("temp_preds"      );
            temp_marker     .SetName("temp_marker"     );
            Scaner          = NULL;
            /*associate_ins[0].SetName("associate_ins[0]");
            associate_ins[1].SetName("associate_ins[1]");
            associate_outs  .SetName("associate_outs"  );  
            associate_orgs  .SetName("associate_orgs"  );  
            out_length      .SetName("out_length"      );  
            in_length    [0].SetName("in_length[0]"    );  
            in_length    [1].SetName("in_length[1]"    );  
            keys_in      [0].SetName("keys_in[0]"      );  
            keys_in      [1].SetName("keys_in[1]"      );
            util::cpu_mt::PrintMessage("DataSlice() end.");*/
        }

        ~DataSlice()
        {
            util::cpu_mt::PrintMessage("~DataSlice() begin.");
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            preds         .Release();
            //keys_in    [0].Release();
            //keys_in    [1].Release();
            visited_mask  .Release();
            scanned_edges .Release();
            temp_preds    .Release();
            temp_marker   .Release();
            delete Scaner; Scaner=NULL;
            /*in_length  [0].Release();
            in_length  [1].Release();
            out_length    .Release();
            associate_orgs.Release();

            if (associate_in != NULL)
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
            }*/
            util::cpu_mt::PrintMessage("~DataSlice() end.");
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            //SizeT num_nodes,
            SizeT num_in_nodes,
            SizeT num_out_nodes)
        {
            //util::cpu_mt::PrintMessage("DataSlice Init() begin.");
            cudaError_t retval = cudaSuccess;
            //this->gpu_idx       = gpu_idx;
            //this->num_associate = num_associate;
            //if (retval = util::SetDevice(gpu_idx))  return retval;
            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph, 
                num_in_nodes,
                num_out_nodes)) return retval;

            // Create SoA on device
            if (retval = labels       .Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = scanned_edges.Allocate(graph->edges,util::DEVICE)) return retval;

            if (_MARK_PREDECESSORS)
            {
                if (retval = preds     .Allocate(graph->nodes,util::DEVICE)) return retval;
                if (retval = temp_preds.Allocate(graph->nodes,util::DEVICE)) return retval;
            }

            if (_ENABLE_IDEMPOTENCE) 
            {
                if (retval = visited_mask.Allocate((graph->nodes +7)/8, util::DEVICE)) return retval;
            } 

            if (num_gpus > 1)
            {
                this->vertex_associate_orgs[0] = labels.GetPointer(util::DEVICE);
                if (_MARK_PREDECESSORS)
                    this->vertex_associate_orgs[1] = preds.GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                if (retval = temp_marker. Allocate(graph->nodes, util::DEVICE)) return retval;
                Scaner = new util::scan::MultiScan<VertexId, SizeT, true, 256, 8>;
            }
            /*if (num_associate != 0)
            {
                if (retval = associate_orgs.Allocate(num_associate, util::HOST | util::DEVICE)) return retval;
                associate_orgs[0]=labels.GetPointer(util::DEVICE);
                if (_MARK_PREDECESSORS)
                     associate_orgs[1]=preds.GetPointer(util::DEVICE);
                if (retval = associate_orgs.Move(util::HOST,util::DEVICE)) return retval;
            }

            if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
            
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
                if (retval = out_length.Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
            }*/

            //util::cpu_mt::PrintMessage("DataSlice Init() end.");
            return retval;
        } // Init

    }; // DataSlice

    // Members
    util::Array1D<SizeT, DataSlice> *data_slices;
    
    // Methods

    /**
     * @brief BFSProblem default constructor
     */

    BFSProblem()
    {
        util::cpu_mt::PrintMessage("BFSProblem() begin.");
        data_slices = NULL;
        util::cpu_mt::PrintMessage("BFSProblem() end.");
    }

    /**
     * @brief BFSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    /*BFSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, Value, SizeT> &graph,
               int         num_gpus,
               int         *gpu_idx,
               std::string partition_method)
    {
        Init(
            stream_from_host,
            graph,
            num_gpus,
            gpu_idx,
            partition_method);
    }*/

    /**
     * @brief BFSProblem default destructor
     */
    ~BFSProblem()
    {
        util::cpu_mt::PrintMessage("~BFSProblem() begin.");
        if (data_slices==NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices;data_slices=NULL;
        util::cpu_mt::PrintMessage("~BFSProblem() end.");
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store predecessor vertex ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;
 
        do {
            printf("num_gpus = %d num_nodes = %d\n", this->num_gpus, this->nodes);
            if (this->num_gpus == 1) {

                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                data_slices[0]->labels.SetPointer(h_labels);
                if (retval = data_slices[0]->labels.Move(util::DEVICE,util::HOST)) return retval;

                if (_MARK_PREDECESSORS) {
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
                    if (_MARK_PREDECESSORS) {
                        if (retval = data_slices[gpu]->preds.Move(util::DEVICE,util::HOST)) return retval;
                        th_preds[gpu]=data_slices[gpu]->preds.GetPointer(util::HOST);
                    }
               } //end for(gpu)

                for (VertexId node=0;node<this->nodes;node++)
                    h_labels[node]=th_labels[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                if (_MARK_PREDECESSORS)
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
     * @brief BFSProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> &graph,
            Csr<VertexId, Value, SizeT> *inversgraph = NULL,
            int         num_gpus = 1,
            int*        gpu_idx  = NULL,
            std::string partition_method ="random",
            cudaStream_t* streams = NULL)
    {
        util::cpu_mt::PrintMessage("BFSProblem Init() begin.");
        ProblemBase<VertexId, SizeT,Value,_USE_DOUBLE_BUFFER>::Init(
            stream_from_host,
            &graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
                if (retval = data_slices[gpu].Allocate(1,util::DEVICE | util::HOST)) return retval;
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
                _data_slice->streams.SetPointer(streams,num_gpus);
                if (this->num_gpus > 1)
                {
                    if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)
                        _data_slice->Init(
                            this->num_gpus,
                            this->gpu_idx[gpu], 
                            2,
                            0,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus]
                              - this->graph_slices[gpu]->out_offset[1]);
                    else _data_slice->Init(
                            this->num_gpus, 
                            this->gpu_idx[gpu], 
                            1,
                            0,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus]
                                - this->graph_slices[gpu]->out_offset[1]);
                } else {
                    _data_slice->Init(
                        this->num_gpus, 
                        this->gpu_idx[gpu], 
                        0,
                        0,
                        &(this->sub_graphs[gpu]), 
                        0, 
                        0);
                }
            } //end for(gpu)
        } while (0);
        
        util::cpu_mt::PrintMessage("BFSProblem Init() end.");
        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BFS problem type. Must be called prior to each BFS run.
     *
     *  @param[in] src Source node for one BFS computing pass.
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
        util::cpu_mt::PrintMessage("BFSProblem Reset() begin.");
        typedef ProblemBase<VertexId, SizeT, Value, _USE_DOUBLE_BUFFER> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output labels if necessary
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE)==NULL)
                if (retval = data_slices[gpu]->labels.Allocate(this->sub_graphs[gpu].nodes,util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->labels.GetPointer(util::DEVICE), -1, this->sub_graphs[gpu].nodes);

            // Allocate preds if necessary
            if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)
            {
                if (data_slices[gpu]->preds.GetPointer(util::DEVICE)==NULL)
                    if (retval = data_slices[gpu]->preds.Allocate(this->sub_graphs[gpu].nodes, util::DEVICE)) return retval;
                util::MemsetKernel<<<128,128>>>(data_slices[gpu]->preds.GetPointer(util::DEVICE), -2, this->sub_graphs[gpu].nodes); 
            }

            if (_ENABLE_IDEMPOTENCE) {
                int visited_mask_bytes  = ((this->sub_graphs[gpu].nodes * sizeof(unsigned char))+7)/8;
                int visited_mask_elements = visited_mask_bytes * sizeof(unsigned char);
                //printf("num_nodes = %d, mask_bytes = %d, mask_slements = %d\n", this->sub_graphs[gpu].nodes, visited_mask_bytes, visited_mask_elements);
                util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->visited_mask.GetPointer(util::DEVICE), (unsigned char)0, visited_mask_elements);
                //util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask ",data_slices[gpu]->visited_mask.GetPointer(util::DEVICE), visited_mask_elements);
            }
            
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }
 
        // Fillin the initial input_queue for BFS problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1)
        {
           gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables[0][src];
            tsrc= this->convertion_tables[0][src];
        }
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[gpu]->frontier_queues.keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                     "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;

       if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE) {
            VertexId src_pred = -1;
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->preds.GetPointer(util::DEVICE)+tsrc,//data_slices[gpu]->d_preds+tsrc,
                            &src_pred,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
       }

       util::cpu_mt::PrintMessage("BFSProblem Reset() end.");
       return retval;
    } // reset

    /** @} */

}; // bfs_problem

} //namespace bfs
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
