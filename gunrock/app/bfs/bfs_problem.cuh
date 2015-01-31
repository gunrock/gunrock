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
                                _MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false>
{

    //static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    //static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;
    //static const bool USE_DOUBLE_BUFFER     = _USE_DOUBLE_BUFFER;
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
        util::Array1D<SizeT, unsigned int  > temp_marker   ;
        util::Array1D<SizeT, VertexId      > temp_preds    ;
        util::Array1D<SizeT, SizeT         > *scanned_edges ;

        DataSlice()
        {   
            //util::cpu_mt::PrintMessage("DataSlice() begin.");
            labels          .SetName("labels"          );  
            preds           .SetName("preds"           );  
            visited_mask    .SetName("visited_mask"    );
            temp_preds      .SetName("temp_preds"      );
            temp_marker     .SetName("temp_marker"     );
            scanned_edges   = NULL;
            //util::cpu_mt::PrintMessage("DataSlice() end.");
        }

        ~DataSlice()
        {
            //util::cpu_mt::PrintMessage("~DataSlice() begin.");
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            preds         .Release();
            visited_mask  .Release();
            for (int gpu=0;gpu<this->num_gpus;gpu++)
                scanned_edges[gpu].Release();
            temp_preds    .Release();
            temp_marker   .Release();
            delete[] scanned_edges;scanned_edges=NULL;
            //util::cpu_mt::PrintMessage("~DataSlice() end.");
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
            float in_sizing = 1.0)
        {
            //util::cpu_mt::PrintMessage("DataSlice Init() begin.");
            cudaError_t retval = cudaSuccess;
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
            if (retval = labels       .Allocate(graph->nodes,util::DEVICE)) return retval;
            scanned_edges=new util::Array1D<SizeT, SizeT>[num_gpus];
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                scanned_edges[gpu].SetName("scanned_edges[]");
                if (retval = scanned_edges[gpu].Allocate(graph->edges,util::DEVICE)) return retval;
            }

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
            }
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
        //util::cpu_mt::PrintMessage("BFSProblem() begin.");
        data_slices = NULL;
        //util::cpu_mt::PrintMessage("BFSProblem() end.");
    }

    /**
     * @brief BFSProblem default destructor
     */
    ~BFSProblem()
    {
        //util::cpu_mt::PrintMessage("~BFSProblem() begin.");
        if (data_slices==NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices;data_slices=NULL;
        //util::cpu_mt::PrintMessage("~BFSProblem() end.");
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
            //printf("num_gpus = %d num_nodes = %d\n", this->num_gpus, this->nodes);fflush(stdout);
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
                //printf("Transfer done\n");fflush(stdout);
                
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
                if (_MARK_PREDECESSORS)
                    for (VertexId node=0;node<this->nodes;node++)
                        h_preds[node]=th_preds[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                //printf("Convertion done\n");fflush(stdout);
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
            Csr<VertexId, Value, SizeT> *graph,
            Csr<VertexId, Value, SizeT> *inversgraph = NULL,
            int         num_gpus         = 1,
            int*        gpu_idx          = NULL,
            std::string partition_method ="random",
            cudaStream_t* streams        = NULL,
            float       queue_sizing     = 2.0,
            float       in_sizing        = 1.0,
            float       partition_factor = -1.0,
            int         partition_seed   = -1)
    {
        //util::cpu_mt::PrintMessage("BFSProblem Init() begin.");
        ProblemBase<VertexId, SizeT,Value,_MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false>::Init(
            stream_from_host,
            graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);

        // No data in DataSlice needs to be copied from host

        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
                if (retval = data_slices[gpu].Allocate(1,util::DEVICE | util::HOST)) return retval;
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
                _data_slice->streams.SetPointer(&streams[gpu*num_gpus*2],num_gpus*2);
                if (this->num_gpus > 1)
                {
                    if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)
                        _data_slice->Init(
                            this->num_gpus,
                            this->gpu_idx[gpu], 
                            2,
                            0,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_counter.GetPointer(util::HOST),
                            this->graph_slices[gpu]->out_counter.GetPointer(util::HOST),
                            queue_sizing,
                            in_sizing);
                    else _data_slice->Init(
                            this->num_gpus, 
                            this->gpu_idx[gpu], 
                            1,
                            0,
                            &(this->sub_graphs[gpu]),
                            this->graph_slices[gpu]->in_counter.GetPointer(util::HOST),
                            this->graph_slices[gpu]->out_counter.GetPointer(util::HOST),
                            queue_sizing,
                            in_sizing);
                } else {
                    _data_slice->Init(
                        this->num_gpus, 
                        this->gpu_idx[gpu], 
                        0,
                        0,
                        &(this->sub_graphs[gpu]), 
                        NULL, 
                        NULL,
                        queue_sizing,
                        in_sizing);
                }
            } //end for(gpu)
        } while (0);
        
        //util::cpu_mt::PrintMessage("BFSProblem Init() end.");
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
        //util::cpu_mt::PrintMessage("BFSProblem Reset() begin.");
        typedef ProblemBase<VertexId, SizeT, Value, _MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output labels if necessary
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE)==NULL)
                if (retval = data_slices[gpu]->labels.Allocate(this->sub_graphs[gpu].nodes,util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->labels.GetPointer(util::DEVICE), _ENABLE_IDEMPOTENCE?-1:(util::MaxValue<Value>()-1), this->sub_graphs[gpu].nodes);

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
                util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->visited_mask.GetPointer(util::DEVICE), (unsigned char)0, visited_mask_elements);
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
                        BaseProblem::graph_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
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

       //util::cpu_mt::PrintMessage("BFSProblem Reset() end.");
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
