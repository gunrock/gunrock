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
    _MARK_PREDECESSORS, 
    _ENABLE_IDEMPOTENCE, 
    _USE_DOUBLE_BUFFER, 
    false, // _ENABLE_BACKWARD
    false, // _KEEP_ORDER
    false> // _KEEP_NODE_NUM
{

    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, VertexId      > labels        ;   
        //util::Array1D<SizeT, VertexId      > preds         ;   
        util::Array1D<SizeT, unsigned char > visited_mask  ;
        util::Array1D<SizeT, unsigned int  > temp_marker   ;
        //util::Array1D<SizeT, VertexId      > temp_preds    ;
        //util::Array1D<SizeT, SizeT         > *scanned_edges ;

        DataSlice()
        {   
            //util::cpu_mt::PrintMessage("DataSlice() begin.");
            labels          .SetName("labels"          );  
            //preds           .SetName("preds"           );  
            visited_mask    .SetName("visited_mask"    );
            //temp_preds      .SetName("temp_preds"      );
            temp_marker     .SetName("temp_marker"     );
            //scanned_edges   = NULL;
            //util::cpu_mt::PrintMessage("DataSlice() end.");
        }

        ~DataSlice()
        {
            //util::cpu_mt::PrintMessage("~DataSlice() begin.");
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            //preds         .Release();
            visited_mask  .Release();
            //for (int gpu=0;gpu<this->num_gpus;gpu++)
            //    scanned_edges[gpu].Release();
            //temp_preds    .Release();
            temp_marker   .Release();
            //delete[] scanned_edges;scanned_edges=NULL;
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
            VertexId *original_vertex,
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
            //scanned_edges=new util::Array1D<SizeT, SizeT>[num_gpus];
            //for (int gpu=0;gpu<num_gpus;gpu++)
            //{
            //    scanned_edges[gpu].SetName("scanned_edges[]");
            //    if (retval = scanned_edges[gpu].Allocate(graph->edges,util::DEVICE)) return retval;
            //}

            if (_MARK_PREDECESSORS)
            {
                if (retval = this->preds     .Allocate(graph->nodes,util::DEVICE)) return retval;
                if (retval = this->temp_preds.Allocate(graph->nodes,util::DEVICE)) return retval;
            }

            if (_ENABLE_IDEMPOTENCE) 
            {
                if (retval = visited_mask.Allocate((graph->nodes +7)/8, util::DEVICE)) return retval;
            } 

            if (num_gpus > 1)
            {
                this->vertex_associate_orgs[0] = labels.GetPointer(util::DEVICE);
                if (_MARK_PREDECESSORS)
                    this->vertex_associate_orgs[1] = this->preds.GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                if (retval = temp_marker. Allocate(graph->nodes, util::DEVICE)) return retval;
            }
            //util::cpu_mt::PrintMessage("DataSlice Init() end.");
            return retval;
        } // Init

        cudaError_t Reset(
            FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
            GraphSlice<SizeT, VertexId, Value>  *graph_slice,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {         
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice->nodes;
            SizeT edges = graph_slice->edges;
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
                        new_frontier_elements[0] = double(this->num_gpus>1? graph_slice->in_counter[peer]:graph_slice->nodes) * queue_sizing_ +2;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case EDGE_FRONTIERS :
                        // O(m) ping-pong global edge frontiers
                        new_frontier_elements[0] = double(graph_slice->edges) * queue_sizing_ +2;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case MIXED_FRONTIERS :
                        // O(n) global vertex frontier, O(m) global edge frontier
                        new_frontier_elements[0] = double(this->num_gpus>1? graph_slice->in_counter[peer]:graph_slice->nodes) * queue_sizing_ +2;
                        new_frontier_elements[1] = double(graph_slice->edges) * queue_sizing_ +2;
                        break;
                }    

                // Iterate through global frontier queue setups
                //for (int i = 0; i < 2; i++) {
                {
                    if (peer == this->num_gpus && i == 1) continue;
                    if (new_frontier_elements[i] > edges + 2 && queue_sizing_ >10) new_frontier_elements[i] = edges+2;
                    if (this->frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]) {

                        // Free if previously allocated
                        if (retval = this->frontier_queues[peer].keys[i].Release()) return retval;

                        // Free if previously allocated
                        if (_USE_DOUBLE_BUFFER) {
                            if (retval = this->frontier_queues[peer].values[i].Release()) return retval;
                        }

                        //frontier_elements[peer][i] = new_frontier_elements[i];

                        if (retval = this->frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                        if (_USE_DOUBLE_BUFFER) {
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
            if (this->labels.GetPointer(util::DEVICE)==NULL)
                if (retval = this->labels.Allocate(nodes,util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(this->labels.GetPointer(util::DEVICE), _ENABLE_IDEMPOTENCE?-1:(util::MaxValue<Value>()-1), nodes);

            // Allocate preds if necessary
            if (_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)
            {
                if (this->preds.GetPointer(util::DEVICE)==NULL)
                    if (retval = this->preds.Allocate(nodes, util::DEVICE)) return retval;
                util::MemsetKernel<<<128,128>>>(this->preds.GetPointer(util::DEVICE), -2, nodes); 
            }

            if (_ENABLE_IDEMPOTENCE) {
                SizeT visited_mask_bytes  = ((nodes * sizeof(unsigned char))+7)/8;
                SizeT visited_mask_elements = visited_mask_bytes * sizeof(unsigned char);
                util::MemsetKernel<<<128, 128>>>(this->visited_mask.GetPointer(util::DEVICE), (unsigned char)0, visited_mask_elements);
            }
            
            return retval;
        } 
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
            float       queue_sizing     = 2.0f,
            float       in_sizing        = 1.0f,
            float       partition_factor = -1.0f,
            int         partition_seed   = -1)
    {
        //util::cpu_mt::PrintMessage("BFSProblem Init() begin.");
        ProblemBase<VertexId, SizeT,Value, _MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false, false, false>::Init(
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
        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
                if (retval = data_slices[gpu].Allocate(1,util::DEVICE | util::HOST)) return retval;
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
                _data_slice->streams.SetPointer(&streams[gpu*num_gpus*2],num_gpus*2);
                if (retval = _data_slice->Init(
                        this->num_gpus,
                        this->gpu_idx[gpu], 
                        this->num_gpus > 1? ((_MARK_PREDECESSORS && !_ENABLE_IDEMPOTENCE)? 2 : 1) : 0,
                        0,
                        &(this->sub_graphs[gpu]),
                        this->num_gpus > 1? this->graph_slices[gpu]->in_counter.GetPointer(util::HOST) : NULL,
                        this->num_gpus > 1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST): NULL,
                        this->num_gpus > 1? this->graph_slices[gpu]->original_vertex.GetPointer(util::HOST) : NULL,
                        queue_sizing,
                        in_sizing)) return retval;
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
            double queue_sizing,                    // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
            double queue_sizing1 = -1.0)
    {
        //util::cpu_mt::PrintMessage("BFSProblem Reset() begin.");
        typedef ProblemBase<VertexId, SizeT, Value, _MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false, false, false> BaseProblem;
        //load ProblemBase Reset
        //BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, queue_sizing1)) return retval;
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
                        data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
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
