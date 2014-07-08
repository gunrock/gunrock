// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_problem.cuh
 *
 * @brief GPU Storage management Structure for BC Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Betweenness centrality problem data structure which stores device-side vectors for doing BC computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    _VertexId,                      
    typename    _SizeT,                        
    typename    _Value,                       
    bool        _MARK_PREDECESSORS,
    bool        _USE_DOUBLE_BUFFER>
struct BCProblem : ProblemBase<_VertexId, _SizeT, _Value,
                                _USE_DOUBLE_BUFFER,true>
{
    typedef _VertexId       VertexId;
    typedef _SizeT          SizeT;
    typedef _Value          Value;
    //typedef ProblemBase<VertexId, SizeT, Value, _USE_DOUBLE_BUFFER, true>::DataSliceBase DataSliceBase;

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = false;
    
    //Helper structures
    
    /** 
     * @brief Data slice structure which contains BC problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, VertexId  >  labels;              /**< Used for source distance */
        util::Array1D<SizeT, VertexId  >  preds;               /**< Used for predecessor */
        util::Array1D<SizeT, Value     >  bc_values;           /**< Used to store final BC values for each node */
        util::Array1D<SizeT, Value     >  ebc_values;          /**< Used to store final BC values for each edge */
        util::Array1D<SizeT, Value     >  sigmas;              /**< Accumulated sigma values for each node */
        util::Array1D<SizeT, Value     >  deltas;              /**< Accumulated delta values for each node */
        util::Array1D<SizeT, VertexId  >  src_node;            /**< Used to store source node ID */

        /*int                               num_vertex_associate,num_value__associate,gpu_idx;
        util::Array1D<SizeT, VertexId  > *vertex_associate_in[2];
        util::Array1D<SizeT, VertexId* >  vertex_associate_ins[2];
        util::Array1D<SizeT, VertexId  > *vertex_associate_out;
        util::Array1D<SizeT, VertexId* >  vertex_associate_outs;
        util::Array1D<SizeT, VertexId* >  vertex_associate_orgs;
        util::Array1D<SizeT, Value     > *value__associate_in[2];
        util::Array1D<SizeT, Value*    >  value__associate_ins[2];
        util::Array1D<SizeT, Value     > *value__associate_out;
        util::Array1D<SizeT, Value*    >  value__associate_outs;
        util::Array1D<SizeT, Value*    >  value__associate_orgs;
        util::Array1D<SizeT, SizeT     >  out_length    ;   
        util::Array1D<SizeT, SizeT     >  in_length[2]  ;   
        util::Array1D<SizeT, VertexId  >  keys_in  [2]  ;*/

        DataSlice()
        {   
            labels      .SetName("labels"      );
            preds       .SetName("preds"       );  
            bc_values   .SetName("bc_values"   );
            ebc_values  .SetName("ebc_values"  );
            sigmas      .SetName("sigmas"      );
            deltas      .SetName("deltas"      );
            src_node    .SetName("src_node"    );  

            /*num_vertex_associate   = 0;
            num_value__associate   = 0;
            gpu_idx                = 0;
            vertex_associate_in[0] = NULL;
            vertex_associate_in[1] = NULL;
            vertex_associate_out   = NULL;
            value__associate_in[0] = NULL;
            value__associate_in[1] = NULL;
            value__associate_out   = NULL;
            vertex_associate_ins[0].SetName("vertex_associate_ins[0]");
            vertex_associate_ins[1].SetName("vertex_associate_ins[1]");
            vertex_associate_outs  .SetName("vertex_associate_outs"  );
            vertex_associate_orgs  .SetName("vertex_associate_orgs"  );
            value__associate_ins[0].SetName("value__associate_ins[0]");
            value__associate_ins[1].SetName("value__associate_ins[1]");
            value__associate_outs  .SetName("value__associate_outs"  );
            value__associate_orgs  .SetName("value__associate_orgs"  );
            out_length             .SetName("out_length"             );
            in_length           [0].SetName("in_length[0]"           );
            in_length           [1].SetName("in_length[1]"           );
            keys_in             [0].SetName("keys_in[0]"             );
            keys_in             [1].SetName("keys_in[1]"             );*/
        }

        ~DataSlice()
        {
            util::cpu_mt::PrintMessage("~DataSlice() begin.");
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            preds         .Release();
            bc_values     .Release();
            ebc_values    .Release();
            sigmas        .Release();
            deltas        .Release();
            src_node      .Release();
/*
            if (vertex_associate_in[0] != NULL)
            {
                for (int i=0;i<num_vertex_associate;i++)
                {
                    vertex_associate_in[0][i].Release();
                    vertex_associate_in[1][i].Release();
                }
                delete[] vertex_associate_in[0];
                delete[] vertex_associate_in[1];
                vertex_associate_in[0]=NULL;
                vertex_associate_in[1]=NULL;
                vertex_associate_ins[0].Release();
                vertex_associate_ins[1].Release();
            }

            if (value__associate_in[0] != NULL)
            {
                for (int i=0;i<num_value__associate;i++)
                {
                    value__associate_in[0][i].Release();
                    value__associate_in[1][i].Release();
                }
                delete[] value__associate_in[0];
                delete[] value__associate_in[1];
                value__associate_in[0]=NULL;
                value__associate_in[1]=NULL;
                value__associate_ins[0].Release();
                value__associate_ins[1].Release();
            }

            if (vertex_associate_out != NULL)
            {
                for (int i=0;i<num_vertex_associate;i++)
                    vertex_associate_out[i].Release();
                delete[] vertex_associate_out;
                vertex_associate_out=NULL;
                vertex_associate_outs.Release();
            }

            if (value__associate_out != NULL)
            {
                for (int i=0;i<num_value__associate;i++)
                    value__associate_out[i].Release();
                delete[] value__associate_out;
                value__associate_out=NULL;
                value__associate_outs.Release();
            }
            
            keys_in    [0].Release();
            keys_in    [1].Release();
            in_length  [0].Release();
            in_length  [1].Release();
            out_length    .Release();
            vertex_associate_orgs.Release();
            value__associate_orgs.Release();
*/
            util::cpu_mt::PrintMessage("~DataSlice() end.");
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT num_in_nodes,
            SizeT num_out_nodes)
        {
            cudaError_t retval         = cudaSuccess;
            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes)) return retval;
            /*this->gpu_idx              = gpu_idx;
            this->num_vertex_associate = num_vertex_associate;
            this->num_value__associate = num_value__associate;
            if (retval = util::SetDevice(gpu_idx))  return retval;
            if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = out_length  .Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
            if (retval = vertex_associate_orgs.Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;

            // Create incoming buffer on device
            if (num_in_nodes > 0)
            for (int t=0;t<2;t++) {
                vertex_associate_in [t] = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
                vertex_associate_ins[t].SetName("vertex_associate_ins");
                if (retval = vertex_associate_ins[t].Allocate(num_vertex_associate, util::DEVICE | util::HOST)) return retval;
                for (int i=0;i<num_vertex_associate;i++)
                {
                    vertex_associate_in[t][i].SetName("vertex_associate_ins[]");
                    if (retval = vertex_associate_in[t][i].Allocate(num_in_nodes,util::DEVICE)) return retval;
                    vertex_associate_ins[t][i] = vertex_associate_in[t][i].GetPointer(util::DEVICE);
                }
                if (retval = vertex_associate_ins[t].Move(util::HOST, util::DEVICE)) return retval;

                value__associate_in [t] = new util::Array1D<SizeT,Value   >[num_value__associate];
                value__associate_ins[t].SetName("value__associate_ins");
                if (retval = value__associate_ins[t].Allocate(num_value__associate, util::DEVICE | util::HOST)) return retval;
                for (int i=0;i<num_value__associate;i++)
                {
                    value__associate_in[t][i].SetName("value__associate_ins[]");
                    if (retval = value__associate_in[t][i].Allocate(num_in_nodes,util::DEVICE)) return retval;
                    value__associate_ins[t][i] = value__associate_in[t][i].GetPointer(util::DEVICE);
                }
                if (retval = value__associate_ins[t].Move(util::HOST, util::DEVICE)) return retval;
                
                if (retval = keys_in[t].Allocate(num_in_nodes,util::DEVICE)) return retval;
            }

            // Create outgoing buffer on device
            if (num_out_nodes > 0)
            {
                vertex_associate_out = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
                vertex_associate_outs.SetName("vertex_associate_outs");
                if (retval = vertex_associate_outs.Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;
                for (int i=0;i<num_vertex_associate;i++)
                {
                    vertex_associate_out[i].SetName("vertex_associate_out[]");
                    if (retval = vertex_associate_out[i].Allocate(num_out_nodes, util::DEVICE)) return retval;
                    vertex_associate_outs[i]=vertex_associate_out[i].GetPointer(util::DEVICE);
                }
                if (retval = vertex_associate_outs.Move(util::HOST, util::DEVICE)) return retval;
                
                value__associate_out = new util::Array1D<SizeT,Value>[num_value__associate];
                value__associate_outs.SetName("value__associate_outs");
                if (retval = value__associate_outs.Allocate(num_value__associate, util::HOST | util::DEVICE)) return retval;
                for (int i=0;i<num_value__associate;i++)
                {
                    value__associate_out[i].SetName("value__associate_out[]");
                    if (retval = value__associate_out[i].Allocate(num_out_nodes, util::DEVICE)) return retval;
                    value__associate_outs[i]=value__associate_out[i].GetPointer(util::DEVICE);
                }
                if (retval = value__associate_outs.Move(util::HOST, util::DEVICE)) return retval;
            }*/
            
            // Create SoA on device
            if (retval = labels    .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = preds     .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = bc_values .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = ebc_values.Allocate(graph->edges, util::DEVICE)) return retval;
            if (retval = sigmas    .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = deltas    .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = src_node  .Allocate(1           , util::DEVICE)) return retval; 
            util::MemsetKernel<<<128, 128>>>( bc_values.GetPointer(util::DEVICE), (Value)0.0f, graph->nodes);
            util::MemsetKernel<<<128, 128>>>(ebc_values.GetPointer(util::DEVICE), (Value)0.0f, graph->edges);

            this->vertex_associate_orgs[0] = labels    .GetPointer(util::DEVICE);
            this->vertex_associate_orgs[1] = preds     .GetPointer(util::DEVICE);
            this->value__associate_orgs[0] = bc_values .GetPointer(util::DEVICE);
            this->value__associate_orgs[1] = sigmas    .GetPointer(util::DEVICE);
            this->value__associate_orgs[2] = deltas    .GetPointer(util::DEVICE);
            if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
            if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;

            return retval;
        } // Init

    };  // DataSlice

    // Members

    // Number of GPUs to be sliced over
    //int                 num_gpus;

    // Size of the graph
    //SizeT               nodes;
    //SizeT               edges;

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>  *data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    //DataSlice           **d_data_slices;

    // Device indices for each data slice
    //int                 *gpu_idx;

    // Methods

    /**
     * @brief BCProblem default constructor
     */

    BCProblem()
    {
        data_slices = NULL;
    }

    /**
     * @brief BCProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    /*BCProblem(bool        stream_from_host,       // Only meaningful for single-GPU
              const Csr<VertexId, Value, SizeT> &graph,
              int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            num_gpus);
    }*/

    /**
     * @brief BCProblem default destructor
     */
    ~BCProblem()
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
     * @brief Copy result per-node BC values and/or sigma values computed on the GPU back to host-side vectors.
     *
     * @param[out] h_sigmas host-side vector to store computed sigma values. (Meaningful only in single-pass BC)
     * @param[out] h_bc_values host-side vector to store Node BC_values.
     *
     * @param[out] h_ebc_values host-side vector to store Edge BC_values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_sigmas, Value *h_bc_values, Value *h_ebc_values)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {

                // Set device
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                data_slices[0]->bc_values.SetPointer(h_bc_values);
                if (retval = data_slices[0]->bc_values.Move(util::DEVICE, util::HOST)) return retval;

                if (h_ebc_values) {
                    data_slices[0]->ebc_values.SetPointer(h_ebc_values);
                    if (retval = data_slices[0]->ebc_values.Move(util::DEVICE, util::HOST)) return retval;
                }

                if (h_sigmas) {
                    data_slices[0]->sigmas.SetPointer(h_sigmas);
                    if (retval = data_slices[0]->sigmas.Move(util::DEVICE, util::HOST)) return retval;
                }

            } else {
                Value **th_bc_values  = new Value*[this->num_gpus];
                Value **th_ebc_values = new Value*[this->num_gpus];
                Value **th_sigmas     = new Value*[this->num_gpus];
                SizeT **th_row_offsets= new SizeT*[this->num_gpus];
                
                for (int gpu=0; gpu< this->num_gpus; gpu++)
                {
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                    if (retval = data_slices[gpu]->bc_values.Move(util::DEVICE,util::HOST)) return retval;
                    th_bc_values[gpu] = data_slices[gpu]->bc_values.GetPointer(util::HOST);
                    if (h_ebc_values) {
                        if (retval = data_slices[gpu]->ebc_values.Move(util::DEVICE,util::HOST)) return retval;
                        th_ebc_values [gpu] = data_slices [gpu]->ebc_values .GetPointer(util::HOST);
                        th_row_offsets[gpu] = this->graph_slices[gpu]->row_offsets.GetPointer(util::HOST);
                    }
                    if (h_sigmas) {
                        if (retval = data_slices[gpu]->sigmas.Move(util::DEVICE, util::HOST)) return retval;
                        th_sigmas[gpu] = data_slices[gpu]->sigmas.GetPointer(util::HOST);
                    }
                } // end for(gpu)

                for (VertexId node=0;node<this->nodes;node++)
                {
                    int      gpu   = this->partition_tables [0][node];
                    VertexId _node = this->convertion_tables[0][node];
                    h_bc_values[node] = th_bc_values[gpu][_node];
                    if (h_sigmas) h_sigmas[node] = th_sigmas[gpu][_node];
                    if (h_ebc_values) {
                        SizeT n_edges=this->org_graph->row_offsets[node+1] - this->org_graph->row_offsets[node];
                        for (SizeT _edge=0;_edge<n_edges;_edge++)
                        {
                            h_ebc_values [ this->org_graph->row_offsets[node] + _edge] =
                                th_ebc_values [gpu][th_row_offsets[gpu][_node] + _edge];
                        }
                    }
                }

                for (int gpu=0; gpu< this->num_gpus; gpu++)
                {
                    if (retval = data_slices[gpu]->bc_values .Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->ebc_values.Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->sigmas    .Release(util::HOST)) return retval;
                }
                delete[] th_row_offsets; th_row_offsets = NULL;
                delete[] th_bc_values  ; th_bc_values   = NULL;
                delete[] th_ebc_values ; th_ebc_values  = NULL;
                delete[] th_sigmas     ; th_sigmas      = NULL;
            } //end if
        } while(0);

        return retval;
    }

    /**
     * @brief BCProblem initialization
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
            int*        gpu_idx   = NULL,
            std::string partition_method = "random")
    {
        /*num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;*/
        ProblemBase<VertexId, SizeT, Value, _USE_DOUBLE_BUFFER, true>::Init(
            stream_from_host,
            &graph,
            inversgraph,
            num_gpus,
            gpu_idx,
            partition_method);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        //data_slices = new DataSlice*[num_gpus];
        //d_data_slices = new DataSlice*[num_gpus];
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        do {
            for (int gpu=0; gpu<this->num_gpus;gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                
                retval = data_slices[gpu]->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    2,
                    3,
                    &(this->sub_graphs[gpu]),
                    this->graph_slices[gpu]-> in_offset[this->num_gpus],
                    this->graph_slices[gpu]->out_offset[this->num_gpus]
                        - this->graph_slices[gpu]->out_offset[1]); 
                if (retval) return retval;
          }
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BC problem type. Must be called prior to each BC run.
     *
     *  @param[in] src Source node for one BC computing pass. If equals to -1 then compute BC value for each node.
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
        typedef ProblemBase<VertexId, SizeT, Value, _USE_DOUBLE_BUFFER,true> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        // Reset all data but d_bc_values and d_ebc_values (Because we need to accumulate them)
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            SizeT nodes = this->sub_graphs[gpu].nodes;
            SizeT edges = this->sub_graphs[gpu].edges;
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            // Allocate output labels if necessary
            if (data_slices[gpu]->labels    .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->labels    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->labels.GetPointer(util::DEVICE), -1, nodes);

            // Allocate preds if necessary
            if (data_slices[gpu]->preds     .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->preds     .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->preds .GetPointer(util::DEVICE), -2, nodes);

            // Allocate bc_values if necessary
            if (data_slices[gpu]->bc_values .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->bc_values .Allocate(nodes, util::DEVICE)) return retval;
            if (data_slices[gpu]->ebc_values.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->ebc_values.Allocate(edges, util::DEVICE)) return retval;

            // Allocate deltas if necessary
            if (data_slices[gpu]->deltas    .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->deltas    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->deltas.GetPointer(util::DEVICE), (Value)0.0f, nodes);

            // Allocate deltas if necessary
            if (data_slices[gpu]->sigmas    .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->sigmas    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->sigmas.GetPointer(util::DEVICE), (Value)0.0f, nodes);

            if (data_slices[gpu]->src_node  .GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->src_node  .Allocate(1    , util::DEVICE)) return retval;
            VertexId tsrc = nodes;
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu]->src_node.GetPointer(util::DEVICE), &tsrc,
                sizeof(VertexId), cudaMemcpyHostToDevice), "BCProblem cudaMemcpy src_node failed", __FILE__, __LINE__)) return retval;

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }
        
        // Fillin the initial input_queue for BC problem
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
                    "BCProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;

        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->src_node.GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy src node failed", __FILE__, __LINE__)) return retval;

        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE) + tsrc,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy labels failed", __FILE__, __LINE__)) return retval;
        
        VertexId src_pred = -1; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->preds.GetPointer(util::DEVICE) + tsrc,
                        &src_pred,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy preds failed", __FILE__, __LINE__)) return retval;
        
        Value src_sigma = 1.0f;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->sigmas.GetPointer(util::DEVICE) + tsrc,
                        &src_sigma,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "BCProblem cudaMemcpy sigmas failed", __FILE__, __LINE__)) return retval;

        return retval;
    }

    /** @} */
};

} //namespace bc
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
