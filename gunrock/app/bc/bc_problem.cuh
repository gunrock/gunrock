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
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Betweenness centrality problem data structure which stores device-side vectors for doing BC computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Whether to mark predecessor value for each node.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        _MARK_PREDECESSORS>
    //bool        _USE_DOUBLE_BUFFER>
struct BCProblem : ProblemBase<VertexId, SizeT, Value,
    _MARK_PREDECESSORS,
    false>   // ENABLE_IDEMPOTENCE
    //_USE_DOUBLE_BUFFER, 
    //true,  // ENABLE_BACKWARD
    //false, // KEEP_ORDER
    //false> // KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 1;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 2;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    bool use_double_buffer;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains BC problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value     >  bc_values;           /**< Used to store final BC values for each node */
        util::Array1D<SizeT, Value     >  sigmas;              /**< Accumulated sigma values for each node */
        util::Array1D<SizeT, Value     >  deltas;              /**< Accumulated delta values for each node */
        util::Array1D<SizeT, VertexId  >  src_node;            /**< Used to store source node ID */
        util::Array1D<SizeT, VertexId  >  *forward_output;     /**< Used to store output noe IDs by the forward pass */
        std::vector<SizeT>                *forward_queue_offsets;
        util::Array1D<SizeT, VertexId  >  original_vertex;
        util::Array1D<int, unsigned char> *barrier_markers;
        util::Array1D<SizeT, bool      >  first_backward_incoming;
        util::Array1D<SizeT, VertexId  >  local_vertices;
        util::Array1D<SizeT, bool      >  middle_event_set;
        util::Array1D<SizeT, cudaEvent_t> middle_events;
        VertexId                          middle_iteration;
        bool                              middle_finish;

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            bc_values   .SetName("bc_values"   );
            sigmas      .SetName("sigmas"      );
            deltas      .SetName("deltas"      );
            src_node    .SetName("src_node"    );
            original_vertex.SetName("original_vertex");
            first_backward_incoming.SetName("first_backward_incoming");
            local_vertices.SetName("local_vertices");
            middle_event_set.SetName("middle_event_set");
            middle_events.SetName("middle_events");
            middle_iteration      = 0;
            middle_finish         = false;
            forward_output        = NULL;
            forward_queue_offsets = NULL;
            barrier_markers       = NULL;
        }

        /*
         * @brief Default destructor
         */
        virtual ~DataSlice()
        {
            Release();
        }

        cudaError_t Release()
        {
            cudaError_t retval = cudaSuccess;
            if (retval = util::SetDevice(this->gpu_idx)) return retval;
            if (retval = bc_values     .Release()) return retval;
            if (retval = sigmas        .Release()) return retval;
            if (retval = deltas        .Release()) return retval;
            if (retval = src_node      .Release()) return retval;
            if (retval = original_vertex.Release()) return retval;
            if (retval = first_backward_incoming.Release()) return retval;
            if (retval = local_vertices.Release()) return retval;
            if (retval = middle_events .Release()) return retval;
            if (retval = middle_event_set.Release()) return retval;
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if (retval = forward_output       [gpu].Release()) return retval;
                forward_queue_offsets[gpu].resize(0);
            }
            if (forward_output != NULL)
            {
                delete[] forward_output       ; forward_output        = NULL;
            }
            if (forward_queue_offsets != NULL)
            {
                delete[] forward_queue_offsets; forward_queue_offsets = NULL;
            }
            barrier_markers = NULL;
            return retval;
        }

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to the GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         * @param[in] keep_node_num Whether to keep node number.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph,
            GraphSlice<VertexId, SizeT, Value> *graph_slice,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0,
            bool  keep_node_num = false)
        {
            cudaError_t retval         = cudaSuccess;
            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                //num_vertex_associate,
                //num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            // Create SoA on device
            if (retval = this->labels    .Allocate(graph->nodes, util::DEVICE | util::HOST)) return retval;
            if (retval = this->preds     .Allocate(graph->nodes, util::DEVICE)) return retval;
            //if (retval = this->temp_preds.Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = bc_values .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = sigmas    .Allocate(graph->nodes, util::DEVICE | util::HOST)) return retval;
            if (retval = deltas    .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = src_node  .Allocate(1           , util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>( bc_values.GetPointer(util::DEVICE), (Value)0.0, graph->nodes);

            forward_queue_offsets = new std::vector<SizeT>[num_gpus];
            forward_output = new util::Array1D<SizeT, VertexId>[num_gpus];
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                forward_queue_offsets[gpu].reserve(graph->nodes);
                forward_queue_offsets[gpu].push_back(0);
                forward_output[gpu].SetName("forward_output[]");
                if (retval = forward_output[gpu].Allocate(graph->nodes, util::DEVICE)) return retval;
            }

            if (num_gpus > 1)
            {
                if (retval = first_backward_incoming.Allocate(num_gpus, util::HOST)) return retval;
                if (num_gpus > 1 && !keep_node_num)
                    original_vertex.SetPointer(
                        graph_slice -> original_vertex.GetPointer(util::DEVICE),
                        graph_slice -> original_vertex.GetSize(),
                        util::DEVICE);

                SizeT local_counter = 0;
                for (VertexId v= 0; v < graph_slice -> nodes; v++)
                if (graph_slice -> partition_table[v] == 0)
                    local_counter ++;
                if (retval = local_vertices.Allocate(local_counter, util::HOST | util::DEVICE)) return retval;
                local_counter = 0;
                for (VertexId v= 0; v < graph_slice -> nodes; v++)
                if (graph_slice -> partition_table[v] == 0)
                {
                    local_vertices[local_counter] = v;
                    local_counter ++;
                }
                if (retval = local_vertices.Move(util::HOST, util::DEVICE)) return retval;

                if (retval = middle_event_set.Allocate(num_gpus, util::HOST)) return retval;
                if (retval = middle_events.Allocate(num_gpus, util::HOST)) return retval;
                for (int gpu = 0; gpu < num_gpus; gpu++)
                {
                    if (retval = util::GRError(
                        cudaEventCreateWithFlags(middle_events + gpu, cudaEventDisableTiming),
                        "cudaEventCreateWithFlag failed", __FILE__, __LINE__))
                        return retval;
                }
            }
            return retval;
        } // Init

        /**
         * @brief Reset problem function. Must be called prior to each run.
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Reset(
            FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
            GraphSlice<VertexId, SizeT, Value>  *graph_slice,
            bool   use_double_buffer,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice->nodes;
            //SizeT edges = graph_slice->edges;
            if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

            if (retval = BaseDataSlice::Reset(
                frontier_type, graph_slice, queue_sizing, 
                use_double_buffer, queue_sizing1)) return retval;

             // Allocate output labels if necessary
            if (this->labels    .GetPointer(util::DEVICE) == NULL)
                if (retval = this->labels    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(this->labels.GetPointer(util::DEVICE), (VertexId)-1, nodes);

            // Allocate preds if necessary
            if (this->preds     .GetPointer(util::DEVICE) == NULL)
                if (retval = this->preds     .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(this->preds .GetPointer(util::DEVICE), (VertexId)-2, nodes);

            // Allocate bc_values if necessary
            if (this->bc_values .GetPointer(util::DEVICE) == NULL)
                if (retval = this->bc_values .Allocate(nodes, util::DEVICE)) return retval;

            // Allocate deltas if necessary
            if (this->deltas    .GetPointer(util::DEVICE) == NULL)
                if (retval = this->deltas    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(this->deltas.GetPointer(util::DEVICE), (Value)0.0, nodes);

            // Allocate deltas if necessary
            if (this->sigmas    .GetPointer(util::DEVICE) == NULL)
                if (retval = this->sigmas    .Allocate(nodes, util::DEVICE)) return retval;
            util::MemsetKernel<<<128, 128>>>(this->sigmas.GetPointer(util::DEVICE), (Value)0.0, nodes);

            if (this->src_node  .GetPointer(util::DEVICE) == NULL)
                if (retval = this->src_node  .Allocate(1    , util::DEVICE)) return retval;
            VertexId tsrc = nodes;
            if (retval = util::GRError(cudaMemcpy(
                this->src_node.GetPointer(util::DEVICE), &tsrc,
                sizeof(VertexId), cudaMemcpyHostToDevice), "BCProblem cudaMemcpy src_node failed", __FILE__, __LINE__)) return retval;

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            //for (int i=0;i<this->num_gpus;i++)
            {
                if (this->forward_output[gpu].GetPointer(util::DEVICE) == NULL)
                    if (retval = this->forward_output[gpu].Allocate(nodes, util::DEVICE)) return retval;

                forward_queue_offsets[gpu].clear();
                forward_queue_offsets[gpu].reserve(nodes);
                forward_queue_offsets[gpu].push_back(0);

                if (this -> num_gpus > 1) middle_event_set[gpu] = false;
            }
            middle_iteration = -1;
            middle_finish    = false;
            return retval;
        }
    };  // DataSlice

    // Members

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>  *data_slices;

    // Methods

    /**
     * @brief BCProblem default constructor
     */
    BCProblem(bool use_double_buffer) : BaseProblem(
        use_double_buffer,
        true , // enable_backward
        false, // keep_order
        true, // keep_node_num
        false, // skip_makeout_selection
        true)  // unified_receive
    {
        data_slices = NULL;
    }

    /**
     * @brief BCProblem default destructor
     */
    virtual ~BCProblem()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices==NULL) return retval;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            if (retval = data_slices[i].Release()) return retval;
        }
        delete[] data_slices;data_slices=NULL;
        return retval;
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
     * @param[out] h_labels host-side vector to store BC labels
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(
        Value *h_sigmas, 
        Value *h_bc_values, 
        VertexId *h_labels)
    {
        cudaError_t retval = cudaSuccess;

        if (this->num_gpus == 1) {

            // Set device
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

            if (h_bc_values) {
                data_slices[0]->bc_values.SetPointer(h_bc_values);
                if (retval = data_slices[0]->bc_values.Move(util::DEVICE, util::HOST)) return retval;
            }

            if (h_sigmas) {
                data_slices[0]->sigmas.SetPointer(h_sigmas);
                if (retval = data_slices[0]->sigmas.Move(util::DEVICE, util::HOST)) return retval;
            }

            if (h_labels) {
                data_slices[0]->labels.SetPointer(h_labels);
                if (retval = data_slices[0]->labels.Move(util::DEVICE, util::HOST)) return retval;
            }
        } else {
            Value **th_bc_values  = new Value*[this->num_gpus];
            Value **th_sigmas     = new Value*[this->num_gpus];
            SizeT **th_row_offsets= new SizeT*[this->num_gpus];
            VertexId **th_labels  = new VertexId*[this->num_gpus];

            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

                if (h_bc_values) {
                    if (retval = data_slices[gpu]->bc_values.Move(util::DEVICE,util::HOST)) return retval;
                    th_bc_values[gpu] = data_slices[gpu]->bc_values.GetPointer(util::HOST);
                }

                if (h_sigmas) {
                    if (retval = data_slices[gpu]->sigmas.Move(util::DEVICE, util::HOST)) return retval;
                    th_sigmas[gpu] = data_slices[gpu]->sigmas.GetPointer(util::HOST);
                }
                if (h_labels) {
                    if (retval = data_slices[gpu]->labels.Move(util::DEVICE, util::HOST)) return retval;
                    th_labels[gpu] = data_slices[gpu]->labels.GetPointer(util::HOST);
                }
            } // end for(gpu)

            for (VertexId node=0;node<this->nodes;node++)
            {
                int      gpu   = this->partition_tables [0][node];
                VertexId _node = this->convertion_tables[0][node];
                if (h_bc_values) h_bc_values[node] = th_bc_values[gpu][_node];
                if (h_sigmas   ) h_sigmas   [node] = th_sigmas   [gpu][_node];
                if (h_labels   ) h_labels   [node] = th_labels   [gpu][_node];
            }

            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                if (retval = data_slices[gpu]->bc_values .Release(util::HOST)) return retval;
                if (retval = data_slices[gpu]->sigmas    .Release(util::HOST)) return retval;
                if (retval = data_slices[gpu]->labels    .Release(util::HOST)) return retval;
            }
            delete[] th_row_offsets; th_row_offsets = NULL;
            delete[] th_bc_values  ; th_bc_values   = NULL;
            delete[] th_sigmas     ; th_sigmas      = NULL;
            delete[] th_labels     ; th_labels      = NULL;
        } //end if

        return retval;
    }

    /**
     * @brief initialization function.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Pointer to the CSR graph object we process on. @see Csr
     * @param[in] inversegraph Pointer to the inversed CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     * @param[in] gpu_idx GPU index used for testing.
     * @param[in] partition_method Partition method to partition input graph.
     * @param[in] streams CUDA stream.
     * @param[in] queue_sizing Maximum queue sizing factor.
     * @param[in] in_sizing
     * @param[in] partition_factor Partition factor for partitioner.
     * @param[in] partition_seed Partition seed used for partitioner.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        bool          stream_from_host,       // Only meaningful for single-GPU
        Csr<VertexId, SizeT, Value>
                     *graph,
        Csr<VertexId, SizeT, Value>
                     *inversegraph     = NULL,
        int           num_gpus         = 1,
        int          *gpu_idx          = NULL,
        std::string   partition_method = "random",
        cudaStream_t *streams          = NULL,
        float         queue_sizing     = 2.0f,
        float         in_sizing        = 1.0f,
        float         partition_factor = -1.0f,
        int           partition_seed   = -1)
    {
        BaseProblem::Init(
            stream_from_host,
            graph,
            inversegraph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);

        // No data in DataSlice needs to be copied from host
        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        for (int gpu=0; gpu<this->num_gpus;gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
            DataSlice* data_slice = data_slices[gpu].GetPointer(util::HOST);
            data_slice->streams.SetPointer(&streams[gpu*num_gpus*2],num_gpus*2);
            GraphSlice<VertexId, SizeT, Value> *graph_slice
                = this -> graph_slices[gpu];

            //if (this->num_gpus > 1)
                retval = data_slice->Init(
                    this -> num_gpus,
                    this -> gpu_idx[gpu],
                    this -> use_double_buffer,
                  &(this -> sub_graphs[gpu]),
                    graph_slice,
                    this -> num_gpus > 1 ? graph_slice -> in_counter.GetPointer(util::HOST) : NULL,
                    this -> num_gpus > 1 ? graph_slice ->out_counter.GetPointer(util::HOST) : NULL,
                    queue_sizing,
                    in_sizing,
                    this -> keep_node_num);
            /*else retval = _data_slice->Init(
                    this -> num_gpus,
                    this -> gpu_idx[gpu],
                    this -> use_double_buffer,
                  &(this -> sub_graphs[gpu]),
                    NULL,
                    NULL,
                    queue_sizing,
                    in_sizing);*/
            if (retval) return retval;
            if (this -> num_gpus > 1)
            {
                if (data_slice -> vertex_associate_out[1].GetSize() < 
                    data_slice -> local_vertices.GetSize())
                {
                    if (retval = data_slice -> vertex_associate_out[1].EnsureSize(
                        data_slice -> local_vertices.GetSize()))
                        return retval;
                    data_slice -> vertex_associate_outs[1] = 
                        data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE); 
                }
                data_slice -> vertex_associate_outs.Move(util::HOST, util::DEVICE);

                if (data_slice -> value__associate_out[1].GetSize() < 
                    data_slice -> local_vertices.GetSize())
                {
                    if (retval = data_slice -> value__associate_out[1].EnsureSize(
                        data_slice -> local_vertices.GetSize()))
                        return retval;
                    data_slice -> value__associate_outs[1] = 
                        data_slice -> value__associate_out[1].GetPointer(util::DEVICE); 
                }
                data_slice -> value__associate_outs.Move(util::HOST, util::DEVICE);
            }
        }

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     *
     * @param[in] src Source node to start. (If -1 BC value for each node.).
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
        VertexId     src,
        FrontierType frontier_type,    // The frontier type (i.e., edge/vertex/mixed)
        double       queue_sizing,     // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
        double       queue_sizing1 = -1.0)
    {
        //typedef ProblemBase<VertexId, SizeT, Value, _MARK_PREDECESSORS, false, _USE_DOUBLE_BUFFER,true> BaseProblem;
        //load ProblemBase Reset
        //BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        // Reset all data but d_bc_values (Because we need to accumulate them)
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            //SizeT nodes = this->sub_graphs[gpu].nodes;
            //SizeT edges = this->sub_graphs[gpu].edges;
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            if (retval = data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu], this -> use_double_buffer, 
                queue_sizing, queue_sizing1)) return retval;

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fill in the initial input_queue for BC problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1)
        {
            gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables [0][src];
            tsrc= this->convertion_tables[0][src];
            //printf("gpu = %d tsrc = %d\n", gpu, tsrc);
        }
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->frontier_queues[0].keys[1].GetPointer(util::DEVICE),
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

        Value src_sigma = 1.0;
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
