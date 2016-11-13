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

#include <cub/cub.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace pr {

template <
    typename SizeT,
    typename Value>
__global__ void Assign_Init_Value_Kernel(
    SizeT num_elements,
    Value init_value,
    SizeT *d_degrees,
    Value *d_rank_current)
{
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

    while (x < num_elements)
    {
        if (d_degrees[x] != 0)
            d_rank_current[x] = init_value / d_degrees[x];
        x += STRIDE;
    }
}

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
    typename    Value,
    bool        NORMALIZED>
struct PRProblem : ProblemBase<VertexId, SizeT, Value,
    true,  // _MARK_PREDECESSORS
    false> // _ENABLE_IDEMPOTENCE
    //false, // _USE_DOUBLE_BUFFER
    //false, // _ENABLE_BACKWARD
    //false, // _KEEP_ORDER
    //true> // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0;  
    static const int  MAX_NUM_VALUE__ASSOCIATES = 1;
    typedef ProblemBase   <VertexId, SizeT, Value,
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem;
    typedef DataSliceBase <VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains PR problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > rank_curr;           /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, Value   > rank_next;           /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, SizeT   > degrees;             /**< Used for keeping out-degree for each vertex */
        //util::Array1D<SizeT, SizeT   > degrees_pong;
        util::Array1D<SizeT, VertexId> node_ids;
        //util::Array1D<SizeT, SizeT   > markers;
        //util::Array1D<SizeT, VertexId> *temp_keys_out;
        util::Array1D<SizeT, VertexId>  local_vertices;
        util::Array1D<SizeT, VertexId> *remote_vertices_out;
        util::Array1D<SizeT, VertexId> *remote_vertices_in;
        Value    threshold;               /**< Used for recording accumulated error */
        Value    delta;
        Value    init_value;
        Value    reset_value;
        VertexId src_node;
        bool     to_continue;
        //SizeT    local_nodes;
        //SizeT    edge_map_queue_len;
        SizeT    max_iter;
        SizeT    num_updated_vertices;
        //SizeT    PR_queue_length;
        //int      PR_queue_selector;
        bool     final_event_set;
        DataSlice* d_data_slice;
        //util::Array1D<SizeT, ContextPtr> context;
        util::Array1D<int, SizeT> in_counters;
        util::Array1D<int, SizeT> out_counters;
        util::Array1D<SizeT, unsigned char> cub_sort_storage;
        util::Array1D<SizeT, VertexId     > temp_vertex;

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice(),
            //temp_keys_out      (NULL),
            threshold          (0),
            delta              (0),
            init_value         (0),
            reset_value        (0),
            src_node           (-1),
            to_continue        (true),
            //local_nodes        (0),
            //edge_map_queue_len (0),
            max_iter           (0),
            num_updated_vertices(0),
            final_event_set    (false),
            d_data_slice       (NULL),
            remote_vertices_in (NULL),
            remote_vertices_out(NULL)
            //PR_queue_length    (0),
            //PR_queue_selector  (0)
       {
            rank_curr   .SetName("rank_curr"   );
            rank_next   .SetName("rank_next"   );
            degrees     .SetName("degrees"     );
            //degrees_pong.SetName("degrees_pong");
            node_ids    .SetName("node_ids"    );
            //markers     .SetName("markers"     );
            //context     .SetName("context"     );
            local_vertices.SetName("local_vertices");
            in_counters   .SetName("in_counters"   );
            out_counters  .SetName("out_counters"  );
            cub_sort_storage.SetName("cub_sort_storage");
            temp_vertex   .SetName("temp_vertex");
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
            if (retval = BaseDataSlice::Release()) return retval;
            if (retval = rank_curr   .Release()) return retval;
            if (retval = rank_next   .Release()) return retval;
            if (retval = degrees     .Release()) return retval;
            //if (retval = degrees_pong.Release()) return retval;
            if (retval = node_ids    .Release()) return retval;
            if (retval = in_counters .Release()) return retval;
            if (retval = out_counters.Release()) return retval;
            if (retval = cub_sort_storage.Release()) return retval;
            if (retval = temp_vertex .Release()) return retval;
            //if (retval = markers     .Release()) return retval;
            //if (temp_keys_out != NULL) {delete[] temp_keys_out; temp_keys_out = NULL;}
            //if (retval = context     .Release()) return retval;
            if (remote_vertices_in != NULL)
            {
                for (int peer = 0; peer < this -> num_gpus; peer++)
                    if (retval = remote_vertices_in[peer].Release()) return retval;
                delete[] remote_vertices_in; remote_vertices_in = NULL;
            }
            if (remote_vertices_out != NULL)
            {
                for (int peer = 0; peer < this -> num_gpus; peer++)
                    if (retval = remote_vertices_out[peer].Release()) return retval;
                delete[] remote_vertices_out; remote_vertices_out = NULL;
            }

            return retval;
        }

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph,
            GraphSlice<VertexId, SizeT, Value>
                   *graph_slice,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT       nodes  = graph->nodes;
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

            /*temp_keys_out = new util::Array1D<SizeT, VertexId>[num_gpus];
            if (num_gpus > 1)
            {
                // printf("Allocating keys_out[0] %d\n", local_nodes);fflush(stdout);
                if (retval = this->keys_out[0].Allocate(local_nodes, util::DEVICE)) return retval;
                this->keys_outs[0] = this->keys_out[0].GetPointer(util::DEVICE);
                for (int peer_ = 0; peer_ < num_gpus; peer_++) 
                {
                    if (peer_ == 0)
                    {// only need the first one, can be reused
                        if (retval = this->keys_marker[peer_].EnsureSize(nodes)) return retval;
                    } else {
                        if (retval = this->keys_marker[peer_].Release()) return retval;
                    }
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
            this->keys_outs.Move(util::HOST, util::DEVICE);*/

            // Create SoA on device
            if (retval = rank_curr   .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = rank_next   .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = degrees     .Allocate(nodes+1, util::DEVICE)) return retval;
            //if (retval = degrees_pong.Allocate(nodes+1, util::DEVICE)) return retval;
            //if (retval = node_ids    .Allocate(nodes, util::DEVICE)) return retval;
            //if (retval = markers     .Allocate(nodes, util::DEVICE)) return retval;

            if (this->num_gpus == 1)
            {
                //local_nodes = nodes;

                //if (retval = this -> frontier_queues[0].keys[0].Release()) return retval;
                //if (this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE) == NULL)
                //    if (retval = this -> frontier_queues[0].keys[0].Allocate(nodes, util::DEVICE)) 
                //        return retval;
                if (retval = local_vertices.Allocate(nodes, util::DEVICE))
                    return retval;
                util::MemsetIdxKernel<<<128, 128>>>(
                    //this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);
                    local_vertices.GetPointer(util::DEVICE), nodes);
            } else {
                out_counters.Allocate(this -> num_gpus, util::HOST);
                in_counters .Allocate(this -> num_gpus, util::HOST);
                remote_vertices_out = new util::Array1D<SizeT, VertexId>[this -> num_gpus];
                remote_vertices_in  = new util::Array1D<SizeT, VertexId>[this -> num_gpus];
                for (int peer = 0; peer < this -> num_gpus; peer++)
                {
                    out_counters[peer] = 0;
                    remote_vertices_out[peer].SetName("remote_vetices_out[]");
                    remote_vertices_in [peer].SetName("remote_vertces_in []");
                }

                for (VertexId v=0; v<graph->nodes; v++)
                    out_counters[graph_slice -> partition_table[v]] ++;
                
                for (int peer = 0; peer < this -> num_gpus; peer++)
                {
                    if (retval = remote_vertices_out[peer].Allocate(
                        out_counters[peer], util::HOST | util::DEVICE))
                        return retval;
                    out_counters[peer] = 0;
                }
                
                for (VertexId v=0; v<graph->nodes; v++)
                {
                    int target = graph_slice -> partition_table[v];
                    remote_vertices_out[target][out_counters[target]] = v;
                    out_counters[target] ++;
                }

                for (int peer = 0; peer < this -> num_gpus; peer++)
                {
                    if (retval = remote_vertices_out[peer].Move(util::HOST, util::DEVICE))
                        return retval;
                }
                if (retval = local_vertices.SetPointer(
                    remote_vertices_out[0].GetPointer(util::HOST), 
                    out_counters[0], util::HOST))
                    return retval;
                if (retval = local_vertices.SetPointer(
                    remote_vertices_out[0].GetPointer(util::DEVICE),
                    out_counters[0], util::DEVICE))
                    return retval;
            }

            return retval;
        }

        cudaError_t Reset(
            VertexId src,
            Value    delta,
            Value    threshold,
            SizeT    max_iter,
            bool     scaled,
            FrontierType frontier_type,
            Csr       <VertexId, SizeT, Value>
                   *org_graph,
            Csr       <VertexId, SizeT, Value>
                   *sub_graph,
            GraphSlice<VertexId, SizeT, Value>
                   *graph_slice,
            double  queue_sizing       = 2.0, 
            bool    use_double_buffer  = false,
            double  queue_sizing1      = -1.0,
            bool    skip_scanned_edges = false)
        {
            cudaError_t retval = cudaSuccess;
            if (retval = BaseDataSlice::Reset(
                frontier_type,
                graph_slice,
                queue_sizing,
                use_double_buffer,
                queue_sizing1,
                skip_scanned_edges))
                return retval;

            SizeT nodes = sub_graph -> nodes;
            //SizeT edges = this->sub_graphs[gpu].edges;
            //SizeT *temp_in_counter = new SizeT[this->num_gpus+1];

            //for (int peer = 1; peer < this->num_gpus; peer++)
            //{
            //    temp_in_counter[peer] = graph_slice -> in_counter[peer];
            //    graph_slice -> in_counter[peer] = 1;
            //}

            // Allocate output page ranks if necessary
            //if (retval = node_ids.Release(util::DEVICE)) return retval;
            if (this -> num_gpus > 1)
            for (int peer = 0; peer < this -> num_gpus; peer++)
            {
                if (retval = this -> keys_out[peer].Release()) return retval;
                if (retval = this -> keys_in[0][peer].Release()) return retval;
                if (retval = this -> keys_in[1][peer].Release()) return retval;
            }

            //for (int peer = 1; peer < this->num_gpus; peer++)
            //    graph_slice -> in_counter[peer] = temp_in_counter[peer];

            if (rank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = rank_curr.Allocate(nodes, util::DEVICE)) return retval;

            if (rank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = rank_next.Allocate(nodes, util::DEVICE)) return retval;

            //if (node_ids .GetPointer(util::DEVICE) == NULL)
            //    if (retval = node_ids .Allocate(nodes, util::DEVICE)) return retval;

            // Allocate degrees if necessary
            if (degrees  .GetPointer(util::DEVICE) == NULL)
                if (retval = degrees  .Allocate(nodes, util::DEVICE)) return retval;

            // Initial rank_next = 0
            //util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_curr,
            //    (Value)1.0/nodes, nodes);
            init_value = NORMALIZED ? 
                (scaled ? 1.0 : (1.0 / (Value)(org_graph->nodes))) 
                : (1.0 - delta);
            reset_value = NORMALIZED ?
                (scaled ? (1.0 - delta) : ((1.0 - delta) / (Value)(org_graph->nodes))) 
                : (1.0 - delta);
            util::MemsetKernel<<<128, 128>>>(
                rank_next.GetPointer(util::DEVICE), 
                NORMALIZED ? (Value) 0.0 : (Value)(1.0 - delta), nodes);
            util::MemsetKernel<<<128, 128>>>(
                rank_curr.GetPointer(util::DEVICE), 
                init_value, nodes);

            // Compute degrees
            //util::MemsetKernel<<<128, 128>>>(
            //    degrees  .GetPointer(util::DEVICE), (SizeT)0, nodes);

            util::MemsetMadVectorKernel <<<128, 128>>>(
                degrees.GetPointer(util::DEVICE),
                graph_slice -> row_offsets.GetPointer(util::DEVICE),
                graph_slice -> row_offsets.GetPointer(util::DEVICE)+1, (SizeT)-1, nodes);

            Assign_Init_Value_Kernel <<<128, 128>>>(
                nodes,
                init_value,
                degrees  .GetPointer(util::DEVICE),
                rank_curr.GetPointer(util::DEVICE));

            //util::MemsetIdxKernel       <<<128, 128>>>(
            //    node_ids    .GetPointer(util::DEVICE), nodes);

            this -> delta       = delta    ; //data_slices[gpu]->delta    .Move(util::HOST, util::DEVICE);
            this -> threshold   = threshold; //data_slices[gpu]->threshold.Move(util::HOST, util::DEVICE);
            this -> src_node    = src      ; //data_slices[gpu]->src_node .Move(util::HOST, util::DEVICE);
            this -> to_continue = true;
            this -> max_iter    = max_iter;
            this -> final_event_set = false;
            //this -> PR_queue_length = 1;
            this -> num_updated_vertices = 1;
            //if (retval = degrees_pong.Release()) return retval; // save sapce when not using R0DIteration
            //if (temp_in_counter != NULL) {delete[] temp_in_counter; temp_in_counter = NULL;}

            return retval;
        }

    };

    // Members

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // whether to use the scaling feature
    bool scaled; 

    // Methods

    /**
     * @brief PRProblem default constructor
     */
    PRProblem(bool _scaled) : BaseProblem(
        false, // use_double_buffer
        false, // enable_backward
        false, // keep_order
        true , // keep_node_num 
        false, // skip_makeout_selection
        true ), // unified_receive
        data_slices(NULL ),
        scaled     (_scaled)
    {}

    /**
     * @brief PRProblem default destructor
     */
    virtual ~PRProblem()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices == NULL) return retval;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            if (retval = util::SetDevice(this->gpu_idx[i])) return retval;
            if (retval = data_slices[i].Release()) return retval;
        }
        delete[] data_slices; data_slices=NULL;
        if (retval = BaseProblem::Release()) return retval;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_rank host-side vector to store page rank values.
     * @param[out] h_node_id host-side vector to store node Vertex ID.
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(Value *h_rank, VertexId *h_node_id)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
        data_slices[0]->rank_curr.SetPointer(h_rank);
        if (retval = data_slices[0]->rank_curr.Move(util::DEVICE, util::HOST)) return retval;
        data_slices[0]->node_ids .SetPointer(h_node_id);
        if (retval = data_slices[0]->node_ids .Move(util::DEVICE, util::HOST)) return retval;

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
     * @param[in] context
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
        ContextPtr   *context          = NULL,
        float         queue_sizing     = 2.0f,
        float         in_sizing        = 1.0f,
        float         partition_factor = -1.0f,
        int           partition_seed   = -1)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseProblem :: Init(
            stream_from_host,
            graph,
            inversegraph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed))
            return retval;

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
        //SizeT *local_nodes = new SizeT[this->num_gpus];

        /*if (this->num_gpus > 1)
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
        }*/

        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this -> gpu_idx[gpu])) 
                return retval;
            if (retval = this -> graph_slices[gpu] -> out_degrees    .Release()) 
                return retval;
            if (retval = this -> graph_slices[gpu] -> original_vertex.Release()) return retval;
            if (retval = this -> graph_slices[gpu] -> convertion_table.Release()) 
                return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) 
                return retval;
            DataSlice* data_slice_ = data_slices[gpu].GetPointer(util::HOST);
            data_slice_ -> d_data_slice = data_slices[gpu].GetPointer(util::DEVICE);
            data_slice_ -> streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);
            //data_slice_ -> context.SetPointer(context + gpu * num_gpus * 2, num_gpus * 2);
            data_slice_ -> init_value = 1.0 / graph->nodes ;
            //if (this->num_gpus > 1) data_slice_->local_nodes = local_nodes[gpu];
            if (retval = data_slice_->Init(
                this->num_gpus,
                this->gpu_idx[gpu],
                this->use_double_buffer,
                //this->num_gpus>1? 0 : 0,
                //this->num_gpus>1? 1 : 0,
                &(this->sub_graphs[gpu]),
                this -> graph_slices[gpu],
                this->num_gpus>1? this->graph_slices[gpu]->in_counter .GetPointer(util::HOST) : NULL,
                this->num_gpus>1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                queue_sizing,
                in_sizing)) return retval;
        }

        if (this -> num_gpus == 1) return retval;

        for (int gpu = 0; gpu < this -> num_gpus; gpu++)
        {
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;
            
            for (int peer = 0; peer < this -> num_gpus; peer++)
            {
                if (peer == gpu) continue;
                int peer_ = (peer < gpu) ? peer + 1 : peer;
                int gpu_  = (peer < gpu) ? gpu : gpu + 1;
                data_slices[gpu] -> in_counters[peer_] = data_slices[peer] -> out_counters[gpu_];
                if (gpu != 0)
                {
                    data_slices[gpu] -> remote_vertices_in[peer_].SetPointer(
                        data_slices[peer] -> remote_vertices_out[gpu_].GetPointer(util::HOST),
                        data_slices[peer] -> remote_vertices_out[gpu_].GetSize(),
                        util::HOST);
                } else {
                    data_slices[gpu] -> remote_vertices_in[peer_].SetPointer(
                        data_slices[peer] -> remote_vertices_out[gpu_].GetPointer(util::HOST),
                        max(data_slices[peer] -> remote_vertices_out[gpu_].GetSize(),
                        data_slices[peer] -> local_vertices.GetSize()),
                        util::HOST);
                }
                if (retval = data_slices[gpu] -> remote_vertices_in[peer_].Move(util::HOST, util::DEVICE, 
                    data_slices[peer] -> remote_vertices_out[gpu_].GetSize()))
                    return retval;

                for (int t=0; t<2; t++)
                {
                    if (data_slices[gpu] -> value__associate_in[t][peer_].GetPointer(util::DEVICE) == NULL)
                    {
                        if (retval = data_slices[gpu] -> value__associate_in[t][peer_].Allocate(
                            data_slices[gpu] -> in_counters[peer_], util::DEVICE))
                            return retval;
                    } else {
                        if (retval = data_slices[gpu] -> value__associate_in[t][peer_].EnsureSize(
                            data_slices[gpu] -> in_counters[peer_]))
                            return retval;
                    }
                }
            }
        }

        for (int gpu = 1; gpu < this -> num_gpus; gpu++)
        {
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;
            if (data_slices[gpu] -> value__associate_out[1].GetPointer(util::DEVICE) == NULL)
            {
                if (retval = data_slices[gpu] -> value__associate_out[1].Allocate(
                    data_slices[gpu] -> local_vertices.GetSize(), util::DEVICE))
                    return retval;
            } else {
                if (retval = data_slices[gpu] -> value__associate_out[1].EnsureSize(
                    data_slices[gpu] -> local_vertices.GetSize()))
                    return retval;
            }
        }

        if (retval = util::SetDevice(this -> gpu_idx[0]))
            return retval;
        for (int gpu = 1; gpu < this -> num_gpus; gpu++)
        {
            if (data_slices[0] -> value__associate_in[0][gpu].GetPointer(util::DEVICE) == NULL)
            {
                if (retval = data_slices[0] -> value__associate_in[0][gpu].Allocate(
                    data_slices[gpu] -> local_vertices.GetSize(), util::DEVICE))
                    return retval;
            } else {
                if (retval = data_slices[0] -> value__associate_in[0][gpu].EnsureSize(
                    data_slices[gpu] -> local_vertices.GetSize()))
                    return retval;
            }
        }
        //delete[] local_nodes; local_nodes = NULL;
        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     *
     * @param[in] src Source node to start.
     * @param[in] delta PageRank delta factor
     * @param[in] threshold Threshold for remove node from PR computation process.
     * @param[in] max_iter Maximum number of iterations.
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1
     * @param[in] skip_scanned_edges Whether to skip scanned edges
     *
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            VertexId src,
            Value    delta,
            Value    threshold,
            SizeT    max_iter,
            FrontierType frontier_type, // The frontier type (i.e., edge/vertex/mixed)
            double   queue_sizing,
            double   queue_sizing1 = -1.0,
            bool     skip_scanned_edges = false)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) 
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->Reset(
                src, delta, threshold, max_iter, scaled,
                frontier_type, this -> org_graph,
                this -> sub_graphs + gpu, this->graph_slices[gpu], 
                queue_sizing, this -> use_double_buffer, 
                queue_sizing1, skip_scanned_edges)) 
                return retval;

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE))
                return retval;

            if (gpu == 0 && this -> num_gpus > 1)
            {
                for (int peer = 1; peer < this -> num_gpus; peer ++)
                if (retval = data_slices[gpu] -> remote_vertices_in[peer].Move(util::HOST, util::DEVICE,
                    data_slices[gpu] -> in_counters[peer]))
                    return retval;
            }
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
