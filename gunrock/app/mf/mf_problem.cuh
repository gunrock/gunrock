// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_problem.cuh
 *
 * @brief GPU Storage management Structure for MF Problem Data
 */

#pragma once

#include <limits>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace mf {

/**
 * @brief Max-Flow Problem structure stores device-side vectors for doing MF computing on the GPU.
 *
 * @tparam VertexId             Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT                Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value                Type of value used for computed values.
 * @tparam MARK_PREDECESSORS    Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam NABLE_IDEMPOTENCE   Boolean type parameter which defines whether to enable idempotence operation for graph traverse.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value>
struct MFProblem : ProblemBase<VertexId, SizeT, Value,
    false, //MARK_PREDECESSORS
    false> //ENABLE_IDEMPOTENCE
{
    static const bool MARK_PREDECESSORS     = false;//MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = false;//ENABLE_IDEMPOTENCE;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = MARK_PREDECESSORS ? 1 : 0;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 1;
    typedef ProblemBase<VertexId, SizeT, Value, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>                  BaseProblem;
    typedef DataSliceBase<VertexId, SizeT, Value, MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains MF problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value>     capacity;           /* Used for storing edge capacity */
        util::Array1D<SizeT, Value>     excess;             /* Used for storing vertex excess */
        util::Array1D<SizeT, Value>     flow;               /* Used for storing edge flow */
        util::Array1D<SizeT, Value>     height;             /* Used for storing vertex height */

        /**
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            capacity        .SetName("capacity"         );
            excess          .SetName("excess"           );
            flow            .SetName("flow"             );
            height          .SetName("height"           );
        }

        /**
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
            if (retval = capacity.Release()) return retval;
            if (retval = excess.Release()) return retval;
            if (retval = flow.Release()) return retval;
            if (retval = height.Release()) return retval;
            return retval;
        }

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to the GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         * @param[in] skip_makeout_selection
         * @param[in] keep_node_num
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
            bool  skip_makeout_selection = false,
            bool  keep_node_num = false)
        {
            cudaError_t retval  = cudaSuccess;

            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing,
                skip_makeout_selection)) return retval;

            if (retval = capacity       .Allocate(graph->edges, util::DEVICE)) return retval;
            if (retval = excess         .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = flow           .Allocate(graph->edges, util::DEVICE)) return retval;
            if (retval = height         .Allocate(graph->nodes, util::DEVICE)) return retval;
            
            //what is graph->edge_values pointer? who sets it? 
            capacity.SetPointer(graph->edge_values, graph->edges, util::HOST);
            if (retval = capacity.Move(util::HOST, util::DEVICE)) return retval;

            //? todo the same what with capacity object
            //set nodes excess and height to 0 except source
            //set source excess to infinity and height to graph->nodes + 1;
            
            if (num_gpus > 1){
                this->value__associate_orgs[0] = capacity.GetPointer(util::DEVICE);
                if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
            }
             
            return retval;
        } // Init

        
        /**
         * @brief Reset problem function. Must be called prior to each run.
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Reset(
            FrontierType frontier_type,
            GraphSlice<VertexId, SizeT, Value> *graph_slice,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice->nodes;
            SizeT edges = graph_slice->edges;
            SizeT new_frontier_elements[2] = {0,0};
            if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;
            // Reset mGPU counters and status
            for (int gpu = 0; gpu < this->num_gpus; gpu++)
                this->wait_marker[gpu] = 0;
            for (int i = 0; i < 4; i++)
                for (int gpu = 0; gpu < this->num_gpus * 2; gpu++)
                    for (int stage = 0; stage < this->num_stages; stage++)
                        this->events_set[i][gpu][stage] = false;
            for (int gpu = 0; gpu < this->num_gpus; gpu++)
                for (int i = 0; i < 2; i++)
                    this->in_length[i][gpu] = 0;
            for (int peer = 0; peer < this->num_gpus; peer++)
                this->out_length[peer] = 1;
            for (int peer = 0; peer < (this->num_gpus > 1 ? this->num_gpus + 1 : 1); peer++){
                for (int i=0; i < 2; i++){
                    double queue_sizing_ = (i == 0 ? queue_sizing : queue_sizing1);
                    switch (frontier_type) {
                        case VERTEX_FRONTIERS :
                            // O(n) ping-pong global vertex frontiers
                            new_frontier_elements[0] = 
                                double(this->num_gpus > 1 ? graph_slice->in_counter[peer] : nodes) * queue_sizing_ + 2;
                            new_frontier_elements[1] = new_frontier_elements[0];
                            break;

                        case EDGE_FRONTIERS :
                            // O(m) ping-pong global edge frontiers
                            new_frontier_elements[0] = double(edges) * queue_sizing_ + 2;
                            new_frontier_elements[1] = new_frontier_elements[0];
                            break;

                        case MIXED_FRONTIERS :
                            // O(n) global vertex frontier, O(m) global edge frontier
                            new_frontier_elements[0] = 
                                double(this->num_gpus > 1 ? graph_slice->in_counter[peer] : nodes) * queue_sizing_ + 2;
                            new_frontier_elements[1] = double(edges) * queue_sizing_ +2;
                            break;
                    }

                    if (peer == this->num_gpus && i == 1) continue;

                    if (new_frontier_elements[i] > edges + 2 && queue_sizing_ > 10) 
                        new_frontier_elements[i] = edges + 2;
                    if (this->frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]){ 
                        // Free if previously allocated
                        if (retval = this->frontier_queues[peer].keys[i].Release()) return retval;
                        if (retval = this->frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                    } //end if

                    if (peer == this->num_gpus || i == 1) continue;

                    SizeT max_elements = new_frontier_elements[0];
                    if (new_frontier_elements[1] > max_elements) 
                        max_elements = new_frontier_elements[1];
                    if (max_elements > nodes) 
                        max_elements = nodes;
                    if (this->scanned_edges[peer].GetSize() < max_elements){
                        if (retval = this->scanned_edges[peer].Release()) return retval;
                        if (retval = this->scanned_edges[peer].Allocate(max_elements, util::DEVICE)) return retval;
                    }
                }
            }

            // todo:
            // Allocate some output arrays?? if necessary
            // in sssp_problem wystepuje thid->distances co chyba nie jest rowne distances?

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>* data_slices;

    // Methods

    /**
     * @brief MFProblem default constructor
     */

    MFProblem() : BaseProblem(
        false, // use_double_buffer
        false, // enable_backward
        false, // keep_order
        true,  // keep_node_num
        false, // skip_makeout_selection
        true)  // unified_receive
    {
        data_slices = NULL;
    }

    /**
     * @brief MFProblem default destructor
     */
    virtual ~MFProblem()
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
        delete[] data_slices; data_slices = NULL;
        if (retval = BaseProblem::Release()) return retval;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result computed on the GPU back to host-side vector.
     *
     * @param[out] h_excess Host-side vector to store computed node excess.
     *                      (Sink excess is an amount of the flow pushed from 
     *                      the source towards the sink)
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(Value *h_excess)
    {
        cudaError_t retval = cudaSuccess;

        if (this->num_gpus == 1){
            //Set device
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
            data_slices[0]->excess.SetPointer(h_excess);
            if (retval = data_slices[0]->excess.Move(util::DEVICE, util::HOST)) return retval;
        }else{
            Value **th_excess = new Value*[this->num_gpus];
            for (int gpu = 0; gpu < this->num_gpus; gpu++){
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu]->excess.Move(util::DEVICE, util::HOST)) return retval;
                th_excess[gpu] = data_slices[gpu]->excess.GetPointer(util::HOST);
            }
            for (VertexId node = 0; node < this->nodes; node++){
                int partition      = this->partition_tables[0][node];
                int convertion   = this->convertion_tables[0][node];
                if (partition >= 0 && partition < this->num_gpus && 
                        convertion >= 0 && convertion < data_slices[partition]->excess.GetSize()){
                    h_excess[node] = th_excess[partition][convertion];
                }else{
                    printf("OutOfBound: node = %d, partition = %d, convertion = %d\n", node, partition, convertion);
                    fflush(stdout);
                }
            }
            for (int gpu = 0; gpu < this->num_gpus; gpu++){
                if (retval = data_slices[gpu]->excess.Release(util::HOST)) return retval;
            }
            delete[] th_excess; th_excess = NULL;
        }
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
            bool                            stream_from_host, // Only meaningful for single-GPU
            Csr<VertexId, SizeT, Value>*    graph,
            Csr<VertexId, SizeT, Value>*    inversegraph      = NULL,
            int                             num_gpus          = 1,
            int*                            gpu_idx           = NULL,
            std::string                     partition_method  = "random",
            cudaStream_t*                   streams           = NULL,
            float                           queue_sizing      = 2.0,
            float                           in_sizing         = 1.0,
            float                           partition_factor  = -1.0,
            int                             partition_seed    = -1)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseProblem::Init(
                    stream_from_host,
                    graph,
                    inversegraph,
                    num_gpus,
                    gpu_idx,
                    partition_method,
                    queue_sizing,
                    partition_factor,
                    partition_seed)) return retval;

        // No data in DataSlice needs to be copied from host
        // todo any data needing to be copy? (check other app, for instance bfs, cc)

        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        for (int gpu = 0; gpu < this->num_gpus; gpu++){
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
            DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
            _data_slice->streams.SetPointer(&streams[gpu * num_gpus * 2], num_gpus * 2);

            if (retval = _data_slice->Init(
                        this->num_gpus,
                        this->gpu_idx[gpu],
                        this->use_double_buffer,
                        &(this->sub_graphs[gpu]),
                        this -> graph_slices[gpu],
                        this->num_gpus > 1? this->graph_slices[gpu]->in_counter.GetPointer(util::HOST) : NULL,
                        this->num_gpus > 1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST): NULL,
                        queue_sizing,
                        in_sizing,
                        this -> skip_makeout_selection,
                        this -> keep_node_num))
                return retval;
        } // end for (gpu)

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     *
     * @param[in] src Source node to start.
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            VertexId src,
            FrontierType frontier_type,
            double queue_sizing,
            double queue_sizing1 = -1)
    {
        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->
                    Reset(frontier_type, this->graph_slices[gpu], queue_sizing, queue_sizing1)) return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fillin the initial input_queue for MF problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1){
            gpu = 0;
            tsrc = src;
        }else{
            gpu = this->partition_tables    [0][src];
            tsrc = this->convertion_tables  [0][src];
        }

        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval = util::GRError(
                    cudaMemcpy(
                        data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        /*
           Value src_excess = 0;
           if (retval = util::GRError(
           cudaMemcpy(
           data_slices[gpu]->excess.GetPointer(util::DEVICE) + tsrc,
           &src_excess,
           sizeof(Value),
           cudaMemcpyHostToDevice),
           "SSSPProblem cudaMemcpy excess failed", __FILE__, __LINE__)) return retval;
         */
        return retval;
    }

    /** @} */

};

} //namespace mf
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
