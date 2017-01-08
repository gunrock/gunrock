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

enum DIRECTION {
    FORWARD  = 0,
    BACKWARD = 1,
    UNDECIDED= 2,
};

/**
 * @brief Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam VertexId             Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT                Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value                Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE  Boolean type parameter which defines whether to enable idempotence operation for graph traverse.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE>//,
    //bool        _USE_DOUBLE_BUFFER>
struct BFSProblem : ProblemBase<VertexId, SizeT, Value,
    _MARK_PREDECESSORS,
    _ENABLE_IDEMPOTENCE>
    //_USE_DOUBLE_BUFFER,
    //false, // _ENABLE_BACKWARD
    //false, // _KEEP_ORDER
    //false> // _KEEP_NODE_NUM
{
    //Helper structures
    static const bool MARK_PREDECESSORS  = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    static const int  MAX_NUM_VERTEX_ASSOCIATES =
        (MARK_PREDECESSORS/* && !_ENABLE_IDEMPOTENCE*/) ? 1 : 0;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 0;
    typedef ProblemBase  <VertexId, SizeT, Value,
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem;
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    //typedef unsigned char MaskT;
    typedef unsigned char MaskT;
    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        //util::Array1D<SizeT, MaskT         > visited_mask  ;
        //util::Array1D<SizeT, unsigned int  > temp_marker   ;
        util::Array1D<SizeT, VertexId      > original_vertex;
        //util::Array1D<SizeT, SizeT         > input_counter;
        //util::Array1D<SizeT, SizeT         > output_counter;
        //util::Array1D<SizeT, int           > edge_marker;
        util::Array1D<SizeT, SizeT         > vertex_markers[2];
        SizeT num_visited_vertices, num_unvisited_vertices;
        bool been_in_backward;
        DIRECTION current_direction, previous_direction;
        util::Array1D<SizeT, VertexId      > unvisited_vertices[2];
        util::Array1D<SizeT, SizeT         > split_lengths;
        util::Array1D<SizeT, VertexId      > local_vertices;
        util::Array1D<SizeT, DIRECTION     > direction_votes;
        util::Array1D<SizeT, MaskT         > old_mask;
        util::Array1D<SizeT, MaskT*        > in_masks;

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            //visited_mask    .SetName("visited_mask"    );
            //temp_marker     .SetName("temp_marker"     );
            original_vertex .SetName("original_vertex" );
            //input_counter   .SetName("input_counter"   );
            //output_counter  .SetName("output_counter"  );
            //edge_marker     .SetName("edge_marker"     );
            vertex_markers[0].SetName("vertex_markers[0]");
            vertex_markers[1].SetName("vertex_markers[1]");
            unvisited_vertices[0].SetName("unvisited_vertices[0]");
            unvisited_vertices[1].SetName("unvisited_vertices[1]");
            local_vertices.SetName("local_vertices");
            split_lengths.SetName("split_length");
            direction_votes.SetName("direction_votes");
            old_mask.SetName("old_mask");
            in_masks.SetName("in_masks");
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
            if (retval = BaseDataSlice::Release())  return retval;
            //if (retval = visited_mask   .Release()) return retval;
            //if (retval = temp_marker    .Release()) return retval;
            if (retval = original_vertex.Release()) return retval;
            //if (retval = input_counter  .Release()) return retval;
            //if (retval = output_counter .Release()) return retval;
            //if (retval = edge_marker    .Release()) return retval;
            if (retval = vertex_markers[0].Release()) return retval;
            if (retval = vertex_markers[1].Release()) return retval;
            if (retval = unvisited_vertices[0].Release()) return retval;
            if (retval = unvisited_vertices[1].Release()) return retval;
            if (retval = split_lengths.Release()) return retval;
            if (retval = local_vertices.Release()) return retval;
            if (retval = direction_votes.Release()) return retval;
            if (retval = old_mask.Release()) return retval;
            if (retval = in_masks.Release()) return retval;
            return retval;
        }

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice
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
            float in_sizing = 1.0,
            bool  skip_makeout_selection = false,
            bool  keep_node_num = false)
        {
            cudaError_t retval = cudaSuccess;
            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                //num_vertex_associate,
                //num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing,
                skip_makeout_selection)) return retval;

            // Create SoA on device
            if (retval = this->labels        .Allocate(graph->nodes, util::DEVICE)) return retval;
            //if (retval = this->input_counter .Allocate(graph->nodes, util::DEVICE)) return retval;
            //if (retval = this->output_counter.Allocate(graph->edges, util::DEVICE)) return retval;
            //if (retval = this->edge_marker   .Allocate(graph->edges, util::DEVICE)) return retval;
            //if (retval = vertex_markers[0].Allocate(graph->nodes + 2, util::DEVICE)) return retval;
            //if (retval = vertex_markers[1].Allocate(graph->nodes + 2, util::DEVICE)) return retval;
            if (retval = unvisited_vertices[0].Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = unvisited_vertices[1].Allocate(graph->nodes, util::DEVICE)) return
            retval;
            if (retval = split_lengths.Init(2, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
                return retval;
            if (retval = direction_votes.Allocate(4, util::HOST));

            if (MARK_PREDECESSORS)
            {
                if (retval = this->preds     .Allocate(graph->nodes,util::DEVICE)) return retval;
                //if (retval = this->temp_preds.Allocate(graph->nodes,util::DEVICE)) return retval;
            }

            if (ENABLE_IDEMPOTENCE)
            {
                if (retval = this -> visited_mask.Allocate(graph->nodes / (sizeof(MaskT)*8) + 2*sizeof(VertexId), util::DEVICE))
                    return retval;
                if (false) //(num_gpus > 1 && !MARK_PREDECESSORS)
                {
                    if (retval = old_mask.Allocate(graph -> nodes / (sizeof(MaskT)*8) + 2*sizeof(VertexId), util::DEVICE))
                        return retval;
                    if (retval = in_masks.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
                    for (int gpu = 0; gpu < num_gpus; gpu++)
                    {
                        if (retval = this -> keys_out[gpu].Release()) return retval;
                        this -> keys_out[gpu].SetPointer(
                            (VertexId*)(this -> visited_mask.GetPointer(util::DEVICE)),
                            (graph -> nodes / (sizeof(MaskT)*8) + 1) * sizeof(MaskT) / sizeof(VertexId) + 1,
                            util::DEVICE);
                        this -> keys_outs[gpu] = (VertexId*)(this -> visited_mask.GetPointer(util::DEVICE)); 
                    }
                    this -> keys_outs.Move(util::HOST, util::DEVICE);
                }
            }

            if (num_gpus > 1)
            {
                //this->vertex_associate_orgs[0] = this->labels.GetPointer(util::DEVICE);
                if (MARK_PREDECESSORS)
                {
                    this->vertex_associate_orgs[0] = this->preds.GetPointer(util::DEVICE);
                    if (!keep_node_num)
                    original_vertex.SetPointer(
                        graph_slice -> original_vertex.GetPointer(util::DEVICE),
                        graph_slice -> original_vertex.GetSize(),
                        util::DEVICE);
                }

                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE))
                    return retval;
                //if (retval = temp_marker. Allocate(graph->nodes, util::DEVICE)) return retval;

                SizeT local_counter = 0;
                for (VertexId v=0; v<graph->nodes; v++)
                if (graph_slice -> partition_table[v] == 0)
                    local_counter ++;
                if (retval = local_vertices.Allocate(local_counter, util::HOST | util::DEVICE))
                    return retval;
                local_counter = 0;
                for (VertexId v=0; v<graph->nodes; v++)
                if (graph_slice -> partition_table[v] == 0)
                {
                    local_vertices[local_counter] = v;
                    local_counter ++;
                }
                if (retval = local_vertices.Move(util::HOST, util::DEVICE))
                    return retval;
                //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("local_vertices", local_vertices.GetPointer(util::HOST), local_counter, gpu_idx);
            }
            return retval;
        } // end Init

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
            GraphSlice<VertexId, SizeT, Value>  *graph_slice,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice->nodes;
            SizeT edges = graph_slice->edges;
            SizeT new_frontier_elements[2] = {0,0};
            SizeT max_queue_length = 0;
            num_visited_vertices = 0;
            num_unvisited_vertices = 0;
            been_in_backward = false;
            current_direction = FORWARD;
            previous_direction = FORWARD;
            if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

            if (retval = util::SetDevice( this -> gpu_idx)) return retval;
            for (int gpu = 0; gpu < this -> num_gpus * 2; gpu++)
                this -> wait_marker[gpu] = 0;
            for (int i=0; i<4; i++)
            for (int gpu = 0; gpu < this -> num_gpus * 2; gpu++)
            for (int stage=0; stage < this -> num_stages; stage++)
                this -> events_set[i][gpu][stage] = false;
            for (int gpu = 0; gpu < this -> num_gpus; gpu++)
            for (int i=0; i<2; i++)
                this -> in_length[i][gpu] = 0;
            for (int peer=0; peer<this->num_gpus; peer++)
                this -> out_length[peer] = 1;
            for (int i=0; i<4; i++)
                direction_votes[i] = UNDECIDED;

            for (int peer=0;peer<(this->num_gpus > 1 ? this->num_gpus+1 : 1);peer++)
            //for (int peer=0;peer< this->num_gpus+1;peer++)
            for (int i=0; i < 2; i++)
            {
                double queue_sizing_ = (i==0 ? queue_sizing : queue_sizing1);
                switch (frontier_type)
                {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    new_frontier_elements[0] = ((this->num_gpus > 1) ?
                        graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ +2;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    new_frontier_elements[0] = graph_slice->edges * queue_sizing_ +2;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    new_frontier_elements[0] = ((this->num_gpus > 1) ?
                        graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ +2;
                    new_frontier_elements[1] = graph_slice->edges * queue_sizing_ +2;
                    break;
                }

                // Iterate through global frontier queue setups
                //for (int i = 0; i < 2; i++) {
                {
                    if (peer == this->num_gpus && i == 1)
                        continue;
                    if (new_frontier_elements[i] > edges + 2 && queue_sizing_ >10)
                        new_frontier_elements[i] = edges+2;
                    if (this->frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i])
                    {
                        // Free if previously allocated
                        if (retval = this->frontier_queues[peer].keys[i].Release())
                            return retval;

                        // Free if previously allocated
                        if (this -> use_double_buffer) {
                            if (retval = this->frontier_queues[peer].values[i].Release())
                                return retval;
                        }

                        //frontier_elements[peer][i] = new_frontier_elements[i];

                        if (retval = this->frontier_queues[peer].keys[i].Allocate(
                            new_frontier_elements[i],util::DEVICE))
                            return retval;
                        if (this -> use_double_buffer)
                        {
                            if (retval = this->frontier_queues[peer].values[i].Allocate(
                                new_frontier_elements[i],util::DEVICE))
                                return retval;
                        }
                    } //end if
                } // end for i<2

                if (new_frontier_elements[0] > max_queue_length)
                    max_queue_length = new_frontier_elements[0];
                if (new_frontier_elements[1] > max_queue_length)
                    max_queue_length = new_frontier_elements[1];
                if (peer == this->num_gpus || i == 1)
                {
                    continue;
                }
                //if (peer == num_gpu) continue;
                SizeT max_elements = new_frontier_elements[0];
                if (new_frontier_elements[1] > max_elements) max_elements=new_frontier_elements[1];
                //if (max_elements > nodes) max_elements = nodes;
                if (this->scanned_edges[peer].GetSize() < max_elements)
                {
                    if (retval = this->scanned_edges[peer].Release())
                        return retval;
                    if (retval = this->scanned_edges[peer].Allocate(max_elements, util::DEVICE))
                        return retval;
                }

                /*SizeT cub_request_size = 0;
                cub::DeviceScan::ExclusiveSum(NULL, cub_request_size, froniter_queue.keys[0], froniter_queue.keys[0], max_queue_length);
                if (cub_scan_space[peer].GetSize() < cub_request_size)
                {
                    if (cub_scan_space[peer].GetPointer(util::DEVICE) != NULL && cub_scan_space[peer].GetSize() != 0)
                    {
                        if (retval = cub_scan_space[peer].EnsureSize(cub_request_size))
                            return retval;
                    } else {
                        if (retval = cub_scan_space[peer].Allocate(cub_request_size, util::DEVICE))
                            return retval;
                    }
                }*/
            }

            // Allocate output labels if necessary
            if (this->labels.GetPointer(util::DEVICE)==NULL)
                if (retval = this->labels.Allocate(nodes, util::DEVICE))
                    return retval;
            util::MemsetKernel<<<128, 128>>>(this->labels.GetPointer(util::DEVICE),
                /*ENABLE_IDEMPOTENCE ? (VertexId)-1 :*/ (util::MaxValue<VertexId>()), nodes);

            // Allocate preds if necessary
            if (MARK_PREDECESSORS)// && !_ENABLE_IDEMPOTENCE)
            {
                if (this->preds.GetPointer(util::DEVICE)==NULL)
                    if (retval = this->preds.Allocate(nodes, util::DEVICE))
                        return retval;
                util::MemsetKernel<<<128,128>>>(
                    this->preds.GetPointer(util::DEVICE), util::InvalidValue<VertexId>()/*(VertexId)-2*/, nodes); 
            }
            //util::MemsetKernel<<<256, 256>>>(
            //    vertex_markers[0].GetPointer(util::DEVICE),
            //    (SizeT)0, nodes + 1);
            //util::MemsetKernel<<<256, 256>>>(
            //    vertex_markers[1].GetPointer(util::DEVICE),
            //    (SizeT)0, nodes + 1);

            if (TO_TRACK)
            {
                if (retval = this -> org_checkpoint.Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_d_out     .Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_offset1   .Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_offset2   .Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_queue_idx .Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_block_idx .Allocate(max_queue_length, util::DEVICE))
                    return retval;
                if (retval = this -> org_thread_idx.Allocate(max_queue_length, util::DEVICE))
                    return retval;
            }
            if (ENABLE_IDEMPOTENCE) {
                //SizeT visited_mask_bytes  = ;
                SizeT visited_mask_elements = this -> visited_mask.GetSize();//nodes / (sizeof(MaskT)*8) + 2*sizeof(VertexId);//visited_mask_bytes * sizeof(unsigned char);

                util::MemsetKernel<<<128, 128>>>(
                    this->visited_mask.GetPointer(util::DEVICE),
                    (MaskT)0, visited_mask_elements);

                if (false) //(this -> num_gpus > 1 && !MARK_PREDECESSORS)
                    util::MemsetKernel<<<128, 128>>>(
                        this -> old_mask.GetPointer(util::DEVICE),
                        (MaskT)0, visited_mask_elements);
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
    BFSProblem(bool direction_optimized, bool undirected) : BaseProblem(
        MARK_PREDECESSORS && ENABLE_IDEMPOTENCE, // use_double_buffer
        false,                                   // enable_backward
        false,                                   // keep_order
        true,                                   // keep_node_num
        direction_optimized,                                  // skip_makeout_selection
        true,                                   // unified_receive
        direction_optimized,                   // use_inv_graph
        undirected), 
        data_slices(NULL)
    {
    }

    /**
     * @brief BFSProblem default destructor
     */
    virtual ~BFSProblem()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices==NULL) return retval;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            if (retval = util::SetDevice(this->gpu_idx[i])) return retval;
            if (retval = data_slices[i].Release()) return retval;
        }
        delete[] data_slices;data_slices=NULL;
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
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store predecessor vertex ids.
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        if (this->num_gpus == 1)
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

            data_slices[0]->labels.SetPointer(h_labels);
            if (retval = data_slices[0]->labels.Move(util::DEVICE,util::HOST)) return retval;

            if (_MARK_PREDECESSORS) {
                data_slices[0]->preds.SetPointer(h_preds);
                if (retval = data_slices[0]->preds.Move(util::DEVICE,util::HOST)) return retval;
            }

        } else {
            VertexId **th_labels = new VertexId*[this->num_gpus];
            VertexId **th_preds  = new VertexId*[this->num_gpus];
            for (int gpu=0; gpu < this->num_gpus; gpu++)
            {
                if (retval = util::SetDevice(this->gpu_idx[gpu]))
                    return retval;
                if (retval = data_slices[gpu]->labels.Move(util::DEVICE,util::HOST))
                    return retval;
                th_labels[gpu]=data_slices[gpu]->labels.GetPointer(util::HOST);
                if (_MARK_PREDECESSORS) {
                    if (retval = data_slices[gpu]->preds.Move(util::DEVICE,util::HOST))
                        return retval;
                    th_preds[gpu]=data_slices[gpu]->preds.GetPointer(util::HOST);
                }
            } //end for(gpu)

            for (VertexId v=0; v < this->nodes; v++)
            {
                int      gpu = this ->  partition_tables[0][v];
                VertexId v_  = this -> convertion_tables[0][v];
                if (gpu >= 0 && gpu <  this->num_gpus &&
                    v_ >= 0 && v_ <  data_slices[gpu]->labels.GetSize())
                {
                    h_labels[v] = th_labels[gpu][v_];
                    if (MARK_PREDECESSORS)
                        h_preds[v] = th_preds[gpu][v_];
                }
                else {
                    printf("OutOfBound: node = %lld, partition = %d, convertion = %lld\n",
                        (long long)v, gpu, (long long)v_);
                    fflush(stdout);
                }
            }

            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if (retval = data_slices[gpu]->labels.Release(util::HOST)) return retval;
                if (retval = data_slices[gpu]->preds.Release(util::HOST)) return retval;
            }
            delete[] th_labels;th_labels=NULL;
            delete[] th_preds ;th_preds =NULL;
        } //end if (data_slices.size() ==1)

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
        bool        stream_from_host,       // Only meaningful for single-GPU
        Csr<VertexId, SizeT, Value> *graph,
        Csr<VertexId, SizeT, Value> *inversegraph = NULL,
        int         num_gpus         = 1,
        int*        gpu_idx          = NULL,
        std::string partition_method ="random",
        cudaStream_t* streams        = NULL,
        float       queue_sizing     = 2.0f,
        float       in_sizing        = 1.0f,
        float       partition_factor = -1.0f,
        int         partition_seed   = -1)
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
            partition_seed))
            return retval;

        // No data in DataSlice needs to be copied from host

        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];

        for (int gpu = 0; gpu < this -> num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
                return retval;
            DataSlice *data_slice
                = data_slices[gpu].GetPointer(util::HOST);
            GraphSlice<VertexId, SizeT, Value> *graph_slice
                = this->graph_slices[gpu];
            data_slice -> streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);

            if (retval = data_slice->Init(
                this -> num_gpus,
                this -> gpu_idx[gpu],
                this -> use_double_buffer,
              &(this -> sub_graphs[gpu]),
                this -> graph_slices[gpu],
                this -> num_gpus > 1? graph_slice -> in_counter     .GetPointer(util::HOST) : NULL,
                this -> num_gpus > 1? graph_slice -> out_counter    .GetPointer(util::HOST) : NULL,
                //this -> num_gpus > 1? graph_slice -> original_vertex.GetPointer(util::HOST) : NULL,
                queue_sizing,
                in_sizing,
                this -> skip_makeout_selection,
                this -> keep_node_num))
                return retval;
        } //end for(gpu)

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     *
     * @param[in] src Source node to start.
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            VertexId    src,
            FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing,                    // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
            double queue_sizing1 = -1.0)
    {
        //typedef ProblemBase<VertexId, SizeT, Value, _MARK_PREDECESSORS, _ENABLE_IDEMPOTENCE, _USE_DOUBLE_BUFFER, false, false, false> BaseProblem;

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu]->Reset(
                frontier_type,
                this->graph_slices[gpu],
                queue_sizing,
                queue_sizing1))
                return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE))
                return retval;
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
            "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__))
            return retval;

        VertexId src_label = 0;
        if (retval = util::GRError(cudaMemcpy(
            data_slices[gpu]->labels.GetPointer(util::DEVICE) + tsrc,
            &src_label,
            sizeof(VertexId),
            cudaMemcpyHostToDevice),
            "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__))
            return retval;

        if (MARK_PREDECESSORS)// && !ENABLE_IDEMPOTENCE)
        {
            VertexId src_pred = -1;
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu]->preds.GetPointer(util::DEVICE) + tsrc,
                &src_pred,
                sizeof(VertexId),
                cudaMemcpyHostToDevice),
                "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__))
                return retval;
        }

        if (ENABLE_IDEMPOTENCE)
        {
            MaskT mask_byte = 1 << (tsrc % (8 * sizeof(MaskT)));
            VertexId mask_pos = tsrc / (8 * sizeof(MaskT));
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu] -> visited_mask.GetPointer(util::DEVICE) + mask_pos,
                &mask_byte,
                sizeof(MaskT),
                cudaMemcpyHostToDevice),
                "BFSProblem cudaMemcpy visited_mask failed", __FILE__, __LINE__))
                return retval;

            if (false)//(this -> num_gpus > 1 && !MARK_PREDECESSORS)
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu] -> old_mask.GetPointer(util::DEVICE) + mask_pos,
                &mask_byte,
                sizeof(MaskT),
                cudaMemcpyHostToDevice),
                "BFSProblem cudaMemcpy visited_mask failed", __FILE__, __LINE__))
                return retval; 
        }

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
