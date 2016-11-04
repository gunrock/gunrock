// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * template_problem.cuh
 *
 * @brief GPU Storage management Structure for Sample Problem Data
 */

#pragma once

#include <limits>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace sample {

/**
 * @brief Sample Problem structure stores device-side vectors for doing SSSP computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of value used for computed values.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value>
struct SampleProblem : ProblemBase<VertexId, SizeT, Value,
    false,//_MARK_PREDECESSORS
    false> //ENABLE_IDEMPOTENCE
{
    static const bool MARK_PREDECESSORS     = false;
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
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays 
        // util::Array1D<SizeT, Value>    user_specific_array;     /**< users can add arvitrary device arrays here. */
        
        util::Array1D<SizeT, VertexId>    original_vertex;    /**< Used for the mapping between original vertex IDs and local vertex IDs on multi-GPUs */

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            original_vertex .SetName("original_vertex" );
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
            if (retval = original_vertex.Release()) return retval;
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

            // labels is a required array that defined in BaseProblem class.
            if (retval = this->labels.Allocate(graph->nodes, util::DEVICE)) return retval;

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
            GraphSlice<VertexId, SizeT, Value>  *graph_slice,
            double queue_sizing = 2.0,
            double queue_sizing1 = -1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice -> nodes;
            SizeT edges = graph_slice -> edges;
            SizeT new_frontier_elements[2] = {0,0};
            if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

            // Reset mGPU counters and status
            for (int gpu = 0; gpu < this -> num_gpus; gpu++)
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

            // Allocate user-specified array if necessary 

            if (this->labels.GetPointer(util::DEVICE) == NULL)
                if (retval = this->labels.Allocate(nodes, util::DEVICE)) return retval;

            util::MemsetKernel<<<128, 128>>>(
                this -> labels .GetPointer(util::DEVICE),
                util::InvalidValue<VertexId>(),
                nodes);

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>          *data_slices;

    // Methods

    /**
     * @brief SampleProblem default constructor
     */

    SampleProblem() : BaseProblem(
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
     * @brief SampleProblem default destructor
     */
    virtual ~SampleProblem()
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
     * @brief Copy result distancess computed on the GPU back to host-side vectors.
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract()
    {
        cudaError_t retval = cudaSuccess;

        // Send in host arrays, copy out results from device arrays 

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
                         *inversegraph      = NULL,
            int           num_gpus          = 1,
            int          *gpu_idx           = NULL,
            std::string   partition_method  = "random",
            cudaStream_t *streams           = NULL,
            float         queue_sizing      = 2.0,
            float         in_sizing         = 1.0,
            float         partition_factor  = -1.0,
            int           partition_seed    = -1)
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

        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
            DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
            _data_slice->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);

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
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            FrontierType frontier_type,
            double queue_sizing,
            double queue_sizing1 = -1)
    {

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu] -> Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing1)) return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fillin the initial input_queue for Sample problem
        return retval;
    }

    /** @} */

};

} //namespace sample
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
