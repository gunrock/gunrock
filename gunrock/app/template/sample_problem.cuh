// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sample_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace sample {

/**
 * @brief Problem structure stores device-side vectors
 *
 * @tparam VertexId            Type of signed integer to use as vertex IDs.
 * @tparam SizeT               Type of int / uint to use for array indexing.
 * @tparam Value               Type of float or double to use for attributes.
 * @tparam _MARK_PREDECESSORS  Whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE Whether to enable idempotent operation.
 * @tparam _USE_DOUBLE_BUFFER  Whether to use double buffer.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool _MARK_PREDECESSORS,
    bool _ENABLE_IDEMPOTENCE>
    //bool _USE_DOUBLE_BUFFER >
struct SampleProblem : ProblemBase<VertexId, SizeT, Value,
    _MARK_PREDECESSORS,
    _ENABLE_IDEMPOTENCE>
    //_USE_DOUBLE_BUFFER,
    //false,   // _ENABLE_BACKWARD
    //false,   // _KEEP_ORDER
    //false>   // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS  =  _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0; 
    static const int  MAX_NUM_VALUE__ASSOCIATES = 0;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;

    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, VertexId> sample_array;
        // TODO(developer): other primitive-specific device arrays here

        DataSlice() : BaseDataSlice()
        {
            sample_array.SetName("sample_array");
            //TODO(developer): primitive-specific array construtor code here
        }

        ~DataSlice()
        {
            if (util::SetDevice(this -> gpu_idx)) return;
            sample_array.Release();
            //TODO(developer): primitive-specific array clean-up code here
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing = 1.0)
        {
            cudaError_t retval = cudaSuccess;
            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            // Create SoA on device
            if (retval = this -> labels.Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = sample_array  .Allocate(graph->nodes, util::DEVICE)) return retval;
            //TODO(developer): primitive-specific array allocation code here

            return retval;
        }

        /**  
         * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
         *
         * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
         * @param[in] graph_slice        Pointer to the corresponding graph slice
         * @param[in] queue_sizing       Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
         * @param[in] _USE_DOUBLE_BUFFER Whether to use double buffer
         * @param[in] queue_sizing1      Scaling factor for frontier_queue1
         *
         * \return cudaError_t object which indicates the success of all CUDA function calls.
         */
        cudaError_t Reset(
            FrontierType 
                    frontier_type,
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

            if (sample_array.GetPointer(util::DEVICE) == NULL)
                if (retval = sample_array.Allocate(graph_slice -> nodes, util::DEVICE))
                    return retval;
            util::MemsetIdxKernel<<<128, 128>>>(
                sample_array.GetPointer(util::DEVICE), graph_slice -> nodes);

            util::MemsetKernel <<< 128, 128>>>(
                this->labels.GetPointer(util::DEVICE),
                ENABLE_IDEMPOTENCE ? (VertexId)-1 : (util::MaxValue<VertexId>() - 1), 
                graph_slice -> nodes);

            //TODO(developer): primitive-specific array reset code here
            return retval;
        }
    };

    //int       num_gpus;
    //SizeT     nodes;
    //SizeT     edges;

    // data slices (one for each GPU)
    //DataSlice **data_slices;
    util::Array1D<SizeT, DataSlice> *data_slices;

    // putting structure on device while keeping the SoA structure
    //DataSlice **d_data_slices;

    // device index for each data slice
    //int       *gpu_idx;

    /**
     * @brief Default constructor
     */
    SampleProblem(bool use_double_buffer) :
        BaseProblem(
            use_double_buffer,
            false, // enable_backward
            false, // keep_order
            false), // keep_node_num
        data_slices(NULL)
    {
    }

    /**
     * @brief Constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    //SampleProblem(bool  stream_from_host,  // only meaningful for single-GPU
    //              const Csr<VertexId, Value, SizeT> &graph,
    //              int   num_gpus) :
    //    num_gpus(num_gpus) 
    //{
    //    Init(stream_from_host, graph, num_gpus);
    //}

    /**
     * @brief Default destructor
     */
    ~SampleProblem() 
    {
        if (data_slices == NULL) return;
        for (int i = 0; i < this -> num_gpus; ++i) 
        {
            util::SetDevice(this -> gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices; data_slices = NULL;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy results computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(VertexId *h_labels)
    {
        cudaError_t retval = cudaSuccess;

        if (this -> num_gpus == 1) 
        {
            int gpu = 0;
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;

            data_slices[gpu] -> labels.SetPointer(h_labels);
            if (retval = data_slices[gpu] -> labels.Move(util::DEVICE, util::HOST))
                return retval;

            // TODO(developer): code to extract other results here

        } else {
            // multi-GPU extension code
        }

        return retval;
    }

    /**
     * @brief Problem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     * @param[in] streams CUDA streams
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
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

        // no data in DataSlice needs to be copied from host

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
                this -> num_gpus > 1? graph_slice -> in_counter     .GetPointer(util::HOST) : NULL,
                this -> num_gpus > 1? graph_slice -> out_counter    .GetPointer(util::HOST) : NULL,
                in_sizing))
                return retval;
        }

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for primitive.
     *
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed).
     *  @param[in] queue_sizing Size scaling factor for work queue allocation.
     *  \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Reset(
        FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
        double queue_sizing,
        double queue_sizing1 = -1.0) 
    {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.

        cudaError_t retval = cudaSuccess;

        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) 
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) 
                return retval;
            if (retval = data_slices[gpu]->Reset(
                frontier_type, 
                this->graph_slices[gpu], 
                queue_sizing, 
                queue_sizing1)) 
                return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // TODO(developer): fill in the initial input_queue if necessary

        return retval;
    }

    /** @} */
};

}  // namespace sample
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
