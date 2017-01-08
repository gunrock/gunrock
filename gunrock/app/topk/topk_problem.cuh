// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * topk_problem.cuh
 *
 * @brief GPU Storage management Structure for Degree Centrality Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace topk {

/**
 * @brief TOPK Problem structure stores device-side vectors for doing TOPK on the GPU.
 *
 * @tparam _VertexId    Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT       Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value       Type of float or double to use for computing degree centrality value.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
struct TOPKProblem : ProblemBase<VertexId, SizeT, Value,
    true,  // MARK_PREDECESSORS
    false> // ENABLE_IDEMPOTENE
    //false, // USE_DOUBLE_BUFFER
    //false, // ENABLE_BACKWORD
    //false, // KEEP_ORDER
    //false> // KEEP_NODE_NUM
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
    * @brief Data slice structure which contains TOPK problem specific data.
    */
    struct DataSlice : BaseDataSlice
    {

        // device storage arrays
        //SizeT       *d_labels;
        util::Array1D<SizeT, VertexId> node_id;   //!< top k node ids
        util::Array1D<SizeT, SizeT   > degrees_s; //!< sum/total degrees
        util::Array1D<SizeT, SizeT   > degrees_i; //!< in-going  degrees
        util::Array1D<SizeT, SizeT   > degrees_o; //!< out-going degrees
        util::Array1D<SizeT, SizeT   > temp_i;    //!< used for sorting in degrees
        util::Array1D<SizeT, SizeT   > temp_o;    //!< used for sorting out degrees

        DataSlice() : BaseDataSlice()
        {
            node_id  .SetName("node_id"  );
            degrees_s.SetName("degrees_s");
            degrees_i.SetName("degrees_i");
            degrees_o.SetName("degrees_o");
            temp_i   .SetName("temp_i"   );
            temp_o   .SetName("temp_o"   );           
        }

        ~DataSlice()
        {
            if (util::SetDevice( this -> gpu_idx)) return;
            node_id  .Release();
            degrees_s.Release();
            degrees_i.Release();
            degrees_o.Release();
            temp_i   .Release();
            temp_o   .Release();
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
            if (retval = node_id  .Allocate(graph -> nodes, util::DEVICE)) return retval;
            if (retval = degrees_s.Allocate(graph -> nodes, util::DEVICE)) return retval;
            if (retval = degrees_i.Allocate(graph -> nodes, util::DEVICE)) return retval;
            if (retval = degrees_o.Allocate(graph -> nodes, util::DEVICE)) return retval;
            if (retval = temp_i   .Allocate(graph -> nodes, util::DEVICE)) return retval;
            if (retval = temp_o   .Allocate(graph -> nodes, util::DEVICE)) return retval;

            return retval;
        }

        /**  
         * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
         *
         * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
         * @param[in] graph_slice        Pointer to the corresponding graph slice
         * @param[in] queue_sizing       Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
         * @param[in] use_double_buffer Whether to use double buffer
         * @param[in] queue_sizing1      Scaling factor for frontier_queue1
         * @param[in] skip_scanned_edges Whether to skip the scanned edges
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

            SizeT nodes = graph_slice -> nodes;

            // Allocate output if necessary
            if (node_id  .GetPointer(util::DEVICE) == NULL)
                if (retval = node_id  .Allocate( nodes, util::DEVICE))
                    return retval;

            if (degrees_s.GetPointer(util::DEVICE) == NULL)
                if (retval = degrees_s.Allocate( nodes, util::DEVICE))
                    return retval;

            if (degrees_i.GetPointer(util::DEVICE) == NULL)
                if (retval = degrees_i.Allocate( nodes, util::DEVICE))
                    return retval;

            if (degrees_o.GetPointer(util::DEVICE) == NULL)
                if (retval = degrees_o.Allocate( nodes, util::DEVICE))
                    return retval;

            if (temp_i   .GetPointer(util::DEVICE) == NULL)
                if (retval = temp_i   .Allocate( nodes, util::DEVICE))
                    return retval;

            if (temp_o   .GetPointer(util::DEVICE) == NULL)
                if (retval = temp_o   .Allocate( nodes, util::DEVICE))
                    return retval;
            // set node ids
            util::MemsetIdxKernel<<<128, 128>>>(
                node_id.GetPointer(util::DEVICE), nodes);

            // count number of out-going degrees for each node
            util::MemsetMadVectorKernel<<<128, 128>>>(
                degrees_o.GetPointer(util::DEVICE),
                graph_slice -> row_offsets.GetPointer(util::DEVICE),
                graph_slice -> row_offsets.GetPointer(util::DEVICE) +1, 
                (SizeT)-1, nodes);

            // count number of in-going degrees for each node
            util::MemsetMadVectorKernel<<<128, 128>>>(
                degrees_i.GetPointer(util::DEVICE),
                graph_slice -> column_offsets.GetPointer(util::DEVICE),
                graph_slice -> column_offsets.GetPointer(util::DEVICE) + 1, 
                (SizeT)-1, nodes);

            return retval;
        }
 
    };
  
    // Members
  
    // Number of GPUs to be sliced over
    //int                 num_gpus;
  
    // Size of the graph
    //SizeT               nodes;
    //SizeT               edges;
  
    // Selector, which d_rank array stores the final page rank?
    SizeT               selector;
  
    // Set of data slices (one for each GPU)
    //DataSlice           **data_slices;
    util::Array1D<SizeT, DataSlice> *data_slices;
  
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    //DataSlice           **d_data_slices;
  
    // Device indices for each data slice
    //int                 *gpu_idx;

    // Methods
  
    /**
    * @brief TOPKProblem default constructor
    */
  
    TOPKProblem():
        BaseProblem(
            false, // use_double_buffer
            false, // enable_backward
            false, // keep_order
            false), // keep_node_num
        data_slices(NULL),
        selector   (0   )
        //nodes(0),
        //edges(0),
        //num_gpus(0) 
    {
    }

    /**
     * @brief TOPKProblem default destructor
     */
    ~TOPKProblem()
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
    * @brief Copy result computed on the GPU back to host-side vectors.
    *
    * @param[out] h_node_id Output node ID array pointer
    * @param[out] h_degrees_i Output node in-degree array pointer
    * @param[out] h_degrees_o Output node out-degree array pointer
    * @param[in] num_nodes Specify the number of nodes
    *
    *\return cudaError_t object which indicates the success of all CUDA function calls.
    */
    cudaError_t Extract(
        VertexId *h_node_id,
        SizeT    *h_degrees_i,
        SizeT    *h_degrees_o,
        SizeT    num_nodes)
    {
        cudaError_t retval = cudaSuccess;
        if (this -> num_gpus == 1)
        {
            // Set device
            int gpu = 0;
            DataSlice *data_slice = data_slices[gpu].GetPointer(util::HOST);
            if (retval = util::SetDevice( this -> gpu_idx[gpu]))
                return retval;

            data_slice -> node_id  .SetPointer(h_node_id);
            if (retval = data_slice -> node_id  .Move(util::DEVICE, util::HOST, num_nodes))
                return retval;

            data_slice -> degrees_i.SetPointer(h_degrees_i);
            if (retval = data_slice -> degrees_i.Move(util::DEVICE, util::HOST, num_nodes))
                return retval;

            data_slice -> degrees_o.SetPointer(h_degrees_o);
            if (retval = data_slice -> degrees_o.Move(util::DEVICE, util::HOST, num_nodes))
                return retval;
        } else
        {
            // TODO: multi-GPU extract result
        }
        return retval;
    }

  /**
   * @brief TOPKProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] org_graph Reference to the CSR graph object we process on. @see Csr
   * @param[in] inv_graph Reference to the inversed CSR graph object we process on. @see Csr   * @param[in] num_gpus Number of the GPUs used.
   * @param[in] gpu_idx
   * @param[in] partition_method
   * @param[in] streams CUDA Streams
   * @param[in] queue_sizing
   * @param[in] in_sizing
   * @param[in] partition_factor
   * @param[in] partition_seed
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Init(
        bool        stream_from_host,       // Only meaningful for single-GPU
        Csr<VertexId, SizeT, Value> *org_graph,
        Csr<VertexId, SizeT, Value> *inv_graph = NULL,
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
            org_graph,
            inv_graph,
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
     *  @brief Performs any initialization work needed for TOPK problem type. 
     *	Must be called prior to each TOPK iteration.
     *
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed).
     *  @param[in] queue_sizing Size scaling factor for work queue allocation.
     *  @param[in] queue_sizing1
     *
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
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fillin the initial input_queue for TOPK problem, this needs to be modified
        // in multi-GPU scene

        return retval;
    }

    /** @} */
};
  
} //namespace topk
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
