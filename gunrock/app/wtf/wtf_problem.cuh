// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * wtf_problem.cuh
 *
 * @brief GPU Storage management Structure for Who-To-Follow framework
 * (combines Personalized PageRank and SALSA/Personalized-SALSA)
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace wtf {

/**
 * @brief WTF Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing WTF value.
 */
template <
    typename VertexId,                       
    typename SizeT,                          
    typename Value>
struct WTFProblem : ProblemBase<VertexId, SizeT, Value,
    true,  // MARK_PREDECESSORS
    false> // ENABLE_IDEMPOTENCE
    //false, // USE_DOUBLE_BUFFER = false
    //false, // ENABLE_BACKWARD
    //false, // KEEP_ORDER
    //false> // KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0;  
    static const int  MAX_NUM_VALUE__ASSOCIATES = 0;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains WTF problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > rank_curr;           /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, Value   > rank_next;           /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, Value   > refscore_curr;
        util::Array1D<SizeT, Value   > refscore_next;
        util::Array1D<SizeT, SizeT   > out_degrees;             /**< Used for keeping out-degree for each vertex */
        util::Array1D<SizeT, SizeT   > in_degrees;
        Value   threshold;               /**< Used for recording accumulated error */
        Value   delta;
        Value   alpha;
        VertexId src_node;
        util::Array1D<SizeT, VertexId> node_ids;
        util::Array1D<SizeT, bool    > cot_map;     /**< Input frontier bitmap */

        DataSlice() : BaseDataSlice()
        {
            rank_curr    .SetName("rank_curr"    );
            rank_next    .SetName("rank_next"    );
            refscore_curr.SetName("refscore_curr");
            refscore_next.SetName("refscore_next");
            out_degrees  .SetName("out_degrees"  );
            in_degrees   .SetName("in_degrees"   );
            node_ids     .SetName("node_ids"     );
            cot_map      .SetName("cot_map"      );
        }

        ~DataSlice()
        {
            if (util::SetDevice(this -> gpu_idx)) return;
            rank_curr    .Release();
            rank_next    .Release();
            refscore_curr.Release();
            refscore_next.Release();
            out_degrees  .Release();
            in_degrees   .Release();
            cot_map      .Release();
            node_ids     .Release();
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
            SizeT nodes = graph -> nodes;
            if (retval = rank_curr    .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = rank_next    .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = refscore_curr.Allocate(nodes, util::DEVICE)) return retval;
            if (retval = refscore_next.Allocate(nodes, util::DEVICE)) return retval;
            if (retval = out_degrees  .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = in_degrees   .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = node_ids     .Allocate(nodes, util::DEVICE)) return retval;
            if (retval = cot_map      .Allocate(nodes, util::DEVICE)) return retval;

            return retval;
        }

        /**
         *  @brief Performs any initialization work needed for primitive.
         */
        cudaError_t Reset(
            VertexId src,
            Value    delta,
            Value    alpha,
            Value    threshold,
            FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
            GraphSlice<VertexId, SizeT, Value>
                    *graph_slice,
            double   queue_sizing       = 1.0,
            bool     use_double_buffer  = false,
            double   queue_sizing1      = -1.0,
            bool     skip_scanned_edges = false)
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
            // Allocate output page ranks if necessary
            if (rank_curr    .GetPointer(util::DEVICE) == NULL)
                if (retval = rank_curr    .Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (rank_next    .GetPointer(util::DEVICE) == NULL)
                if (retval = rank_next    .Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (refscore_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = refscore_curr.Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (refscore_next.GetPointer(util::DEVICE) == NULL)
                if (retval = refscore_next.Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (node_ids     .GetPointer(util::DEVICE) == NULL)
                if (retval = node_ids     .Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (out_degrees  .GetPointer(util::DEVICE) == NULL)
                if (retval = out_degrees  .Allocate(nodes, util::DEVICE)) 
                    return retval;

            // Allocate d_in_degrees if necessary
            if (in_degrees   .GetPointer(util::DEVICE) == NULL)
                if (retval = in_degrees   .Allocate(nodes, util::DEVICE)) 
                    return retval;

            if (cot_map      .GetPointer(util::DEVICE) == NULL)
                if (retval = cot_map      .Allocate(nodes, util::DEVICE)) 
                    return retval;
 
            util::MemsetKernel<<<128, 128>>>(
                rank_next    .GetPointer(util::DEVICE), (Value)0.0      , nodes);

            util::MemsetKernel<<<128, 128>>>(
                rank_curr    .GetPointer(util::DEVICE), (Value)1.0/nodes, nodes);

            util::MemsetKernel<<<128, 128>>>(
                refscore_curr.GetPointer(util::DEVICE), (Value)0.0      , nodes);

            util::MemsetKernel<<<128, 128>>>(
                refscore_next.GetPointer(util::DEVICE), (Value)0.0      , nodes);
          
            // Compute degrees
            util::MemsetKernel<<<128, 128>>>(
                out_degrees  .GetPointer(util::DEVICE), (SizeT)0        , nodes);

            util::MemsetMadVectorKernel<<<128, 128>>>(
                out_degrees  .GetPointer(util::DEVICE), 
                graph_slice -> row_offsets.GetPointer(util::DEVICE), 
                graph_slice -> row_offsets.GetPointer(util::DEVICE) +1, 
                (SizeT)-1, nodes);

            util::MemsetKernel<<<128, 128>>>(
                in_degrees   .GetPointer(util::DEVICE), (SizeT)0        , nodes);

            //util::MemsetMadVectorKernel<<<128, 128>>>(
            //    in_degrees   .GetPointer(util::DEVICE), 
            //    graph_slice -> column_offsets.GetPointer(util::DEVICE), 
            //    graph_slice -> column_offsets.GetPointer(util::DEVICE) +1, -1, nodes);
 
            util::MemsetIdxKernel<<<128, 128>>>(
                node_ids  .GetPointer(util::DEVICE), nodes);

            util::MemsetKernel<<<128, 128>>>(
                cot_map   .GetPointer(util::DEVICE), false, nodes);

            // Fillin the initial input_queue for WTF problem, this needs to be modified
            // in multi-GPU scene
            // Put every vertex in there
            util::MemsetIdxKernel<<<128, 128>>>(
                this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);

            this->delta     = delta;
            this->alpha     = alpha;
            this->threshold = threshold;
            this->src_node  = src;

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
     * @brief WTFProblem default constructor
     */

    WTFProblem() :
        BaseProblem(
            false, // use_double_buffer
            false, // enable_backward
            false, // keep_order
            false), // keep_node_num
        data_slices(NULL),
        selector(0)
    {
    }

    /**
     * @brief WTFProblem default destructor
     */
    ~WTFProblem()
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
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_rank host-side vector to store page rank values.
     * @param[out] h_node_id host-side vector to store node IDs.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_rank, VertexId *h_node_id)
    {
        cudaError_t retval = cudaSuccess;

        if (this -> num_gpus == 1) 
        {
            int gpu = 0;
            // Set device
            if (retval = util::SetDevice( this -> gpu_idx[gpu]))
                return retval;

            data_slices[gpu]->refscore_curr.SetPointer(h_rank);
            if (retval = data_slices[gpu]->refscore_curr.Move(util::DEVICE, util::HOST)) 
                return retval;

            data_slices[gpu]->node_ids.SetPointer(h_node_id);
            if (retval = data_slices[gpu]->node_ids.Move(util::DEVICE, util::HOST))
                return retval;

        } else {
            // TODO: multi-GPU extract result
        } //end if (data_slices.size() ==1)

        return retval;
    }

    /**
     * @brief WTFProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] inv_graph Reference to the CSC graph object we process on. @see Csr
     * @param[in] num_gpus Number of the GPUs used.
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
        Csr<VertexId, SizeT, Value> *graph,
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
            graph,
            inv_graph,
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
     *  @brief Performs any initialization work needed for WTF problem type. Must be called prior to each WTF iteration.
     *
     *  @param[in] src Source node for one WTF computing pass.
     *  @param[in] delta Delta in SALSA equation.
     *  @param[in] alpha Alpha in SALSA equation.
     *  @param[in] threshold Threshold for convergence.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Queue sizing of the frontier.
     *  @param[in] queue_sizing1
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        VertexId src,
        Value    delta,
        Value    alpha,
        Value    threshold,
        FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
        double   queue_sizing  = 1.0,
        double   queue_sizing1 = -1.0)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this -> num_gpus; ++gpu) 
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu]->Reset(
                src,
                delta,
                alpha,
                threshold,
                frontier_type,
                this->graph_slices[gpu],
                queue_sizing,
                queue_sizing1))
                return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) 
                return retval;
        }
        
        return retval;
    }

    /** @} */

};

} //namespace wtf
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
