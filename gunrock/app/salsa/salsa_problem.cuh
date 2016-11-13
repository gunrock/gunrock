// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * salsa_problem.cuh
 *
 * @brief GPU Storage management Structure for SALSA(Stochastic Approach for Link-Structure Analysis) Problem Data
 */

#pragma once

#include <cub/cub.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace salsa {

/**
 * @brief SALSA Problem structure stores device-side vectors for doing SALSA Algorithm on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing SALSA rank value.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value>
struct SALSAProblem : ProblemBase<VertexId, SizeT, Value,
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
    typedef ProblemBase   <VertexId, SizeT, Value,
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem;
    typedef DataSliceBase <VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains SALSA problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > hrank_curr;          /**< Used for ping-pong hub rank value */
        util::Array1D<SizeT, Value   > arank_curr;          /**< Used for ping-pong authority rank value */
        util::Array1D<SizeT, Value   > hrank_next;          /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, Value   > arank_next;          /**< Used for ping-pong page rank value */
        util::Array1D<SizeT, SizeT   > in_degrees;          /**< Used for keeping in-degree for each vertex */
        util::Array1D<SizeT, SizeT   > out_degrees;         /**< Used for keeping out-degree for each vertex */
        util::Array1D<SizeT, VertexId> hub_predecessors;        /**< Used for keeping the predecessor for each vertex */
        util::Array1D<SizeT, VertexId> auth_predecessors;        /**< Used for keeping the predecessor for each vertex */
    };

    // Members
    
    // Number of GPUs to be sliced over
    //int                 num_gpus;

    // Size of the graph
    //SizeT               nodes;
    //SizeT               edges;
    SizeT               out_nodes;
    SizeT               in_nodes;

    // Selector, which d_rank array stores the final page rank?
    SizeT               selector;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    //int                 *gpu_idx;

    // Methods

    /**
     * @brief SALSAProblem default constructor
     */

    SALSAProblem() : BaseProblem(
        false, // use_double_buffer
        false, // enable_backward
        false, // keep_order
        false), // keep_node_num
        //nodes(0),
        //edges(0),
        out_nodes(0),
        in_nodes(0),
        //num_gpus(0),
        selector(0),
        data_slices(0),
        d_data_slices(0)
    {
    }

    /**
     * @brief SALSAProblem default destructor
     */
    ~SALSAProblem()
    {
        for (int i = 0; i < this -> num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(this -> gpu_idx[i]),
                "~SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            data_slices[i]->hrank_curr.Release();
            data_slices[i]->arank_curr.Release();
            data_slices[i]->hrank_next.Release();
            data_slices[i]->arank_next.Release();
            data_slices[i]->in_degrees.Release();
            data_slices[i]->out_degrees.Release();
            data_slices[i]->hub_predecessors.Release();
            data_slices[i]->auth_predecessors.Release();
            if (d_data_slices[i])                 
                util::GRError(cudaFree(d_data_slices[i]), 
                    "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices)  delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_hrank host-side vector to store hub rank values.
     *
     * @param[out] h_arank host-side vector to store authority rank values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_hrank, Value *h_arank)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this -> num_gpus == 1) 
            {
                // Set device
                if (util::GRError(cudaSetDevice(this -> gpu_idx[0]),
                    "SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                data_slices[0]->hrank_curr.SetPointer(h_hrank);
                if (retval = data_slices[0]->hrank_curr.Move(util::DEVICE, util::HOST)) 
                    return retval;

                data_slices[0]->arank_curr.SetPointer(h_arank);
                if (retval = data_slices[0]->arank_curr.Move(util::DEVICE, util::HOST)) 
                    return retval;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief SALSAProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] hub_graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] auth_graph Reference to the CSC graph object we process on.
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
        bool          stream_from_host,       // Only meaningful for single-GPU
        Csr<VertexId, SizeT, Value> *hub_graph,
        Csr<VertexId, SizeT, Value> *auth_graph,
        int           num_gpus         = 1,
        int          *gpu_idx          = NULL,
        std::string   partition_method = "random",
        cudaStream_t *streams          = NULL,
        float         queue_sizing     = 2.0f,
        float         in_sizing        = 1.0f,
        float         partition_factor = -1.0f,
        int           partition_seed   = -1)
    {
        BaseProblem :: Init(
            stream_from_host,
            hub_graph,
            auth_graph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);

        out_nodes = hub_graph -> out_nodes;
        in_nodes = auth_graph -> out_nodes;

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];
        if (streams == NULL) {streams = new cudaStream_t[num_gpus]; streams[0] = 0;}

        do {
            if (num_gpus <= 1) 
            {
                // Create a single data slice for the currently-set gpu
                int gpu = 0;
                if (retval = util::SetDevice(this -> gpu_idx[gpu])) return retval;

                data_slices[gpu] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                    (void**)&d_data_slices[gpu],
                    sizeof(DataSlice)),
                    "SALSAProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) 
                    return retval;

                data_slices[gpu][0].streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);
                if (retval = data_slices[gpu]->Init(
                    this -> num_gpus,
                    this -> gpu_idx[gpu],
                    this -> use_double_buffer,
                    //0,
                    //0,
                    hub_graph,
                    NULL,
                    NULL,
                    in_sizing))
                    return retval;

                // Create SoA on device
                data_slices[gpu] -> hrank_curr.SetName("hrank_curr");
                if (retval = data_slices[gpu] -> hrank_curr.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> arank_curr.SetName("arank_curr");
                if (retval = data_slices[gpu] -> arank_curr.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> hrank_next.SetName("hrank_next");
                if (retval = data_slices[gpu] -> hrank_next .Allocate( this->nodes, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> arank_next.SetName("arank_next");
                if (retval = data_slices[gpu] -> arank_next .Allocate( this->nodes, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> in_degrees.SetName("in_degrees");
                if (retval = data_slices[gpu] -> in_degrees .Allocate( this->nodes, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> hub_predecessors.SetName("hub_predecessors");
                if (retval = data_slices[gpu] -> hub_predecessors.Allocate( this->edges, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> auth_predecessors.SetName("auth_predecessors");
                if (retval = data_slices[gpu] -> auth_predecessors.Allocate( this->edges, util::DEVICE)) 
                    return retval;

                data_slices[gpu] -> out_degrees.SetName("out_degrees");
                if (retval = data_slices[gpu] -> out_degrees.Allocate( this->nodes, util::DEVICE)) 
                    return retval;
            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for PR problem type. Must be called prior to each PR iteration.
     *
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Queue sizing of the frontier
     *  @param[in] queue_sizing1
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,
        double queue_sizing = 1.0,             // The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing1 = -1.0)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) 
        {
            // Set device
            if (retval = util::GRError(cudaSetDevice( this -> gpu_idx[gpu]),
                "SALSAProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
            //printf("Reseting dataslice queue_sizing = %lf, %lf\n",
            //    (double) queue_sizing, (double) queue_sizing1);
            data_slices[gpu] -> Reset(
                frontier_type, 
                this->graph_slices[gpu], 
                queue_sizing,
                this->use_double_buffer, 
                queue_sizing1);

            if (data_slices[gpu] -> hrank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> hrank_curr.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> arank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> arank_curr.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> hrank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> hrank_next.Allocate( this->nodes, util::DEVICE))
                    return retval;

            if (data_slices[gpu] -> arank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> arank_next.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> in_degrees.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> in_degrees.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> hub_predecessors.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> hub_predecessors.Allocate( this->edges, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> auth_predecessors.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> auth_predecessors.Allocate( this->edges, util::DEVICE)) 
                    return retval;

            if (data_slices[gpu] -> out_degrees.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu] -> out_degrees.Allocate( this->nodes, util::DEVICE)) 
                    return retval;

            // Initial rank_curr = 0 
            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> hrank_curr.GetPointer(util::DEVICE), 
                (Value)1.0/out_nodes, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> arank_curr.GetPointer(util::DEVICE), 
                (Value)1.0/ in_nodes, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> hrank_next.GetPointer(util::DEVICE), 
                (Value)0, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> arank_next.GetPointer(util::DEVICE), 
                (Value)0, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> out_degrees.GetPointer(util::DEVICE), 
                (SizeT)0, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> in_degrees.GetPointer(util::DEVICE), 
                (SizeT)0, this -> nodes);

            util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[gpu] -> out_degrees.GetPointer(util::DEVICE), 
                this->graph_slices[gpu] -> row_offsets.GetPointer(util::DEVICE), 
                this->graph_slices[gpu] -> row_offsets.GetPointer(util::DEVICE) + 1, 
                (SizeT)-1, this -> nodes);

            util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[gpu] -> in_degrees.GetPointer(util::DEVICE), 
                this->graph_slices[gpu] -> column_offsets.GetPointer(util::DEVICE), 
                this->graph_slices[gpu] -> column_offsets.GetPointer(util::DEVICE) +1, 
                (SizeT)-1, this -> nodes);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> hub_predecessors.GetPointer(util::DEVICE), 
                (VertexId)-1, this -> edges);

            util::MemsetKernel<<<128, 128>>>(
                data_slices[gpu] -> auth_predecessors.GetPointer(util::DEVICE), 
                (VertexId)-1, this -> edges);
            
            if (retval = util::GRError(cudaMemcpy(
                d_data_slices[gpu],
                data_slices[gpu],
                sizeof(DataSlice),
                cudaMemcpyHostToDevice),
                "SALSAProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__))
                return retval;
        }
        
        // Fillin the initial input_queue for SALSA problem, this needs to be modified
        // in multi-GPU scene

        return retval;
    }

    /** @} */

};

} //namespace salsa
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

