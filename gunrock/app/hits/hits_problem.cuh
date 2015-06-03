// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_problem.cuh
 *
 * @brief GPU Storage management Structure for HITS(Hyperlink-Induced Topic Search) Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace hits {

/**
 * @brief HITS Problem structure stores device-side vectors for doing HITS Algorithm on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing HITS rank value.
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value>
struct HITSProblem : ProblemBase<_VertexId, _SizeT, _Value,
    true,  // MARK_PREDECESSORS
    false, // ENABLE_IDEMPOTENCE
    false, // USE_DOUBLE_BUFFER = false
    false, // ENABLE_BACKWARD
    false, // KEEP_ORDER
    false> // KEEP_NODE_NUM
{

    typedef _VertexId 			VertexId;
	typedef _SizeT			    SizeT;
	typedef _Value              Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains HITS problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, Value   > hrank_curr;           /**< Used for ping-pong hub rank value */
        util::Array1D<SizeT, Value   > arank_curr;           /**< Used for ping-pong authority rank value */
        util::Array1D<SizeT, Value   > hrank_next;           /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, Value   > arank_next;           /**< Used for ping-pong page rank value */       
        util::Array1D<SizeT, VertexId> in_degrees;          /**< Used for keeping in-degree for each vertex */
        util::Array1D<SizeT, VertexId> out_degrees;         /**< Used for keeping out-degree for each vertex */
        Value                          delta;
        VertexId                       src_node;
        util::Array1D<SizeT, SizeT   > labels;
    };

    // Members
    
    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Selector, which d_rank array stores the final page rank?
    SizeT               selector;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    int                 *gpu_idx;

    // Methods

    /**
     * @brief HITSProblem default constructor
     */

    HITSProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief HITSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] inv_graph Reference to the inversed CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    HITSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, Value, SizeT> &graph,
               const Csr<VertexId, Value, SizeT> &inv_graph,
               int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            inv_graph,
            num_gpus);
    }

    /**
     * @brief HITSProblem default destructor
     */
    ~HITSProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~HITSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            data_slices[i]->hrank_curr.Release();
            data_slices[i]->arank_curr.Release();
            data_slices[i]->hrank_next.Release();
            data_slices[i]->arank_next.Release();
            data_slices[i]->in_degrees.Release();
            data_slices[i]->out_degrees.Release();

            //if (data_slices[i]->d_hrank_curr)      util::GRError(cudaFree(data_slices[i]->d_hrank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_arank_curr)      util::GRError(cudaFree(data_slices[i]->d_arank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_hrank_next)      util::GRError(cudaFree(data_slices[i]->d_hrank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_arank_next)      util::GRError(cudaFree(data_slices[i]->d_arank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_in_degrees)      util::GRError(cudaFree(data_slices[i]->d_in_degrees), "GpuSlice cudaFree d_in_degrees failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_out_degrees)      util::GRError(cudaFree(data_slices[i]->d_out_degrees), "GpuSlice cudaFree d_out_degrees failed", __FILE__, __LINE__); 
            //if (data_slices[i]->d_delta)            util::GRError(cudaFree(data_slices[i]->d_delta), "GpuSlice cudaFree d_delta failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_src_node)            util::GRError(cudaFree(data_slices[i]->d_src_node), "GpuSlice cudaFree d_src_node failed", __FILE__, __LINE__);
            if (d_data_slices[i])                 util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
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
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "HITSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                /*if (retval = util::GRError(cudaMemcpy(
                                h_hrank,
                                data_slices[0]->d_hrank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "HITSProblem cudaMemcpy d_hranks failed", __FILE__, __LINE__)) break;*/
                data_slices[0]->hrank_curr.SetPointer(h_hrank);
                if (retval = data_slices[0]->hrank_curr.Move(util::DEVICE, util::HOST)) return retval;

                /*if (retval = util::GRError(cudaMemcpy(
                                h_arank,
                                data_slices[0]->d_arank_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "HITSProblem cudaMemcpy d_aranks failed", __FILE__, __LINE__)) break;*/
                data_slices[0]->arank_curr.SetPointer(h_arank);
                if (retval = data_slices[0]->arank_curr.Move(util::DEVICE, util::HOST)) return retval;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief HITSProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] hub_graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] auth_graph Reference to the CSC graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> &hub_graph,
            Csr<VertexId, Value, SizeT> &auth_graph,
            int         _num_gpus,
            cudaStream_t* streams = NULL)
    {
        num_gpus = _num_gpus;
        nodes = hub_graph.nodes;
        edges = hub_graph.edges;
        //SizeT *h_row_offsets = hub_graph.row_offsets;
        //VertexId *h_column_indices = hub_graph.column_indices;
        //SizeT *h_col_offsets = auth_graph.row_offsets;
        //VertexId *h_row_indices = auth_graph.column_indices;
        ProblemBase<VertexId, SizeT, Value,
            true, false, false, false, false, false>
          ::Init(stream_from_host,
                 &hub_graph,
                 &auth_graph,
                 num_gpus,
                 NULL,
                 "random");

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];
        if (streams == NULL) {streams = new cudaStream_t[num_gpus]; streams[0] = 0;}

        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "HITSProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "HITSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;
                data_slices[0][0].streams.SetPointer(streams, 1);
                data_slices[0]->Init(
                    1,
                    gpu_idx[0],
                    0,
                    0,
                    &hub_graph,
                    NULL,
                    NULL);

                // Create SoA on device
                /*Value    *d_hrank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_hrank1,
                        nodes * sizeof(Value)),
                    "HITSProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_hrank_curr = d_hrank1;*/
                data_slices[0]->hrank_curr.SetName("hrank_curr");
                if (retval = data_slices[0]->hrank_curr.Allocate(nodes, util::DEVICE)) return retval;

                /*Value    *d_arank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_arank1,
                        nodes * sizeof(Value)),
                    "HITSProblem cudaMalloc d_arank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_arank_curr = d_arank1;*/
                data_slices[0]->arank_curr.SetName("arank_curr");
                if (retval = data_slices[0]->arank_curr.Allocate(nodes, util::DEVICE)) return retval;

                /*Value    *d_hrank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_hrank2,
                        nodes * sizeof(Value)),
                    "HITSProblem cudaMalloc d_hrank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_hrank_next = d_hrank2;*/
                data_slices[0]->hrank_next.SetName("hrank_next");
                if (retval = data_slices[0]->hrank_next.Allocate(nodes, util::DEVICE)) return retval;

                /*Value    *d_arank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_arank2,
                        nodes * sizeof(Value)),
                    "HITSProblem cudaMalloc d_arank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_arank_next = d_arank2;*/
                data_slices[0]->arank_next.SetName("arank_next");
                if (retval = data_slices[0]->arank_next.Allocate(nodes, util::DEVICE)) return retval;

                /*VertexId   *d_in_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_in_degrees,
                        nodes * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_in_degrees = d_in_degrees;*/
                data_slices[0]->in_degrees.SetName("in_degrees");
                if (retval = data_slices[0]->in_degrees.Allocate(nodes, util::DEVICE)) return retval;

                /*VertexId   *d_out_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_out_degrees,
                        nodes * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_out_degrees = d_out_degrees;*/
                data_slices[0]->out_degrees.SetName("out_degrees");
                if (retval = data_slices[0]->out_degrees.Allocate(nodes, util::DEVICE)) return retval;

                /*Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_delta,
                        1 * sizeof(Value)),
                    "PRProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_delta = d_delta;

                VertexId    *d_src_node;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_src_node,
                        1 * sizeof(VertexId)),
                    "PRProblem cudaMalloc d_src_node failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_src_node = d_src_node;
 
                data_slices[0]->d_labels = NULL;*/

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for PR problem type. Must be called prior to each PR iteration.
     *
     *  @param[in] src Source node for one PR computing pass.
     *  @param[in] delta Parameter for PR value distribution equation
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Frontier queue scaling factor
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId        src,
            Value           delta,
            FrontierType    frontier_type,
            double queue_sizing = 1.0)             // The frontier type (i.e., edge/vertex/mixed)
    {
        //typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
        //load ProblemBase Reset
        //BaseProblem::Reset(frontier_type, queue_sizing); // Default queue sizing is 1.0

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "HITSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
            data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, queue_sizing);

            // Allocate output page ranks if necessary
            /*if (!data_slices[gpu]->d_hrank_curr) {
                Value    *d_hrank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_hrank1,
                                nodes * sizeof(Value)),
                            "HITSProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_hrank_curr = d_hrank1;
            }*/
            if (data_slices[gpu]->hrank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->hrank_curr.Allocate(nodes, util::DEVICE)) return retval;

            /*if (!data_slices[gpu]->d_arank_curr) {
                Value    *d_arank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_arank1,
                                nodes * sizeof(Value)),
                            "HITSProblem cudaMalloc d_hrank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_arank_curr = d_arank1;
            }*/
            if (data_slices[gpu]->arank_curr.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->arank_curr.Allocate(nodes, util::DEVICE)) return retval;

            /*if (!data_slices[gpu]->d_hrank_next) {
                Value    *d_hrank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_hrank2,
                                nodes * sizeof(Value)),
                            "HITSProblem cudaMalloc d_hrank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_hrank_next = d_hrank2;
            }*/
            if (data_slices[gpu]->hrank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->hrank_next.Allocate(nodes, util::DEVICE)) return retval; 

            /*if (!data_slices[gpu]->d_arank_next) {
                Value    *d_arank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_arank2,
                                nodes * sizeof(Value)),
                            "HITSProblem cudaMalloc d_arank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_arank_next = d_arank2;
            }*/
            if (data_slices[gpu]->arank_next.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->arank_next.Allocate(nodes, util::DEVICE)) return retval;

            // Allocate d_degrees if necessary
            /*if (!data_slices[gpu]->d_in_degrees) {
                VertexId    *d_in_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_in_degrees,
                                nodes * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_in_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_in_degrees = d_in_degrees;
            }*/
            if (data_slices[gpu]->in_degrees.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->in_degrees.Allocate(nodes, util::DEVICE)) return retval;

            // Allocate d_degrees if necessary
            /*if (!data_slices[gpu]->d_out_degrees) {
                VertexId    *d_out_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_out_degrees,
                                nodes * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_out_degrees = d_out_degrees;
            }*/
            if (data_slices[gpu]->out_degrees.GetPointer(util::DEVICE) == NULL)
                if (retval = data_slices[gpu]->out_degrees.Allocate(nodes, util::DEVICE)) return retval;

            /*if (!data_slices[gpu]->d_delta) {
                Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_delta,
                                1 * sizeof(Value)),
                            "PRProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_delta = d_delta;
            }

            if (!data_slices[gpu]->d_src_node) {
                VertexId    *d_src_node;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_src_node,
                                1 * sizeof(VertexId)),
                            "PRProblem cudaMalloc d_src_node failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_src_node = d_src_node;
            }

            data_slices[gpu]->d_labels = NULL;*/

            // Initial rank_curr = 0 
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->hrank_curr.GetPointer(util::DEVICE), (Value)0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->arank_curr.GetPointer(util::DEVICE), (Value)0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->hrank_next.GetPointer(util::DEVICE), (Value)0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->arank_next.GetPointer(util::DEVICE), (Value)0, nodes);

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->out_degrees.GetPointer(util::DEVICE), 0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->in_degrees.GetPointer(util::DEVICE), 0, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[gpu]->out_degrees.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE)+1, 
                -1, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[gpu]->in_degrees.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->column_offsets.GetPointer(util::DEVICE), 
                this->graph_slices[gpu]->column_offsets.GetPointer(util::DEVICE)+1, 
                -1, nodes);
            //util::DisplayDeviceResults(data_slices[gpu]->d_out_degrees, nodes);

            data_slices[gpu]->delta = delta;
            data_slices[gpu]->src_node = src;
              
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "HITSProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }
        
        // Fillin the initial input_queue for PR problem, this needs to be modified
        // in multi-GPU scene
        Value init_score = 1.0;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->hrank_curr.GetPointer(util::DEVICE) + src,
                        &init_score,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy d_hrank_curr[src] failed", __FILE__, __LINE__)) return retval;

        /*if (retval = util::GRError(cudaMemcpy(
                        data_slices[0]->d_delta,
                        &delta,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy d_delta failed", __FILE__, __LINE__)) return retval;

        if (retval = util::GRError(cudaMemcpy(
                            data_slices[0]->d_src_node,
                            &src,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "CCProblem cudaMemcpy src to d_src_node failed", __FILE__, __LINE__)) return retval;
        */

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(this->data_slices[0]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);

        return retval;
    }

    /** @} */

};

} //namespace hits
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
