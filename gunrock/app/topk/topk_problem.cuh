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
  typename    _VertexId,
  typename    _SizeT,
  typename    _Value>
struct TOPKProblem : ProblemBase<_VertexId, _SizeT, _Value,
  true,  // MARK_PREDECESSORS
  false, // ENABLE_IDEMPOTENE
  false, // USE_DOUBLE_BUFFER
  false, // ENABLE_BACKWORD
  false, // KEEP_ORDER
  false> // KEEP_NODE_NUM
{
  typedef _VertexId   VertexId;
  typedef _SizeT      SizeT;
  typedef _Value      Value;
  
  static const bool MARK_PREDECESSORS     = true;
  static const bool ENABLE_IDEMPOTENCE    = false;

  //Helper structures
  
  /**
   * @brief Data slice structure which contains TOPK problem specific data.
   */
  struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
  {

    // device storage arrays
    //SizeT       *d_labels;
    VertexId    *d_node_id;   //!< top k node ids
    Value       *d_degrees_s; //!< sum/total degrees
    Value       *d_degrees_i; //!< in-going  degrees
    Value       *d_degrees_o; //!< out-going degrees
    Value       *d_temp_i;    //!< used for sorting in degrees
    Value       *d_temp_o;    //!< used for sorting out degrees
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
   * @brief TOPKProblem default constructor
   */
  
  TOPKProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

  /**
   * @brief TOPKProblem constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph_original Reference to the CSR graph object we process on.
   * @param[in] graph_reversed Reference to the inversed CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  TOPKProblem(
    bool stream_from_host,       // Only meaningful for single-GPU
    const Csr<VertexId, Value, SizeT> &graph_original,
    const Csr<VertexId, Value, SizeT> &graph_reversed,
    int  num_gpus) :
    num_gpus(num_gpus)
  {
    Init(stream_from_host, graph_original, graph_reversed, num_gpus);
  }

  /**
   * @brief TOPKProblem default destructor
   */
  ~TOPKProblem()
  {
    for (int i = 0; i < num_gpus; ++i)
    {
      if (util::GRError(cudaSetDevice(gpu_idx[i]),
      "~TOPKProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
      if (data_slices[i]->d_node_id)
        util::GRError(cudaFree(data_slices[i]->d_node_id),
        "GpuSlice cudaFree d_node_id failed", __FILE__, __LINE__);
      if (data_slices[i]->d_degrees_s)
        util::GRError(cudaFree(data_slices[i]->d_degrees_s),
        "GpuSlice cudaFree d_degrees_s failed", __FILE__, __LINE__);
      if (data_slices[i]->d_degrees_i)
        util::GRError(cudaFree(data_slices[i]->d_degrees_i),
        "GpuSlice cudaFree d_degrees_i failed", __FILE__, __LINE__);
      if (data_slices[i]->d_degrees_o)
        util::GRError(cudaFree(data_slices[i]->d_degrees_o),
        "GpuSlice cudaFree d_degrees_o failed", __FILE__, __LINE__);
      if (data_slices[i]->d_temp_i)
        util::GRError(cudaFree(data_slices[i]->d_temp_i),
        "GpuSlice cudaFree d_temp_i failed", __FILE__, __LINE__);
      if (data_slices[i]->d_temp_o)
        util::GRError(cudaFree(data_slices[i]->d_temp_o),
        "GpuSlice cudaFree d_temp_o failed", __FILE__, __LINE__);
      if (d_data_slices[i])
        util::GRError(cudaFree(d_data_slices[i]),
        "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
    }
    if (d_data_slices) delete[] d_data_slices;
    if (data_slices)   delete[] data_slices;
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
    Value    *h_degrees_i,
    Value    *h_degrees_o,
    SizeT    num_nodes)
  {
    cudaError_t retval = cudaSuccess;
    do
    {
      if (num_gpus == 1)
      {
        // Set device
        if (util::GRError(cudaSetDevice(gpu_idx[0]),
          "TOPKProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

        if (retval = util::GRError(cudaMemcpy(
          h_node_id,
          data_slices[0]->d_node_id,
          sizeof(VertexId) * num_nodes,
          cudaMemcpyDeviceToHost),
          "TOPK cudaMemcpy d_node_id failed", __FILE__, __LINE__)) break;

        if (retval = util::GRError(cudaMemcpy(
          h_degrees_i,
          data_slices[0]->d_degrees_i,
          sizeof(Value) * num_nodes,
          cudaMemcpyDeviceToHost),
          "TOPK cudaMemcpy d_degrees_i failed", __FILE__, __LINE__)) break;

        if (retval = util::GRError(cudaMemcpy(
          h_degrees_o,
          data_slices[0]->d_degrees_o,
          sizeof(Value) * num_nodes,
          cudaMemcpyDeviceToHost),
          "TOPK cudaMemcpy d_degrees_i failed", __FILE__, __LINE__)) break;
      }
      else
      {
        // TODO: multi-GPU extract result
      }
    } while(0);
    return retval;
  }

  /**
   * @brief TOPKProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph_original Reference to the CSR graph object we process on. @see Csr
   * @param[in] graph_reversed Reference to the inversed CSR graph object we process on. @see Csr
   * @param[in] _num_gpus Number of the GPUs used.
   * @param[in] streams pointer to CUDA Streams.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Init(
    bool stream_from_host, // Only meaningful for single-GPU
    Csr<VertexId, Value, SizeT> &graph_original,
    Csr<VertexId, Value, SizeT> &graph_reversed,
    int  _num_gpus,
    cudaStream_t *streams = NULL)
  { 
    // non-inversed graph
    num_gpus = _num_gpus;
    nodes    = graph_original.nodes;
    edges    = graph_original.edges;
    //VertexId *h_row_offsets_i = graph_i.row_offsets;
    //VertexId *h_col_indices_i = graph_i.column_indices;
    //VertexId *h_row_offsets_o = graph_o.row_offsets;
    //VertexId *h_col_indices_o = graph_o.column_indices;
    ProblemBase<VertexId, SizeT, Value, 
        true, false, false, false, false, false>
      ::Init(stream_from_host,
        &graph_original,
        &graph_reversed,
        num_gpus,
        NULL,
        "random");

    // No data in DataSlice needs to be copied from host
    
    /**
     * Allocate output labels/preds
     */
    cudaError_t retval = cudaSuccess;

    data_slices   = new DataSlice*[num_gpus];
    d_data_slices = new DataSlice*[num_gpus];
    if (streams == NULL) {streams = new cudaStream_t[num_gpus]; streams[0] = 0;}
 
    do {
        if (num_gpus <= 1) 
        {
            gpu_idx = (int*)malloc(sizeof(int));
            // Create a single data slice for the currently-set gpu
            int gpu;
            if (retval = util::GRError(cudaGetDevice(&gpu), 
            "TOPKProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
            gpu_idx[0] = gpu;

            data_slices[0] = new DataSlice;
            if (retval = util::GRError(cudaMalloc((void**)&d_data_slices[0],
                            sizeof(DataSlice)),
                         "TOPKProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;
            data_slices[0][0].streams.SetPointer(streams,1);
            data_slices[0]->Init(
                1,  
                gpu_idx[0],
                0,  
                0,  
                &graph_original,
                NULL,
                NULL);


            // Create SoA on device
            VertexId    *d_node_id;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_node_id,
              nodes * sizeof(VertexId)),
              "TOPK cudaMalloc d_node_id failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_node_id = d_node_id;

            Value *d_degrees_s;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_degrees_s,
              nodes * sizeof(Value)),
              "TOPK cudaMalloc d_degrees_s failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_degrees_s = d_degrees_s;

            Value *d_degrees_i;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_degrees_i,
              nodes * sizeof(Value)),
              "TOPK cudaMalloc d_degrees_i failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_degrees_i = d_degrees_i;

            Value *d_degrees_o;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_degrees_o,
              nodes * sizeof(Value)),
              "TOPK cudaMalloc d_degrees_o failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_degrees_o = d_degrees_o;

            Value *d_temp_i;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_temp_i,
              nodes * sizeof(Value)),
              "TOPK cudaMalloc d_temp_i failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_temp_i = d_temp_i;

            Value *d_temp_o;
            if (retval = util::GRError(cudaMalloc(
              (void**)&d_temp_o,
              nodes * sizeof(Value)),
              "TOPK cudaMalloc d_temp_o failed",
              __FILE__, __LINE__)) return retval;
            data_slices[0]->d_temp_o = d_temp_o;

	  
	    }
        //TODO: add multi-GPU allocation code
    } while (0);
    
    return retval;
  }
  
  /**
   *  @brief Performs any initialization work needed for TOPK problem type. 
   *	Must be called prior to each TOPK iteration.
   *
   *  @param[in] frontier_type The f rontier type (i.e., edge/vertex/mixed)
   * 
   *  \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Reset(FrontierType frontier_type)             // The frontier type (i.e., edge/vertex/mixed)
  {
    //typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
    //load ProblemBase Reset
    //BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0
    
    cudaError_t retval = cudaSuccess;
    
    for (int gpu = 0; gpu < num_gpus; ++gpu) 
    {
	    // Set device
	    if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
				   "TOPKProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
	
        data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], 1.0f, 1.0f);
        // Allocate output if necessary
        if (!data_slices[gpu]->d_node_id)
        {
          VertexId    *d_node_id;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_node_id,
            nodes * sizeof(VertexId)),
            "TOPK cudaMalloc d_node_id failed", __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_node_id = d_node_id;
        }

        if (!data_slices[gpu]->d_degrees_s)
        {
          Value *d_degrees_s;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_degrees_s,
            nodes * sizeof(Value)),
            "TOPK cudaMalloc d_degrees_s failed",
            __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_degrees_s = d_degrees_s;
        }

        if (!data_slices[gpu]->d_degrees_i)
        {
          Value *d_degrees_i;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_degrees_i,
            nodes * sizeof(Value)),
            "TOPK cudaMalloc d_degrees_i failed",
            __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_degrees_i = d_degrees_i;
        }

        if (!data_slices[gpu]->d_degrees_o)
        {
          Value *d_degrees_o;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_degrees_o,
            nodes * sizeof(Value)),
            "TOPK cudaMalloc d_degrees_o failed",
            __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_degrees_o = d_degrees_o;
        }

        if (!data_slices[gpu]->d_temp_i)
        {
          Value *d_temp_i;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_temp_i,
            nodes * sizeof(Value)),
            "TOPK cudaMalloc d_temp_i failed",
            __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_temp_i = d_temp_i;
        }

        if (!data_slices[gpu]->d_temp_o)
        {
          Value *d_temp_o;
          if (retval = util::GRError(cudaMalloc(
            (void**)&d_temp_o,
            nodes * sizeof(Value)),
            "TOPK cudaMalloc d_temp_o failed",
            __FILE__, __LINE__)) return retval;
          data_slices[gpu]->d_temp_o = d_temp_o;
        }

        if (retval = util::GRError(cudaMemcpy(
          d_data_slices[gpu],
          data_slices[gpu],
          sizeof(DataSlice),
          cudaMemcpyHostToDevice),
          "TOPK cudaMemcpy data_slices to d_data_slices failed",
          __FILE__, __LINE__)) return retval;
    }

    // Fillin the initial input_queue for TOPK problem, this needs to be modified
    // in multi-GPU scene

    // set node ids
    util::MemsetIdxKernel<<<128, 128>>>(data_slices[0]->d_node_id, nodes);

    // count number of out-going degrees for each node
    util::MemsetMadVectorKernel<<<128, 128>>>(
      data_slices[0]->d_degrees_o,
      this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE),
      this->graph_slices[0]->row_offsets.GetPointer(util::DEVICE)+1, -1, nodes);

    // count number of in-going degrees for each node
    util::MemsetMadVectorKernel<<<128, 128>>>(
      data_slices[0]->d_degrees_i,
      this->graph_slices[0]->column_offsets.GetPointer(util::DEVICE),
      this->graph_slices[0]->column_offsets.GetPointer(util::DEVICE)+1, -1, nodes);

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
