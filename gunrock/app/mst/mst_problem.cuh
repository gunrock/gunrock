// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * mst_problem.cuh
 *
 * @brief GPU Storage management Structure for MST Problem Data
 */

#pragma once

#include <limits>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace mst {

/**
 * @brief MST Problem structure stores device-side vectors for
 * doing Minimum Spanning Tree on the GPU.
 *
 * @tparam _VertexId
 * Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT
 * Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value
 * Type of float or double to use for computing MST value.
 * @tparam _MARK_PREDECESSORS  Whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE Whether to enable idempotent operation.
 * @tparam _USE_DOUBLE_BUFFER  Defines whether to use double buffer.
 *
 */
template <
  typename VertexId,
  typename SizeT,
  typename Value,
  bool _MARK_PREDECESSORS,
  bool _ENABLE_IDEMPOTENCE,
  bool _USE_DOUBLE_BUFFER>
struct MSTProblem : ProblemBase <
  VertexId, SizeT, Value,
  _MARK_PREDECESSORS,
  _ENABLE_IDEMPOTENCE,
  _USE_DOUBLE_BUFFER,
  false,                // _ENABLE_BACKWARD
  false,                // _KEEP_ORDER
  false >               // _KEEP_NODE_NUM
{
  static const bool MARK_PREDECESSORS  =  _MARK_PREDECESSORS;
  static const bool ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;

  // helper structures

  /**
   * @brief Data slice structure which contains MST problem specific data.
   *
   * @tparam VertexId Type of signed integer to use as vertex IDs.
   * @tparam SizeT    Type of int / uint to use for array indexing.
   * @tparam Value    Type of float or double to use for attributes.
   */
  struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
  {
    //
    // data slice device-side arrays
    //

    util::Array1D<SizeT, int> done_flags;  // complete flag for jumping kernels
    util::Array1D<SizeT, int> mst_output;  // mask indicates selected MST edges
    util::Array1D<SizeT, unsigned int> flag_array;  // one for start of segment
    util::Array1D<SizeT, unsigned int> edge_flags;  // output of segmented sort
    util::Array1D<SizeT, VertexId> keys_array;  // prefix sum of the flag array
    util::Array1D<SizeT, VertexId> reduce_key;  // segmented reduced keys array
    util::Array1D<SizeT, VertexId> successors;  // destinations have min weight
    util::Array1D<SizeT, VertexId> original_n;  // tracking original vertex IDs
    util::Array1D<SizeT, VertexId> original_e;  // tracking original edge IDs
    util::Array1D<SizeT, VertexId> super_edge;  // super edges next iteration
    util::Array1D<SizeT, VertexId> colindices;  // column indices of CSR graph
    util::Array1D<SizeT, VertexId> temp_index;  // used for storing temp index
    util::Array1D<SizeT, Value>    temp_value;  // used for storing temp value
    util::Array1D<SizeT, Value>    reduce_val;  // store reduced minimum weight
    util::Array1D<SizeT, Value>    edge_value;  // store weight values per edge
    util::Array1D<SizeT, SizeT>    row_offset;  // row offsets arr of CSR graph
    util::Array1D<SizeT, SizeT>    super_idxs;  // vertex ids scanned from flag

    /*
     * @brief Default constructor
     */
    DataSlice()
    {
      done_flags.SetName("done_flags");
      mst_output.SetName("mst_output");
      flag_array.SetName("flag_array");
      edge_flags.SetName("edge_flags");
      keys_array.SetName("keys_array");
      reduce_key.SetName("reduce_key");
      successors.SetName("successors");
      original_n.SetName("original_n");
      original_e.SetName("original_e");
      super_edge.SetName("super_edge");
      colindices.SetName("colindices");
      temp_index.SetName("temp_index");
      temp_value.SetName("temp_value");
      reduce_val.SetName("reduce_val");
      edge_value.SetName("edge_value");
      row_offset.SetName("row_offset");
      super_idxs.SetName("super_idxs");
    }

    /*
     * @brief Default destructor
     */
    ~DataSlice()
    {
      mst_output.Release();
      done_flags.Release();
      flag_array.Release();
      edge_flags.Release();
      keys_array.Release();
      reduce_key.Release();
      successors.Release();
      original_n.Release();
      original_e.Release();
      super_edge.Release();
      colindices.Release();
      temp_index.Release();
      temp_value.Release();
      reduce_val.Release();
      edge_value.Release();
      row_offset.Release();
      super_idxs.Release();
    }
  };

  // Members

  // Number of GPUs to be sliced over
  //int            num_gpus;

  // Size of the graph
  //SizeT          nodes;
  //SizeT          edges;

  // Set of data slices (one for each GPU)
  DataSlice      **data_slices;

  // Nasty method for putting structure on device
  // while keeping the SoA structure
  DataSlice      **d_data_slices;

  // Device indices for each data slice
  //int            *gpu_idx;

  // Methods

  /**
   * @brief MSTProblem default constructor
   */

  MSTProblem() : //nodes(0), edges(0), num_gpus(0) {}
    data_slices  (NULL),
    d_data_slices(NULL)
  {
  }

  /**
   * @brief MSTProblem constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  /*MSTProblem(
    bool  stream_from_host,
    const Csr<VertexId, Value, SizeT> &graph,
    int   num_gpus) : num_gpus(num_gpus)
  {
    Init(stream_from_host, graph, num_gpus);
  }*/

  /**
   * @brief MSTProblem default destructor
   */
  ~MSTProblem()
  {
    for (int i = 0; i < this -> num_gpus; ++i)
    {
      if (util::GRError(
        cudaSetDevice(this -> gpu_idx[i]),
        "~MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

      if (d_data_slices[i])
        util::GRError(cudaFree(d_data_slices[i]),
          "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
    }

    if (d_data_slices) delete[] d_data_slices;
    if (data_slices)   delete[] data_slices;

  }

  /**
   * \addtogroup Public Interface
   * @{
   */

  /**
   * @brief Copy result labels and / or predecessors computed
   * on the GPU back to host-side vectors.
   *
   * @param[out] h_mst_output host-side vector to store MST results.
   *
   *\return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Extract(SizeT *h_mst_output)
  {
    cudaError_t ret = cudaSuccess;
    do
    {
      if ( this -> num_gpus == 1)
      {
        // Set device
        if (util::GRError(cudaSetDevice(this -> gpu_idx[0]),
          "MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

        data_slices[0]->mst_output.SetPointer(h_mst_output);
        if (ret = data_slices[0]->mst_output.Move(
          util::DEVICE, util::HOST)) return ret;
      }
      else
      {
        // TODO: multi-GPU extract result
      }
    } while(0);
    return ret;
  }

  /**
   * @brief MSTProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] _num_gpus Number of the GPUs used.
   * @param[in] streams pointer to CUDA Streams used (NULL by default)
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Init(
    bool          stream_from_host,
    Csr<VertexId, Value, SizeT> 
                 &graph,
    int           num_gpus = 1,
    int          *gpu_idx  = NULL,
    cudaStream_t *streams  = NULL)
  {
    //nodes = graph.nodes;
    //edges = graph.edges;

    ProblemBase < VertexId, SizeT, Value,
      _MARK_PREDECESSORS,
      _ENABLE_IDEMPOTENCE,
      _USE_DOUBLE_BUFFER,
      false,  // _ENABLE_BACKWARD
      false,  // _KEEP_ORDER
      false >::Init(
        stream_from_host, 
        &graph, 
        NULL, 
        num_gpus, 
        gpu_idx, 
        "random");

    // No data in DataSlice needs to be copied from host

    //
    // Allocate output labels.
    //

    cudaError_t ret = cudaSuccess;
    data_slices   = new DataSlice * [num_gpus];
    d_data_slices = new DataSlice * [num_gpus];

    /*if (streams == NULL)
    {
      streams = new cudaStream_t[num_gpus];
      streams[0] = 0;
    }*/

    do
    {
      if (num_gpus <= 1)
      {
        int gpu = 0;
        if (ret = util::SetDevice(this -> gpu_idx[0])) return ret;

        data_slices[0] = new DataSlice;
        if (ret = util::GRError(cudaMalloc(
          (void**)&d_data_slices[0],
          sizeof(DataSlice)),
          "MSTProblem cudaMalloc d_data_slices failed",
          __FILE__, __LINE__)) return ret;

        data_slices[0][0].streams.SetPointer(streams, 1 * 2);
        data_slices[0]->Init(
          1,           // Number of GPUs
          gpu_idx[0],  // GPU indices
          0,           // Number of vertex associate
          0,           // Number of value associate
          &graph,      // Pointer to CSR graph
          NULL,        // Number of in vertices
          NULL);       // Number of out vertices

        //
        // allocate SoA on device
        //

        if (ret = data_slices[gpu]->done_flags.Allocate(            1, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->mst_output.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->flag_array.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->edge_flags.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->keys_array.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->reduce_key.Allocate(this -> nodes, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->successors.Allocate(this -> nodes, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->original_n.Allocate(this -> nodes, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->original_e.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->super_edge.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->colindices.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->temp_index.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->temp_value.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->reduce_val.Allocate(this -> nodes, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->edge_value.Allocate(this -> edges, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->row_offset.Allocate(this -> nodes+1, util::DEVICE))
          return ret;
        if (ret = data_slices[gpu]->super_idxs.Allocate(this -> nodes, util::DEVICE))
          return ret;

        //
        // initialize if necessary
        //

        // initialize done flag for kernel calls
        int *vertex_flag = new int; vertex_flag[0] = 1;
        data_slices[gpu]->done_flags.SetPointer(vertex_flag);
        if (ret = data_slices[gpu]->done_flags.Move(util::HOST, util::DEVICE))
          return ret;

        // initialize mst_output to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->mst_output.GetPointer(util::DEVICE),
          (int)0, this -> edges);

        // initialize flag_array to a vector of zeros
        util::MemsetKernel<unsigned int><<<128, 128>>>(
          data_slices[gpu]->flag_array.GetPointer(util::DEVICE),
          (unsigned int)0, this -> edges);

        // initialize edge_flags to a vector of zeros
        util::MemsetKernel<unsigned int><<<128, 128>>>(
          data_slices[gpu]->edge_flags.GetPointer(util::DEVICE),
          (unsigned int)0, this -> edges);

        // initialize keys_array to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->keys_array.GetPointer(util::DEVICE),
          (VertexId)0, this -> edges);

        // initialize reduce_key to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->reduce_key.GetPointer(util::DEVICE),
          (VertexId)0, this -> edges);

        // initialize successors to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->successors.GetPointer(util::DEVICE),
          (VertexId)0, this -> edges);

        // initialize original node IDs from 0 to nodes
        util::MemsetIdxKernel<<<128, 128>>>(
          data_slices[gpu]->original_n.GetPointer(util::DEVICE), this -> nodes);

        // initialize original edge IDs from 0 to edges
        util::MemsetIdxKernel<<<128, 128>>>(
          data_slices[gpu]->original_e.GetPointer(util::DEVICE), this -> edges);

        // initialize super edges with graph.column_indices
        data_slices[gpu]->super_edge.SetPointer(graph.column_indices);
        if (ret = data_slices[gpu]->super_edge.Move(util::HOST, util::DEVICE))
          return ret;

        // initialize col_indices with graph.column_indices
        data_slices[gpu]->colindices.SetPointer(graph.column_indices);
        if (ret = data_slices[gpu]->colindices.Move(util::HOST, util::DEVICE))
          return ret;

        // initialize temp_index to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->temp_index.GetPointer(util::DEVICE),
          (VertexId)0, this -> edges);

        // initialize temp_value to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->temp_value.GetPointer(util::DEVICE),
          (Value)0, this -> edges);

        // initialize reduce_val to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->reduce_val.GetPointer(util::DEVICE),
          (Value)0, this -> edges);

        // initialize edge_value with graph.column_indices
        data_slices[gpu]->edge_value.SetPointer(graph.edge_values);
        if (ret = data_slices[gpu]->edge_value.Move(util::HOST, util::DEVICE))
          return ret;

        // initialize row_offset with graph.column_indices
        data_slices[gpu]->row_offset.SetPointer(graph.row_offsets);
        if (ret = data_slices[gpu]->row_offset.Move(util::HOST, util::DEVICE))
          return ret;

        // initialize super_idxs to a vector of zeros
        util::MemsetKernel<<<128, 128>>>(
          data_slices[gpu]->super_idxs.GetPointer(util::DEVICE),
          (SizeT)0, this -> nodes);

        if (ret = data_slices[0]->labels.Allocate(this -> nodes, util::DEVICE))
        {
          return ret;
        }

        delete vertex_flag;
      }
    } while (0);
    return ret;
  }

  /**
   * @brief Performs any initialization work needed for MST problem type.
   * Must be called prior to each MST iteration.
   *
   * @param[in] frontier_type The frontier type (i.e., edge / vertex / mixed)
   * @param[in] queue_sizing Queue sizing factor
   *
   *  \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Reset(FrontierType frontier_type, double queue_sizing)
  {
    cudaError_t ret = cudaSuccess;

    for (int gpu = 0; gpu < this -> num_gpus; ++gpu)
    {
      // Set device
      if (ret = util::GRError(
        cudaSetDevice( this -> gpu_idx[gpu]),
        "MSTProblem cudaSetDevice failed",
        __FILE__, __LINE__)) return ret;

      data_slices[gpu]->Reset(
        frontier_type,
        this->graph_slices[gpu],
        queue_sizing,
        queue_sizing);

      //
      // Allocate outputs if necessary
      //

      if (data_slices[gpu]->done_flags.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->done_flags.Allocate(    1, util::DEVICE))
          return ret;

      if (data_slices[gpu]->mst_output.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->mst_output.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->flag_array.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->flag_array.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->edge_flags.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->edge_flags.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->keys_array.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->keys_array.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->reduce_key.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->reduce_key.Allocate(this -> nodes, util::DEVICE))
          return ret;

      if (data_slices[gpu]->successors.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->successors.Allocate(this -> nodes, util::DEVICE))
          return ret;

      if (data_slices[gpu]->original_n.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->original_n.Allocate(this -> nodes, util::DEVICE))
          return ret;

      if (data_slices[gpu]->original_e.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->original_e.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->super_edge.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->super_edge.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->colindices.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->colindices.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->temp_index.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->temp_index.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->temp_value.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->temp_value.Allocate(this -> edges, util::DEVICE))
          return ret;

      if (data_slices[gpu]->reduce_val.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->reduce_val.Allocate(this -> nodes, util::DEVICE))
          return ret;

      if (data_slices[gpu]->edge_value.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->edge_value.Allocate(this -> edges, util::DEVICE))

          return ret;

      if (data_slices[gpu]->row_offset.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->row_offset.Allocate(this -> nodes+1, util::DEVICE))
          return ret;

      if (data_slices[gpu]->super_idxs.GetPointer(util::DEVICE) == NULL)
        if (ret = data_slices[gpu]->super_idxs.Allocate(this -> nodes, util::DEVICE))
          return ret;


      if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL)
      {
        if (ret = data_slices[gpu]->labels.Allocate(this -> nodes, util::DEVICE))
        {
          return ret;
        }
      }

      // put every vertex frontier queue used for mappings
      util::MemsetIdxKernel<<<128, 128>>>(
        data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
        this -> nodes);

      // put every edges frontier queue used for mappings
      util::MemsetIdxKernel<<<128, 128>>>(
        data_slices[gpu]->frontier_queues[0].values[0].GetPointer(util::DEVICE),
        this -> edges);

      if (ret = util::GRError(cudaMemcpy(d_data_slices[gpu],
          data_slices[gpu], sizeof(DataSlice), cudaMemcpyHostToDevice),
          "MSTProblem cudaMemcpy data_slices to d_data_slices failed",
          __FILE__, __LINE__)) return ret;

    }
    return ret;
  }

  /** @} */

};

} //namespace mst
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
