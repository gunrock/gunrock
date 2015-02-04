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
 * @tparam _USE_DOUBLE_BUFFER
 * Boolean type parameter which defines whether to use double buffer.
 *
 */
template <
  typename _VertexId,
  typename _SizeT,
  typename _Value,
  bool     _USE_DOUBLE_BUFFER>
struct MSTProblem : ProblemBase<_VertexId, _SizeT, _USE_DOUBLE_BUFFER>
{
  typedef _VertexId VertexId;
  typedef _SizeT    SizeT;
  typedef _Value    Value;

  static const bool MARK_PREDECESSORS  = true;
  static const bool ENABLE_IDEMPOTENCE = false;

  // helper structures

  /**
   * @brief Data slice structure which contains MST problem specific data.
   */
  struct DataSlice
  {
    SizeT    *d_labels;

    // device storage arrays
    int          *d_vertex_flag;  // finish flag for per-vertex kernels
    int          *d_mst_output;   // mark selected edges with 1
    unsigned int *d_flags_array;  // flags 1 start of segment, 0 otherwise
    unsigned int *d_edge_flags;   // flags from the output of segment sort
    VertexId     *d_keys_array;   // keys array - scan of the flags array
    VertexId     *d_reduced_keys; // reduced keys array
    VertexId     *d_successors;   // destination vertices that have min weight
    VertexId     *d_origin_nodes; // used to track origin vertex ids
    VertexId     *d_origin_edges; // origin edge list keep track of e_ids
    VertexId     *d_super_edges;  // super edge list for next iteration
    VertexId     *d_col_indices;  // column indices of CSR graph (edges)
    Value        *d_reduced_vals; // store reduced minimum weights
    Value        *d_edge_weights; // store weights per edge
    Value        *d_temp_storage; // used for storing temp arrays
    Value        *d_tmp_storage;  // used for storing temp arrays
    SizeT        *d_supervtx_ids; // super vertex ids scanned from flags
    SizeT        *d_row_offsets;  // row offsets of CSR graph
  };

  // Members

  // Number of GPUs to be sliced over
  int            num_gpus;

  // Size of the graph
  SizeT          nodes;
  SizeT          edges;

  // Selector
  SizeT          selector;

  // Set of data slices (one for each GPU)
  DataSlice      **data_slices;

  // Nasty method for putting structure on device
  // while keeping the SoA structure
  DataSlice      **d_data_slices;

  // Device indices for each data slice
  int            *gpu_idx;

  // Methods

  /**
   * @brief MSTProblem default constructor
   */

  MSTProblem():
  nodes(0),
  edges(0),
  num_gpus(0) {}

  /**
   * @brief MSTProblem constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  MSTProblem(
    bool  stream_from_host,
    const Csr<VertexId, Value, SizeT> &graph,
    int   num_gpus) : num_gpus(num_gpus)
  {
    Init(stream_from_host, graph, num_gpus);
  }

  /**
   * @brief MSTProblem default destructor
   */
  ~MSTProblem()
  {
    for (int i = 0; i < num_gpus; ++i)
    {
      if (util::GRError(
        cudaSetDevice(gpu_idx[i]),
        "~MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

      if (data_slices[i]->d_col_indices)
        util::GRError(cudaFree(data_slices[i]->d_col_indices),
          "GpuSlice cudaFree d_col_indices  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_edge_weights)
        util::GRError(cudaFree(data_slices[i]->d_edge_weights),
          "GpuSlice cudaFree d_edge_weights failed", __FILE__, __LINE__);
      if (data_slices[i]->d_reduced_vals)
        util::GRError(cudaFree(data_slices[i]->d_reduced_vals),
          "GpuSlice cudaFree d_reduced_vals failed", __FILE__, __LINE__);
      if (data_slices[i]->d_flags_array)
        util::GRError(cudaFree(data_slices[i]->d_flags_array),
          "GpuSlice cudaFree  d_flags_array failed", __FILE__, __LINE__);
      if (data_slices[i]->d_keys_array)
        util::GRError(cudaFree(data_slices[i]->d_keys_array),
          "GpuSlice cudaFree  d_keys_array  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_temp_storage)
        util::GRError(cudaFree(data_slices[i]->d_temp_storage),
          "GpuSlice cudaFree d_temp_storage failed", __FILE__, __LINE__);
      if (data_slices[i]->d_reduced_keys)
        util::GRError(cudaFree(data_slices[i]->d_reduced_keys),
          "GpuSlice cudaFree d_reduced_keys failed", __FILE__, __LINE__);
      if (data_slices[i]->d_successors)
        util::GRError(cudaFree(data_slices[i]->d_successors),
          "GpuSlice cudaFree  d_successors  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_row_offsets)
        util::GRError(cudaFree(data_slices[i]->d_row_offsets),
          "GpuSlice cudaFree d_row_offsets  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_vertex_flag)
        util::GRError(cudaFree(data_slices[i]->d_vertex_flag),
          "GpuSlice cudaFree d_vertex_flag  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_origin_nodes)
        util::GRError(cudaFree(data_slices[i]->d_origin_nodes),
          "GpuSlice cudaFree d_origin_nodes failed", __FILE__, __LINE__);
      if (data_slices[i]->d_supervtx_ids)
        util::GRError(cudaFree(data_slices[i]->d_supervtx_ids),
          "GpuSlice cudaFree d_supervtx_ids failed", __FILE__, __LINE__);
      if (data_slices[i]->d_origin_edges)
        util::GRError(cudaFree(data_slices[i]->d_origin_edges),
          "GpuSlice cudaFree d_origin_edges failed", __FILE__, __LINE__);
      if (data_slices[i]->d_mst_output)
        util::GRError(cudaFree(data_slices[i]->d_mst_output),
          "GpuSlice cudaFree  d_mst_output  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_edge_flags)
        util::GRError(cudaFree(data_slices[i]->d_edge_flags),
          "GpuSlice cudaFree  d_edge_flags  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_tmp_storage)
        util::GRError(cudaFree(data_slices[i]->d_tmp_storage),
          "GpuSlice cudaFree d_tmp_storage  failed", __FILE__, __LINE__);
      if (data_slices[i]->d_super_edges)
        util::GRError(cudaFree(data_slices[i]->d_super_edges),
          "GpuSlice cudaFree d_super_edges  failed", __FILE__, __LINE__);

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
   * @brief Copy result labels and / or predecessors computed
   * on the GPU back to host-side vectors.
   *
   * @param[out] h_mst_output host-side vector to store MST results.
   *
   *\return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  //TODO: write extract function
  cudaError_t Extract(SizeT *h_mst_output)
  {
    cudaError_t retval = cudaSuccess;
    do
    {
      if (num_gpus == 1)
      {
        // Set device
        if (util::GRError(cudaSetDevice(gpu_idx[0]),
          "MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

        if (retval = util::GRError(cudaMemcpy(
          h_mst_output, data_slices[0]->d_mst_output,
          sizeof(SizeT) * edges, cudaMemcpyDeviceToHost),
          "MSTProblem cudaMemcpy selector failed", __FILE__, __LINE__)) break;
      }
      else
      {
        // TODO: multi-GPU extract result
      }
    } while(0);
    return retval;
  }

  /**
   * @brief MSTProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] _num_gpus Number of the GPUs used.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Init(
    bool  stream_from_host,
    const Csr<VertexId, Value, SizeT> &graph,
    int   _num_gpus)
  {
    num_gpus = _num_gpus;

    nodes = graph.nodes;
    edges = graph.edges;

    VertexId *h_row_offsets = graph.row_offsets;
    VertexId *h_col_indices = graph.column_indices;

    ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>::Init(stream_from_host,
      nodes, edges, h_row_offsets, h_col_indices, NULL, NULL, num_gpus);

    // No data in DataSlice needs to be copied from host

    /**
     * Allocate output labels / predecessors
     */
    cudaError_t retval = cudaSuccess;
    data_slices   = new DataSlice*[num_gpus];
    d_data_slices = new DataSlice*[num_gpus];

    do
    {
      if (num_gpus <= 1)
      {
        gpu_idx = (int*)malloc(sizeof(int));

        // create a single data slice for the currently-set GPU
        int gpu;
        if (retval = util::GRError(cudaGetDevice(&gpu),
          "MSTProblem cudaGetDevice failed",
          __FILE__, __LINE__)) break;
        gpu_idx[0] = gpu;

        data_slices[0] = new DataSlice;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_data_slices[0],
          sizeof(DataSlice)),
          "MSTProblem cudaMalloc d_data_slices failed",
          __FILE__, __LINE__)) return retval;

        // create SoA on device
        VertexId *d_col_indices;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_col_indices,
          edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_col_indices failed",
          __FILE__, __LINE__)) return retval;
        if (retval = util::GRError(cudaMemcpy(
          d_col_indices,
          graph.column_indices,
          edges * sizeof(VertexId),
          cudaMemcpyHostToDevice),
          "ProblemBase cudaMemcpy d_col_indices failed",
          __FILE__, __LINE__)) return retval;
        data_slices[0]->d_col_indices = d_col_indices;

        VertexId *d_super_edges;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_super_edges,
          edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_super_edges failed",
          __FILE__, __LINE__)) return retval;
        if (retval = util::GRError(cudaMemcpy(
          d_super_edges,
          graph.column_indices,
          edges * sizeof(VertexId),
          cudaMemcpyHostToDevice),
          "ProblemBase cudaMemcpy d_super_edges failed",
          __FILE__, __LINE__)) return retval;
        data_slices[0]->d_super_edges = d_super_edges;

        Value *d_edge_weights;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_edge_weights,
          edges * sizeof(Value)),
          "MSTProblem cudaMalloc d_edge_weights failed",
          __FILE__, __LINE__)) return retval;
        if (retval = util::GRError(cudaMemcpy(
          d_edge_weights,
          graph.edge_values,
          edges * sizeof(Value),
          cudaMemcpyHostToDevice),
          "ProblemBase cudaMemcpy d_edge_weights failed",
          __FILE__, __LINE__)) return retval;
        data_slices[0]->d_edge_weights = d_edge_weights;

        Value *d_reduced_vals;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_reduced_vals,
          nodes * sizeof(Value)),
          "MSTProblem cudaMalloc d_reduced_vals failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_reduced_vals = d_reduced_vals;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_reduced_vals, 0, nodes);

        unsigned int *d_flags_array;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_flags_array,
          edges * sizeof(unsigned int)),
          "MSTProblem cudaMalloc d_flags_array Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_flags_array = d_flags_array;
        util::MemsetKernel<unsigned int><<<128, 128>>>(
          data_slices[0]->d_flags_array, 0, edges);

        VertexId *d_keys_array;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_keys_array,
          edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_keys_array Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_keys_array = d_keys_array;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_keys_array, 0, edges);

        SizeT *d_temp_storage;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_temp_storage,
          edges * sizeof(SizeT)),
          "MSTProblem cudaMalloc d_temp_storage Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_temp_storage = d_temp_storage;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_temp_storage, 0, edges);

        VertexId *d_reduced_keys;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_reduced_keys,
          nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_reduced_keys Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_reduced_keys = d_reduced_keys;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_reduced_keys, 0, nodes);

        VertexId *d_successors;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_successors,
          nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_successors Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_successors = d_successors;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_successors, 0, nodes);

        SizeT *d_row_offsets;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_row_offsets,
          (nodes + 1) * sizeof(SizeT)),
          "MSTProblem cudaMalloc d_row_offsets Failed",
          __FILE__, __LINE__)) return retval;
        if (retval = util::GRError(cudaMemcpy(
          d_row_offsets,
          graph.row_offsets,
          (nodes + 1) * sizeof(SizeT),
          cudaMemcpyHostToDevice),
          "ProblemBase cudaMemcpy d_row_offsets failed",
          __FILE__, __LINE__)) return retval;
        data_slices[0]->d_row_offsets = d_row_offsets;

        int *d_vertex_flag;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_vertex_flag,
          sizeof(int)),
          "MSTProblem cudaMalloc d_vertex_flag failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_vertex_flag = d_vertex_flag;

        VertexId *d_origin_nodes;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_origin_nodes,
          nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_origin_nodes failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_origin_nodes = d_origin_nodes;
        util::MemsetIdxKernel<<<128, 128>>>(
          data_slices[0]->d_origin_nodes, nodes);

        VertexId *d_supervtx_ids;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_supervtx_ids,
          nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_supervtx_ids Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_supervtx_ids = d_supervtx_ids;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_supervtx_ids, 0, nodes);

        VertexId *d_origin_edges;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_origin_edges,
          edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_origin_edges Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_origin_edges = d_origin_edges;
        util::MemsetIdxKernel<<<128, 128>>>(
          data_slices[0]->d_origin_edges, edges);

        int *d_mst_output;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_mst_output,
          edges * sizeof(SizeT)),
          "MSTProblem cudaMalloc d_mst_output Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_mst_output = d_mst_output;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_mst_output, 0, edges);

        unsigned int *d_edge_flags;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_edge_flags,
          edges * sizeof(unsigned int)),
          "MSTProblem cudaMalloc d_edge_flags Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_edge_flags = d_edge_flags;
        util::MemsetKernel<unsigned int><<<128, 128>>>(
          data_slices[0]->d_edge_flags, 0, edges);

        Value *d_tmp_storage;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_tmp_storage,
          edges * sizeof(Value)),
          "MSTProblem cudaMalloc d_tmp_storage Failed",
          __FILE__, __LINE__)) return retval;
          data_slices[0]->d_tmp_storage = d_tmp_storage;
        util::MemsetKernel<<<128, 128>>>(
          data_slices[0]->d_tmp_storage, 0, edges);

        data_slices[0]->d_labels = NULL;
      }
    } while (0);
    return retval;
  }

  /**
   * @brief Performs any initialization work needed for MST problem type.
   * Must be called prior to each MST iteration.
   *
   * @param[in] frontier_type The frontier type (i.e., edge / vertex / mixed)
   *
   *  \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Reset(FrontierType frontier_type)
  {
    typedef ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER> BaseProblem;

    // load ProblemBase Reset
    BaseProblem::Reset(frontier_type, 1.0f); // default queue sizing is 1.0

    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
      // Set device
      if (retval = util::GRError(
        cudaSetDevice(gpu_idx[gpu]),
        "MSTProblem cudaSetDevice failed",
        __FILE__, __LINE__)) return retval;

      // Allocate output if necessary
      if (!data_slices[gpu]->d_col_indices)
      {
        VertexId *d_col_indices;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_col_indices, edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_col_indices failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_col_indices = d_col_indices;
      }

      if (!data_slices[gpu]->d_super_edges)
      {
        VertexId *d_super_edges;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_super_edges, edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_super_edges failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_super_edges = d_super_edges;
      }

      if (!data_slices[gpu]->d_edge_weights)
      {
        Value *d_edge_weights;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_edge_weights, edges * sizeof(Value)),
          "MSTProblem cudaMalloc d_edge_weights failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_edge_weights = d_edge_weights;
      }

      if(!data_slices[gpu]->d_reduced_vals)
      {
        Value *d_reduced_vals;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_reduced_vals, nodes * sizeof(Value)),
          "MSTProblem cudaMalloc d_reduced_vals failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_reduced_vals = d_reduced_vals;
      }

      if (!data_slices[gpu]->d_flags_array)
      {
        unsigned int *d_flags_array;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_flags_array, edges * sizeof(unsigned int)),
          "MSTProblem cudaMalloc d_flags_array Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_flags_array = d_flags_array;
      }

      if (!data_slices[gpu]->d_keys_array)
      {
        VertexId *d_keys_array;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_keys_array, edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_keys_array Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_keys_array = d_keys_array;
      }

      if (!data_slices[gpu]->d_temp_storage)
      {
        SizeT *d_temp_storage;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_temp_storage, edges * sizeof(SizeT)),
          "MSTProblem cudaMalloc d_temp_storage Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_temp_storage = d_temp_storage;
      }

      if (!data_slices[gpu]->d_successors)
      {
        VertexId *d_successors;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_successors, nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_successors Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_successors = d_successors;
      }

      if (!data_slices[gpu]->d_reduced_keys)
      {
        VertexId *d_reduced_keys;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_reduced_keys, nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_reduced_keys Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_reduced_keys = d_reduced_keys;
      }

      if (!data_slices[gpu]->d_row_offsets)
      {
        SizeT *d_row_offsets;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_row_offsets, (nodes+1) * sizeof(SizeT)),
          "MSTProblem cudaMalloc d_row_offsets Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_row_offsets = d_row_offsets;
      }

      int *vertex_flag = new int;
      // allocate vertex_flag if necessary
      if (!data_slices[gpu]->d_vertex_flag)
      {
        int *d_vertex_flag;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_vertex_flag, sizeof(int)),
          "MSTProblem cudaMalloc d_vertex_flag failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_vertex_flag = d_vertex_flag;
      }
      vertex_flag[0] = 1;
      if (retval = util::GRError(cudaMemcpy(
        data_slices[gpu]->d_vertex_flag,
        vertex_flag, sizeof(int), cudaMemcpyHostToDevice),
        "MSTProblem cudaMemcpy vertex_flag to d_vertex_flag failed",
        __FILE__, __LINE__)) return retval;
      delete vertex_flag;

      if (!data_slices[gpu]->d_origin_nodes)
      {
        VertexId *d_origin_nodes;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_origin_nodes, nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_origin_nodes Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_origin_nodes = d_origin_nodes;
      }

      if (!data_slices[gpu]->d_supervtx_ids)
      {
        VertexId *d_supervtx_ids;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_supervtx_ids, nodes * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_supervtx_ids Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_supervtx_ids = d_supervtx_ids;
      }

      if (!data_slices[gpu]->d_origin_edges)
      {
        VertexId *d_origin_edges;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_origin_edges, edges * sizeof(VertexId)),
          "MSTProblem cudaMalloc d_origin_edges Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_origin_edges = d_origin_edges;
      }

      if (!data_slices[gpu]->d_mst_output)
      {
        int *d_mst_output;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_mst_output, edges * sizeof(int)),
          "MSTProblem cudaMalloc d_mst_output Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_mst_output = d_mst_output;
      }

      if (!data_slices[gpu]->d_edge_flags)
      {
        unsigned int *d_edge_flags;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_edge_flags, edges * sizeof(unsigned int)),
          "MSTProblem cudaMalloc d_edge_flags Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_edge_flags = d_edge_flags;
      }

      if (!data_slices[gpu]->d_tmp_storage)
      {
        Value *d_tmp_storage;
        if (retval = util::GRError(cudaMalloc(
          (void**)&d_tmp_storage, edges * sizeof(Value)),
          "MSTProblem cudaMalloc d_tmp_storage Failed",
          __FILE__, __LINE__)) return retval;
        data_slices[gpu]->d_tmp_storage = d_tmp_storage;
      }

      data_slices[0]->d_labels = NULL;

      if (retval = util::GRError(cudaMemcpy(d_data_slices[gpu],
        data_slices[gpu], sizeof(DataSlice), cudaMemcpyHostToDevice),
        "MSTProblem cudaMemcpy data_slices to d_data_slices failed",
        __FILE__, __LINE__)) return retval;

    }

    // put every vertex / edges in frontier queue used for mappings
    util::MemsetIdxKernel<<<128, 128>>>(
      BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

    util::MemsetIdxKernel<<<128, 128>>>(
      BaseProblem::graph_slices[0]->frontier_queues.d_values[0], edges);

    return retval;

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