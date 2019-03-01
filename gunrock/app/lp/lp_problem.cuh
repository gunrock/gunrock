// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file lp_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <cub/cub.cuh>
#include <moderngpu.cuh>

namespace gunrock {
namespace app {
namespace lp {

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
template <typename VertexId, typename SizeT, typename Value>
// bool _MARK_PREDECESSORS,
// bool _ENABLE_IDEMPOTENCE>
// bool _USE_DOUBLE_BUFFER >
struct LpProblem : ProblemBase<VertexId, SizeT, Value,
                               true,   //_MARK_PREDECESSORS,
                               false>  //_ENABLE_IDEMPOTENCE>
                                       //_USE_DOUBLE_BUFFER,
                                       // false,   // _ENABLE_BACKWARD
// false,   // _KEEP_ORDER
// false>   // _KEEP_NODE_NUM
{
  static const bool MARK_PREDECESSORS = true;
  static const bool ENABLE_IDEMPOTENCE = false;
  static const int MAX_NUM_VERTEX_ASSOCIATES = 0;
  static const int MAX_NUM_VALUE__ASSOCIATES = 0;
  typedef ProblemBase<VertexId, SizeT, Value, MARK_PREDECESSORS,
                      ENABLE_IDEMPOTENCE>
      BaseProblem;
  typedef DataSliceBase<VertexId, SizeT, Value, MAX_NUM_VERTEX_ASSOCIATES,
                        MAX_NUM_VALUE__ASSOCIATES>
      BaseDataSlice;

  typedef unsigned char MaskT;

  typedef cub::KeyValuePair<SizeT, Value> KVPair;

  /**
   * @brief Data slice structure which contains problem specific data.
   *
   * @tparam VertexId Type of signed integer to use as vertex IDs.
   * @tparam SizeT    Type of int / uint to use for array indexing.
   * @tparam Value    Type of float or double to use for attributes.
   */
  struct DataSlice : BaseDataSlice {
    // device storage arrays
    util::Array1D<SizeT, Value>
        labels_argmax;  // Updated community ID (argmax of neighgor weights
    util::Array1D<SizeT, Value> node_weights;  // Weight value per node
    util::Array1D<SizeT, Value>
        edge_weights;  // Weight value per edge (reused as updated weights in
                       // the computation
    util::Array1D<SizeT, Value>
        final_weights;  // final weights edge_weight*weight_reg
    util::Array1D<SizeT, Value> weight_reg;  // Regularizer of weights
    util::Array1D<SizeT, Value>
        degrees; /**< Used for keeping out-degree for each vertex */
    util::Array1D<SizeT, VertexId> froms;  // Edge source node ID
    util::Array1D<SizeT, VertexId> tos;    // Edge destination node ID
    util::Array1D<SizeT, SizeT> offsets;   // Point to graph_slice's row_offsets
    util::Array1D<SizeT, KVPair> argmax_kv;
    util::Array1D<SizeT, int>
        stable_flag; /**< Finish flag for label propagation, indicates that all
                        labels are stable.*/
    SizeT max_iter;

    DataSlice() : BaseDataSlice(), max_iter(0) {
      labels_argmax.SetName("labels_argmax");
      node_weights.SetName("node_weights");
      edge_weights.SetName("edge_weights");
      final_weights.SetName("final_weights");
      weight_reg.SetName("weight_reg");
      degrees.SetName("degrees");
      froms.SetName("froms");
      tos.SetName("tos");
      argmax_kv.SetName("argmax_kv");
      offsets.SetName("offsets");
      stable_flag.SetName("stable_flag");
    }

    ~DataSlice() {
      if (util::SetDevice(this->gpu_idx)) return;
      labels_argmax.Release();
      node_weights.Release();
      edge_weights.Release();
      final_weights.Release();
      weight_reg.Release();
      degrees.Release();
      froms.Release();
      tos.Release();
      argmax_kv.Release();
      offsets.Release();
      stable_flag.Release();
    }

    cudaError_t Init(int num_gpus, int gpu_idx, bool use_double_buffer,
                     ContextPtr context, Csr<VertexId, SizeT, Value> *graph,
                     GraphSlice<VertexId, SizeT, Value> *graph_slice,
                     SizeT *num_in_nodes, SizeT *num_out_nodes,
                     float queue_sizing = 2.0, float in_sizing = 1.0) {
      cudaError_t retval = cudaSuccess;
      if (retval =
              BaseDataSlice::Init(num_gpus, gpu_idx, use_double_buffer, graph,
                                  num_in_nodes, num_out_nodes, in_sizing))
        return retval;

      if (retval = this->frontier_queues[0].keys[0].Release()) return retval;
      if (this->frontier_queues[0].keys[0].GetPointer(util::DEVICE) == NULL)
        if (retval = this->frontier_queues[0].keys[0].Allocate(graph->edges,
                                                               util::DEVICE))
          return retval;
      // Create SoA on device
      if (retval = this->labels.Allocate(graph->nodes, util::DEVICE))
        return retval;
      if (retval = labels_argmax.Allocate(graph->nodes, util::DEVICE))
        return retval;
      if (retval = node_weights.Allocate(graph->nodes, util::DEVICE))
        return retval;
      if (retval = edge_weights.Allocate(graph->edges, util::DEVICE))
        return retval;
      if (retval = final_weights.Allocate(graph->edges, util::DEVICE))
        return retval;
      if (retval = weight_reg.Allocate(graph->nodes, util::DEVICE))
        return retval;
      if (retval = degrees.Allocate(graph->nodes, util::DEVICE)) return retval;
      if (retval = stable_flag.Allocate(1, util::HOST | util::DEVICE))
        return retval;

      edge_weights.SetPointer(graph->edge_values, graph->edges, util::HOST);
      if (retval = edge_weights.Move(util::HOST, util::DEVICE)) return retval;
      if (retval = froms.Allocate(graph->edges, util::DEVICE)) return retval;

      if (retval = argmax_kv.Allocate(graph->nodes, util::DEVICE))
        return retval;

      if (retval = tos.SetPointer(
              graph_slice->column_indices.GetPointer(util::DEVICE),
              graph_slice->edges, util::DEVICE))
        return retval;

      if (retval = offsets.SetPointer(
              graph_slice->row_offsets.GetPointer(util::DEVICE),
              graph_slice->nodes + 1, util::DEVICE))
        return retval;

      /*for (VertexId node = 0; node < graph->nodes; ++node) {
          SizeT start = graph->row_offsets[node];
          SizeT end = graph->row_offsets[node+1];
          for (SizeT edge = start; edge < end; ++edge) {
              froms[edge] = node;
          }
      }*/
      // if (retval = froms.Move(util::HOST, util::DEVICE)) return retval;

      util::Array1D<SizeT, VertexId>
          tmp_val;  // Updated community ID (argmax of neighgor weights
      if (retval = tmp_val.Allocate(graph->edges, util::DEVICE)) return retval;

      util::MemsetIdxKernel<<<128, 128>>>(tmp_val.GetPointer(util::DEVICE),
                                          graph->edges);

      mgpu::IntervalExpand(graph->edges,
                           graph_slice->row_offsets.GetPointer(util::DEVICE),
                           tmp_val.GetPointer(util::DEVICE), graph->nodes,
                           froms.GetPointer(util::DEVICE), context[0]);

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      VertexId *key = froms.GetPointer(util::DEVICE);
      Value *value = edge_weights.GetPointer(util::DEVICE);
      Value *uniq_out = labels_argmax.GetPointer(util::DEVICE);
      Value *output = node_weights.GetPointer(util::DEVICE);
      VertexId *num_out = stable_flag.GetPointer(util::DEVICE);
      cub::DeviceReduce::ReduceByKey(
          d_temp_storage, temp_storage_bytes, key, (VertexId *)uniq_out, value,
          output, (VertexId *)num_out, cub::Sum(), graph->edges);
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      cub::DeviceReduce::ReduceByKey(
          d_temp_storage, temp_storage_bytes, key, (VertexId *)uniq_out, value,
          output, (VertexId *)num_out, cub::Sum(), graph->edges);

      mgpu::IntervalExpand(graph->edges,
                           graph_slice->row_offsets.GetPointer(util::DEVICE),
                           node_weights.GetPointer(util::DEVICE), graph->nodes,
                           tmp_val.GetPointer(util::DEVICE), context[0]);

      util::MemsetDivVectorKernel<<<128, 128>>>(
          edge_weights.GetPointer(util::DEVICE),
          (Value *)tmp_val.GetPointer(util::DEVICE), graph->edges);
      cub::DeviceReduce::ReduceByKey(
          d_temp_storage, temp_storage_bytes, key, (VertexId *)uniq_out, value,
          output, (VertexId *)num_out, cub::Max(), graph->edges);

      tmp_val.Release();

      return retval;
    }

    /**
     * @brief Performs reset work needed for DataSliceBase. Must be called prior
     * to each search
     *
     * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
     * @param[in] graph_slice        Pointer to the corresponding graph slice
     * @param[in] queue_sizing       Sizing scaling factor for work queue
     * allocation. 1.0 by default. Reserved for future use.
     * @param[in] _USE_DOUBLE_BUFFER Whether to use double buffer
     * @param[in] queue_sizing1      Scaling factor for frontier_queue1
     *
     * \return cudaError_t object which indicates the success of all CUDA
     * function calls.
     */
    cudaError_t Reset(FrontierType frontier_type,
                      GraphSlice<VertexId, SizeT, Value> *graph_slice,
                      SizeT max_iter, double queue_sizing = 2.0,
                      double queue_sizing1 = -1.0,
                      bool use_double_buffer = false,
                      bool skip_scanned_edges = false) {
      cudaError_t retval = cudaSuccess;
      if (retval = BaseDataSlice::Reset(frontier_type, graph_slice,
                                        queue_sizing, use_double_buffer,
                                        queue_sizing1, skip_scanned_edges))
        return retval;

      if (labels_argmax.GetPointer(util::DEVICE) == NULL)
        if (retval = labels_argmax.Allocate(graph_slice->nodes, util::DEVICE))
          return retval;
      // for reset edge_weights array will be used as a |V| length array.
      if (edge_weights.GetPointer(util::DEVICE) == NULL)
        if (retval = edge_weights.Allocate(graph_slice->nodes, util::DEVICE))
          return retval;
      if (final_weights.GetPointer(util::DEVICE) == NULL)
        if (retval = final_weights.Allocate(graph_slice->edges, util::DEVICE))
          return retval;
      if (weight_reg.GetPointer(util::DEVICE) == NULL)
        if (retval = weight_reg.Allocate(graph_slice->nodes, util::DEVICE))
          return retval;

      if (degrees.GetPointer(util::DEVICE) == NULL)
        if (retval = degrees.Allocate(graph_slice->nodes, util::DEVICE))
          return retval;

      util::MemsetIdxKernel<<<128, 128>>>(this->labels.GetPointer(util::DEVICE),
                                          graph_slice->nodes);
      util::MemsetKernel<<<128, 128>>>(edge_weights.GetPointer(util::DEVICE),
                                       0.0f, graph_slice->nodes);
      util::MemsetKernel<<<128, 128>>>(weight_reg.GetPointer(util::DEVICE),
                                       1.0f, graph_slice->nodes);

      util::MemsetMadVectorKernel<<<128, 128>>>(
          degrees.GetPointer(util::DEVICE),
          (Value *)graph_slice->row_offsets.GetPointer(util::DEVICE),
          (Value *)graph_slice->row_offsets.GetPointer(util::DEVICE) + 1, -1.0f,
          graph_slice->nodes);
      Value norm = 1.0f / (2 * graph_slice->edges);
      util::MemsetScaleKernel<<<128, 128>>>(degrees.GetPointer(util::DEVICE),
                                            norm, graph_slice->nodes);

      stable_flag[0] = 0;
      if (retval = stable_flag.Move(util::HOST, util::DEVICE)) return retval;

      this->max_iter = max_iter;

      return retval;
    }
  };

  // int       num_gpus;
  // SizeT     nodes;
  // SizeT     edges;
  SizeT num_components;

  // data slices (one for each GPU)
  // DataSlice **data_slices;
  util::Array1D<SizeT, DataSlice> *data_slices;

  // putting structure on device while keeping the SoA structure
  // DataSlice **d_data_slices;

  // device index for each data slice
  // int       *gpu_idx;

  /**
   * @brief Default constructor
   */
  LpProblem(bool use_double_buffer)
      : BaseProblem(use_double_buffer,
                    false,   // enable_backward
                    false,   // keep_order
                    false),  // keep_node_num
        num_components(0),
        data_slices(NULL) {}

  /**
   * @brief Constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  // LpProblem(bool  stream_from_host,  // only meaningful for single-GPU
  //              const Csr<VertexId, Value, SizeT> &graph,
  //              int   num_gpus) :
  //    num_gpus(num_gpus)
  //{
  //    Init(stream_from_host, graph, num_gpus);
  //}

  /**
   * @brief Default destructor
   */
  ~LpProblem() {
    if (data_slices == NULL) return;
    for (int i = 0; i < this->num_gpus; ++i) {
      util::SetDevice(this->gpu_idx[i]);
      data_slices[i].Release();
    }
    delete[] data_slices;
    data_slices = NULL;
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
  cudaError_t Extract(VertexId *h_labels) {
    cudaError_t retval = cudaSuccess;
    int *marker = new int[this->nodes];
    memset(marker, 0, sizeof(int) * this->nodes);

    if (this->num_gpus == 1) {
      int gpu = 0;
      if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
      data_slices[gpu]->labels.SetPointer(h_labels);
      if (retval = data_slices[gpu]->labels.Move(util::DEVICE, util::HOST))
        return retval;
      num_components = 0;
      for (int node = 0; node < this->nodes; ++node) {
        if (marker[h_labels[node]] == 0) {
          num_components++;
          marker[h_labels[node]] = 1;
        }
      }

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
  cudaError_t Init(bool stream_from_host,  // Only meaningful for single-GPU
                   Csr<VertexId, SizeT, Value> *graph, ContextPtr context,
                   Csr<VertexId, SizeT, Value> *inversegraph = NULL,
                   int num_gpus = 1, int *gpu_idx = NULL,
                   std::string partition_method = "random",
                   cudaStream_t *streams = NULL, float queue_sizing = 2.0f,
                   float in_sizing = 1.0f, float partition_factor = -1.0f,
                   int partition_seed = -1) {
    cudaError_t retval = cudaSuccess;
    if (retval = BaseProblem::Init(
            stream_from_host, graph, inversegraph, num_gpus, gpu_idx,
            partition_method, queue_sizing, partition_factor, partition_seed))
      return retval;

    // no data in DataSlice needs to be copied from host

    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[]");
      if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
      if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
        return retval;
      DataSlice *data_slice = data_slices[gpu].GetPointer(util::HOST);
      GraphSlice<VertexId, SizeT, Value> *graph_slice = this->graph_slices[gpu];
      data_slice->streams.SetPointer(streams + gpu * num_gpus * 2,
                                     num_gpus * 2);

      if (retval = data_slice->Init(
              this->num_gpus, this->gpu_idx[gpu], this->use_double_buffer,
              context, &(this->sub_graphs[gpu]), graph_slice,
              this->num_gpus > 1
                  ? graph_slice->in_counter.GetPointer(util::HOST)
                  : NULL,
              this->num_gpus > 1
                  ? graph_slice->out_counter.GetPointer(util::HOST)
                  : NULL,
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
      SizeT max_iter, double queue_sizing, double queue_sizing1 = -1.0) {
    // size scaling factor for work queue allocation (e.g., 1.0 creates
    // n-element and m-element vertex and edge frontiers, respectively).
    // 0.0 is unspecified.

    cudaError_t retval = cudaSuccess;

    if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
      if (retval =
              data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu],
                                      max_iter, queue_sizing, queue_sizing1))
        return retval;
      if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE))
        return retval;
    }

    return retval;
  }

  /** @} */
};

}  // namespace lp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
