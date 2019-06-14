// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_problem.cuh
 *
 * @brief GPU Storage management Structure for CC Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/track_utils.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

namespace gunrock {
namespace app {
namespace cc {

/**
 * @brief Connected Component Problem structure stores device-side vectors for
 * doing connected component computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id
 * (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array
 * indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC
 * value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to
 * use double buffer
 */
template <typename VertexId, typename SizeT, typename Value>
// bool        _USE_DOUBLE_BUFFER>
struct CCProblem : ProblemBase<VertexId, SizeT, Value,
                               false,  // _MARK_PREDECESSORS
                               false>  // _ENABLE_IDEMPOTENCE,
                                       //_USE_DOUBLE_BUFFER,
                                       // false, // _EnABLE_BACKWARD
// false, // _KEEP_ORDER
// true>  // _KEEP_NODE_NUM
{
  static const bool MARK_PREDECESSORS = false;
  static const bool ENABLE_IDEMPOTENCE = false;
  static const int MAX_NUM_VERTEX_ASSOCIATES = 1;
  static const int MAX_NUM_VALUE__ASSOCIATES = 0;
  typedef ProblemBase<VertexId, SizeT, Value, MARK_PREDECESSORS,
                      ENABLE_IDEMPOTENCE>
      BaseProblem;
  typedef DataSliceBase<VertexId, SizeT, Value, MAX_NUM_VERTEX_ASSOCIATES,
                        MAX_NUM_VALUE__ASSOCIATES>
      BaseDataSlice;
  typedef unsigned char MaskT;

  // Helper structures

  /**
   * @brief Data slice structure which contains CC problem specific data.
   */
  struct DataSlice : BaseDataSlice {
    // device storage arrays
    util::Array1D<SizeT, VertexId> component_ids; /**< Used for component id */
    util::Array1D<SizeT, VertexId> old_c_ids;
    // util::Array1D<SizeT, SizeT   > CID_markers;
    util::Array1D<SizeT, signed char>
        masks; /**< Size equals to node number, show if a node is the root */
    util::Array1D<SizeT, bool>
        marks; /**< Size equals to edge number, show if two vertices belong to
                  the same component */
    util::Array1D<SizeT, VertexId>
        froms; /**< Size equals to edge number, from vertex of one edge */
    util::Array1D<SizeT, VertexId>
        tos; /**< Size equals to edge number, to vertex of one edge */
    util::Array1D<SizeT, int>
        vertex_flag; /**< Finish flag for per-vertex kernels in CC algorithm */
    util::Array1D<SizeT, int>
        edge_flag; /**< Finish flag for per-edge kernels in CC algorithm */
    util::Array1D<SizeT, VertexId> local_vertices;
    util::Array1D<SizeT, VertexId *> vertex_associate_ins;
    int turn;
    // DataSlice *d_pointer;
    bool has_change, previous_change;
    bool scanned_queue_computed;
    VertexId *temp_vertex_out;
    VertexId *temp_comp_out;
    // util::CtaWorkProgressLifetime *work_progress;

    /*
     * @brief Default constructor
     */
    DataSlice() {
      component_ids.SetName("component_ids");
      old_c_ids.SetName("old_c_ids");
      // CID_markers  .SetName("CID_markers"  );
      masks.SetName("masks");
      marks.SetName("marks");
      froms.SetName("froms");
      tos.SetName("tos");
      vertex_flag.SetName("vertex_flag");
      edge_flag.SetName("edge_flag");
      // local_vertices.SetName("local_vertices");
      vertex_associate_ins.SetName("vertex_associate_ins");
      turn = 0;
      // d_pointer     = NULL;
      // work_progress = NULL;
      has_change = true;
      previous_change = true;
      scanned_queue_computed = false;
      temp_vertex_out = NULL;
      temp_comp_out = NULL;
    }

    /*
     * @brief Default destructor
     */
    ~DataSlice() {
      if (util::SetDevice(this->gpu_idx)) return;
      component_ids.Release();
      old_c_ids.Release();
      // CID_markers  .Release();
      masks.Release();
      marks.Release();
      froms.Release();
      tos.Release();
      vertex_flag.Release();
      edge_flag.Release();
      // local_vertices.Release();
      vertex_associate_ins.Release();
      // d_pointer     = NULL;
      // work_progress = NULL;
    }

    /**
     * @brief initialization function.
     *
     * @param[in] num_gpus Number of the GPUs used.
     * @param[in] gpu_idx GPU index used for testing.
     * @param[in] use_double_buffer Whether to use double buffer.
     * @param[in] graph Pointer to the graph we process on.
     * @param[in] graph_slice Pointer to GraphSlice object.
     * @param[in] num_in_nodes
     * @param[in] num_out_nodes
     * @param[in] original_vertex
     * @param[in] queue_sizing Maximum queue sizing factor.
     * @param[in] in_sizing
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(int num_gpus, int gpu_idx, bool use_double_buffer,
                     Csr<VertexId, SizeT, Value> *graph,
                     GraphSlice<VertexId, SizeT, Value> *graph_slice,
                     SizeT *num_in_nodes, SizeT *num_out_nodes,
                     VertexId *original_vertex, float queue_sizing = 2.0,
                     float in_sizing = 1.0) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = graph->nodes;
      SizeT edges = graph->edges;

      if (num_gpus > 1)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
          num_in_nodes[gpu] = nodes;
          num_out_nodes[gpu] = gpu == 1 ? nodes : 0;
        }

      if (retval = BaseDataSlice::Init(num_gpus, gpu_idx, use_double_buffer,
                                       // num_vertex_associate,
                                       // num_value__associate,
                                       graph, num_in_nodes, num_out_nodes,
                                       in_sizing))
        return retval;

      for (int peer_ = 2; peer_ < num_gpus; peer_++) {
        this->keys_out[peer_].SetPointer(
            this->keys_out[1].GetPointer(util::DEVICE),
            this->keys_out[1].GetSize(), util::DEVICE);
        this->keys_outs[peer_] = this->keys_out[1].GetPointer(util::DEVICE);
        this->vertex_associate_out[peer_].SetPointer(
            this->vertex_associate_out[1].GetPointer(util::DEVICE),
            this->vertex_associate_out[1].GetSize(), util::DEVICE);
        this->vertex_associate_outs[peer_] =
            this->vertex_associate_out[1].GetPointer(util::DEVICE);
      }
      if (retval = this->vertex_associate_outs.Move(util::HOST, util::DEVICE))
        return retval;

      // printf("@ gpu %d: nodes = %d, edges = %d\n", gpu_idx, nodes, edges);
      // Create a single data slice for the currently-set gpu
      if (retval = froms.Allocate(edges, util::HOST | util::DEVICE))
        return retval;
      // if (retval = tos   .Allocate(edges, util::DEVICE)) return retval;
      if (retval = tos.SetPointer(
              graph_slice->column_indices.GetPointer(util::DEVICE),
              graph_slice->edges, util::DEVICE))
        return retval;
      // Construct coo from/to edge list from row_offsets and column_indices
      for (VertexId node = 0; node < graph->nodes; node++) {
        SizeT start_edge = graph->row_offsets[node];
        SizeT end_edge = graph->row_offsets[node + 1];
        if (TO_TRACK)
          if (util::to_track(node))
            printf("node %lld @ gpu %d : %lld -> %lld\n", (long long)node,
                   gpu_idx, (long long)start_edge, (long long)end_edge);
        for (SizeT edge = start_edge; edge < end_edge; ++edge) {
          froms[edge] = node;
          // tos  [edge] = graph->column_indices[edge];
          if (TO_TRACK)
            if (util::to_track(node) || util::to_track(tos[edge]))
              printf("edge %lld @ gpu %d : %lld -> %lld\n", (long long)edge,
                     gpu_idx, (long long)froms[edge], (long long)tos[edge]);
        }
      }
      if (retval = froms.Move(util::HOST, util::DEVICE)) return retval;
      // if (retval = tos  .Move(util::HOST, util::DEVICE)) return retval;
      if (retval = froms.Release(util::HOST)) return retval;
      // if (retval = tos  .Release(util::HOST)) return retval;

      // Create SoA on device
      if (retval = component_ids.Allocate(nodes, util::DEVICE)) return retval;
      if (num_gpus > 1)
        if (retval = old_c_ids.Allocate(nodes, util::DEVICE)) return retval;
      // if (retval = CID_markers  .Allocate(nodes+1, util::DEVICE)) return
      // retval;
      if (retval = masks.Allocate(nodes, util::DEVICE)) return retval;
      if (retval = marks.Allocate(edges, util::DEVICE)) return retval;
      if (retval = vertex_flag.Allocate(1, util::HOST | util::DEVICE))
        return retval;
      if (retval = edge_flag.Allocate(1, util::HOST | util::DEVICE))
        return retval;
      // if (retval = this -> frontier_queues[0].keys  [0].Allocate(nodes + 2,
      // util::DEVICE)) return retval; if (retval = this ->
      // scanned_edges[0].Allocate(nodes + 2, util::DEVICE)) return retval;
      if (retval = vertex_associate_ins.Allocate(num_gpus,
                                                 util::HOST | util::DEVICE))
        return retval;
      scanned_queue_computed = false;
      // if (retval = this->frontier_queues[0].keys  [0].Allocate(edges+2,
      // util::DEVICE)) return retval; if (retval =
      // this->frontier_queues[0].keys [1].Allocate(edges+2, util::DEVICE))
      // return retval; if (retval =
      // this->frontier_queues[0].values[0].Allocate(nodes+2, util::DEVICE))
      // return retval; if (retval =
      // this->frontier_queues[0].values[1].Allocate(nodes+2, util::DEVICE))
      // return retval;
      /*if (num_gpus > 1) {
          this->frontier_queues[num_gpus].keys  [0].SetPointer(
              this->frontier_queues[0].keys  [0].GetPointer(util::DEVICE),
      edges+2, util::DEVICE);
          //this->frontier_queues[num_gpus].keys
      [1].SetPointer(this->frontier_queues[0].keys [1].GetPointer(util::DEVICE),
      edges+2, util::DEVICE);
          this->frontier_queues[num_gpus].values[0].SetPointer(
              this->frontier_queues[0].values[0].GetPointer(util::DEVICE),
      nodes+2, util::DEVICE);
          //this->frontier_queues[num_gpus].values[1].SetPointer(this->frontier_queues[0].values[1].GetPointer(util::DEVICE),
      nodes+2, util::DEVICE);
      }*/

      if (false)  //(num_gpus > 1)
      {
        SizeT num_local_vertices = 0;
        for (VertexId v = 0; v < nodes; v++)
          if (graph_slice->partition_table[v] == 0) num_local_vertices++;
        if (retval = local_vertices.Allocate(num_local_vertices,
                                             util::HOST | util::DEVICE))
          return retval;
        num_local_vertices = 0;
        for (VertexId v = 0; v < nodes; v++)
          if (graph_slice->partition_table[v] == 0) {
            local_vertices[num_local_vertices] = v;
            num_local_vertices++;
          }
        if (retval = local_vertices.Move(util::HOST, util::DEVICE))
          return retval;
      }
      return retval;
    }

    cudaError_t Reset(GraphSlice<VertexId, SizeT, Value> *graph_slice) {
      SizeT nodes = graph_slice->nodes;
      SizeT edges = graph_slice->edges;
      cudaError_t retval = cudaSuccess;
      for (int gpu = 0; gpu < this->num_gpus * 2; gpu++)
        this->wait_marker[gpu] = 0;
      for (int i = 0; i < 4; i++)
        for (int gpu = 0; gpu < this->num_gpus * 2; gpu++)
          for (int stage = 0; stage < this->num_stages; stage++)
            this->events_set[i][gpu][stage] = false;
      for (int gpu = 0; gpu < this->num_gpus; gpu++)
        for (int i = 0; i < 2; i++) this->in_length[i][gpu] = 0;
      for (int peer = 0; peer < this->num_gpus; peer++)
        this->out_length[peer] = 1;
      turn = 0;
      has_change = true;
      previous_change = true;

      // Set device
      if (retval = util::SetDevice(this->gpu_idx)) return retval;

      // if (retval = data_slices[gpu]->Reset(frontier_type,
      // this->graph_slices[gpu], queue_sizing, _USE_DOUBLE_BUFFER)) return
      // retval; if (retval = this -> frontier_queues[0].keys
      // [0].EnsureSize(edges+2)) return retval; if (retval = this ->
      // frontier_queues[0].keys  [1].EnsureSize(edges+2)) return retval; if
      // (retval = this -> frontier_queues[0].values[0].EnsureSize(nodes+2))
      // return retval; if (retval = this ->
      // frontier_queues[0].values[1].EnsureSize(nodes+2)) return retval; if
      // (retval = this -> frontier_queues[0].keys[0].EnsureSize(nodes + 2))
      // return retval;

      // Allocate output component_ids if necessary
      util::MemsetIdxKernel<<<128, 128>>>(
          component_ids.GetPointer(util::DEVICE), nodes);

      // Allocate marks if necessary
      util::MemsetKernel<<<128, 128>>>(marks.GetPointer(util::DEVICE), false,
                                       edges);

      // Allocate masks if necessary
      util::MemsetKernel<<<128, 128>>>(masks.GetPointer(util::DEVICE),
                                       (signed char)0, nodes);

      // Allocate vertex_flag if necessary
      vertex_flag[0] = 1;
      if (retval = vertex_flag.Move(util::HOST, util::DEVICE)) return retval;

      // Allocate edge_flag if necessary
      edge_flag[0] = 1;
      if (retval = edge_flag.Move(util::HOST, util::DEVICE)) return retval;

      // Initialize edge frontier_queue
      // util::MemsetIdxKernel<<<128, 128>>>(this -> frontier_queues[0].keys
      // [0].GetPointer(util::DEVICE), edges);

      // Initialize vertex frontier queue
      // util::MemsetIdxKernel<<<128, 128>>>(this ->
      // frontier_queues[0].values[0].GetPointer(util::DEVICE), nodes);
      // util::MemsetIdxKernel<<<240, 512>>>(this ->
      // frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);

      if (this->num_gpus > 1)
        util::MemsetIdxKernel<<<240, 512>>>(old_c_ids.GetPointer(util::DEVICE),
                                            nodes);
      util::MemsetKernel<<<240, 512>>>(marks.GetPointer(util::DEVICE), false,
                                       edges);
      return retval;
    }
  };

  // Members
  SizeT num_components;

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief CCProblem default constructor
   */
  CCProblem()
      : BaseProblem(false,  // use_double_buffer
                    false,  // enable_backward
                    false,  // keep_order
                    true,   // keep_node_num
                    false,  // skip_makeout_selection
                    true)   // unified_receive
  {
    num_components = 0;
    data_slices = NULL;
  }

  /**
   * @brief CCProblem default destructor
   */
  ~CCProblem() {
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
   * @brief Copy result component ids computed on the GPU back to a host-side
   *vector.
   *
   * @param[out] h_component_ids host-side vector to store computed component
   *ids.
   *
   *\return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Extract(VertexId *h_component_ids) {
    cudaError_t retval = cudaSuccess;
    int *marker = new int[this->nodes];
    memset(marker, 0, sizeof(int) * this->nodes);

    if (this->num_gpus == 1) {
      if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
      data_slices[0]->component_ids.SetPointer(h_component_ids);
      if (retval = data_slices[0]->component_ids.Move(util::DEVICE, util::HOST))
        return retval;
      num_components = 0;
      for (int node = 0; node < this->nodes; node++)
        if (marker[h_component_ids[node]] == 0) {
          num_components++;
          // printf("%d\t ",node);
          marker[h_component_ids[node]] = 1;
        }

    } else {
      VertexId **th_component_ids = new VertexId *[this->num_gpus];
      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval =
                data_slices[gpu]->component_ids.Move(util::DEVICE, util::HOST))
          return retval;
        th_component_ids[gpu] =
            data_slices[gpu]->component_ids.GetPointer(util::HOST);
      }

      num_components = 0;
      for (VertexId node = 0; node < this->nodes; node++) {
        h_component_ids[node] =
            th_component_ids[this->partition_tables[0][node]]
                            [this->convertion_tables[0][node]];
        if (marker[h_component_ids[node]] == 0) {
          num_components++;
          // printf("%d ",node);
          marker[h_component_ids[node]] = 1;
        }
      }

      VertexId **temp_cids = new VertexId *[this->num_gpus];
      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        temp_cids[gpu] = new VertexId[this->nodes];
        for (SizeT i = 0; i < this->nodes; i++)
          temp_cids[gpu][i] = util::InvalidValue<VertexId>();
        for (SizeT v_ = 0; v_ < this->graph_slices[gpu]->nodes; v_++) {
          temp_cids[gpu][this->graph_slices[gpu]->original_vertex[v_]] =
              th_component_ids[gpu][v_];
        }
      }

      SizeT num_diff = 0;
      for (VertexId node = 0; node < this->nodes; node++) {
        VertexId host_cid = h_component_ids[node];
        bool difference_found = false;
        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
          if (temp_cids[gpu][node] != util::InvalidValue<VertexId>() &&
              temp_cids[gpu][node] != host_cid) {
            difference_found = true;
            break;
          }
        }
        if (difference_found) {
          if (num_diff < 10) {
            printf("Node %d : ", node);
            for (int gpu = 0; gpu < this->num_gpus; gpu++) {
              if (gpu != 0) printf(", ");
              printf("%d -> %d", temp_cids[gpu][node],
                     temp_cids[gpu][temp_cids[gpu][node]]);
            }
            printf(" host = %d\n", this->partition_tables[0][node]);
          }
          num_diff++;
        }
      }
      if (num_diff != 0)
        printf("Number of differences = %lld\n", (long long)num_diff);
    }  // end if

    return retval;
  }

  /**
   * @brief Compute histogram for component ids.
   *
   * @param[in] h_component_ids host-side vector stores  component ids.
   * @param[out] h_roots host-side vector to store root node id for each
   * component.
   * @param[out] h_histograms host-side vector to store histograms.
   *
   */
  void ComputeCCHistogram(VertexId *h_component_ids, VertexId *h_roots,
                          SizeT *h_histograms) {
    // Get roots for each component and the total number of component
    // VertexId *min_nodes = new VertexId[this->nodes];
    VertexId *counter = new VertexId[this->nodes];
    for (SizeT i = 0; i < this->nodes; i++) {
      // min_nodes[i] = this->nodes;
      counter[i] = 0;
    }
    // for (int i = 0; i < this->nodes; i++)
    //    if (min_nodes[h_component_ids[i]] > i) min_nodes[h_component_ids[i]] =
    //    i;
    num_components = 0;
    for (SizeT i = 0; i < this->nodes; i++) {
      if (counter[h_component_ids[i]] == 0) {
        // h_histograms[num_components] = counter[h_component_ids[i]];
        h_roots[num_components] = i;
        ++num_components;
        // printf("%d\t", i);
      }
      counter[h_component_ids[i]]++;
    }
    for (SizeT i = 0; i < num_components; i++)
      h_histograms[i] = counter[h_component_ids[h_roots[i]]];
    /*for (int i = 0; i < this->nodes; ++i)
    {
        if (h_component_ids[i] == i)
        {
           h_roots[num_components] = i;
           h_histograms[num_components] = counter[h_component_ids[i]];
           ++num_components;
        }
    }*/

    /*for (int i = 0; i < this->nodes; ++i)
    {
        for (int j = 0; j < num_components; ++j)
        {
            if (h_component_ids[i] == h_roots[j])
            {
                ++h_histograms[j];
                break;
            }
        }
    }*/
    // delete[] min_nodes; min_nodes = NULL;
    delete[] counter;
    counter = NULL;
  }

  /**
   * @brief initialization function.
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Pointer to the CSR graph object we process on. @see Csr
   * @param[in] inversegraph Pointer to the inversed CSR graph object we process
   * on.
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
  cudaError_t Init(bool stream_from_host,  // Only meaningful for single-GPU
                   Csr<VertexId, SizeT, Value> *graph,
                   Csr<VertexId, SizeT, Value> *inversegraph = NULL,
                   int num_gpus = 1, int *gpu_idx = NULL,
                   std::string partition_method = "random",
                   cudaStream_t *streams = NULL, float queue_sizing = 2.0f,
                   float in_sizing = 1.0f, float partition_factor = -1.0f,
                   int partition_seed = -1) {
    BaseProblem::Init(stream_from_host, graph, inversegraph, num_gpus, gpu_idx,
                      partition_method, queue_sizing, partition_factor,
                      partition_seed);

    // No data in DataSlice needs to be copied from host

    /**
     * Allocate output labels/preds
     */
    cudaError_t retval = cudaSuccess;
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    do {
      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        data_slices[gpu].SetName("data_slices[]");
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
          return retval;
        DataSlice *data_slice_ = data_slices[gpu].GetPointer(util::HOST);
        // data_slice_->d_pointer = data_slices[gpu].GetPointer(util::DEVICE);
        data_slice_->streams.SetPointer(&streams[gpu * num_gpus * 2],
                                        num_gpus * 2);
        if (retval = data_slice_->Init(
                this->num_gpus, this->gpu_idx[gpu], this->use_double_buffer,
                // this->num_gpus>1? 1:0,
                // 0,
                &(this->sub_graphs[gpu]), this->graph_slices[gpu],
                this->num_gpus > 1
                    ? this->graph_slices[gpu]->in_counter.GetPointer(util::HOST)
                    : NULL,
                this->num_gpus > 1
                    ? this->graph_slices[gpu]->out_counter.GetPointer(
                          util::HOST)
                    : NULL,
                this->num_gpus > 1
                    ? this->graph_slices[gpu]->original_vertex.GetPointer(
                          util::HOST)
                    : NULL,
                queue_sizing, in_sizing))
          return retval;
      }
    } while (0);

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   *
   * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
   * @param[in] queue_sizing Size scaling factor for work queue allocation
   * (e.g., 1.0 creates n-element and m-element vertex and edge frontiers,
   * respectively).
   *
   *  \return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Reset(FrontierType frontier_type,  // The frontier type (i.e.,
                                                 // edge/vertex/mixed)
                    double queue_sizing) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (retval = data_slices[gpu]->Reset(this->graph_slices[gpu]))
        return retval;
      if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE))
        return retval;
    }

    return retval;
  }

  /** @} */
};

}  // namespace cc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
