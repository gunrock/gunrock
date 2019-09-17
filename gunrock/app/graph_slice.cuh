// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * graph_slice.cuh
 *
 * @brief Structure for graph slice. Only for temp dummping of code, will be
 * refactored latter
 */
#pragma once

/**
 * @brief Graph slice structure which contains common graph structural data.
 *
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g.,
 * uint32)
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value    Type to use as vertex / edge associated values
 */
template <typename VertexId, typename SizeT, typename Value>
struct GraphSlice {
  int num_gpus;         // Number of GPUs
  int index;            // Slice index
  VertexId nodes;       // Number of nodes in slice
  SizeT edges;          // Number of edges in slice
  SizeT inverse_edges;  // Number of inverse_edges in slice

  Csr<VertexId, SizeT, Value>* graph;       // Pointer to CSR format subgraph
  util::Array1D<SizeT, SizeT> row_offsets;  // CSR format row offset
  util::Array1D<SizeT, VertexId> column_indices;  // CSR format column indices
  util::Array1D<SizeT, SizeT> out_degrees;
  util::Array1D<SizeT, SizeT> column_offsets;  // CSR format column offset
  util::Array1D<SizeT, VertexId> row_indices;  // CSR format row indices
  util::Array1D<SizeT, SizeT> in_degrees;
  util::Array1D<SizeT, int>
      partition_table;  // Partition number for vertices, local is always 0
  util::Array1D<SizeT, VertexId>
      convertion_table;  // IDs of vertices in their hosting partition
  util::Array1D<SizeT, VertexId> original_vertex;  // Original IDs of vertices
  util::Array1D<SizeT, SizeT>
      in_counter;  // Incoming vertices counter from peers
  util::Array1D<SizeT, SizeT> out_offset;   // Outgoing vertices offsets
  util::Array1D<SizeT, SizeT> out_counter;  // Outgoing vertices counter
  util::Array1D<SizeT, SizeT>
      backward_offset;  // Backward offsets for partition and conversion tables
  util::Array1D<SizeT, int>
      backward_partition;  // Remote peers having the same vertices
  util::Array1D<SizeT, VertexId>
      backward_convertion;  // IDs of vertices in remote peers

  /**
   * @brief GraphSlice Constructor
   *
   * @param[in] index GPU index.
   */
  GraphSlice(int index)
      : index(index), graph(NULL), num_gpus(0), nodes(0), edges(0) {
    row_offsets.SetName("row_offsets");
    column_indices.SetName("column_indices");
    out_degrees.SetName("out_degrees");
    column_offsets.SetName("column_offsets");
    row_indices.SetName("row_indices");
    in_degrees.SetName("in_degrees");
    partition_table.SetName("partition_table");
    convertion_table.SetName("convertion_table");
    original_vertex.SetName("original_vertex");
    in_counter.SetName("in_counter");
    out_offset.SetName("out_offset");
    out_counter.SetName("out_counter");
    backward_offset.SetName("backward_offset");
    backward_partition.SetName("backward_partition");
    backward_convertion.SetName("backward_convertion");
  }  // end GraphSlice(int index)

  /**
   * @brief GraphSlice Destructor to free all device memories.
   */
  virtual ~GraphSlice() { Release(); }

  cudaError_t Release() {
    cudaError_t retval = cudaSuccess;

    // Set device (use slice index)
    if (retval = util::SetDevice(index)) return retval;

    // Release allocated host / device memory
    if (retval = row_offsets.Release()) return retval;
    if (retval = column_indices.Release()) return retval;
    if (retval = out_degrees.Release()) return retval;
    if (retval = column_offsets.Release()) return retval;
    if (retval = row_indices.Release()) return retval;
    if (retval = in_degrees.Release()) return retval;
    if (retval = partition_table.Release()) return retval;
    if (retval = convertion_table.Release()) return retval;
    if (retval = original_vertex.Release()) return retval;
    if (retval = in_counter.Release()) return retval;
    if (retval = out_offset.Release()) return retval;
    if (retval = out_counter.Release()) return retval;
    if (retval = backward_offset.Release()) return retval;
    if (retval = backward_partition.Release()) return retval;
    if (retval = backward_convertion.Release()) return retval;

    return retval;
  }  // end ~GraphSlice()

  /**
   * @brief Initialize graph slice
   *
   * @param[in] stream_from_host    Whether to stream data from host
   * @param[in] num_gpus            Number of GPUs
   * @param[in] graph               Pointer to the sub graph
   * @param[in] inverstgraph        Pointer to the invert graph
   * @param[in] partition_table     The partition table
   * @param[in] convertion_table    The conversion table
   * @param[in] original_vertex     The original vertex table
   * @param[in] in_counter          In_counters
   * @param[in] out_offset          Out_offsets
   * @param[in] out_counter         Out_counters
   * @param[in] backward_offsets    Backward_offsets
   * @param[in] backward_partition  The backward partition table
   * @param[in] backward_convertion The backward conversion table
   * \return cudaError_t            Object indicating the success of all CUDA
   * function calls
   */
  cudaError_t Init(
      bool stream_from_host, int num_gpus, Csr<VertexId, SizeT, Value>* graph,
      Csr<VertexId, SizeT, Value>* inverstgraph, int* partition_table,
      VertexId* convertion_table, VertexId* original_vertex, SizeT* in_counter,
      SizeT* out_offset, SizeT* out_counter, SizeT* backward_offsets = NULL,
      int* backward_partition = NULL, VertexId* backward_convertion = NULL) {
    cudaError_t retval = cudaSuccess;

    // Set local variables / array pointers
    this->num_gpus = num_gpus;
    this->graph = graph;
    this->nodes = graph->nodes;
    this->edges = graph->edges;
    if (inverstgraph != NULL)
      this->inverse_edges = inverstgraph->edges;
    else
      this->inverse_edges = 0;
    if (partition_table != NULL)
      this->partition_table.SetPointer(partition_table, nodes);
    if (convertion_table != NULL)
      this->convertion_table.SetPointer(convertion_table, nodes);
    if (original_vertex != NULL)
      this->original_vertex.SetPointer(original_vertex, nodes);
    if (in_counter != NULL)
      this->in_counter.SetPointer(in_counter, num_gpus + 1);
    if (out_offset != NULL)
      this->out_offset.SetPointer(out_offset, num_gpus + 1);
    if (out_counter != NULL)
      this->out_counter.SetPointer(out_counter, num_gpus + 1);
    this->row_offsets.SetPointer(graph->row_offsets, nodes + 1);
    this->column_indices.SetPointer(graph->column_indices, edges);
    if (inverstgraph != NULL) {
      this->column_offsets.SetPointer(inverstgraph->row_offsets, nodes + 1);
      this->row_indices.SetPointer(inverstgraph->column_indices,
                                   inverstgraph->edges);
    }

    // Set device using slice index
    if (retval = util::SetDevice(index)) return retval;

    // Allocate and initialize row_offsets
    if (retval = this->row_offsets.Allocate(nodes + 1, util::DEVICE))
      return retval;
    if (retval = this->row_offsets.Move(util::HOST, util::DEVICE))
      return retval;

    // Allocate and initialize column_indices
    if (retval = this->column_indices.Allocate(edges, util::DEVICE))
      return retval;
    if (retval = this->column_indices.Move(util::HOST, util::DEVICE))
      return retval;

    // Allocate out degrees for each node
    if (retval = this->out_degrees.Allocate(nodes, util::DEVICE)) return retval;
    // count number of out-going degrees for each node
    util::MemsetMadVectorKernel<<<128, 128>>>(
        this->out_degrees.GetPointer(util::DEVICE),
        this->row_offsets.GetPointer(util::DEVICE),
        this->row_offsets.GetPointer(util::DEVICE) + 1, (SizeT)-1, nodes);

    if (inverstgraph != NULL) {
      // Allocate and initialize column_offsets
      if (retval = this->column_offsets.Allocate(nodes + 1, util::DEVICE))
        return retval;
      if (retval = this->column_offsets.Move(util::HOST, util::DEVICE))
        return retval;

      // Allocate and initialize row_indices
      if (retval =
              this->row_indices.Allocate(inverstgraph->edges, util::DEVICE))
        return retval;
      if (retval = this->row_indices.Move(util::HOST, util::DEVICE))
        return retval;

      if (retval = this->in_degrees.Allocate(nodes, util::DEVICE))
        return retval;
      // count number of in-going degrees for each node
      util::MemsetMadVectorKernel<<<128, 128>>>(
          this->in_degrees.GetPointer(util::DEVICE),
          this->column_offsets.GetPointer(util::DEVICE),
          this->column_offsets.GetPointer(util::DEVICE) + 1, (SizeT)-1, nodes);
    }

    // For multi-GPU cases
    if (num_gpus > 1) {
      // Allocate and initialize convertion_table
      if (retval = this->partition_table.Allocate(nodes, util::DEVICE))
        return retval;
      if (partition_table != NULL)
        if (retval = this->partition_table.Move(util::HOST, util::DEVICE))
          return retval;

      // Allocate and initialize convertion_table
      if (retval = this->convertion_table.Allocate(nodes, util::DEVICE))
        return retval;
      if (convertion_table != NULL)
        if (retval = this->convertion_table.Move(util::HOST, util::DEVICE))
          return retval;

      // Allocate and initialize original_vertex
      if (retval = this->original_vertex.Allocate(nodes, util::DEVICE))
        return retval;
      if (original_vertex != NULL)
        if (retval = this->original_vertex.Move(util::HOST, util::DEVICE))
          return retval;

      // If need backward information proration
      if (backward_offsets != NULL) {
        // Allocate and initialize backward_offset
        this->backward_offset.SetPointer(backward_offsets, nodes + 1);
        if (retval = this->backward_offset.Allocate(nodes + 1, util::DEVICE))
          return retval;
        if (retval = this->backward_offset.Move(util::HOST, util::DEVICE))
          return retval;

        // Allocate and initialize backward_partition
        this->backward_partition.SetPointer(backward_partition,
                                            backward_offsets[nodes]);
        if (retval = this->backward_partition.Allocate(backward_offsets[nodes],
                                                       util::DEVICE))
          return retval;
        if (retval = this->backward_partition.Move(util::HOST, util::DEVICE))
          return retval;

        // Allocate and initialize backward_convertion
        this->backward_convertion.SetPointer(backward_convertion,
                                             backward_offsets[nodes]);
        if (retval = this->backward_convertion.Allocate(backward_offsets[nodes],
                                                        util::DEVICE))
          return retval;
        if (retval = this->backward_convertion.Move(util::HOST, util::DEVICE))
          return retval;
      }
    }  // end if num_gpu>1

    return retval;
  }  // end of Init(...)

  /**
   * @brief overloaded = operator
   *
   * @param[in] other GraphSlice to copy from
   *
   * \return GraphSlice& a copy of local GraphSlice
   */
  GraphSlice& operator=(GraphSlice other) {
    num_gpus = other.num_gpus;
    index = other.index;
    nodes = other.nodes;
    edges = other.edges;
    graph = other.graph;
    row_offsets = other.row_offsets;
    column_indices = other.column_indices;
    column_offsets = other.column_offsets;
    row_indices = other.row_indices;
    partition_table = other.partition_table;
    convertion_table = other.convertion_table;
    original_vertex = other.original_vertex;
    in_counter = other.in_counter;
    out_offset = other.out_offset;
    out_counter = other.out_counter;
    backward_offset = other.backward_offset;
    backward_partition = other.backward_partition;
    backward_convertion = other.backward_convertion;
    return *this;
  }  // end operator=()

};  // end GraphSlice

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
