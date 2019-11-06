// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gp.cuh
 *
 * @brief Partition info for graph
 */

#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {

namespace partitioner {

using PartitionFlag = uint32_t;

enum : PartitionFlag {
  PARTITION_NONE = 0x00,

  Enable_Backward = 0x10,
  Keep_Order = 0x20,
  Keep_Node_Num = 0x40,
  Use_Original_Vertex = 0x80,

  Org_Graph_Mark = 0x100,
  Sub_Graph_Mark = 0x200,

  PARTITION_RESERVE = 0xF0,
};
}  // namespace partitioner

namespace graph {

/**
 * @brief GP data structure to store partition info
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE | HAS_GP,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Gp : public GraphBase<_VertexT, _SizeT, _ValueT, _FLAG | HAS_GP,
                             cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG | HAS_GP;
  static const util::ArrayFlag ARRAY_FLAG =
      util::If_Val<(FLAG & GRAPH_PINNED) != 0,
                   (FLAG & ARRAY_RESERVE) | util::PINNED,
                   FLAG & ARRAY_RESERVE>::Value;
  typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
      BaseGraph;
  typedef Gp<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag> GpT;

  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag>
      partition_table;  // Partition tables indicating which GPU the vertices
                        // are hosted
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      convertion_table;  // Conversions tables indicating vertex IDs on local /
                         // remote GPUs
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      original_vertex;  // Vertex IDs in the original graph
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      in_counter;  // Number of in vertices
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      out_offset;  // Out offsets for data communication
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      out_counter;  // Number of out vertices
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      backward_offset;  // Offsets for backward propagation
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag>
      backward_partition;  // Partition tables for backward propagation
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      backward_convertion;  // Conversion tables for backward propagation

  /**
   * @brief GP Constructor
   */
  Gp() : BaseGraph() {
    partition_table.SetName("partition_table");
    convertion_table.SetName("convertion_table");
    original_vertex.SetName("original_vertex");
    in_counter.SetName("in_counter");
    out_offset.SetName("out_offset");
    out_counter.SetName("out_counter");
    backward_offset.SetName("backward_offset");
    backward_partition.SetName("backward_partition");
    backward_convertion.SetName("backward_convertion");
  }

  /**
   * @brief GP destructor
   */
  __device__ __host__ ~Gp() {
    // Release();
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (retval = partition_table.Release(target)) return retval;
    if (retval = convertion_table.Release(target)) return retval;
    if (retval = original_vertex.Release(target)) return retval;
    if (retval = in_counter.Release(target)) return retval;
    if (retval = out_offset.Release(target)) return retval;
    if (retval = out_counter.Release(target)) return retval;
    if (retval = backward_offset.Release(target)) return retval;
    if (retval = backward_partition.Release(target)) return retval;
    if (retval = backward_convertion.Release(target)) return retval;
    return retval;
  }

  cudaError_t Allocate(
      SizeT nodes, SizeT edges, int num_subgraphs = 1,
      partitioner::PartitionFlag flag = partitioner::PARTITION_NONE,
      util::Location target = GRAPH_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    if (num_subgraphs == 1) return retval;

    retval = BaseGraph ::Allocate(nodes, edges, target);
    if (retval) return retval;
    retval = partition_table.Allocate(nodes, target);
    if (retval) return retval;
    if ((flag & partitioner::Keep_Node_Num) == 0) {
      retval = convertion_table.Allocate(nodes, target);
      if (retval) return retval;
    }

    if ((flag & partitioner::Org_Graph_Mark)) return retval;
    // if (retval = in_counter      .Allocate(num_subgraphs, target))
    //    return retval;
    // if (retval = out_counter     .Allocate(num_subgraphs, target))
    //    return retval;
    if (flag & partitioner::Use_Original_Vertex) {
      retval = original_vertex.Allocate(nodes, target);
      if (retval) return retval;
    }
    if (flag & partitioner::Enable_Backward) {
      retval = backward_offset.Allocate(nodes + 1, target);
      if (retval) return retval;
      retval = backward_partition.Allocate(in_counter[num_subgraphs], target);
      if (retval) return retval;
      if ((flag & partitioner::Keep_Node_Num) == 0)
        retval =
            backward_convertion.Allocate(in_counter[num_subgraphs], target);
      if (retval) return retval;
    }
    return retval;
  }
};  // Gp

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
