// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dup_partitioner.cuh
 *
 * @brief Implementation of random partitioner
 */

#pragma once

#include <gunrock/partitioner/partitioner_base.cuh>

namespace gunrock {
namespace partitioner {
namespace duplicate {

template <typename GraphT>
cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                      util::Parameters &parameters, int num_subgraphs = 1,
                      PartitionFlag flag = PARTITION_NONE,
                      util::Location target = util::HOST,
                      float *weitage = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::GpT GpT;

  cudaError_t retval = cudaSuccess;
  auto &partition_table = org_graph.partition_table;

  partition_table[0] = 0;
  SizeT org_nodes = (org_graph.nodes - 1) / num_subgraphs;
  // printf("org_nodes = %lld\n", (long long)org_nodes);
  for (int i = 0; i < num_subgraphs; i++) {
    VertexT base_v = 1 + org_nodes * i;
    for (VertexT org_v = 0; org_v < org_nodes; org_v++) {
      partition_table[base_v + org_v] = i;
    }
  }

  return retval;
}

}  // namespace duplicate
}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
