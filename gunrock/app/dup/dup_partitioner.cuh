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

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <random>

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace app {
namespace dup {

template <
    typename VertexId,
    typename SizeT,
    typename Value/*,
    bool     ENABLE_BACKWARD = false,
    bool     KEEP_ORDER      = false,
    bool     KEEP_NODE_NUM   = false*/ >
struct DuplicatePartitioner :
    PartitionerBase<VertexId, SizeT, Value/*,
    ENABLE_BACKWARD, KEEP_ORDER, KEEP_NODE_NUM*/>
{
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  DuplicatePartitioner(const GraphT &graph, int num_gpus, float *weitage = NULL,
                       bool _enable_backward = false, bool _keep_order = false,
                       bool _keep_node_num = false)
      : BasePartitioner(_enable_backward, _keep_order, _keep_node_num) {
    this->Init(graph, num_gpus);
  }

  ~DuplicatePartitioner() {}

  cudaError_t Partition(GraphT *&sub_graphs, int **&partition_tables,
                        VertexId **&convertion_tables,
                        VertexId **&original_vertexes,
                        // SizeT**    &in_offsets,
                        SizeT **&in_counter, SizeT **&out_offsets,
                        SizeT **&out_counter, SizeT **&backward_offsets,
                        int **&backward_partitions,
                        VertexId **&backward_convertions, float factor = -1,
                        int seed = -1) {
    cudaError_t retval = cudaSuccess;
    int *tpartition_table = this->partition_tables[0];

    tpartition_table[0] = 0;
    SizeT org_nodes = (this->graph->nodes - 1) / this->num_gpus;
    // printf("org_nodes = %lld\n", (long long)org_nodes);
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      VertexId base_v = 1 + org_nodes * gpu;
      for (VertexId org_v = 0; org_v < org_nodes; org_v++) {
        tpartition_table[base_v + org_v] = gpu;
      }
    }

    retval = this->MakeSubGraph();
    sub_graphs = this->sub_graphs;
    partition_tables = this->partition_tables;
    convertion_tables = this->convertion_tables;
    original_vertexes = this->original_vertexes;
    // in_offsets          = this->in_offsets;
    in_counter = this->in_counter;
    out_offsets = this->out_offsets;
    out_counter = this->out_counter;
    backward_offsets = this->backward_offsets;
    backward_partitions = this->backward_partitions;
    backward_convertions = this->backward_convertions;
    return retval;
  }
};

}  // namespace dup
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
