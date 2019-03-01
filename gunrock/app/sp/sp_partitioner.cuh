// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rp_partitioner.cuh
 *
 * @brief Implementation of random partitioner
 */

#pragma once

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace app {
namespace sp {

template <
    typename VertexId,
    typename SizeT,
    typename Value/*,
    bool     ENABLE_BACKWARD = false,
    bool     KEEP_ORDER      = false,
    bool     KEEP_NODE_NUM   = false*/>
struct StaticPartitioner : PartitionerBase<VertexId,SizeT,Value/*,
    ENABLE_BACKWARD,KEEP_ORDER,KEEP_NODE_NUM*/>
{
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  // Members
  float *weitage;

  // Methods
  /*StaticPartitioner()
  {
      weitage=NULL;
  }*/

  StaticPartitioner(const GraphT &graph, int num_gpus, float *weitage = NULL,
                    bool _enable_backward = false, bool _keep_order = false,
                    bool _keep_node_num = false)
      : BasePartitioner(_enable_backward, _keep_order, _keep_node_num) {
    Init2(graph, num_gpus, weitage);
  }

  void Init2(const GraphT &graph, int num_gpus, float *weitage) {
    this->Init(graph, num_gpus);
    this->weitage = new float[num_gpus + 1];
    if (weitage == NULL)
      for (int gpu = 0; gpu < num_gpus; gpu++)
        this->weitage[gpu] = 1.0f / num_gpus;
    else {
      float sum = 0;
      for (int gpu = 0; gpu < num_gpus; gpu++) sum += weitage[gpu];
      for (int gpu = 0; gpu < num_gpus; gpu++)
        this->weitage[gpu] = weitage[gpu] / sum;
    }
    for (int gpu = 0; gpu < num_gpus; gpu++)
      this->weitage[gpu + 1] += this->weitage[gpu];
  }

  ~StaticPartitioner() {
    if (weitage != NULL) {
      delete[] weitage;
      weitage = NULL;
    }
  }

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
    // time_t      t = time(NULL);
    SizeT nodes = this->graph->nodes;

    for (SizeT node = 0; node < nodes; node++)
      tpartition_table[node] = (node % this->num_gpus);
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

}  // namespace sp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
