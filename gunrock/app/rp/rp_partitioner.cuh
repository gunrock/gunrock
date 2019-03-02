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
#include <random>

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace app {
namespace rp {

typedef std::mt19937 Engine;
typedef std::uniform_int_distribution<int> Distribution;

template <typename SizeT>
struct sort_node {
 public:
  SizeT posit;
  int value;

  bool operator==(const sort_node &node) const { return (node.value == value); }

  bool operator<(const sort_node &node) const { return (node.value < value); }

  sort_node &operator=(const sort_node &rhs) {
    this->posit = rhs.posit;
    this->value = rhs.value;
    return *this;
  }
};

template <typename SizeT>
bool compare_sort_node(sort_node<SizeT> A, sort_node<SizeT> B) {
  return (A.value < B.value);
}

template <
    typename VertexId,
    typename SizeT,
    typename Value/*,
    bool     ENABLE_BACKWARD = false,
    bool     KEEP_ORDER      = false,
    bool     KEEP_NODE_NUM   = false*/ >
struct RandomPartitioner :
    PartitionerBase<VertexId, SizeT, Value/*,
    ENABLE_BACKWARD, KEEP_ORDER, KEEP_NODE_NUM*/>
{
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  // Members
  float *weitage;

  // Methods
  /*RandomPartitioner()
  {
      weitage = NULL;
  }*/

  RandomPartitioner(const GraphT &graph, int num_gpus, float *weitage = NULL,
                    bool _enable_backward = false, bool _keep_order = false,
                    bool _keep_node_num = false)
      : BasePartitioner(_enable_backward, _keep_order, _keep_node_num) {
    Init2(graph, num_gpus, weitage);
  }

  void Init2(const GraphT &graph, int num_gpus, float *weitage) {
    this->Init(graph, num_gpus);
    this->weitage = new float[num_gpus + 1];
    if (weitage == NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        this->weitage[gpu] = 1.0f / num_gpus;
      }
    } else {
      float sum = 0;
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        sum += weitage[gpu];
      }
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        this->weitage[gpu] = weitage[gpu] / sum;
      }
    }
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      this->weitage[gpu + 1] += this->weitage[gpu];
    }
  }

  ~RandomPartitioner() {
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
    SizeT nodes = this->graph->nodes;
    sort_node<SizeT> *sort_list = new sort_node<SizeT>[nodes];

    if (seed < 0)
      this->seed = time(NULL);
    else
      this->seed = seed;
    printf("Partition begin. seed=%d\n", this->seed);
    fflush(stdout);

#pragma omp parallel
    {
      int thread_num = omp_get_thread_num();
      int num_threads = omp_get_num_threads();
      SizeT i_start = (long long)(nodes)*thread_num / num_threads;
      SizeT i_end = (long long)(nodes) * (thread_num + 1) / num_threads;
      unsigned int seed_ = this->seed + 754 * thread_num;
      Engine engine(seed_);
      Distribution distribution(0, util::MaxValue<int>());
      for (SizeT i = i_start; i < i_end; i++) {
        long int x;
        x = distribution(engine);
        sort_list[i].value = x;
        sort_list[i].posit = i;
      }
    }

    util::omp_sort(sort_list, nodes, compare_sort_node<SizeT>);
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      SizeT begin_pos = gpu == 0 ? 0 : weitage[gpu - 1] * nodes;
      SizeT end_pos = weitage[gpu] * nodes;
      for (SizeT pos = begin_pos; pos < end_pos; pos++) {
        tpartition_table[sort_list[pos].posit] = gpu;
      }
    }

    delete[] sort_list;
    sort_list = NULL;
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

}  // namespace rp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
