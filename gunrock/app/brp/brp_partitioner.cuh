// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * brp_partitioner.cuh
 *
 * @brief Implementation of biased random partitioner
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
namespace brp {

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

template <typename VertexId, typename SizeT, typename Value>
// bool     ENABLE_BACKWARD = false,
// bool     KEEP_ORDER      = false,
// bool     KEEP_NODE_NUM   = false>
struct BiasRandomPartitioner : PartitionerBase<VertexId, SizeT,
                                               Value /*,
ENABLE_BACKWARD, KEEP_ORDER, KEEP_NODE_NUM*/> {
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  // Members
  float *weitage;

  // Methods
  /*BiasRandomPartitioner()
  {
      weitage=NULL;
  }*/

  BiasRandomPartitioner(const GraphT &graph, int num_gpus,
                        float *weitage = NULL, bool _enable_backward = false,
                        bool _keep_order = false, bool _keep_node_num = false)
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
  }

  ~BiasRandomPartitioner() {
    if (weitage != NULL) {
      delete[] weitage;
      weitage = NULL;
    }
  }

  cudaError_t Partition(GraphT *&sub_graphs, int **&partition_tables,
                        VertexId **&convertion_tables,
                        VertexId **&original_vertexes, SizeT **&in_counter,
                        SizeT **&out_offsets, SizeT **&out_counter,
                        SizeT **&backward_offsets, int **&backward_partitions,
                        VertexId **&backward_convertions, float factor = -1,
                        int seed = -1) {
    cudaError_t retval = cudaSuccess;
    int *tpartition_table = this->partition_tables[0];
    SizeT nodes = this->graph->nodes;
    sort_node<SizeT> *sort_list = new sort_node<SizeT>[nodes];
    VertexId *t_queue = new VertexId[this->graph->nodes];
    VertexId *marker = new VertexId[this->graph->nodes];
    SizeT total_count = 0, current = 0, tail = 0, level = 0;
    SizeT *counter = new SizeT[this->num_gpus + 1];
    SizeT n1 = 1;  //, n2 = 1;
    SizeT target_level = n1;
    SizeT *level_tail = new SizeT[target_level + 1];
    float *gpu_percentage = new float[this->num_gpus + 1];
    SizeT *current_count = new SizeT[this->num_gpus];
    VertexId StartId, EndId;
    SizeT *row_offsets = this->graph->row_offsets;
    VertexId *column_indices = this->graph->column_indices;

    if (seed < 0)
      this->seed = time(NULL);
    else
      this->seed = seed;
    srand(this->seed);
    printf("Partition begin. seed = %d\n", this->seed);
    fflush(stdout);

    if (factor < 0)
      this->factor = 0.5;
    else
      this->factor = factor;
    // printf("partition_factor = %f\n", this->factor);fflush(stdout);

    target_level = n1;  //(n1<n2? n2:n1);
    for (SizeT node = 0; node < nodes; node++) {
      sort_list[node].value =
          this->graph->row_offsets[node + 1] - this->graph->row_offsets[node];
      sort_list[node].posit = node;
      tpartition_table[node] = this->num_gpus;
    }
    for (int i = 0; i < this->num_gpus; i++) current_count[i] = 0;
    memset(marker, 0, sizeof(VertexId) * nodes);
    std::vector<sort_node<SizeT> > sort_vector(sort_list, sort_list + nodes);
    std::sort(sort_vector.begin(), sort_vector.end());

    for (SizeT pos = 0; pos < nodes; pos++) {
      VertexId node = sort_vector[pos].posit;
      if (tpartition_table[node] != this->num_gpus) continue;
      current = 0;
      tail = 0;
      level = 0;
      total_count = 0;
      t_queue[current] = node;
      marker[node] = node;
      for (SizeT i = 0; i <= this->num_gpus; i++) counter[i] = 0;
      for (level = 0; level < target_level; level++) {
        level_tail[level] = tail;
        for (; current <= level_tail[level]; current++) {
          VertexId t_node = t_queue[current];
          StartId = row_offsets[t_node];
          EndId = row_offsets[t_node + 1];
          for (VertexId i = StartId; i < EndId; i++) {
            VertexId neibor = column_indices[i];
            if (marker[neibor] == node) continue;
            if (tpartition_table[neibor] < this->num_gpus) {
              if (level < n1) {
                counter[tpartition_table[neibor]]++;
              }
            }
            marker[neibor] = node;
            tail++;
            t_queue[tail] = neibor;
          }
        }
      }
      level_tail[level] = tail;

      total_count = 0;
      for (int i = 0; i < this->num_gpus; i++) {
        total_count += counter[i];
      }
      for (int i = 0; i < this->num_gpus; i++) {
        gpu_percentage[i] =
            (total_count == 0 ? 0 : (this->factor * counter[i] / total_count));
      }
      total_count = 0;
      for (int i = 0; i < this->num_gpus; i++) {
        SizeT e = nodes * weitage[i] - current_count[i];
        total_count += (e >= 0 ? e : 0);
      }
      for (int i = 0; i < this->num_gpus; i++) {
        SizeT e = nodes * weitage[i] - current_count[i];
        gpu_percentage[i] +=
            (e > 0 ? ((1 - this->factor) * e / total_count) : 0);
      }
      float total_percentage = 0;
      for (int i = 0; i < this->num_gpus; i++)
        total_percentage += gpu_percentage[i];
      for (int i = 0; i < this->num_gpus; i++) {
        gpu_percentage[i] = gpu_percentage[i] / total_percentage;
      }
      gpu_percentage[this->num_gpus] = 1;
      for (int i = this->num_gpus - 1; i >= 0; i--)
        gpu_percentage[i] = gpu_percentage[i + 1] - gpu_percentage[i];
      float x = 1.0f * rand() / RAND_MAX;
      for (int i = 0; i < this->num_gpus; i++)
        if (x >= gpu_percentage[i] && x < gpu_percentage[i + 1]) {
          current_count[i]++;
          tpartition_table[node] = i;
          break;
        }
      if (tpartition_table[node] >= this->num_gpus) {
        tpartition_table[node] = (rand() % (this->num_gpus));
        current_count[tpartition_table[node]]++;
      }
    }

    delete[] sort_list;
    sort_list = NULL;
    delete[] t_queue;
    t_queue = NULL;
    delete[] counter;
    counter = NULL;
    delete[] marker;
    marker = NULL;
    delete[] level_tail;
    level_tail = NULL;
    delete[] current_count;
    current_count = NULL;
    delete[] gpu_percentage;
    gpu_percentage = NULL;
    retval = this->MakeSubGraph();
    sub_graphs = this->sub_graphs;
    partition_tables = this->partition_tables;
    convertion_tables = this->convertion_tables;
    original_vertexes = this->original_vertexes;
    in_counter = this->in_counter;
    out_offsets = this->out_offsets;
    out_counter = this->out_counter;
    backward_offsets = this->backward_offsets;
    backward_partitions = this->backward_partitions;
    backward_convertions = this->backward_convertions;
    return retval;
  }
};

}  // namespace brp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
