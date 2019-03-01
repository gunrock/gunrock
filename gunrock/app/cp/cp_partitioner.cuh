// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cp_partitioner.cuh
 *
 * @brief Implementation of cluster partitioner
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
namespace cp {

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
    bool     KEEP_NODE_NUM   = false*/>
struct ClusterPartitioner : PartitionerBase<VertexId,SizeT,Value/*,
    ENABLE_BACKWARD,KEEP_ORDER,KEEP_NODE_NUM*/>
{
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  // Members
  float *weitage;

  // Methods
  /*ClusterPartitioner()
  {
      weitage=NULL;
  }*/

  ClusterPartitioner(const GraphT &graph, int num_gpus, float *weitage = NULL,
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

  ~ClusterPartitioner() {
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
    VertexId *t_queue = new VertexId[this->graph->nodes];
    VertexId *marker = new VertexId[this->graph->nodes];
    SizeT total_count = 0, current = 0, tail = 0, level = 0, target_level;
    SizeT *counter = new SizeT[this->num_gpus + 1];
    SizeT n1 = 1, n2 = 1;
    SizeT *level_tail = new SizeT[(n1 < n2 ? n2 : n1) + 1];
    // float       f1 = 1.0/this->num_gpus;
    SizeT *current_count = new SizeT[this->num_gpus];
    VertexId StartId, EndId;
    SizeT *row_offsets = this->graph->row_offsets;
    VertexId *column_indices = this->graph->column_indices;

    if (factor < 0)
      this->factor = 1.0 / this->num_gpus;
    else
      this->factor = factor;
    printf("partition_factor = %f\n", this->factor);

    target_level = (n1 < n2 ? n2 : n1);
    for (SizeT node = 0; node < nodes; node++) {
      sort_list[node].value =
          this->graph->row_offsets[node + 1] - this->graph->row_offsets[node];
      sort_list[node].posit = node;
      tpartition_table[node] = this->num_gpus;
    }
    for (int i = 0; i < this->num_gpus; i++) current_count[i] = 0;
    std::vector<sort_node<SizeT> > sort_vector(sort_list, sort_list + nodes);
    std::sort(sort_vector.begin(), sort_vector.end());

    // printf("1");fflush(stdout);
    for (SizeT pos = 0; pos <= nodes; pos++) {
      VertexId node = sort_vector[pos].posit;
      if (tpartition_table[node] != this->num_gpus) continue;
      // printf("node = %d, value =%d\t",node,
      // sort_vector[pos].value);fflush(stdout);
      current = 0;
      tail = 0;
      level = 0;
      total_count = 0;
      t_queue[current] = node;
      marker[node] = node;
      tpartition_table[node] = this->num_gpus + 1;
      for (SizeT i = 0; i <= this->num_gpus; i++) counter[i] = 0;
      counter[this->num_gpus] = 1;
      // memset(marker,0,sizeof(int)*nodes);
      // while (level < (n1<n2? n2:n1))
      for (level = 0; level < target_level; level++) {
        level_tail[level] = tail;
        // printf("level = %d\t",level);fflush(stdout);
        // while (current <= level_tail[level])
        for (; current <= level_tail[level]; current++) {
          VertexId t_node = t_queue[current];
          StartId = row_offsets[t_node];
          EndId = row_offsets[t_node + 1];
          // printf("t_node = %d\t",t_node);fflush(stdout);
          for (VertexId i = StartId; i < EndId; i++) {
            VertexId neibor = column_indices[i];
            if (marker[neibor] == node) continue;
            if (tpartition_table[neibor] < this->num_gpus) {
              if (level < n1) {
                counter[tpartition_table[neibor]]++;
                total_count++;
              }
            } else {
              if (level < n2) {
                counter[this->num_gpus]++;
                tpartition_table[neibor] = this->num_gpus + 1;
                // printf("%d\t",neibor);
              }
              if (level < n1) total_count++;
            }
            marker[neibor] = node;
            tail++;
            t_queue[tail] = neibor;
          }
          // current ++;
        }
        // level++;
      }
      level_tail[level] = tail;

      // printf(" -2");fflush(stdout);
      SizeT Max_Count = 0;
      int Max_GPU = -1, Set_GPU = -1;
      for (int i = 0; i < this->num_gpus; i++)
        if (counter[i] > Max_Count) {
          Max_Count = counter[i];
          Max_GPU = i;
        }
      // printf("Max_Count = %d, total_count = %d",Max_Count,total_count);
      if (Max_GPU != -1 && 1.0 * Max_Count / total_count > this->factor &&
          current_count[Max_GPU] + counter[this->num_gpus] <=
              nodes * weitage[Max_GPU])
        Set_GPU = Max_GPU;
      else {
        float max_empty = 0;
        for (int i = 0; i < this->num_gpus; i++)
          if (1 - 1.0 * current_count[i] / nodes / weitage[i] > max_empty) {
            max_empty = 1 - 1.0 * current_count[i] / nodes / weitage[i];
            Set_GPU = i;
          }
        // printf(", max_empty = %f", max_empty);
      }

      // printf(", Set_GPU = %d, tail = %d\n", Set_GPU,
      // level_tail[n2]);fflush(stdout);
      for (VertexId i = 0; i <= level_tail[n2]; i++)
        if (tpartition_table[t_queue[i]] == this->num_gpus + 1) {
          tpartition_table[t_queue[i]] = Set_GPU;
          // printf("%d->%d \t",t_queue[i],Set_GPU);
        }
      // printf("\n");fflush(stdout);
      current_count[Set_GPU] += counter[this->num_gpus];
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

}  // namespace cp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
