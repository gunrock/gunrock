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

#include <gunrock/partitioner/partitioner_base.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace partitioner {
namespace cluster {

template <typename GraphT, bool CSR_SWITCH>
struct CsrSwitch {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    return util::GRError(
        cudaErrorUnknown,
        "Cluster partitioner not defined for non-CSR graph representations",
        __FILE__, __LINE__);
  }
};

template <typename GraphT>
struct CsrSwitch<GraphT, true> {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::GpT GpT;

    cudaError_t retval = cudaSuccess;
    SizeT total_count = 0;
    SizeT current = 0;
    SizeT tail = 0;
    SizeT level = 0;
    SizeT target_level;
    SizeT n1 = 1;
    SizeT n2 = 1;
    // float       f1 = 1.0/this->num_gpus;
    SizeT start_edge, end_edge;
    double partition_factor = 0;

    auto &partition_table = org_graph.GpT::partition_table;
    SizeT nodes = org_graph.nodes;
    auto &row_offsets = org_graph.CsrT::row_offsets;
    auto &column_indices = org_graph.CsrT::column_indices;
    util::Array1D<SizeT, SortNode<SizeT, SizeT> > sort_list;
    util::Array1D<SizeT, VertexT> t_queue;
    util::Array1D<SizeT, VertexT> marker;
    util::Array1D<SizeT, SizeT> counter;
    util::Array1D<SizeT, SizeT> level_tail;
    util::Array1D<SizeT, SizeT> current_count;

    if (parameters.UseDefault("partition-factor"))
      partition_factor = 1.0 / num_subgraphs;
    else
      partition_factor = parameters.Get<double>("partition-factor");

    bool quiet = parameters.Get<bool>("quiet");
    if (!quiet)
      util::PrintMsg("Cluster partition begin. factor = " +
                     std::to_string(partition_factor));

    sort_list.SetName("partitioner::cluster::sort_list");
    retval = sort_list.Allocate(nodes, target);
    if (retval) return retval;

    t_queue.SetName("partitioner::cluster::t_queue");
    retval = t_queue.Allocate(nodes, target);
    if (retval) return retval;

    marker.SetName("partitioner::cluster::marker");
    retval = marker.Allocate(nodes, target);
    if (retval) return retval;

    counter.SetName("partitioner::cluster::counter");
    retval = counter.Allocate(num_subgraphs + 1, target);
    if (retval) return retval;

    level_tail.SetName("partitioner::cluster::level_tail");
    retval = level_tail.Allocate((n1 < n2 ? n2 : n1) + 1, target);
    if (retval) return retval;

    current_count.SetName("partitioner::cluster::current_count");
    retval = current_count.Allocate(num_subgraphs, target);
    if (retval) return retval;

    target_level = (n1 < n2 ? n2 : n1);
    retval = sort_list.ForAll(
        row_offsets,
        [] __host__ __device__(SortNode<SizeT, SizeT> * sort_list_,
                               SizeT * row_offsets_, SizeT node) {
          sort_list_[node].value = row_offsets_[node + 1] - row_offsets_[node];
          sort_list_[node].posit = node;
        },
        nodes, target);
    if (retval) return retval;

    retval = partition_table.ForEach(
        [num_subgraphs] __host__ __device__(int &partition) {
          partition = num_subgraphs;
        },
        nodes, target);
    if (retval) return retval;

    for (int i = 0; i < num_subgraphs; i++) current_count[i] = 0;
    std::vector<SortNode<SizeT, SizeT> > sort_vector(sort_list + 0,
                                                     sort_list + nodes);
    std::sort(sort_vector.begin(), sort_vector.end());

    // printf("1");fflush(stdout);
    for (SizeT pos = 0; pos <= nodes; pos++) {
      VertexT node = sort_vector[pos].posit;
      if (partition_table[node] != num_subgraphs) continue;
      // printf("node = %d, value =%d\t",node,
      // sort_vector[pos].value);fflush(stdout);
      current = 0;
      tail = 0;
      level = 0;
      total_count = 0;
      t_queue[current] = node;
      marker[node] = node;
      partition_table[node] = num_subgraphs + 1;
      for (SizeT i = 0; i <= num_subgraphs; i++) counter[i] = 0;
      counter[num_subgraphs] = 1;
      // memset(marker,0,sizeof(int)*nodes);
      // while (level < (n1<n2? n2:n1))
      for (level = 0; level < target_level; level++) {
        level_tail[level] = tail;
        // printf("level = %d\t",level);fflush(stdout);
        // while (current <= level_tail[level])
        for (; current <= level_tail[level]; current++) {
          VertexT t_node = t_queue[current];
          start_edge = row_offsets[t_node];
          end_edge = row_offsets[t_node + 1];
          // printf("t_node = %d\t",t_node);fflush(stdout);
          for (SizeT e = start_edge; e < end_edge; e++) {
            VertexT neibor = column_indices[e];
            if (marker[neibor] == node) continue;
            if (partition_table[neibor] < num_subgraphs) {
              if (level < n1) {
                counter[partition_table[neibor]]++;
                total_count++;
              }
            } else {
              if (level < n2) {
                counter[num_subgraphs]++;
                partition_table[neibor] = num_subgraphs + 1;
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
      SizeT max_count = 0;
      int max_GPU = -1, set_GPU = -1;
      for (int i = 0; i < num_subgraphs; i++)
        if (counter[i] > max_count) {
          max_count = counter[i];
          max_GPU = i;
        }
      // printf("Max_Count = %d, total_count = %d",Max_Count,total_count);
      if (max_GPU != -1 && (1.0 * max_count / total_count) > partition_factor &&
          current_count[max_GPU] + counter[num_subgraphs] <=
              nodes * weitage[max_GPU])
        set_GPU = max_GPU;
      else {
        float max_empty = 0;
        for (int i = 0; i < num_subgraphs; i++) {
          float current_empty = 1 - 1.0 * current_count[i] / nodes / weitage[i];
          if (current_empty > max_empty) {
            max_empty = current_empty;
            set_GPU = i;
          }
        }
        // printf(", max_empty = %f", max_empty);
      }

      // printf(", Set_GPU = %d, tail = %d\n", Set_GPU,
      // level_tail[n2]);fflush(stdout);
      for (VertexT i = 0; i <= level_tail[n2]; i++)
        if (partition_table[t_queue[i]] == num_subgraphs + 1) {
          partition_table[t_queue[i]] = set_GPU;
          // printf("%d->%d \t",t_queue[i],Set_GPU);
        }
      // printf("\n");fflush(stdout);
      current_count[set_GPU] += counter[num_subgraphs];
    }

    if (retval = sort_list.Release()) return retval;
    if (retval = t_queue.Release()) return retval;
    if (retval = counter.Release()) return retval;
    if (retval = marker.Release()) return retval;
    if (retval = level_tail.Release()) return retval;
    if (retval = current_count.Release()) return retval;

    return retval;
  }
};

template <typename GraphT>
cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                      util::Parameters &parameters, int num_subgraphs = 1,
                      PartitionFlag flag = PARTITION_NONE,
                      util::Location target = util::HOST,
                      float *weitage = NULL) {
  return CsrSwitch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSR) !=
                               0>::Partition(org_graph, sub_graphs, parameters,
                                             num_subgraphs, flag, target,
                                             weitage);
}

}  // namespace cluster
}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
