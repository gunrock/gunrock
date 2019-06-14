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

#include <gunrock/partitioner/partitioner_base.cuh>

namespace gunrock {
namespace partitioner {
namespace biased_random {

template <typename GraphT, bool CSR_SWITCH>
struct CsrSwitch {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    return util::GRError(cudaErrorUnknown,
                         "BiasedRandom partitioner not defined for non-CSR "
                         "graph representations",
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
    SizeT n1 = 1;  //, n2 = 1;
    SizeT target_level = n1;
    SizeT start_edge, end_edge;
    double partition_factor = 0.5;
    long partition_seed = 0;

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
    util::Array1D<SizeT, float> subgraph_percentage;

    if (!parameters.UseDefault("partition-factor"))
      partition_factor = parameters.Get<double>("partition-factor");

    if (parameters.UseDefault("partition-seed"))
      partition_seed = time(NULL);
    else
      partition_seed = parameters.Get<long>("partition-seed");

    bool quiet = parameters.Get<bool>("quiet");
    if (!quiet)
      util::PrintMsg("Biased random partition begin. factor = " +
                     std::to_string(partition_factor) +
                     ", seed = " + std::to_string(partition_seed));

    sort_list.SetName("partitioner::biased_random::sort_list");
    retval = sort_list.Allocate(nodes, target);
    if (retval) return retval;

    t_queue.SetName("partitioner::biased_random::t_queue");
    retval = t_queue.Allocate(nodes, target);
    if (retval) return retval;

    marker.SetName("partitioner::biased_random::marker");
    retval = marker.Allocate(nodes, target);
    if (retval) return retval;

    counter.SetName("partitioner::biased_random::counter");
    retval = counter.Allocate(num_subgraphs + 1, target);
    if (retval) return retval;

    level_tail.SetName("partitioner::biased_random::level_tail");
    retval = level_tail.Allocate(target_level + 1, target);
    if (retval) return retval;

    current_count.SetName("partitioner::biased_random::current_count");
    retval = current_count.Allocate(num_subgraphs, target);
    if (retval) return retval;

    subgraph_percentage.SetName(
        "partitioner::biased_random::subgraph_percentage");
    retval = subgraph_percentage.Allocate(num_subgraphs + 1, target);
    if (retval) return retval;

    target_level = n1;  //(n1<n2? n2:n1);
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
    memset(marker + 0, 0, sizeof(VertexT) * nodes);
    std::vector<SortNode<SizeT, SizeT> > sort_vector(sort_list + 0,
                                                     sort_list + nodes);
    std::sort(sort_vector.begin(), sort_vector.end());

    for (SizeT pos = 0; pos < nodes; pos++) {
      VertexT node = sort_vector[pos].posit;
      if (partition_table[node] != num_subgraphs) continue;
      current = 0;
      tail = 0;
      level = 0;
      total_count = 0;
      t_queue[current] = node;
      marker[node] = node;
      for (int i = 0; i <= num_subgraphs; i++) counter[i] = 0;
      for (level = 0; level < target_level; level++) {
        level_tail[level] = tail;
        for (; current <= level_tail[level]; current++) {
          VertexT t_node = t_queue[current];
          start_edge = row_offsets[t_node];
          end_edge = row_offsets[t_node + 1];
          for (SizeT e = start_edge; e < end_edge; e++) {
            VertexT neibor = column_indices[e];
            if (marker[neibor] == node) continue;
            if (partition_table[neibor] < num_subgraphs) {
              if (level < n1) {
                counter[partition_table[neibor]]++;
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
      for (int i = 0; i < num_subgraphs; i++) {
        total_count += counter[i];
      }
      for (int i = 0; i < num_subgraphs; i++) {
        subgraph_percentage[i] =
            (total_count == 0 ? 0
                              : (partition_factor * counter[i] / total_count));
      }
      total_count = 0;
      for (int i = 0; i < num_subgraphs; i++) {
        SizeT e = nodes * weitage[i] - current_count[i];
        total_count += (util::atLeastZero(e) ? e : 0);
      }
      for (int i = 0; i < num_subgraphs; i++) {
        SizeT e = nodes * weitage[i] - current_count[i];
        subgraph_percentage[i] +=
            (e > 0 ? ((1 - partition_factor) * e / total_count) : 0);
      }
      float total_percentage = 0;
      for (int i = 0; i < num_subgraphs; i++)
        total_percentage += subgraph_percentage[i];
      for (int i = 0; i < num_subgraphs; i++) {
        subgraph_percentage[i] = subgraph_percentage[i] / total_percentage;
      }
      subgraph_percentage[num_subgraphs] = 1;
      for (int i = num_subgraphs - 1; i >= 0; i--)
        subgraph_percentage[i] =
            subgraph_percentage[i + 1] - subgraph_percentage[i];
      float x = 1.0f * rand() / RAND_MAX;
      for (int i = 0; i < num_subgraphs; i++)
        if (x >= subgraph_percentage[i] && x < subgraph_percentage[i + 1]) {
          current_count[i]++;
          partition_table[node] = i;
          break;
        }
      if (partition_table[node] >= num_subgraphs) {
        partition_table[node] = (rand() % (num_subgraphs));
        current_count[partition_table[node]]++;
      }
    }

    if (retval = sort_list.Release()) return retval;
    if (retval = t_queue.Release()) return retval;
    if (retval = counter.Release()) return retval;
    if (retval = marker.Release()) return retval;
    if (retval = level_tail.Release()) return retval;
    if (retval = current_count.Release()) return retval;
    if (retval = subgraph_percentage.Release()) return retval;

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

}  // namespace biased_random
}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
