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

#include <gunrock/partitioner/partitioner_base.cuh>

namespace gunrock {
namespace partitioner {
namespace random {

typedef std::mt19937 Engine;
typedef std::uniform_int_distribution<int> Distribution;

template <typename GraphT>
cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                      util::Parameters &parameters, int num_subgraphs = 1,
                      PartitionFlag flag = PARTITION_NONE,
                      util::Location target = util::HOST,
                      float *weitage = NULL) {
  // typedef _GraphT  GraphT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::GpT GpT;

  cudaError_t retval = cudaSuccess;
  auto &partition_table = org_graph.GpT::partition_table;
  SizeT nodes = org_graph.nodes;
  util::Array1D<SizeT, SortNode<SizeT, long int> > sort_list;
  sort_list.SetName("partitioner::random::sort_list");

  int partition_seed = parameters.Get<int>("partition-seed");
  if (parameters.UseDefault("partition-seed")) partition_seed = time(NULL);

  bool quiet = parameters.Get<bool>("quiet");
  if (!quiet)
    util::PrintMsg("Random partition begin. seed = " +
                   std::to_string(partition_seed));

  retval = sort_list.Allocate(nodes, target);
  if (retval) return retval;

#pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    SizeT node_start = (long long)(nodes)*thread_num / num_threads;
    SizeT node_end = (long long)(nodes) * (thread_num + 1) / num_threads;
    unsigned int thread_seed = partition_seed + 754 * thread_num;
    Engine engine(thread_seed);
    Distribution distribution(0, util::PreDefinedValues<int>::MaxValue);
    for (SizeT v = node_start; v < node_end; v++) {
      long int x;
      x = distribution(engine);
      sort_list[v].value = x;
      sort_list[v].posit = v;
    }
  }

  util::omp_sort(sort_list + 0, nodes, Compare_SortNode<SizeT, long int>);

  for (int i = 0; i < num_subgraphs; i++) {
    SizeT begin_pos = (i == 0 ? 0 : weitage[i - 1] * nodes);
    SizeT end_pos = weitage[i] * nodes;
    for (SizeT pos = begin_pos; pos < end_pos; pos++)
      partition_table[sort_list[pos].posit] = i;
  }

  if (retval = sort_list.Release()) return retval;

  return retval;
}  // end of Partition

}  // namespace random
}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
