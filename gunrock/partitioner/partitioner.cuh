// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * partitioner.cuh
 *
 * @brief Common interface for all partitioners
 */

#pragma once

#include <string>

#include <gunrock/util/parameters.h>
#include <gunrock/partitioner/random.cuh>
#include <gunrock/partitioner/static.cuh>
#include <gunrock/partitioner/metis.cuh>
#include <gunrock/partitioner/cluster.cuh>
#include <gunrock/partitioner/biased_random.cuh>
#include <gunrock/partitioner/duplicate.cuh>

namespace gunrock {
namespace partitioner {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  retval = parameters.Use<std::string>(
      "partition-method",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "random",
      "partitioning method, can be one of {random, biasrandom, cluster, metis, "
      "static, duplicate}",
      __FILE__, __LINE__);
  if (retval) return retval;

  retval = parameters.Use<float>(
      "partition-factor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.5, "partitioning factor", __FILE__, __LINE__);
  if (retval) return retval;

  retval = parameters.Use<int>(
      "partition-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "partitioning seed, default is time(NULL)", __FILE__, __LINE__);
  if (retval) return retval;

  return retval;
}

template <typename GraphT>
cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                      util::Parameters &parameters, int num_subgraphs = 1,
                      PartitionFlag flag = PARTITION_NONE,
                      util::Location target = util::HOST,
                      float *weitage = NULL) {
  typedef typename GraphT::GpT GpT;

  cudaError_t retval = cudaSuccess;
  std::string partition_method =
      parameters.Get<std::string>("partition-method");

  retval =
      org_graph.GpT::Allocate(org_graph.nodes, org_graph.edges, num_subgraphs,
                              flag & Org_Graph_Mark, target);
  if (retval) return retval;

  bool weitage_allocated = false;
  if (weitage == NULL) {
    weitage_allocated = true;
    weitage = new float[num_subgraphs + 1];
    for (int i = 0; i < num_subgraphs; i++) weitage[i] = 1.0f / num_subgraphs;
  } else {
    float sum = 0;
    for (int i = 0; i < num_subgraphs; i++) sum += weitage[i];
    for (int i = 0; i < num_subgraphs; i++) weitage[i] = weitage[i] / sum;
  }
  for (int i = 0; i < num_subgraphs; i++) weitage[i + 1] += weitage[i];

  util::Location target_ = util::HOST;
  if (partition_method == "random")
    retval = random::Partition(org_graph, sub_graphs, parameters, num_subgraphs,
                               flag, target_, weitage);
  else if (partition_method == "static")
    retval = static_p::Partition(org_graph, sub_graphs, parameters,
                                 num_subgraphs, flag, target_, weitage);
  else if (partition_method == "metis")
    retval = metis::Partition(org_graph, sub_graphs, parameters, num_subgraphs,
                              flag, target_, weitage);
  else if (partition_method == "cluster")
    retval = cluster::Partition(org_graph, sub_graphs, parameters,
                                num_subgraphs, flag, target_, weitage);
  else if (partition_method == "biasrandom")
    retval = biased_random::Partition(org_graph, sub_graphs, parameters,
                                      num_subgraphs, flag, target_, weitage);
  else if (partition_method == "duplicate")
    retval = duplicate::Partition(org_graph, sub_graphs, parameters,
                                  num_subgraphs, flag, target_, weitage);
  else
    retval = util::GRError(cudaErrorUnknown,
                           "Unknown partitioning method " + partition_method,
                           __FILE__, __LINE__);
  if (retval) return retval;

  if (weitage_allocated) {
    delete[] weitage;
    weitage = NULL;
  }
  retval = MakeSubGraph(org_graph, sub_graphs, parameters, num_subgraphs, flag,
                        target);
  if (retval) return retval;
  return retval;
}

}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
