// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graphio.cuh
 *
 * @brief high level, graph type independent routines for graphio
 */

#pragma once

#include <gunrock/util/parameters.h>
#include <gunrock/graph/csr.cuh>
#include <gunrock/graph/coo.cuh>
#include <gunrock/graph/csc.cuh>
#include <gunrock/graph/gp.cuh>
#include <gunrock/graphio/csv.cuh>
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>
#include <gunrock/graphio/small_world.cuh>

namespace gunrock {
namespace graphio {

template <typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.template Use<std::string>(
      graph_prefix + "graph-type",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "",
      graph_prefix + " graph type, be one of market, csv, rgg,"
                     " rmat, grmat or smallworld",
      __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<std::string>(
      graph_prefix + "graph-file",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", graph_prefix + " graph file, empty points to STDIN", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "undirected",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether " + graph_prefix + " graph is undirected", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "random-edge-values",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false,
      "If true, " + graph_prefix +
          " graph edge values are randomly generated when missing. " +
          "If false, they are set to 1.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<float>(
      graph_prefix + "edge-value-range",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      64, "Range of edge values when randomly generated", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<float>(
      graph_prefix + "edge-value-min",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Minimum value of edge values when randomly generated", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "vertex-start-from-zero",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true,
      "Whether the vertex Id in " + graph_prefix +
          " starts from 0 instead of 1",
      __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<long>(
      graph_prefix + "edge-value-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Rand seed to generate edge values, default is time(NULL)", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<long long>(
      graph_prefix + "graph-nodes",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1 << 10, "Number of nodes", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<long long>(
      graph_prefix + "graph-edges",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      (1 << 10) * 48, "Number of edges", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<long long>(
      graph_prefix + "graph-scale",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      10, "Vertex scale", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<double>(
      graph_prefix + "graph-edgefactor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      48, "Edge factor", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<int>(
      graph_prefix + "graph-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Rand seed to generate the graph, default is time(NULL)", __FILE__,
      __LINE__));

  if (graph_prefix == "") {
    GUARD_CU(parameters.template Use<bool>(
        "64bit-VertexT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit VertexT", __FILE__, __LINE__));

    GUARD_CU(parameters.template Use<bool>(
        "64bit-SizeT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit SizeT", __FILE__, __LINE__));

    GUARD_CU(parameters.template Use<bool>(
        "64bit-ValueT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit ValueT", __FILE__, __LINE__));

    GUARD_CU(parameters.template Use<std::string>(
        "dataset",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        "", "Name of dataset, default value is set by graph reader / generator",
        __FILE__, __LINE__));
  }

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "remove-duplicate-edges",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to remove duplicate edges", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "remove-self-loops",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to remove self loops", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "read-from-binary",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true,
      "Whether to read a graph from binary file, if supported and file "
      "available",
      __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "store-to-binary",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to store the graph to binary file, if supported", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<std::string>(
      graph_prefix + "binary-prefix",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "Prefix to store a binary copy of the graph, default is the value of " +
          graph_prefix + "-graph-file",
      __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<bool>(
      graph_prefix + "sort-csr",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to sort CSR edges per vertex", __FILE__, __LINE__));

  GUARD_CU(market ::UseParameters(parameters, graph_prefix));
  GUARD_CU(rgg ::UseParameters(parameters, graph_prefix));
  GUARD_CU(small_world::UseParameters(parameters, graph_prefix));
  GUARD_CU(rmat ::UseParameters(parameters, graph_prefix));
  return retval;
}

/**
 * @brief Utility function to load input graph.
 *
 * @tparam EDGE_VALUE
 * @tparam INVERSE_GRAPH
 *
 * @param[in] args Command line arguments.
 * @param[in] csr_ref Reference to the CSR graph.
 *
 * \return int whether successfully loaded the graph (0 success, 1 error).
 */
template <typename GraphT>
cudaError_t LoadGraph(util::Parameters &parameters, GraphT &graph,
                      std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;
  std::string graph_type =
      parameters.Get<std::string>(graph_prefix + "graph-type");

  if (graph_type == "market")  // Matrix-market graph
  {
    GUARD_CU(market::Load(parameters, graph, graph_prefix));
  }

  else if (graph_type == "csv") // comma-separated values graph
  {
      parameters.Set("vertex-start-from-zero", false);
      GUARD_CU(csv::Load(parameters, graph, graph_prefix));
  }

  else if (graph_type == "rmat") {
    GUARD_CU(rmat::Load(parameters, graph, graph_prefix));
  }

  else if (graph_type == "rgg") {
    GUARD_CU(rgg::Load(parameters, graph, graph_prefix));
  }

  else if (graph_type == "smallworld") {
    GUARD_CU(small_world::Load(parameters, graph, graph_prefix));
  }

  else if (graph_type == "by-pass") {
  }

  else {
    return util::GRError(cudaErrorUnknown,
                         "Unspecified graph type " + graph_type, __FILE__,
                         __LINE__);
  }

  if ((graph.FLAG & gunrock::graph::HAS_CSR) &&
      parameters.Get<bool>("sort-csr")) {
    graph.csr().Sort();
  }

  //Histogram functions assume the input data is stored on host
  if (!parameters.Get<bool>("quiet") && parameters.Get<gunrock::util::Location>("mem-space")==gunrock::util::HOST) {
    typedef typename GraphT::SizeT SizeT;
    util::Array1D<SizeT, SizeT> histogram;
    GUARD_CU(graph::GetHistogram(graph, histogram));
    GUARD_CU(graph::PrintHistogram(graph, histogram));
    GUARD_CU(histogram.Release());
    util::PrintMsg("");
  }
  return retval;
}

}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
