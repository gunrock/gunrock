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
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>
#include <gunrock/graphio/small_world.cuh>

namespace gunrock {
namespace graphio {

cudaError_t UseParameters(util::Parameters &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<std::string>(
      graph_prefix + "graph-type",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "",
      graph_prefix + " graph type, be one of market, rgg,"
                     " rmat, grmat or smallworld",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      graph_prefix + "graph-file",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", graph_prefix + " graph file, empty points to STDIN", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "undirected",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether " + graph_prefix + " graph is undirected", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "random-edge-values",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false,
      "If true, " + graph_prefix +
          " graph edge values are randomly generated when missing. " +
          "If false, they are set to 1.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      graph_prefix + "edge-value-range",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      64, "Range of edge values when randomly generated", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      graph_prefix + "edge-value-min",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Minimum value of edge values when randomly generated", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "vertex-start-from-zero",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true,
      "Whether the vertex Id in " + graph_prefix +
          " starts from 0 instead of 1",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<long>(
      graph_prefix + "edge-value-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Rand seed to generate edge values, default is time(NULL)", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<long long>(
      graph_prefix + "graph-nodes",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1 << 10, "Number of nodes", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<long long>(
      graph_prefix + "graph-edges",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      (1 << 10) * 48, "Number of edges", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<long long>(
      graph_prefix + "graph-scale",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      10, "Vertex scale", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      graph_prefix + "graph-edgefactor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      48, "Edge factor", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      graph_prefix + "graph-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Rand seed to generate the graph, default is time(NULL)", __FILE__,
      __LINE__));

  if (graph_prefix == "") {
    GUARD_CU(parameters.Use<bool>(
        "64bit-VertexT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit VertexT", __FILE__, __LINE__));

    GUARD_CU(parameters.Use<bool>(
        "64bit-SizeT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit SizeT", __FILE__, __LINE__));

    GUARD_CU(parameters.Use<bool>(
        "64bit-ValueT",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false, "Whether to use 64-bit ValueT", __FILE__, __LINE__));

    GUARD_CU(parameters.Use<std::string>(
        "dataset",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        "", "Name of dataset, default value is set by graph reader / generator",
        __FILE__, __LINE__));
  }

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "remove-duplicate-edges",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to remove duplicate edges", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "remove-self-loops",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to remove self loops", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "read-from-binary",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true,
      "Whether to read a graph from binary file, if supported and file "
      "available",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      graph_prefix + "store-to-binary",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to store the graph to binary file, if supported", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      graph_prefix + "binary-prefix",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "Prefix to store a binary copy of the graph, default is the value of " +
          graph_prefix + "-graph-file",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
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

  else if (graph_type == "rmat") {
    GUARD_CU(rmat::Load(parameters, graph, graph_prefix));
  }

  /*else if (graph_type == "rmat" || graph_type == "grmat" || graph_type ==
  "metarmat")  // R-MAT graph
  {
      if (!args.CheckCmdLineFlag("quiet"))
      {
          printf("Generating R-MAT graph ...\n");
      }
      // parse R-MAT parameters
      SizeT rmat_nodes = 1 << 10;
      SizeT rmat_edges = 1 << 10;
      SizeT rmat_scale = 10;
      SizeT rmat_edgefactor = 48;
      double rmat_a = 0.57;
      double rmat_b = 0.19;
      double rmat_c = 0.19;
      double rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
      double rmat_vmin = 1;
      double rmat_vmultipiler = 64;
      int rmat_seed = -1;

      args.GetCmdLineArgument("rmat_scale", rmat_scale);
      rmat_nodes = 1 << rmat_scale;
      args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
      args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
      rmat_edges = rmat_nodes * rmat_edgefactor;
      args.GetCmdLineArgument("rmat_edges", rmat_edges);
      args.GetCmdLineArgument("rmat_a", rmat_a);
      args.GetCmdLineArgument("rmat_b", rmat_b);
      args.GetCmdLineArgument("rmat_c", rmat_c);
      rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
      args.GetCmdLineArgument("rmat_d", rmat_d);
      args.GetCmdLineArgument("rmat_seed", rmat_seed);
      args.GetCmdLineArgument("rmat_vmin", rmat_vmin);
      args.GetCmdLineArgument("rmat_vmultipiler", rmat_vmultipiler);

      std::vector<int> temp_devices;
      if (args.CheckCmdLineFlag("device"))  // parse device list
      {
          args.GetCmdLineArguments<int>("device", temp_devices);
          num_gpus = temp_devices.size();
      }
      else  // use single device with index 0
      {
          num_gpus = 1;
          int gpu_idx;
          util::GRError(cudaGetDevice(&gpu_idx),
              "cudaGetDevice failed", __FILE__, __LINE__);
          temp_devices.push_back(gpu_idx);
      }
      int *gpu_idx = new int[temp_devices.size()];
      for (int i=0; i<temp_devices.size(); i++)
          gpu_idx[i] = temp_devices[i];

      // put everything into mObject info
      info["rmat_a"] = rmat_a;
      info["rmat_b"] = rmat_b;
      info["rmat_c"] = rmat_c;
      info["rmat_d"] = rmat_d;
      info["rmat_seed"] = rmat_seed;
      info["rmat_scale"] = (int64_t)rmat_scale;
      info["rmat_nodes"] = (int64_t)rmat_nodes;
      info["rmat_edges"] = (int64_t)rmat_edges;
      info["rmat_edgefactor"] = (int64_t)rmat_edgefactor;
      info["rmat_vmin"] = rmat_vmin;
      info["rmat_vmultipiler"] = rmat_vmultipiler;
      //can use to_string since c++11 is required, niiiice.
      file_stem = "rmat_" +
          (args.CheckCmdLineFlag("rmat_scale") ?
              ("n" + std::to_string(rmat_scale)) : std::to_string(rmat_nodes))
         + "_" + (args.CheckCmdLineFlag("rmat_edgefactor") ?
              ("e" + std::to_string(rmat_edgefactor)) :
  std::to_string(rmat_edges)); info["dataset"] = file_stem;

      util::CpuTimer cpu_timer;
      cpu_timer.Start();

      // generate R-MAT graph
      if (graph_type == "rmat")
      {
          if (graphio::rmat::BuildRmatGraph<EDGE_VALUE>(
              rmat_nodes,
              rmat_edges,
              csr_ref,
              info["undirected"].get_bool(),
              rmat_a,
              rmat_b,
              rmat_c,
              rmat_d,
              rmat_vmultipiler,
              rmat_vmin,
              rmat_seed,
              args.CheckCmdLineFlag("quiet")) != 0)
          {
              return 1;
          }
      } else if (graph_type == "grmat")
      {
          if (graphio::grmat::BuildRmatGraph<EDGE_VALUE>(
              rmat_nodes,
              rmat_edges,
              csr_ref,
              info["undirected"].get_bool(),
              rmat_a,
              rmat_b,
              rmat_c,
              rmat_d,
              rmat_vmultipiler,
              rmat_vmin,
              rmat_seed,
              args.CheckCmdLineFlag("quiet"),
              temp_devices.size(),
              gpu_idx) != 0)
          {
              return 1;
          }
      } else // must be metarmat
      {
          if (graphio::grmat::BuildMetaRmatGraph<EDGE_VALUE>(
              rmat_nodes,
              rmat_edges,
              csr_ref,
              info["undirected"].get_bool(),
              rmat_a,
              rmat_b,
              rmat_c,
              rmat_d,
              rmat_vmultipiler,
              rmat_vmin,
              rmat_seed,
              args.CheckCmdLineFlag("quiet"),
              temp_devices.size(),
              gpu_idx) != 0)
          {
              return 1;
          }
      }

      cpu_timer.Stop();
      float elapsed = cpu_timer.ElapsedMillis();
      delete[] gpu_idx; gpu_idx = NULL;

      if (!args.CheckCmdLineFlag("quiet"))
      {
          printf("R-MAT graph generated in %.3f ms, "
                 "a = %.3f, b = %.3f, c = %.3f, d = %.3f\n",
                 elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
      }
  }*/
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

  if (!parameters.Get<bool>("quiet")) {
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
