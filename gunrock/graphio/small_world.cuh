// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * grmat.cuh
 *
 * @brief gpu based R-MAT Graph Construction Routines
 */

#pragma once

#ifdef BOOST_FOUND
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/small_world_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#endif

#include <gunrock/util/parameters.h>

namespace gunrock {
namespace graphio {
namespace small_world {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

cudaError_t UseParameters(util::Parameters &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<double>(
      graph_prefix + "sw-p",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.00, "p", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<long long>(
      graph_prefix + "sw-k",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      6, "k", __FILE__, __LINE__));

  return retval;
}

template <typename GraphT>
cudaError_t Build(util::Parameters &parameters, GraphT &graph,
                  std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CooT CooT;

#ifndef BOOST_FOUND
  retval = util::GRError(
      "Small world graph generator requires boost, but it's not found",
      __FILE__, __LINE__);
  return retval;

#else
  // using namespace boost;
  bool quiet = parameters.Get<bool>("quiet");
  std::string dataset = "smallworld_";
  // bool undirected = !parameters.Get<bool>(graph_prefix + "directed");
  SizeT scale = parameters.Get<SizeT>(graph_prefix + "graph-scale");
  SizeT num_nodes = 0;
  if (!parameters.UseDefault(graph_prefix + "graph-nodes")) {
    num_nodes = parameters.Get<SizeT>(graph_prefix + "graph-nodes");
    dataset = dataset + std::to_string(num_nodes) + "_";
  } else {
    num_nodes = 1 << scale;
    dataset = dataset + "n" + std::to_string(scale) + "_";
  }
  double p = parameters.Get<double>(graph_prefix + "sw-p");
  SizeT k = parameters.Get<SizeT>(graph_prefix + "sw-k");
  dataset = dataset + "_k" + std::to_string(k) + "_p" + std::to_string(p);
  if (parameters.UseDefault("dataset"))
    parameters.Set<std::string>("dataset", dataset);

  bool random_edge_values =
      parameters.Get<bool>(graph_prefix + "random-edge-values");
  double edge_value_range =
      parameters.Get<double>(graph_prefix + "edge-value-range");
  double edge_value_min =
      parameters.Get<double>(graph_prefix + "edge-value-min");

  int seed = time(NULL);
  if (parameters.UseDefault(graph_prefix + "graph-seed"))
    seed = parameters.Get<int>(graph_prefix + "graph-seed");
  unsigned int seed_ = seed + 2244;
  Engine engine(seed_);
  Distribution distribution(0.0, 1.0);

  util::PrintMsg("Generating Small World " + graph_prefix + "graph, p = " +
                     std::to_string(p) + ", k = " + std::to_string(k) +
                     ", seed = " + std::to_string(seed) + "...",
                 !quiet);
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  util::Location target = util::HOST;
  typedef boost::adjacency_list<> BGraph;
  typedef boost::small_world_iterator<boost::minstd_rand, BGraph> SWGen;

  boost::minstd_rand gen;
  BGraph g(SWGen(gen, num_nodes, k, p), SWGen(), num_nodes);

  boost::property_map<BGraph, boost::vertex_index_t>::type vi =
      boost::get(boost::vertex_index, g);

  SizeT num_edges = boost::num_edges(g);
  GUARD_CU(graph.CooT::Allocate(num_nodes, num_edges, target));

  boost::graph_traits<BGraph>::edge_iterator ei;
  boost::graph_traits<BGraph>::edge_iterator ei_end;
  SizeT edge_counter = 0;
  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
    VertexT v = vi[boost::source(*ei, g)];
    VertexT u = vi[boost::target(*ei, g)];
    graph.CooT::edge_pairs[edge_counter].x = v;
    graph.CooT::edge_pairs[edge_counter].y = u;
    if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
      if (random_edge_values) {
        graph.CooT::edge_values[edge_counter] =
            distribution(engine) * edge_value_range + edge_value_min;
      } else {
        graph.CooT::edge_values[edge_counter] = 1;
      }
    }
    edge_counter++;
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  util::PrintMsg("Small world generated in " + std::to_string(elapsed) + " ms.",
                 !quiet);

  return retval;
#endif
}

template <typename GraphT, bool COO_SWITCH>
struct CooSwitch {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(Build(parameters, graph, graph_prefix));

    bool remove_self_loops =
        parameters.Get<bool>(graph_prefix + "remove-self-loops");
    bool remove_duplicate_edges =
        parameters.Get<bool>(graph_prefix + "remove-duplicate-edges");
    bool quiet = parameters.Get<bool>("quiet");

    if (remove_self_loops && remove_duplicate_edges) {
      GUARD_CU(graph.RemoveSelfLoops_DuplicateEdges(
          gunrock::graph::BY_ROW_ASCENDING, util::HOST, 0, quiet));
    } else if (remove_self_loops) {
      GUARD_CU(
          graph.RemoveSelfLoops(gunrock::graph::BY_ROW_ASCENDING, util::HOST));
    } else if (remove_duplicate_edges) {
      GUARD_CU(graph.RemoveDuplicateEdges(gunrock::graph::BY_ROW_ASCENDING,
                                          util::HOST));
    }

    GUARD_CU(graph.FromCoo(graph, util::HOST, 0, quiet, true));
    return retval;
  }
};

template <typename GraphT>
struct CooSwitch<GraphT, false> {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    typedef graph::Coo<typename GraphT::VertexT, typename GraphT::SizeT,
                       typename GraphT::ValueT, GraphT::FLAG | graph::HAS_COO,
                       GraphT::cudaHostRegisterFlag>
        CooT;
    cudaError_t retval = cudaSuccess;

    CooT coo;
    GUARD_CU(Build(parameters, coo, graph_prefix));

    bool remove_self_loops =
        parameters.Get<bool>(graph_prefix + "remove-self-loops");
    bool remove_duplicate_edges =
        parameters.Get<bool>(graph_prefix + "remove-duplicate-edges");
    bool quiet = parameters.Get<bool>("quiet");
    if (remove_self_loops && remove_duplicate_edges) {
      GUARD_CU(coo.RemoveSelfLoops_DuplicateEdges(
          gunrock::graph::BY_ROW_ASCENDING, util::HOST, 0, quiet));
    } else if (remove_self_loops) {
      GUARD_CU(coo.RemoveSelfLoops(gunrock::graph::BY_ROW_ASCENDING, util::HOST,
                                   0, quiet));
    } else if (remove_duplicate_edges) {
      GUARD_CU(coo.RemoveDuplicateEdges(gunrock::graph::BY_ROW_ASCENDING,
                                        util::HOST, 0, quiet));
    }

    GUARD_CU(graph.FromCoo(coo, util::HOST, 0, quiet, false));
    GUARD_CU(coo.Release());
    return retval;
  }
};

template <typename GraphT>
cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                 std::string graph_prefix = "") {
  return CooSwitch<GraphT, (GraphT::FLAG & graph::HAS_COO) != 0>::Load(
      parameters, graph, graph_prefix);
}

}  // namespace small_world
}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
