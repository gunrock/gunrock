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

cudaError_t UseParameters(
    util::Parameters &parameters,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;

    retval = parameters.Use<long long>(
        graph_prefix + "sw-nodes",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1 << 10,
        "Number of nodes",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<long long>(
        graph_prefix + "sw-scale",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        10,
        "Vertex scale",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<double>(
        graph_prefix + "sw-p",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0.00,
        "p",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<long long>(
        graph_prefix + "sw-k",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        6,
        "k",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<int>(
        graph_prefix + "sw-seed",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "rand seed to generate the small world graph, default is time(NULL)",
        __FILE__, __LINE__);
    if (retval) return retval;

    return retval;
}

template <typename GraphT>
cudaError_t Build(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::CooT    CooT;

#ifndef BOOST_FOUND
    retval = util::GRError("Small world graph generator requires boost, but it's not found", __FILE__, __LINE__);
    return retval;

#else
    //using namespace boost;
    bool quiet = parameters.Get<bool>("quiet");
    //bool undirected = !parameters.Get<bool>(graph_prefix + "directed");
    SizeT scale = parameters.Get<SizeT>(graph_prefix + "rgg-scale");
    SizeT num_nodes = 1 << scale;
    if (!parameters.UseDefault(graph_prefix + "sw-nodes"))
        num_nodes = parameters.Get<SizeT>(graph_prefix + "sw-nodes");

    double p = parameters.Get<double>(graph_prefix + "sw-p");
    SizeT k = parameters.Get<SizeT>(graph_prefix + "sw-k");

    double edge_value_range = parameters.Get<double>(graph_prefix + "edge-value-range");
    double edge_value_min   = parameters.Get<double>(graph_prefix + "edge-value-min");

    int seed = time(NULL);
    if (parameters.UseDefault(graph_prefix + "sw-seed"))
        seed = parameters.Get<int>(graph_prefix + "sw-seed");
    unsigned int seed_ = seed + 2244;
    Engine engine(seed_);
    Distribution distribution(0.0, 1.0);

    if (!quiet)
        util::PrintMsg("Generating Small World " + graph_prefix +
            "graph, p = " + std::to_string(p) +
            ", k = " + std::to_string(k) +
            ", seed = " + std::to_string(seed) + "...");
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    util::Location target = util::HOST;
    typedef boost::adjacency_list<> BGraph;
    typedef boost::small_world_iterator<boost::minstd_rand, BGraph> SWGen;

    boost::minstd_rand gen;
    BGraph g(SWGen(gen, num_nodes, k, p), SWGen(), num_nodes);

    boost::property_map<BGraph, boost::vertex_index_t>::type vi = boost::get(boost::vertex_index, g);

    SizeT num_edges = boost::num_edges(g);
    if (retval = graph.CooT::Allocate(num_nodes, num_edges, target))
        return retval;

    boost::graph_traits<BGraph>::edge_iterator ei;
    boost::graph_traits<BGraph>::edge_iterator ei_end;
    SizeT edge_counter = 0;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++)
    {
        VertexT v = vi[boost::source(*ei, g)];
        VertexT u = vi[boost::target(*ei, g)];
        graph.CooT::edge_pairs[edge_counter].x = v;
        graph.CooT::edge_pairs[edge_counter].y = u;
        if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
        {
            graph.CooT::edge_values[edge_counter] = distribution(engine) * edge_value_range + edge_value_min;
        }
        edge_counter ++;
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    if (!quiet)
    {
        util::PrintMsg("Small world generated in " +
            std::to_string(elapsed) + " ms.");
    }

    return retval;
#endif
}

template <typename GraphT, bool COO_SWITCH>
struct CooSwitch
{
static cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    retval = Build(parameters, graph, graph_prefix);
    if (retval) return retval;
    retval = graph.FromCoo(graph, true);
    return retval;
}
};

template <typename GraphT>
struct CooSwitch<GraphT, false>
{
cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    typedef graph::Csr<typename GraphT::VertexT,
        typename GraphT::SizeT,
        typename GraphT::ValueT,
        GraphT::FLAG | graph::HAS_COO, GraphT::cudaHostRegisterFlag> CooT;
    cudaError_t retval = cudaSuccess;

    CooT coo;
    retval = Build(parameters, coo, graph_prefix);
    if (retval) return retval;
    retval = graph.FromCoo(coo);
    if (retval) return retval;
    retval = coo.Release();
    return retval;
}
};

template <typename GraphT>
cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    return CooSwitch<GraphT, (GraphT::FLAG & graph::HAS_COO) != 0>
        ::Load(parameters, graph, graph_prefix);
}

} // namespace small_world
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
