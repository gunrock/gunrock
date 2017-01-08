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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/small_world_generator.hpp>
#include <boost/random/linear_congruential.hpp>

namespace gunrock {
namespace graphio {
namespace small_world {

template <bool WITH_EDGE_VALUES, typename VertexId, typename SizeT, typename Value>
cudaError_t BuildSWGraph(
    SizeT sw_nodes,
    Csr<VertexId, SizeT, Value> &graph,
    SizeT k,
    double p,
    bool undirected,
    double vmultipiler = 1.00,
    double vmin = 1.00,
    int seed = -1,
    bool quiet = false)
{
    using namespace boost;
    cudaError_t retval = cudaSuccess;

    //typedef adjacency_list<vecS, vecS, bidirectionalS, property<vertex_index_t, VertexId>,
    //    property<edge_index_t, SizeT> > BGraph;
    typedef adjacency_list<> BGraph;
    typedef small_world_iterator<minstd_rand, BGraph> SWGen;
    //typedef graph_traits<Graph>::vertex_descriptor Vertex;

    minstd_rand gen;
    BGraph g(SWGen(gen, sw_nodes, k, p), SWGen(), sw_nodes);

    property_map<BGraph, vertex_index_t>::type vi = get(vertex_index, g);

    SizeT sw_edges = num_edges(g);
    graph. template FromScratch<WITH_EDGE_VALUES, false>(sw_nodes, undirected ? sw_edges * 2 : sw_edges);
    SizeT *edge_counters = new SizeT[sw_nodes + 1];
    for (SizeT v=0; v<sw_nodes + 1; v++)
        edge_counters[v] = 0;

    graph_traits<BGraph>::edge_iterator ei;
    graph_traits<BGraph>::edge_iterator ei_end;
    for (tie(ei, ei_end) = edges(g); ei != ei_end; ei++)
    {
        edge_counters[vi[source(*ei, g)]] ++;
        if (undirected)
            edge_counters[vi[target(*ei, g)]] ++;
    }
    graph.row_offsets[0] = 0;
    for (SizeT v=0; v<sw_nodes; v++)
    {
        graph.row_offsets[v+1] = graph.row_offsets[v] + edge_counters[v];
        edge_counters[v] = 0;
    }
    for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ei++)
    {
        //Vertex bv = source(*ei, g);
        //Vertex bu = target(*ei, g);
        VertexId v = vi[source(*ei, g)];
        VertexId u = vi[target(*ei, g)];
        graph.column_indices[graph.row_offsets[v] + edge_counters[v]] = u;
        edge_counters[v] ++;
        if (undirected)
        {
            graph.column_indices[graph.row_offsets[u] + edge_counters[u]] = v;
            edge_counters[u] ++;
        }
    }

    delete[] edge_counters; edge_counters = NULL;
    return retval;    
}

} // namespace small_world
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:


