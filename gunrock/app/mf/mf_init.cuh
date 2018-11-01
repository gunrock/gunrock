// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_d_spatial.cuh
 *
 * @brief Device Spatial helpers for geolocation app
 */

//#define debug_aml(a...) printf(a)
#define debug_aml(a...)

#pragma once

namespace gunrock {
namespace app {
namespace mf {


template <typename GraphT, typename VertexT>
__device__ __host__ void init_reverse(GraphT &graph, VertexT* reverse)
{
    typedef typename GraphT::CsrT CsrT;

    for (auto u = 0; u < graph.nodes; ++u)
    {
        auto e_start = graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e)
        {
            auto v = graph.CsrT::GetEdgeDest(e);
            auto f_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors2 = 
                graph.CsrT::GetNeighborListLength(v);
            auto f_end = f_start + num_neighbors2;
            for (auto f = f_start; f < f_end; ++f)
            {
                auto z = graph.CsrT::GetEdgeDest(f);
                if (z == u)
                {
                    reverse[e] = f;
                    reverse[f] = e;
                    break;
                }
            }
        }
    }
}


template <typename GraphT>
__device__ __host__ void correct_capacity_for_undirected_graph(
        GraphT &undirected_graph, 
        GraphT &directed_graph)
{
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::ValueT ValueT;

    // Correct capacity values on reverse edges
    for (auto u = 0; u < undirected_graph.nodes; ++u)
    {
        auto e_start = undirected_graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = undirected_graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        debug_aml("vertex %d\nnumber of neighbors %d", u, 
                num_neighbors);
        for (auto e = e_start; e < e_end; ++e)
        {
            undirected_graph.CsrT::edge_values[e] = (ValueT)0;
            auto v = undirected_graph.CsrT::GetEdgeDest(e);
            // Looking for edge u->v in directed graph
            auto f_start = directed_graph.CsrT::GetNeighborListOffset(u);
            auto num_neighbors2 = 
                directed_graph.CsrT::GetNeighborListLength(u);
            auto f_end = f_start + num_neighbors2;
            for (auto f = f_start; f < f_end; ++f)
            {
                auto z = directed_graph.CsrT::GetEdgeDest(f);
                if (z == v and directed_graph.CsrT::edge_values[f] > 0)
                {
                    undirected_graph.CsrT::edge_values[e]  = 
                        directed_graph.CsrT::edge_values[f];
                    debug_aml("edge (%d, %d) cap = %lf\n", u, v, \
                            undirected_graph.CsrT::edge_values[e]);
                    break;
                }
            }
        }
    }
}

}
}
}


