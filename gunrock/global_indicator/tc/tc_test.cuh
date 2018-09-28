// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * tc_test.cuh
 *
 * @brief Test related functions for TC
 */

#pragma once

// TODO: change to other includes, according to the problem
#ifdef BOOST_FOUND
    #include <boost/config.hpp>
    #include <boost/graph/graph_traits.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/property_map/property_map.hpp>
#else
    #include <queue>
    #include <vector>
    #include <utility>
#endif

namespace gunrock {
namespace app {
namespace tc {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length)
{
    if (length > 40)
        length = 40;

    util::PrintMsg("[", true, false);
    for (SizeT i = 0; i < length; ++i)
    {
        util::PrintMsg(std::to_string(i) + ":"
            + std::to_string(array[i]) + " ", true, false);
    }
    util::PrintMsg("]");
}

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SSSP
 */
template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
double CPU_Reference(
    const    GraphT          &graph,
    // TODO: add problem specific inputs and outputs, e.g.:
                     ValueT  *excess,
    typename GraphT::VertexT  src,
    bool                      quiet)
{
#ifdef BOOST_FOUND
    using namespace boost;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  GValueT;
    // TODO: change to other graph representation, if not using Csr
    typedef typename GraphT::CsrT    CsrT;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
            property <edge_weight_t, GValueT> > BGraphT;

    typedef typename graph_traits<BGraphT>::vertex_descriptor vertex_descriptor;
    typedef typename graph_traits<BGraphT>::edge_descriptor edge_descriptor;

    typedef std::pair<VertexT, VertexT> EdgeT;

    EdgeT   *edges = ( EdgeT*)malloc(sizeof( EdgeT) * graph.edges);
    GValueT *weight = (GValueT*)malloc(sizeof(GValueT) * graph.edges);

    for (VertexT v = 0; v < graph.nodes; ++v)
    {
        SizeT edge_start = graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
        for (SizeT e = 0; e < num_neighbors; ++e)
        {
            edges [e + edge_start] = EdgeT(v, graph.CsrT::GetEdgeDest(e + edge_start));
            weight[e + edge_start] = graph.CsrT::edge_values[e + edge_start];
        }
    }

    BGraphT g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<ValueT>            d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    typename property_map<BGraphT, vertex_index_t>::type
        indexmap = get(vertex_index, g);

    //
    // Perform Boost reference
    //

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    // TODO: use boost routine to do computation on CPU, e.g.:
    // dijkstra_shortest_paths(g, s,
    //        distance_map(boost::make_iterator_property_map(
    //            d.begin(), get(boost::vertex_index, g))));
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    // TODO: extract results on CPU, e.g.:
    // typedef std::pair<VertexT, ValueT> PairT;
    // PairT* sort_dist = new PairT[graph.nodes];
    // typename graph_traits <BGraphT>::vertex_iterator vi, vend;
    // for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    // {
    //    sort_dist[(*vi)].first  = (*vi);
    //    sort_dist[(*vi)].second = d[(*vi)];
    // }
    // std::stable_sort(
    //    sort_dist, sort_dist + graph.nodes,
    //    [](const PairT &a, const PairT &b) -> bool
    //    {
    //        return a.first < b.first;
    //    });
    // for (VertexT v = 0; v < graph.nodes; ++v)
    //    distances[v] = sort_dist[v].second;
    delete[] sort_dist; sort_dist = NULL;

    return elapsed;
#else
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    return elapsed;
#endif
}
/**
 * @brief Validation of TC results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */

template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
    // TODO: add problem specific data for validation, e.g.:
        typename GraphT::VertexT   src,
                    ValueT   *h_excess,
         ValueT   *ref_excess = NULL,
         bool      verbose       = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    // TODO: change to other representation, if not using CSR
    typedef typename GraphT::CsrT    CsrT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    return num_errors;
}

} // namespace tc
} // namespace app
} // namespace gunrock
