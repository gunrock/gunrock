// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once

// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

namespace gunrock {
namespace app {
namespace sssp {

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
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference SSSP ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for each node
 * @param[in] src Source node where SSSP starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename GraphT>
void CPU_Reference(
    const    GraphT          &graph,
    typename GraphT::ValueT  *distances,
    typename GraphT::VertexT *preds,
    typename GraphT::VertexT  src,
    bool                      quiet,
    bool                      mark_preds)
{
    using namespace boost;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::CsrT    CsrT;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
            property <edge_weight_t, ValueT> > BGraphT;

    typedef typename graph_traits<BGraphT>::vertex_descriptor vertex_descriptor;
    typedef typename graph_traits<BGraphT>::edge_descriptor edge_descriptor;

    typedef std::pair<VertexT, VertexT> EdgeT;

    EdgeT   *edges = ( EdgeT*)malloc(sizeof( EdgeT) * graph.edges);
    ValueT *weight = (ValueT*)malloc(sizeof(ValueT) * graph.edges);

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
    // Perform SSSP
    //

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    if (mark_preds)
    {
        dijkstra_shortest_paths(g, s,
            predecessor_map(boost::make_iterator_property_map(
                p.begin(), get(boost::vertex_index, g))).distance_map(
                    boost::make_iterator_property_map(
                        d.begin(), get(boost::vertex_index, g))));
    }
    else
    {
        dijkstra_shortest_paths(g, s,
            distance_map(boost::make_iterator_property_map(
                d.begin(), get(boost::vertex_index, g))));
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    util::PrintMsg("CPU SSSP finished in " + std::to_string(elapsed)
        + " msec.", !quiet);

    typedef std::pair<VertexT, ValueT> PairT;
    PairT* sort_dist = new PairT[graph.nodes];
    typename graph_traits <BGraphT>::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].first  = (*vi);
        sort_dist[(*vi)].second = d[(*vi)];
    }
    std::stable_sort(
        sort_dist, sort_dist + graph.nodes,
        //RowFirstTupleCompare<Coo<Value, Value> >);
        [](const PairT &a, const PairT &b) -> bool
        {
            return a.first < b.first;
        });
    for (VertexT v = 0; v < graph.nodes; ++v)
        distances[v] = sort_dist[v].second;
    delete[] sort_dist; sort_dist = NULL;

    if (mark_preds)
    {
        typedef std::pair<VertexT, VertexT> VPairT;
        VPairT* sort_pred = new VPairT[graph.nodes];
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].first  = (*vi);
            sort_pred[(*vi)].second = p[(*vi)];
        }
        std::stable_sort(
            sort_pred, sort_pred + graph.nodes,
            //RowFirstTupleCompare< Coo<VertexId, VertexId> >);
            [](const VPairT &a, const VPairT &b) -> bool
            {
                return a.first < b.first;
            });
        for (VertexT v = 0; v < graph.nodes; ++v)
            preds[v] = sort_pred[v].second;
        delete[] sort_pred; sort_pred = NULL;
    }
}

template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
    typename GraphT::VertexT   src,
                     ValueT   *h_distances,
    typename GraphT::VertexT  *h_preds,
                     ValueT   *ref_distances = NULL,
    typename GraphT::VertexT  *ref_preds     = NULL,
                     bool      verbose       = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");
    bool mark_pred = parameters.Get<bool>("mark-pred");

    // Verify the result
    if (!quick && ref_distances != NULL)
    {
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            if (!util::isValid(ref_distances[v]))
                ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
        }

        util::PrintMsg("Distance Validity: ", !quiet, false);
        SizeT errors_num = util::CompareResults(
            h_distances, ref_distances,
            graph.nodes, true, quiet);
        if (errors_num > 0)
        {
            util::PrintMsg(
                std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        }
    }
    else if (!quick && ref_distances == NULL)
    {
        util::PrintMsg("Distance Validity: ", !quiet, false);
        SizeT errors_num = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            ValueT v_distance = h_distances[v];
            if (!util::isValid(v_distance))
                continue;
            SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
            SizeT e_end = e_start + num_neighbors;
            for (SizeT e = e_start; e < e_end; e++)
            {
                VertexT u = graph.CsrT::GetEdgeDest(e);
                ValueT u_distance = h_distances[u];
                ValueT e_value = graph.CsrT::edge_values[e];
                if (v_distance + e_value >= u_distance)
                    continue;
                errors_num ++;
                if (errors_num > 1)
                    continue;

                util::PrintMsg("FAIL: v[" + std::to_string(v)
                    + "] ("    + std::to_string(v_distance)
                    + ") + e[" + std::to_string(e)
                    + "] ("    + std::to_string(e_value)
                    + ") < u[" + std::to_string(u)
                    + "] ("    + std::to_string(u_distance) + ")", !quiet);
            }
        }
        if (errors_num > 0)
        {
            util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    }

    if (!quiet && verbose)
    {
        // Display Solution
        util::PrintMsg("First 40 distances of the GPU result:");
        DisplaySolution(h_distances, graph.nodes);
        if (ref_distances != NULL)
        {
            util::PrintMsg("First 40 distances of the reference CPU result.");
            DisplaySolution(ref_distances, graph.nodes);
        }
        util::PrintMsg("");
    }

    if (mark_pred && !quick)
    {
        util::PrintMsg("Predecessors Validity: ", !quiet, false);
        SizeT errors_num = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            VertexT pred          = h_preds[v];
            if (!util::isValid(pred) || v == src)
                continue;
            ValueT  v_distance    = h_distances[v];
            if (v_distance == util::PreDefinedValues<ValueT>::MaxValue)
                continue;
            ValueT  pred_distance = h_distances[pred];
            bool edge_found = false;
            SizeT edge_start = graph.CsrT::GetNeighborListOffset(pred);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(pred);

            for (SizeT e = edge_start; e < edge_start + num_neighbors; e++)
            {
                if (v == graph.CsrT::GetEdgeDest(e) &&
                    std::abs((pred_distance + graph.CsrT::edge_values[e]
                    - v_distance) * 1.0) < 1e-6)
                {
                    edge_found = true;
                    break;
                }
            }
            if (edge_found)
                continue;
            errors_num ++;
            if (errors_num > 1)
                continue;

            util::PrintMsg("FAIL: [" + std::to_string(pred)
                + "] ("    + std::to_string(pred_distance)
                + ") -> [" + std::to_string(v)
                + "] ("    + std::to_string(v_distance)
                + ") can't find the corresponding edge.", !quiet);
        }
        if (errors_num > 0)
        {
            util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    }

    if (!quiet && mark_pred && verbose)
    {
        util::PrintMsg("First 40 preds of the GPU result:");
        DisplaySolution(h_preds, graph.nodes);
        if (ref_preds != NULL)
        {
            util::PrintMsg("First 40 preds of the reference CPU result "
                "(could be different because the paths are not unique):");
            DisplaySolution(ref_preds, graph.nodes);
        }
        util::PrintMsg("");
    }

    return num_errors;
}

} // namespace sssp
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
