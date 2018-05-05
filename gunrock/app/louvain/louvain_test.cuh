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

#include <map>
#include <unordered_map>
#include <set>

namespace gunrock {
namespace app {
namespace louvain {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the community detection result (i.e. communities of vertices)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] community for each node.
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
 * Louvain Testing Routines
 *****************************************************************************/

template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
ValueT Get_Modularity(
    const GraphT &graph,
    typename GraphT::VertexT *communities = NULL)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT nodes = graph.nodes;
    ValueT *w_v2 = new ValueT[nodes];
    ValueT *w_c  = new ValueT[nodes];
    ValueT m2 = 0;
    ValueT w_in = 0;

    for (VertexT v = 0; v < nodes; v++)
    {
        w_v2[v] = 0;
        w_c [v] = 0;
    }

    //#pragma omp parallel for //reduction(+:m)
    for (VertexT v = 0; v < nodes; v++)
    {
        SizeT start_e = graph.GetNeighborListOffset(v);
        SizeT degree  = graph.GetNeighborListLength(v);
        VertexT c_v   = (communities == NULL) ? v : communities[v];

        for (SizeT k = 0; k < degree; k++)
        {
            SizeT   e = start_e + k;
            VertexT u = graph.GetEdgeDest(e);
            ValueT  w = graph.edge_values[e];
            w_v2[v] += w;
            w_c[c_v] += w;

            VertexT c_u = (communities == NULL) ? u : communities[u];
            if (c_v != c_u) continue;
            w_in += w;
        }
    }

    ValueT q = 0;
    for (VertexT v = 0; v < nodes; v++)
    {
        m2 += w_v2[v];
        if (w_c[v] != 0)
            q += w_c[v] * w_c[v];
    }

    delete[] w_v2; w_v2 = NULL;
    delete[] w_c ; w_c  = NULL;
    return (w_in - q/m2)/m2;

    /*
    q = 0;
    ValueT w1 = 0, w2 = 0;
    //#pragma omp parallel for //reduction(+:q)
    for (VertexT v = 0; v < nodes; v++)
    {
        VertexT comm_v = (communities == NULL) ? v : communities[v];
        q_v[v] = 0;
        SizeT start_e = graph.GetNeighborListOffset(v);
        SizeT degree  = graph.GetNeighborListLength(v);
        ValueT w_v2_v = w_v2[v];
        std::unordered_map<VertexT, ValueT> w_v2v;

        for (SizeT k = 0; k < degree; k++)
        {
            SizeT   e = start_e + k;
            VertexT u = graph.GetEdgeDest(e);
            if (comm_v != ((communities == NULL) ? u : communities[u]))
                continue;
            ValueT  w = graph.edge_values[e];
            auto it = w_v2v.find(u);
            if (it == w_v2v.end())
                w_v2v[u] = w;
            else
                it -> second += w;
        }

        auto &comm = comms[comm_v];
        for (auto it = comm.begin(); it != comm.end(); it++)
        {
            VertexT u = *it;
            auto it2 = w_v2v.find(u);
            ValueT  w = 0;
            if (it2 != w_v2v.end())
                w = w_v2v[u];
            w1 += w;
            w2 += w_v2_v * w_v2[u];
            q_v[v] += (w - w_v2_v * w_v2[u] / m);
        }
        //q += q_v;
        w_v2v.clear();
    }

    for (VertexT v = 0; v < nodes; v++)
        q += q_v[v];
    util::PrintMsg("w1 = " + std::to_string(w1) +
        + ", w2 = " + std::to_string(w2)
        + ", w_c^2 / m = " + std::to_string(w1 - q));
    q /= m;

    delete[] q_v ; q_v  = NULL;
    delete[] w_2v; w_2v = NULL;
    delete[] w_v2; w_v2 = NULL;
    return q;
    */
}

/**
 * @brief Simple CPU-based reference Louvain Community Detection implementation
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   parameters    Input parameters
 * @param[in]   graph         Input graph
 * @param[out]  communities   Community IDs for each vertex
 * \return      double        Time taken for the Louvain implementation
 */
template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
double CPU_Reference(
    util::Parameters         &parameters,
             GraphT          &graph,
    typename GraphT::VertexT *communities,
    std::vector<std::vector<typename GraphT::VertexT>* > *pass_communities = NULL,
    std::vector<GraphT*> *pass_graphs = NULL)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;

    VertexT max_passes = parameters.Get<VertexT>("max-passes");
    VertexT max_iters  = parameters.Get<VertexT>("max-iters");
    bool    pass_stats = parameters.Get<bool   >("pass-stats");
    bool    iter_stats = parameters.Get<bool   >("iter-stats");
    ValueT  pass_gain_threshold = parameters.Get<ValueT>("pass-th");
    ValueT  iter_gain_threshold = parameters.Get<ValueT>("iter-th");

    bool has_pass_communities = false;
    if (pass_communities != NULL)
        has_pass_communities = true;
    else
        pass_communities = new std::vector<std::vector<VertexT>* >;
    pass_communities -> clear();
    bool has_pass_graphs = false;
    if (pass_graphs != NULL)
        has_pass_graphs = true;

    ValueT q = Get_Modularity(graph);
    ValueT  *w_v2c          = new ValueT [graph.nodes];
    VertexT *neighbor_comms = new VertexT[graph.nodes];
    VertexT *comm_convert   = new VertexT[graph.nodes];
    ValueT  *w_v2self       = new ValueT [graph.nodes];
    ValueT  *w_v2           = new ValueT [graph.nodes];
    ValueT  *w_c2           = new ValueT [graph.nodes];
    typedef std::pair<VertexT, ValueT> pairT;
    std::vector<pairT> *w_c2c = new std::vector<pairT>[graph.nodes];

    GraphT temp_graph;
    temp_graph.Allocate(graph.nodes, graph.edges, util::HOST);
    auto &temp_row_offsets    = temp_graph.CsrT::row_offsets;
    auto &temp_column_indices = temp_graph.CsrT::column_indices;
    auto &temp_edge_values    = temp_graph.CsrT::edge_values;

    auto c_graph = &graph;
    auto n_graph = c_graph;
    n_graph = NULL;

    ValueT m2 = 0;
    for (SizeT e = 0; e < graph.edges; e++)
        m2 += graph.CsrT::edge_values[e];
    for (SizeT v = 0; v < graph.nodes; v++)
        w_v2c[v] = util::PreDefinedValues<ValueT>::InvalidValue;

    util::CpuTimer cpu_timer, pass_timer, iter_timer;
    cpu_timer.Start();

    int pass_num = 0;
    while (pass_num < max_passes)
    {
        // Pass initialization
        if (pass_stats)
            pass_timer.Start();
        if (iter_stats)
            iter_timer.Start();

        auto &current_graph = *c_graph;
        SizeT nodes = current_graph.nodes;
        auto c_communities = new std::vector<VertexT>;
        auto &current_communities = *c_communities;
        current_communities.reserve(nodes);

        for (VertexT v = 0; v < nodes; v++)
        {
            current_communities[v] = v;
            w_v2[v] = 0;
            w_v2self[v] = 0;
            SizeT start_e = current_graph.GetNeighborListOffset(v);
            SizeT degree  = current_graph.GetNeighborListLength(v);

            for (SizeT k = 0; k < degree; k++)
            {
                SizeT   e = start_e + k;
                VertexT u = current_graph.GetEdgeDest(e);
                ValueT  w = current_graph.edge_values[e];
                w_v2[v] += w;
                if (u == v)
                    w_v2self[v] += w;
            }
            w_c2[v] = w_v2[v];
        }
        if (iter_stats)
            iter_timer.Stop();
        util::PrintMsg("pass " + std::to_string(pass_num)
            + ", pre-iter, elapsed = "
            + std::to_string(iter_timer.ElapsedMillis()), iter_stats);

        // Modulation Optimization
        int iter_num = 0;
        ValueT pass_gain = 0;
        while (iter_num < max_iters)
        {
            if (iter_stats)
                iter_timer.Start();
            ValueT iter_gain = 0;
            for (VertexT v = 0; v < nodes; v++)
            {
                SizeT start_e = current_graph.GetNeighborListOffset(v);
                SizeT degree  = current_graph.GetNeighborListLength(v);

                VertexT num_neighbor_comms = 0;
                for (SizeT k = 0; k < degree; k++)
                {
                    SizeT   e = start_e + k;
                    VertexT u = current_graph.GetEdgeDest(e);
                    ValueT  w = current_graph.edge_values[e];
                    VertexT c = current_communities[u];

                    if (!util::isValid(w_v2c[c]))
                    {
                        w_v2c[c] = w;
                        neighbor_comms[num_neighbor_comms] = c;
                        num_neighbor_comms ++;
                    } else
                        w_v2c[c] += w;
                }

                VertexT org_comm = current_communities[v];
                VertexT new_comm = org_comm;
                ValueT  w_v2c_org = 0;
                if (util::isValid(w_v2c[org_comm]))
                    w_v2c_org = w_v2c[org_comm];
                ValueT  w_v2_v   = w_v2[v];
                ValueT  gain_base = w_v2self[v] - w_v2c_org
                    - (w_v2[v] - w_c2[org_comm]) * w_v2_v / m2;
                ValueT  max_gain = 0;

                for (VertexT i = 0; i < num_neighbor_comms; i ++)
                {
                    VertexT c = neighbor_comms[i];
                    if (c == org_comm)
                    {
                        w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
                        continue;
                    }
                    ValueT  w_v2c_c = w_v2c[c];
                    w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
                    ValueT gain = gain_base + w_v2c_c - w_c2[c] * w_v2_v / m2;

                    if (gain > max_gain)
                    {
                        max_gain = gain;
                        new_comm = c;
                    }
                }

                if (max_gain > 0 && new_comm != current_communities[v])
                {
                    iter_gain += max_gain;
                    current_communities[v] = new_comm;
                    w_c2[new_comm] += w_v2[v];
                    w_c2[org_comm] -= w_v2[v];
                }
            }

            iter_num ++;
            iter_gain *= 2;
            iter_gain /= m2;
            q += iter_gain;
            pass_gain += iter_gain;
            if (iter_stats)
                iter_timer.Stop();
            util::PrintMsg("pass " + std::to_string(pass_num)
                + ", iter " + std::to_string(iter_num)
                + ", q = " + std::to_string(q)
                + ", iter_gain = " + std::to_string(iter_gain)
                + ", pass_gain = " + std::to_string(pass_gain)
                + ", elapsed = "
                + std::to_string(iter_timer.ElapsedMillis()), iter_stats);
            if (iter_gain < iter_gain_threshold)
                break;
        }

        // Community Aggregation
        if (iter_stats)
            iter_timer.Start();
        VertexT num_comms = 0;
        for (VertexT v = 0; v < nodes; v++)
            comm_convert[v] = 0;
        for (VertexT v = 0; v < nodes; v++)
            comm_convert[current_communities[v]] = 1;
        for (VertexT v = 0; v < nodes; v++)
        {
            if (comm_convert[v] == 0)
                continue;
            comm_convert[v] = num_comms;
            num_comms ++;
        }

        for (VertexT v = 0; v < nodes; v++)
            current_communities[v] = comm_convert[current_communities[v]];
        pass_communities -> push_back(c_communities);

        for (VertexT v = 0; v < nodes; v++)
        {
            SizeT start_e = current_graph.GetNeighborListOffset(v);
            SizeT degree  = current_graph.GetNeighborListLength(v);
            VertexT comm_v = current_communities[v];
            auto &w_c2c_c = w_c2c[comm_v];

            for (SizeT k = 0; k < degree; k++)
            {
                SizeT   e = start_e + k;
                VertexT u = current_graph.GetEdgeDest(e);
                ValueT  w = current_graph.edge_values[e];
                VertexT comm_u = current_communities[u];

                w_c2c_c.push_back(std::make_pair(comm_u, w));
            }
        }

        SizeT num_edges = 0;
        auto &w_2c = w_v2c;
        temp_row_offsets[0] = 0;
        for (VertexT c = 0; c < num_comms; c++)
        {
            auto &w_c2c_c = w_c2c[c];
            VertexT num_neighbor_comms = 0;
            for (auto it = w_c2c_c.begin(); it != w_c2c_c.end(); it++)
            {
                VertexT u_c = it -> first;
                ValueT  w = it -> second;
                if (!util::isValid(w_2c[u_c]))
                {
                    w_2c[u_c] = w;
                    neighbor_comms[num_neighbor_comms] = u_c;
                    num_neighbor_comms ++;
                } else
                    w_2c[u_c] += w;
            }
            w_c2c_c.clear();

            for (VertexT i = 0; i < num_neighbor_comms; i++)
            {
                VertexT u_c = neighbor_comms[i];
                ValueT  w = w_2c[u_c];
                temp_column_indices[num_edges + i] = u_c;
                temp_edge_values   [num_edges + i] = w;
                w_2c[u_c] = util::PreDefinedValues<ValueT>::InvalidValue;
            }
            num_edges += num_neighbor_comms;
            temp_row_offsets[c+1] = num_edges;
        }

        n_graph = new GraphT;
        auto &next_graph = *n_graph;
        if (has_pass_graphs)
            pass_graphs -> push_back(n_graph);
        next_graph.Allocate(num_comms, num_edges, util::HOST);
        memcpy(next_graph.CsrT::row_offsets    + 0, temp_row_offsets    + 0,
            sizeof(SizeT  ) * (num_comms + 1));
        memcpy(next_graph.CsrT::column_indices + 0, temp_column_indices + 0,
            sizeof(VertexT) * num_edges);
        memcpy(next_graph.CsrT::edge_values    + 0, temp_edge_values    + 0,
            sizeof(ValueT ) * num_edges);

        if (iter_stats)
            iter_timer.Stop();
        util::PrintMsg("pass " + std::to_string(pass_num)
            + ", graph compaction, elapsed = "
            + std::to_string(iter_timer.ElapsedMillis()), iter_stats);

        if (pass_stats)
            pass_timer.Stop();
        util::PrintMsg("pass " + std::to_string(pass_num)
            + ", #v = " + std::to_string(nodes) + " -> " + std::to_string(num_comms)
            + ", #e = " + std::to_string(current_graph.edges)
            + " -> " + std::to_string(num_edges)
            + ", #iter = " + std::to_string(iter_num)
            + ", q = " + std::to_string(q)
            + ", pass_gain = " + std::to_string(pass_gain)
            + ", elapsed = "
            + std::to_string(pass_timer.ElapsedMillis()), pass_stats);

        if (pass_num != 0 && !has_pass_graphs)
        {
            current_graph.Release(util::HOST);
            delete c_graph;
        }
        c_graph = n_graph;
        n_graph = NULL;

        pass_num ++;
        if (pass_gain < pass_gain_threshold)
            break;
    }

    // Assining communities to vertices in original graph
    for (VertexT v = 0; v < graph.nodes; v ++)
        communities[v] = v;
    pass_num = 0;
    for (auto it = pass_communities -> begin(); it != pass_communities -> end(); it++)
    {
        auto &v2c = *(*it);
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            communities[v] = v2c[communities[v]];
        }
        pass_num ++;
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    // Clearn-up
    if (!has_pass_communities)
    {
        for (auto it = pass_communities -> begin(); it != pass_communities -> end(); it++)
        {
            (*it)->clear();
            delete *it;
        }
        pass_communities -> clear();
        delete pass_communities; pass_communities = NULL;
    }

    temp_graph.Release();
    delete[] comm_convert; comm_convert = NULL;
    delete[] w_c2c;        w_c2c        = NULL;
    delete[] w_v2self;     w_v2self     = NULL;
    delete[] w_v2;         w_v2         = NULL;
    delete[] w_c2;         w_c2         = NULL;
    delete[] w_v2c;        w_v2c        = NULL;
    delete[] neighbor_comms; neighbor_comms = NULL;
    return elapsed;
}

/**
 * @brief Validation of SSSP results
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
    // typename GraphT::VertexT   src,
    //                  ValueT   *h_distances,
    //                  ValueT   *ref_distances = NULL,
                     bool      verbose       = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    // TODO: change to other representation, if not using CSR
    typedef typename GraphT::CsrT    CsrT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // Verify the result
    // TODO: result validation and display, e.g.:
    // if (ref_distances != NULL)
    // {
    //    for (VertexT v = 0; v < graph.nodes; v++)
    //    {
    //        if (!util::isValid(ref_distances[v]))
    //            ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
    //    }
    //
    //    util::PrintMsg("Distance Validity: ", !quiet, false);
    //    SizeT errors_num = util::CompareResults(
    //        h_distances, ref_distances,
    //        graph.nodes, true, quiet);
    //    if (errors_num > 0)
    //    {
    //        util::PrintMsg(
    //            std::to_string(errors_num) + " errors occurred.", !quiet);
    //        num_errors += errors_num;
    //    }
    // }
    // else if (ref_distances == NULL)
    // {
    //    util::PrintMsg("Distance Validity: ", !quiet, false);
    //    SizeT errors_num = 0;
    //    for (VertexT v = 0; v < graph.nodes; v++)
    //    {
    //        ValueT v_distance = h_distances[v];
    //        if (!util::isValid(v_distance))
    //            continue;
    //        SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
    //        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
    //        SizeT e_end = e_start + num_neighbors;
    //        for (SizeT e = e_start; e < e_end; e++)
    //        {
    //            VertexT u = graph.CsrT::GetEdgeDest(e);
    //            ValueT u_distance = h_distances[u];
    //            ValueT e_value = graph.CsrT::edge_values[e];
    //            if (v_distance + e_value >= u_distance)
    //                continue;
    //            errors_num ++;
    //            if (errors_num > 1)
    //                continue;
    //
    //            util::PrintMsg("FAIL: v[" + std::to_string(v)
    //                + "] ("    + std::to_string(v_distance)
    //                + ") + e[" + std::to_string(e)
    //                + "] ("    + std::to_string(e_value)
    //                + ") < u[" + std::to_string(u)
    //                + "] ("    + std::to_string(u_distance) + ")", !quiet);
    //        }
    //    }
    //    if (errors_num > 0)
    //    {
    //        util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
    //        num_errors += errors_num;
    //    } else {
    //        util::PrintMsg("PASS", !quiet);
    //    }
    // }
    //
    // if (!quiet && verbose)
    // {
    //    // Display Solution
    //    util::PrintMsg("First 40 distances of the GPU result:");
    //    DisplaySolution(h_distances, graph.nodes);
    //    if (ref_distances != NULL)
    //    {
    //        util::PrintMsg("First 40 distances of the reference CPU result.");
    //        DisplaySolution(ref_distances, graph.nodes);
    //    }
    //    util::PrintMsg("");
    // }

    return num_errors;
}

} // namespace louvain
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
