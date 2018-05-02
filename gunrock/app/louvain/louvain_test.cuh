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
    const    GraphT          &graph,
    typename GraphT::VertexT *communities,
    std::vector<typename GraphT::VertexT*> *pass_communities = NULL,
    std::vector<GraphT*> *pass_graphs = NULL)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    VertexT max_passes = parameters.Get<VertexT>("max-passes");
    VertexT max_iters  = parameters.Get<VertexT>("max-iters");

    bool has_pass_communities = false;
    if (pass_communities != NULL)
        has_pass_communities = true;
    else
        pass_communities = new std::vector<VertexT*>;
    bool has_pass_graphs = false;
    if (pass_graphs != NULL)
        has_pass_graphs = true;

    std::unordered_map<VertexT, ValueT> w_v2c;
    std::set<VertexT> comm_sets;
    VertexT *comm_convert = new VertexT[graph.nodes];
    std::unordered_map<VertexT, ValueT> *w_c2c
        = new std::unordered_map<VertexT, ValueT>[graph.nodes];
    ValueT *w_v2self = new ValueT[graph.nodes];

    auto c_graph = &graph;
    auto n_graph = c_graph;
    n_graph = NULL;

    ValueT m = 0;
    for (SizeT e = 0; e < graph.edges; e++)
    {
        m += graph.CsrT::edge_values[e];
    }

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    int pass_num = 0;
    while (pass_num < max_passes)
    {
        // Pass initialization
        auto &current_graph = *c_graph;
        SizeT nodes = current_graph.nodes;
        util::Print_Msg("pass " + std::to_string(pass_num)
            + ", #v = " + std::to_string(nodes)
            + ", #e = " + std::to_string(current_graph.edges));
        VertexT *current_communities = new VertexT[nodes];
        pass_communities.push_back(current_communities);
        for (VertexT v = 0; v < nodes; v++)
        {
            current_communities[v] = v;
            w_2v[v] = 0;
            w_v2self[v] = 0;
        }
        for (VertexT v = 0; v < nodes; v++)
        {
            SizeT start_e = current_graph.GetNeighborListOffset(v);
            SizeT degree  = current_graph.GetNeighborListLength(v);

            for (SizeT k = 0; k < degree; k++)
            {
                SizeT   e = start_e + k;
                VertexT u = current_graph.GetEdgeDest(e);
                ValueT  w = current_graph.edge_values[e];
                w_2v[u] += w;
                if (u == v)
                    w_v2self[v] += w;
            }
        }
        for (VertexT v = 0; v < nodes; v++)
        {
            w_2c[v] = w_2v[v];
        }

        // Modulation Optimization
        int iter_num = 0;
        ValueT pass_gain = 0;
        while (iter_num < max_iters)
        {
            ValueT iter_gain = 0;
            for (VertexT v = 0; v < nodes; v++)
            {
                w_v2c.clear();
                SizeT start_e = current_graph.GetNeighborListOffset(v);
                SizeT degree  = current_graph.GetNeighborListLength(v);

                for (SizeT k = 0; k < degree; k++)
                {
                    SizeT   e = start_e + k;
                    VertexT u = current_graph.GetEdgeDest(e);
                    ValueT  w = current_graph.edge_values[e];
                    VertexT c = current_communities[u];

                    auto it = w_v2c.find(c);
                    if (it == w_v2c.end())
                        w_v2c[c] = w;
                    else
                        it -> second += w;
                }

                ValueT  max_gain = 0;
                VertexT new_comm = current_communities[v];
                VertexT org_comm = new_comm;
                ValueT  w_v2c_org = 0;
                auto it = w_v2c.find(org_comm);
                if (it != w_v2c.end())
                    w_v2c_org = it -> second;
                ValueT  w_2c_org = w_2c[org_comm];
                ValueT  w_2v_v   = w_2v[v];

                for (auto it = w_v2c.begin(); it != w_v2c.end(); it++)
                {
                    if (it -> first == org_comm)
                        continue;

                    ValueT gain = 0;
                    gain += it -> second - w_v2c_org + w_v2self;
                    gain -= (w_2c[it -> first] - w_2c_org + w_v2self) * w_2v_v / m;
                    if (gain > max_gain)
                    {
                        max_gain = gain;
                        new_comm = it -> first;
                    }
                }
                if (max_gain > 0 && new_comm != current_communities[v])
                {
                    iter_gain += max_gain;
                    current_communities[v] = new_comm;
                    w_2c[new_comm] += w_v2c[new_comm] + w_v2self;
                    w_2c[org_comm] -= w_v2c[org_comm];
                }
            }

            iter_num ++;
            iter_gain /= m;
            pass_gain += iter_gain;
            util::PrintMsg("pass " + std::to_string(pass_num)
                + ", iter " + std::to_string(iter_num)
                + ", iter_gain = " + std::to_string(iter_gain)
                + ", pass_gain = " + std::to_string(pass_gain));
        }
        util::PrintMsg("pass " + std::to_string(pass_num)
            + ", #iter = " + std::to_string(iter_num)
            + ", pass_gain = " + std::to_string(pass_gain));

        // Community Aggregation
        w_v2c.clear();
        comm_sets.clear();
        for (VertexT v = 0; v < nodes; v++)
        {
            comm_sets.insert(current_communities[v]);
        }

        VertexT num_comms = comm_sets.size();
        VertexT comm_counter = 0;
        for (auto it = comm_sets.begin(); it != comm_sets.end(); it++)
        {
            comm_convert[*it] = comm_counter;
            comm_counter ++;
        }
        comm_sets.clear();

        for (VertexT v = 0; v < nodes; v++)
        {
            SizeT start_e = current_graph.GetNeighborListOffset(v);
            SizeT degree  = current_graph.GetNeighborListLength(v);
            VertexT comm_v = comm_convert[current_communities[v]];
            auto &w_c2c_v = w_c2c[comm_v];

            for (SizeT k = 0; k < degree; k++)
            {
                SizeT   e = start_e + k;
                VertexT u = current_graph.GetEdgeDest(e);
                ValueT  w = current_graph.edge_values[e];
                VertexT comm_u = comm_convert[current_communities[u]];

                auto it = w_c2c_v.find(comm_u);
                if (it == w_c2c_v.end())
                    w_v2c_v[comm_u] = w;
                else
                    it -> second += w;
            }
        }

        SizeT num_edges = 0;
        for (VertexT c = 0; c < num_comms; c++)
            num_edges += w_c2c[c].size();

        n_graph = new GraphT;
        auto &next_graph = *n_graph;
        if (has_pass_graphs)
            pass_graphs.push_back(n_graph);
        next_graph.Allocate(num_comms, num_edges, util::HOST);
        auto &row_offsets    = next_graph.CsrT::row_offsets;
        auto &column_indices = next_graph.CsrT::column_indices;
        auto &edge_values    = next_graph.CsrT::edge_values;
        SizeT edge_counter = 0;
        for (VertexT c = 0; c < num_comms; c++)
        {
            row_offsets[c] = edge_counter;
            auto &w_c2c_c = w_c2c[c];
            SizeT degree = w_c2c_c.size();
            SizeT k = 0;

            for (auto it = w_c2c_c.begin(); it != w_c2c_c.end(); it++)
            {
                SizeT e = edge_counter + k;
                VertexT u = it -> first;
                ValueT  w = it -> second;
                column_indices[e] = u;
                edge_values   [e] = w;
                k ++;
            }
            edge_counter += degree;
            w_c2c_c.clear();
        }
        row_offsets[num_comms] = num_edges;

        if (pass_num != 0 && !has_pass_graphs)
        {
            current_graph.Release(util::HOST);
            delete c_graph;
        }
        c_graph = n_graph;
        n_graph = NULL;

        pass_num ++;
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    delete[] comm_convert; comm_convert = NULL;
    delete[] w_c2c;        w_c2c        = NULL;
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
