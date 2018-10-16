// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * gtf_test.cu
 *
 * @brief Test related functions for Max Flow algorithm.
 */

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a); \
    printf("\n");}

#pragma once

#include <gunrock/app/mf/mf_test.cuh>
#include <queue>

namespace gunrock {
namespace app {
namespace gtf {

/*****************************************************************************
 * Housekeeping Routines
 ****************************************************************************/

cudaError_t UseParameters_test(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(parameters.Use<std::string>(
        "weights",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
        "",
        "Filename of edge weights from the source or to the sink, "
        "with #nodes elements in floating point. The first line are two numbers: "
        "#nodes and 20, with 20 indicating the type of elements is double. "
        "Elements larger than 0 indicating capacities from the source, "
        "and elements less than 0 indicating capacities to the sink.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "lambda",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        6,
        "Parameter controlling how heavily non-connected solutions are penalized.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "gamma",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        3,
        "Parameter controling how heavily non-sparsity is penalized.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "error_threshold",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1e-6,
        "Error threshold to compare floating point values",
        __FILE__, __LINE__));

    return retval;
}

/**
 * @brief Displays the GTF result
 *
 * @tparam ValueT     Type of capacity/flow/excess
 * @tparam VertxeT    Type of vertex
 *
 * @param[in] h_flow  Flow calculated on edges
 * @param[in] source  Index of source vertex
 * @param[in] nodes   Number of nodes
 */
template<typename GraphT, typename ValueT, typename VertexT>
void DisplaySolution(GraphT graph, ValueT* h_flow, VertexT* reverse,
	VertexT sink, VertexT nodes)
{
}

/****************************************************************************
 * GTF Testing Routines
 ***************************************************************************/

template <typename GraphT>
cudaError_t MinCut(
    util::Parameters         &parameters,
    GraphT                   &graph,
    typename GraphT::SizeT   *reverse_edges,
    typename GraphT::VertexT  source,
    typename GraphT::VertexT  dest,
    typename GraphT::ValueT  *edge_flows,
    typename GraphT::ValueT  *edge_residuals,
    bool                     *vertex_reachabilities)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    cudaError_t retval = cudaSuccess;
    double error_threshold = parameters.Get<double>("error_threshold");

    ValueT max_flow = 0;
    mf::CPU_Reference(parameters, graph,
        source, dest, max_flow, reverse_edges, edge_flows);
    auto &edge_capacities = graph.edge_values;

    for (auto e = 0; e < graph.edges; e++)
        edge_residuals[e] = edge_capacities[e] - edge_flows[e];

    VertexT *queue   = new VertexT[graph.nodes];
    char    *markers = new char   [graph.nodes];
    for (char t = 1; t < 3; t++)
    {
        VertexT head = 0;
        VertexT tail = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
            markers[v] = 0;
        queue[head] = (t == 1) ? source : dest;

        while (tail <= head)
        {
            VertexT v = queue[tail];
            auto e_start = graph.GetNeighborListOffset(v);
            auto num_neighbors = graph.GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; e++)
            {
                VertexT u = graph.GetEdgeDest(e);
                if (markers[u] != 0)
                    continue;
                if (edge_residuals[e] < error_threshold)
                    continue;

                head ++;
                queue[head] = u;
                markers[u] = 1;
                vertex_reachabilities[u] = t;
            }
            tail ++;
        }
    }
    delete[] queue  ; queue   = NULL;
    delete[] markers; markers = NULL;
    return retval;
}

/*-----------------------------------------------*/
template<typename ValueT, typename VertexT>
int bfs(ValueT** rGraph, VertexT s, VertexT t, VertexT parent[], const int V) {
    // Create a visited array and mark all vertices as not visited
    bool *visited = new bool[V];
    memset(visited, 0, V*sizeof(visited[0]));

    // Create a queue, enqueue source vertex and mark source vertex
    // as visited
    std::queue <VertexT> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    // Standard BFS Loop
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (visited[v] == false && rGraph[u][v] > 0) {
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    // If we reached sink in BFS starting from source, then return
    // true, else false
    return (visited[t] == true);
}

// A DFS based function to find all reachable vertices from s.  The function
// marks visited[i] as true if i is reachable from s.  The initial values in
// visited[] must be false. We can also use BFS to find reachable vertices
template<typename ValueT, typename VertexT>
void dfs(ValueT** rGraph, VertexT s, bool visited[], const int V) {
    visited[s] = true;
    for (int i = 0; i < V; i++)
       if (abs(rGraph[s][i]) > 1e-6 && !visited[i])
           dfs(rGraph, i, visited, V);
}

// Prints the minimum s-t cut
template<typename ValueT, typename VertexT, typename GraphT>
void minCut(GraphT graph, VertexT s, VertexT t, bool *visited, ValueT *edge_residuals, const int V) {
    ValueT max_flow = 0;
    // Create a residual graph and fill the residual graph with
    // given capacities in the original graph as residual capacities
    // in residual graph
    ValueT** rGraph = new ValueT*[V];  // rGraph[i][j] indicates residual capacity of edge i-j
    for (int u = 0; u < V; u++) {
        rGraph[u] = new ValueT[V];
        for (int v = 0; v < V; v++) {
             rGraph[u][v] = 0;
        }
    }

    for (auto u = 0; u < graph.nodes; ++u)
    {
        auto e_start = graph.GraphT::CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.GraphT::CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e)
        {
            auto v = graph.GraphT::CsrT::GetEdgeDest(e);
            rGraph[int(u)][int(v)] = graph.edge_values[e];
        }
    }
    /*
    printf("\n we are before maxflow \n");
    for (int u = 0; u < V; u++) {
          for (int v = 0; v < V; v++) {
            printf("%5.2f ", rGraph[u][v]);
          }
          printf("\n");
    }
    printf("\n");
    */

    VertexT *parent = new VertexT[V];  // This array is filled by BFS and to store path

    // Augment the flow while there is a path from source to sink
    int counter = 0;
    while (bfs(rGraph, s, t, parent, V)) {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        ValueT path_flow = INT_MAX;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // update residual capacities of the edges and reverse edges
        // along the path
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            //printf("%d -> %d\n", u, v);
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }
        counter++;
        max_flow += path_flow;
    }

    // Flow is maximum now, find vertices reachable from s
    memset(visited, false, V*sizeof(visited[0]));
    dfs(rGraph, s, visited, V);

    int tem_i = 0;
    for (auto u = 0; u < graph.nodes; ++u)
    {
        auto e_start = graph.GraphT::CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.GraphT::CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e)
        {
            auto v = graph.GraphT::CsrT::GetEdgeDest(e);
            edge_residuals[tem_i] = rGraph[int(u)][int(v)];
            //printf("inside graph loaded as %d (%d -> %d) = %f\n", tem_i, u, v, edge_residuals[tem_i]);
            tem_i++;
        }
    }
    /*
    for (int u = 0; u < V; u++) {
          for (int v = 0; v < V; v++) {
            printf("%5.2f ", rGraph[u][v]);
          }
          printf("\n");
    }
    */

}
/*----------------------------------------------*/

/**
 * @brief Simple CPU-based reference GTF implementations
 *
 * @tparam GraphT   Type of the graph
 * @tparam VertexT  Type of the vertex
 * @tparam ValueT   Type of the capacity/flow/excess
 * @param[in]  parameters Running parameters
 * @param[in]  graph      Input graph
 * @param[in]  src        The source vertex
 * @param[in]  sin        The sink vertex
 * @param[out] maxflow	  Value of computed maxflow reached sink
 * @param[out] reverse	  Computed reverse
 * @param[out] edges_flow Computed flows on edges
 *
 * \return     double      Time taken for the GTF
 */
template <typename GraphT, typename ArrayT>
cudaError_t CPU_Reference(
	util::Parameters &parameters,
	GraphT  &graph,
	ArrayT  &reverse_edges,
    double  &elapsed)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;

    cudaError_t retval = cudaSuccess;
    auto     num_nodes        = graph.nodes; // n + 2 = V
    auto     num_org_nodes    = num_nodes-2; // n
    auto     num_edges        = graph.edges; // m + n*4
    VertexT  source           = num_org_nodes; // originally 0
    VertexT  dest             = num_org_nodes+1; // originally 1
    double   lambda           = parameters.Get<double>("lambda");
    double   error_threshold  = parameters.Get<double>("error_threshold");
    VertexT  num_comms        = 1; // nlab
    VertexT *next_communities = new VertexT[num_nodes]; // nextlabel
    VertexT *curr_communities = new VertexT[num_nodes]; // label
    VertexT *community_sizes  = new VertexT[num_nodes]; // nums
    ValueT  *community_weights= new ValueT [num_nodes]; // averages
    bool    *community_active = new bool   [num_nodes]; // !inactivelable
    ValueT  *community_accus  = new ValueT [num_nodes]; // values
    bool    *vertex_active    = new bool   [num_nodes]; // alive
    bool    *vertex_reachabilities = new bool[num_nodes];
        // visited: 1 reachable from source, 2 reachable from dest, 0 otherwise
    ValueT  *edge_residuals   = new ValueT [num_edges]; // graph
    ValueT  *edge_flows       = new ValueT [num_edges]; // edge flows
    double   sum_weights_source_sink = 0;               // moy
    util::CpuTimer cpu_timer;

    // Normalization and resets
    SizeT offset = num_edges - num_org_nodes * 2;
    printf("offset is %d num edges %d \n", offset, num_edges);
    for (VertexT v = 0; v < num_org_nodes; v++)
    {
        sum_weights_source_sink += graph.edge_values[offset + v];
        SizeT e = graph.GetNeighborListOffset(v) + graph.GetNeighborListLength(v) - 1;
        sum_weights_source_sink -= graph.edge_values[e];

        vertex_active   [v] = true;
        community_active[v] = true;
        curr_communities[v] = 0;
        next_communities[v] = 0; //extra
    }

    auto avg_weights_source_sink = sum_weights_source_sink / num_org_nodes;
    community_accus[0] = avg_weights_source_sink;
    printf("avg is %f \n", avg_weights_source_sink);
    for (VertexT v = 0; v < num_org_nodes; v++)
    {
        SizeT  e = graph.GetNeighborListOffset(v) + graph.GetNeighborListLength(v) - 1;
        ValueT val = graph.edge_values[offset + v] - graph.edge_values[e]
            - avg_weights_source_sink;

        if (val > 0)
        {
            graph.edge_values[offset + v] = val;
            graph.edge_values[e] = 0;
        } else {
            graph.edge_values[offset + v] = 0;
            graph.edge_values[e] = -1 * val;
        }

    }

    cpu_timer.Start();
    VertexT iteration   = 0;    // iter
    bool    to_continue = true; // flagstop
    unsigned int comm;
    while (to_continue)
    {
        printf("Iteration %d\n", iteration);
        iteration++;

        GUARD_CU(MinCut(parameters, graph, reverse_edges + 0, source, dest,
            edge_flows, edge_residuals, vertex_reachabilities));
        //minCut(graph, source, dest, vertex_reachabilities, edge_residuals, num_nodes);

        auto &edge_capacities = graph.edge_values;

        for (comm = 0; comm < num_comms; comm++)
        {
            community_weights[comm] = 0;
            community_sizes  [comm] = 0;
            next_communities [comm] = 0;
        }
        auto pervious_num_comms = num_comms;

        for (VertexT v = 0; v < num_org_nodes; v++)
        {
            if (!vertex_active[v])
                continue;
            if (vertex_reachabilities[v] == 1)
            { // reachable by source
                comm = next_communities[curr_communities[v]];
                if (comm == 0)
                { // not assigned yet
                    comm = num_comms;
                    next_communities[curr_communities[v]] = num_comms;
                    community_active [comm] = true;
                    num_comms ++;
                    community_weights[comm] = 0;
                    community_sizes  [comm] = 0;
                    next_communities [comm] = 0;
                    community_accus  [comm] = community_accus[curr_communities[v]];
                }
                curr_communities[v] = comm;
                community_weights[comm] +=
                    edge_residuals[num_edges - num_org_nodes * 2 + v];
                community_sizes  [comm] ++;
            }

            else { // otherwise
                comm = curr_communities[v];
                SizeT e_start = graph.GetNeighborListOffset(v);
                SizeT num_neighbors = graph.GetNeighborListLength(v);
                community_weights[comm] -= edge_residuals[e_start + num_neighbors - 1];
                community_sizes  [comm] ++;

                auto e_end = e_start + num_neighbors - 2;
                for (auto e = e_start; e < e_end; e++)
                {
                    VertexT u = graph.GetEdgeDest(e);
                    if (vertex_reachabilities[u] == 1)
                    {
                        edge_residuals[e] = 0;
                    }
                }
            }
            //printf("%d %f %f\n", comm, community_weights[comm], community_accus[comm]);
        } // end of for v

        for (comm = 0; comm < pervious_num_comms; comm ++)
        {
            if (community_active[comm])
            {
                if (next_communities[comm] == 0)
                {
                    community_weights[comm] = 0;
                    community_active [comm] = false;
                } else if (community_sizes[comm] == 0) {
                    community_active [comm] = false;
                    community_active [next_communities[comm]] = false;
                    community_weights[next_communities[comm]] = 0;
                } else {
                    //printf("values: comm: %d, sizes: %d, weights: %f, accus: %f.\n", comm, community_sizes[comm], community_weights[comm], community_accus[comm]);
                    community_weights[comm] /= community_sizes  [comm];
                    community_accus  [comm] += community_weights[comm];
                }
            } else {
                community_weights[comm] = 0;
            }
        }
        for (; comm < num_comms; comm++)
        {
            community_weights[comm] /= community_sizes  [comm];
            community_accus  [comm] += community_weights[comm];
            //printf("values: comm: %d, sizes: %d, weights: %f, accus: %f.\n", comm, community_sizes[comm], community_weights[comm], community_accus[comm]);
        }

        to_continue = false;
        for (VertexT v = 0; v < num_org_nodes; v++)
        {
            if (!vertex_active[v])
                continue;

            auto comm = curr_communities[v];
            if (!community_active[comm] ||
                abs(community_weights[comm]) < error_threshold)
            {
                if (vertex_reachabilities[v] == 1)
                    edge_residuals[num_edges - num_org_nodes * 2 + v] = 0;
                if (vertex_reachabilities[v] != 1)
                {
                    SizeT  e = graph.GetNeighborListOffset(v)
                        + graph.GetNeighborListLength(v) - 1;
                    edge_residuals[e] = 0;
                }
                vertex_active[v] = false;
                community_active[comm] = false;
            }

            else {
                to_continue = true;
                SizeT e_from_src = num_edges - num_org_nodes * 2 + v;
                SizeT e_to_dest  = graph.GetNeighborListOffset(v)
                    + graph.GetNeighborListLength(v) - 1;
                if (vertex_reachabilities[v] == 1)
                {
                    edge_residuals[e_from_src] -= community_weights[comm];
                    if (edge_residuals[e_from_src] < 0)
                    {
                        double temp = -1 * edge_residuals[e_from_src];
                        edge_residuals[e_from_src] = edge_residuals[e_to_dest];
                        edge_residuals[e_to_dest] = temp;
                    }
                } else {
                    edge_residuals[e_to_dest] += community_weights[comm];
                    if (edge_residuals[e_to_dest] < 0)
                    {
                        double temp = -1 * edge_residuals[e_to_dest];
                        edge_residuals[e_to_dest] = edge_residuals[e_from_src];
                        edge_residuals[e_from_src] = temp;
                    }
                }
            }
        } // end of for v

        for (SizeT e = 0; e < graph.edges; e ++)
            edge_capacities[e] = edge_residuals[e];
    } // end of while
    cpu_timer.Stop();
    elapsed = cpu_timer.ElapsedMillis();

    std::ofstream out_pr( "./output_pr.txt" );
    for(int i = 0; i < num_org_nodes; i++) out_pr << (double) community_accus[curr_communities[i]] << std::endl;
    out_pr.close();

    delete[] next_communities ; next_communities  = NULL;
    delete[] curr_communities ; curr_communities  = NULL;
    delete[] community_sizes  ; community_sizes   = NULL;
    delete[] community_weights; community_weights = NULL;
    delete[] community_active ; community_active  = NULL;
    delete[] community_accus  ; community_accus   = NULL;
    delete[] vertex_active    ; vertex_active     = NULL;
    delete[] vertex_reachabilities; vertex_reachabilities = NULL;
    delete[] edge_residuals   ; edge_residuals    = NULL;
    return retval;
}

/**
 * @brief Validation of GTF results
 *
 * @tparam     GraphT	      Type of the graph
 * @tparam     ValueT	      Type of the distances
 *
 * @param[in]  parameters     Excution parameters
 * @param[in]  graph	      Input graph
 * @param[in]  source	      The source vertex
 * @param[in]  sink           The sink vertex
 * @param[in]  h_flow	      Computed flow on edges
 * @param[in]  ref_flow	      Reference flow on edges
 * @param[in]  verbose	      Whether to output detail comparsions
 *
 * \return     int  Number of errors
 */
template <typename GraphT, typename ValueT, typename VertexT>
int Validate_Results(
	util::Parameters  &parameters,
        GraphT		  &graph,
    	VertexT		  source,
    	VertexT		  sink,
        ValueT		  *h_flow,
	VertexT		  *reverse,
        ValueT		  *ref_flow = NULL,
	bool		  verbose = true)
{
    typedef typename GraphT::SizeT SizeT;
    SizeT num_errors = 0;

    return num_errors;
}

} // namespace gtf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
