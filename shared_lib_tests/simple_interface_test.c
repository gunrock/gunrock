/**
 * @brief Simple test for shared library simple interface
 * @file simple_interface_test.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    ///////////////////////////////////////////////////////////////////////////
    int nodes = 2 << 9;   // number of nodes
    int edges = 2 << 13;  // number of edges
    // row_offsets, column_indices, and per edge values (weight)
    int rows[nodes + 1], cols[edges]; unsigned int vals[edges];
    // generate random R-MAT input graph with default parameters
    rmat_graph(rows, cols, vals, nodes, edges, false, 0.57, 0.19, 0.19);

    ///////////////////////////////////////////////////////////////////////////
    // allocate host arrays to store test results
    int*   bfs_label = (  int*)malloc(sizeof(  int) * nodes);
    float* bc_scores = (float*)malloc(sizeof(float) * nodes);
    int*   conn_comp = (  int*)malloc(sizeof(  int) * nodes);
    unsigned int *sssp_dist =
        (unsigned int*)malloc(sizeof( unsigned int) * nodes);
    int*   top_nodes = (  int*)malloc(sizeof(  int) * nodes);
    float* top_ranks = (float*)malloc(sizeof(float) * nodes);

    ///////////////////////////////////////////////////////////////////////////
    bfs(bfs_label, nodes, edges, rows, cols, 0);        // breath-first search
    bc(bc_scores, nodes, edges, rows, cols, -1);     // betweenness centrality
    int c = cc(conn_comp, nodes, edges, rows, cols);   // connected components
    sssp(sssp_dist, nodes, edges, rows, cols, vals, 0);       // shortest path
    pagerank(top_nodes, top_ranks, nodes, edges, rows, cols);      // pagerank

    ///////////////////////////////////////////////////////////////////////////
    // example demo outputs
    int node; nodes = nodes < 10 ? nodes : 10;
    printf("\n breath-first search:\n");
    for (node = 0; node < nodes; ++node)
        printf(" node: [%d] | label: [%d]\n", node, bfs_label[node]);

    printf("\n betweenness centrality:\n");
    for (node = 0; node < nodes; ++node)
        printf(" node: [%d] | score: [%.4f]\n", node, bc_scores[node]);

    printf("\n connected components:\n");
    printf(" number of components: %d\n", c);
    for (node = 0; node < nodes; ++node)
        printf(" node: [%d] | component: [%d]\n", node, conn_comp[node]);

    printf("\n single-source shortest path:\n");
    for (node = 0; node < nodes; ++node)
        printf(" node: [%d] | component: [%d]\n", node, sssp_dist[node]);

    printf("\n top pagerank:\n");
    for (node = 0; node < nodes; ++node)
        printf(" node: [%d] | rank: [%.4f]\n",top_nodes[node],top_ranks[node]);

    // TODO: add other primitives here

    // clean ups
    if (bfs_label) free(bfs_label);
    if (bc_scores) free(bc_scores);
    if (conn_comp) free(conn_comp);
    if (sssp_dist) free(sssp_dist);
    if (top_nodes) free(top_nodes);
    if (top_ranks) free(top_ranks);

    return 0;
}
