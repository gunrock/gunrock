/**
 * @brief Simple test for shared library simple interface
 * @file simple_interface_test.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {

    ///////////////////////////////////////////////////////////////////////////
    // define input graph
    int row_offsets[] = {
        0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[] = {
        1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
        5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};
    unsigned int edge_values[] = {
        3, 4, 5, 3, 5, 7, 4, 5, 7, 8, 9, 5, 7, 10,
        11, 7, 8, 11, 12, 9, 10, 11, 13, 11, 12, 13};

    // nodes = length of row offsets-1, edges = length of column indices
    size_t num_nodes = sizeof(row_offsets) / sizeof(row_offsets[0]) - 1;
    size_t num_edges = sizeof(col_indices) / sizeof(col_indices[0]);

    ///////////////////////////////////////////////////////////////////////////
    // allocate host arrays to store test results
    int*   bfs_label = (  int*)malloc(sizeof(  int) * num_nodes);
    float* bc_scores = (float*)malloc(sizeof(float) * num_nodes);
    int*   conn_comp = (  int*)malloc(sizeof(  int) * num_nodes);
    unsigned int *sssp_dist =
        (unsigned int*)malloc(sizeof( unsigned int) * num_nodes);
    int*    pr_nodes = (  int*)malloc(sizeof(  int) * num_nodes);
    float*  pr_ranks = (float*)malloc(sizeof(float) * num_nodes);

    ///////////////////////////////////////////////////////////////////////////
    printf("\n testing breath-first search ...\n");
    bfs(bfs_label, num_nodes, num_edges, row_offsets, col_indices, 0);
    int node; for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | label (depth): [%d]\n", node, bfs_label[node]);
    }

    ///////////////////////////////////////////////////////////////////////////
    printf("\n testing betweenness centrality ...\n");
    bc(bc_scores, num_nodes, num_edges, row_offsets, col_indices, -1);
    for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | score: [%.4f]\n", node, bc_scores[node]);
    }

    ///////////////////////////////////////////////////////////////////////////
    printf("\n testing connected components ...\n");
    int num_comp = cc(conn_comp, num_nodes, num_edges, row_offsets, col_indices);
    printf(" total number of components: %d\n", num_comp);
    for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | component: [%d]\n", node, conn_comp[node]);
    }

    ///////////////////////////////////////////////////////////////////////////
    printf("\n testing single-source shortest path ...\n");
    sssp(sssp_dist, num_nodes, num_edges, row_offsets, col_indices, edge_values, 0);
    for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | component: [%d]\n", node, sssp_dist[node]);
    }

    ///////////////////////////////////////////////////////////////////////////
    printf("\n testing pagerank ...\n");
    pagerank(pr_nodes, pr_ranks, num_nodes, num_edges, row_offsets, col_indices);
    for (node = 0; node < num_nodes; ++node) {
      printf(" node: [%d] | rank: [%.4f]\n", pr_nodes[node], pr_ranks[node]);
    }

    // TODO(ydwu): add other primitive tests

    // clean ups
    if (bfs_label) free(bfs_label);
    if (bc_scores) free(bc_scores);
    if (conn_comp) free(conn_comp);
    if (sssp_dist) free(sssp_dist);
    if (pr_nodes)   free(pr_nodes);
    if (pr_ranks)   free(pr_ranks);

    return 0;
}
