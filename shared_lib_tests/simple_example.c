// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @brief Simple test for shared library simple interface.
 * @file simple_example.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // define input graph require input of row-offsets and column-indices
    // for primitives use per edge weight value also requires edge-values
    int rows[] =
    {
        0, 3, 6, 11, 15, 19, 23, 26
    };
    int cols[] =
    {
        1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2, 5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5
    };
    unsigned int vals[] =
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    int nodes = sizeof(rows) / sizeof(rows[0]) - 1;  // number of nodes
    int edges = sizeof(cols) / sizeof(cols[0]);      // number of edges

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
    // run different primitive tests
    // graph traversal from given source return integer labels
    bfs(bfs_label, 0, nodes, edges, rows, cols, 1,/*source=*/ 0, 0, false, false);
    // node betweenness centrality from given source
    // store computed results to bc_scores of floats
    bc(bc_scores, nodes, edges, rows, cols, /*source=*/ 0);
    // return number of component and per node component ID
    int num_components = cc(conn_comp, nodes, edges, rows, cols);
    // return shortest distance for each vertex from given source
    sssp(sssp_dist, 0, nodes, edges, rows, cols, vals, 1,/*source=*/ 0,false);


    // return top-ranked nodes and their PageRank values of floats
    pagerank(top_nodes, top_ranks, nodes, edges, rows, cols, false);

    ///////////////////////////////////////////////////////////////////////////
    // demo prints allow at most ten nodes
    int i; nodes = nodes < 10 ? nodes : 10;

    printf("\n Breath-first search labels:\n");
    for (i = 0; i < nodes; ++i)
        printf(" i: [%d] | label: [%d]\n", i, bfs_label[i]);

    printf("\n Node betweenness centrality:\n");
    for (i = 0; i < nodes; ++i)
        printf(" i: [%d] | score: [%.4f]\n", i, bc_scores[i]);

    printf("\n Connected components IDs:\n");
    printf(" Total number of components: %d\n", num_components);
    for (i = 0; i < nodes; ++i)
        printf(" i: [%d] | component: [%d]\n", i, conn_comp[i]);

    printf("\n Single-source shortest path:\n");
    for (i = 0; i < nodes; ++i)
        printf(" i: [%d] | distance: [%d]\n", i, sssp_dist[i]);

    printf("\n Top-ranked nodes and their ranks:\n");
    for (i = 0; i < nodes; ++i)
        printf(" i: [%d] | rank: [%.4f]\n", top_nodes[i], top_ranks[i]);

    // clean up
    free(bfs_label);
    free(bc_scores);
    free(conn_comp);
    free(sssp_dist);
    free(top_nodes);
    free(top_ranks);

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
