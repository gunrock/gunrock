/**
 * @brief Simple test for shared library simple interface
 * @file simple_interface_test.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    int row_offsets[] = {0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};
    size_t num_nodes = sizeof(row_offsets) / sizeof(row_offsets[0]) - 1;
    size_t num_edges = sizeof(col_indices) / sizeof(col_indices[0]);

    printf("\n testing breath-first search ...\n");
    int *labels = (int*)malloc(sizeof(int) * num_nodes);
    bfs(labels, num_nodes, num_edges, row_offsets, col_indices, 0);
    int node; for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | label (depth): [%d]\n", node, labels[node]);
    }

    printf("\n testing betweenness centrality ...\n");
    float *scores = (float*)malloc(sizeof(float) * num_nodes);
    bc(scores, num_nodes, num_edges, row_offsets, col_indices, -1);
    for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | score: [%.4f]\n", node, scores[node]);
    }

    printf("\n testing connected components ...\n");
    int *components = (int*)malloc(sizeof(int) * num_nodes);
    int ret = cc(components, num_nodes, num_edges, row_offsets, col_indices);
    printf(" total number of components: %d\n", ret);
    for (node = 0; node < num_nodes; ++node) {
      printf(" node: [%d] | component: [%d]\n", node, components[node]);
    }

    // TODO(ydwu): add other primitive tests

    if (labels) { free(labels); }
    return 0;
}
