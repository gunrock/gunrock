/**
 * @brief Simple test for shared library simple interface
 * @file simple_interface_test.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    int row_offsets[] = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
    size_t num_nodes = sizeof(row_offsets) / sizeof(row_offsets[0]) - 1;
    size_t num_edges = sizeof(col_indices) / sizeof(col_indices[0]);

    int *labels = (int*)malloc(sizeof(int) * num_nodes);

    printf(" testing breath-first search ...\n");  // test bfs
    bfs(labels, num_nodes, num_edges, row_offsets, col_indices, 0, 0);
    printf("-------------------- outputs --------------------\n");
    int node; for (node = 0; node < num_nodes; ++node) {
        printf(" node: [%d] | label (depth): [%d]\n", node, labels[node]);
    }
    printf("------------------- completed -------------------\n");

    if (labels) { free(labels); }
    return 0;
}
