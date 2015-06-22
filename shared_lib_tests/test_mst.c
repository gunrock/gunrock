/**
 * @brief MST test for shared library
 * @file test_mst.c
 *
 * set input graph, configs and call function gunrock_mst
 * return per node or per edge values in graph_out node_values
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // set problem data types
  struct GunrockDataType dt;
  dt.VTXID_TYPE = VTXID_INT;
  dt.SIZET_TYPE = SIZET_INT;
  dt.VALUE_TYPE = VALUE_INT;

  // configurations (optional)
  struct GunrockConfig configs;
  configs.device = 0;

  // tiny sample graph
  size_t num_nodes = 7;
  size_t num_edges = 26;
  int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
  int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};
  int edge_values[26] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // build graph as input
  struct GunrockGraph *graph_input =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  graph_input->num_nodes   = num_nodes;
  graph_input->num_edges   = num_edges;
  graph_input->row_offsets = (void*)&row_offsets[0];
  graph_input->col_indices = (void*)&col_indices[0];
  graph_input->edge_values = (void*)&edge_values[0];

  // malloc output graph
  struct GunrockGraph *graph_output =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));

  // call MST
  gunrock_mst(graph_output, graph_input, configs, dt);

  // demo test print
  printf("Demo Outputs:\n");
  int *mst_mask = (int*)malloc(sizeof(int) * num_edges);
  mst_mask = (int*)graph_output->edge_values;
  int edge;
  for (edge = 0; edge < num_edges; ++edge) {
    printf("Edge ID [%d] : Label [%d]\n", edge, mst_mask[edge]);
  }

  if (graph_input)  { free(graph_input);  }
  if (graph_output) { free(graph_output); }

  return 0;
}
