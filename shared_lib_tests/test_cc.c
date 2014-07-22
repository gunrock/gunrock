/**
 * @brief cc test for bfs shared library
 * @file test_cc.c
 *
 * set input graph, configs and call function gunrock_bfs
 * return per node label values in graph_out node_values
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_UINT;

  // connected component configurations
  struct GunrockConfig configs;
  configs.device = 0;

  // define graph
  size_t num_nodes = 7;
  size_t num_edges = 15;
  unsigned int row_offsets[8] = {0,3,6,9,11,14,15,15};
  int col_indices[15] = {1,2,3,0,2,4,3,4,5,5,6,2,5,6,6};

  // build graph as input
  struct GunrockGraph *graph_input =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  graph_input->num_nodes   = num_nodes;
  graph_input->num_edges   = num_edges;
  graph_input->row_offsets = (void*)&row_offsets[0];
  graph_input->col_indices = (void*)&col_indices[0];

  // malloc output graph
  struct GunrockGraph *graph_output =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));

  // run connected component calculations
  gunrock_cc(
    (struct GunrockGraph*)graph_output,
    (const struct GunrockGraph*)graph_input,
    configs,
    data_type);

  // test print
  int i;
  int *component_ids = (int*)malloc(sizeof(int) * graph_input->num_nodes);
  component_ids = (int*)graph_output->node_values;
  printf("\n[Node_ID:Component_ID] \n[");
  for (i = 0; i < graph_input->num_nodes; ++i)
  {
    printf("%d:%d ", i, component_ids[i]);
  }
  printf("]\n");

  if (graph_input)  { free(graph_input);  }
  if (graph_output) { free(graph_output); }

  return 0;
}
