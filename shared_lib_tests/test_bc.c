/**
 * @brief BC test for shared library
 * @file test_bc.c
 *
 * set input graph, configs and call function gunrock_bc_func
 * return per node label values in graph_out node_values
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_INT;
  data_type.VALUE_TYPE = VALUE_FLOAT;

  // bc configurations (optional)
  struct GunrockConfig bc_config;
  bc_config.device       =    0;
  bc_config.src_node     =   -1;     //!< source vertex to begin search
  bc_config.queue_size   = 1.0f;
  bc_config.src_mode = manually;

  // define graph (undirected graph)
  size_t num_nodes = 7;
  size_t num_edges = 26;
  int row_offsets[8] = {0, 3, 6, 11, 15, 19, 23, 26};
  int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

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

  // run bc calculations
  gunrock_bc_func(
    graph_output,
    graph_input,
    bc_config,
    data_type);

  // test print
  int i;
  printf("Demo Outputs:\n");
  // print per node betweeness centrality values
  float *bc_vals = (float*)malloc(sizeof(float) * graph_input->num_nodes);
  bc_vals = (float*)graph_output->node_values;
  for (i = 0; i < graph_input->num_nodes; ++i)
  {
    printf("Node_ID [%d] : BC[%f]\n", i, bc_vals[i]);
  }
  printf("\n");
  // print per edge betweeness centrality values
  float *ebc_vals = (float*)malloc(sizeof(float)*graph_input->num_edges);
  ebc_vals = (float*)graph_output->edge_values;
  for (i = 0; i < graph_input->num_edges; ++i)
  {
    printf("Edge_ID [%d] : EBC[%f]\n", i, ebc_vals[i]);
  }

  if (graph_input)  { free(graph_input);  }
  if (graph_output) { free(graph_output); }

  return 0;
}
