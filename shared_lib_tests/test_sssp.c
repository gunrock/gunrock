/**
 * @brief SSSP test for shared library
 * @file test_sssp.c
 *
 * set input graph, configs and call function gunrock_sssp_func
 * return per node or per edge values in graph_out node_values
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_INT;
  data_type.VALUE_TYPE = VALUE_UINT;

  // pr configurations (optional)
  struct GunrockConfig sssp_config;
  sssp_config.device       =    0;
  sssp_config.mark_pred    = true;
  sssp_config.queue_size   = 1.0f;
  sssp_config.delta_factor =    1;
  sssp_config.src_mode     = randomize;
  //sssp_config.src_node     =    1;

  // define graph
  size_t num_nodes = 7;
  size_t num_edges = 15;

  int row_offsets[8]           = {0,3,6,9,11,14,15,15};
  int col_indices[15]          = {1,2,3,0,2,4,3,4,5,5,6,2,5,6,6};
  unsigned int edge_values[15] = {39,6,41,51,63,17,10,44,41,13,58,43,50,59,35};

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
  int *predecessor = (int*)malloc(sizeof(int) * num_nodes);

  // run sssp calculations
  gunrock_sssp_func(
    graph_output,
    predecessor,
    graph_input,
    sssp_config,
    data_type);

  // test print
  int i;
  printf("Demo Outputs:\n");
  int *label = (int*)malloc(sizeof(int) * num_nodes);
  label = (int*)graph_output->node_values;
  for (i = 0; i < num_nodes; ++i)
  {
    printf("Node ID [%d] : Label [%d] : Predecessor [%d]\n",
           i, label[i], predecessor[i]);
  }

  if (predecessor)  { free(predecessor);  }
  if (graph_input)  { free(graph_input);  }
  if (graph_output) { free(graph_output); }

  return 0;
}
