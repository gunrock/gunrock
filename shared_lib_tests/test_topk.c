#include <gunrock/gunrock.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_INT;
  data_type.VALUE_TYPE = VALUE_INT;

  struct GunrockConfig topk_config;
  topk_config.device    = 0;
  topk_config.top_nodes = 3;

  // define graph (directed, reversed and non-reversed)
  size_t num_nodes = 7;
  size_t num_edges = 15;

  int row_offsets[8] = {0,3,6,9,11,14,15,15};
  int col_indices[15] = {1,2,3,0,2,4,3,4,5,5,6,2,5,6,6};

  int col_offsets[8] = {0,1,2,5,7,9,12,15};
  int row_indices[15] = {1,0,0,1,4,0,2,1,2,2,3,4,3,4,5};

  // build graph as input
  struct GunrockGraph *graph_input =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  graph_input->num_nodes = num_nodes;
  graph_input->num_edges = num_edges;
  graph_input->row_offsets = (void*)&row_offsets[0];
  graph_input->col_indices = (void*)&col_indices[0];
  graph_input->col_offsets = (void*)&col_offsets[0];
  graph_input->row_indices = (void*)&row_indices[0];

  // malloc output result arrays
  struct GunrockGraph *graph_output =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  int *node_ids    = (int*)malloc(sizeof(int) * topk_config.top_nodes);
  int *in_degrees  = (int*)malloc(sizeof(int) * topk_config.top_nodes);
  int *out_degrees = (int*)malloc(sizeof(int) * topk_config.top_nodes);

  // run topk calculations
  gunrock_topk_func(
    graph_output,
    node_ids,
    in_degrees,
    out_degrees,
    graph_input,
    topk_config,
    data_type);

  // print results for check correctness
  int i;
  printf("Demo Outputs:\n");
  for (i = 0; i < topk_config.top_nodes; ++i)
  {
    printf("Node ID [%d] : in_degrees [%d] : out_degrees [%d] \n",
      node_ids[i], in_degrees[i], out_degrees[i]);
  }

  if (in_degrees)   free(in_degrees);
  if (out_degrees)  free(out_degrees);
  if (node_ids)     free(node_ids);
  if (graph_input)  free(graph_input);
  if (graph_output) free(graph_output);
  return 0;
}