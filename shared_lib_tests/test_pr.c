/**
 * @brief pr test for pr shared library
 * @file test_pr.c
 *
 * set input graph, configs and call function gunrock_pr
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
  data_type.VALUE_TYPE = VALUE_FLOAT;

  // pr configurations (optional)
  struct GunrockConfig pr_config;
  pr_config.device    =     0;
  pr_config.delta     = 0.85f;
  pr_config.error     = 0.01f;
  pr_config.max_iter  =    20;
  pr_config.top_nodes =    10;
  pr_config.src_node  =    -1;
  pr_config.src_mode  = manually;

  // define graph (undirected graph)
  size_t num_nodes = 4;
  size_t num_edges = 8;
  int row_offsets[5] = {0, 3, 5, 6, 8};
  int col_indices[8] = {1, 2, 3, 2, 3, 0, 0, 2};

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
  int   *node_ids  = ( int* )malloc(sizeof( int ) * pr_config.top_nodes);
  float *page_rank = (float*)malloc(sizeof(float) * pr_config.top_nodes);

  // run pr calculations
  gunrock_pr(
    graph_output,
    node_ids,
    page_rank,
    graph_input,
    pr_config,
    data_type);

  // test print
  int i;
  printf("Demo Outputs:\n");
  if (pr_config.top_nodes > num_nodes) pr_config.top_nodes = num_nodes;
  for (i = 0; i < pr_config.top_nodes; ++i)
  {
    printf("Node ID [%d] : Page Rank [%f] \n", node_ids[i], page_rank[i]);
  }

  if (node_ids)     { free(node_ids);     }
  if (page_rank)    { free(page_rank);    }
  if (graph_input)  { free(graph_input);  }
  if (graph_output) { free(graph_output); }

  return 0;
}
