/**
 * @brief PR test for shared library
 * @file test_pr.c
 *
 * set input graph, configs and call function gunrock_pr_func
 * return per node or per edge values in graph_out node_values
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;   //!< integer type vertex_ids
  data_type.SIZET_TYPE = SIZET_INT;   //!< integer type graph size
  data_type.VALUE_TYPE = VALUE_FLOAT; //!< float type value for pr

  // pr configurations (optional)
  struct GunrockConfig pr_config;
  pr_config.device    =     0; //!< use device 0
  pr_config.delta     = 0.85f; //!< default delta value
  pr_config.error     = 0.01f; //!< default error threshold
  pr_config.max_iter  =    20; //!< maximum number of iterations
  pr_config.top_nodes =    10; //!< number of top nodes
  pr_config.src_node  =     0; //!< source node to begin page rank
  pr_config.src_mode  = manually; //!< set source node manually

  // define graph (undirected graph)
  size_t num_nodes = 7;
  size_t num_edges = 15;
  int row_offsets[8] = {0,3,6,9,11,14,15,15};
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
  int   *node_ids  = (int*)malloc(sizeof(int) * pr_config.top_nodes);
  float *page_rank = (float*)malloc(sizeof(float) * pr_config.top_nodes);

  // run pr calculations
  gunrock_pr_func(
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
