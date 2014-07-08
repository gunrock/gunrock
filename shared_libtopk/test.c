#include <gunrock/gunrock.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_UINT;
  data_type.VALUE_TYPE = VALUE_INT;

  // define graph
  size_t num_nodes = 7;
  size_t num_edges = 15;
  size_t top_nodes = 7;

  unsigned int row_offsets[8] = {0,3,6,9,11,14,15,15};
  int col_indices[15] = {1,2,3,0,2,4,3,4,5,5,6,2,5,6,6};

  unsigned int col_offsets[8] = {0,1,2,5,7,9,12,15};
  int row_indices[15] = {1,0,0,1,4,0,2,1,2,2,3,4,3,4,5};

  // build graph as input
  struct GunrockGraph *graph = (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  graph->num_nodes = num_nodes;
  graph->num_edges = num_edges;
  graph->row_offsets = (void*)&row_offsets[0];
  graph->col_indices = (void*)&col_indices[0];
  graph->col_offsets = (void*)&col_offsets[0];
  graph->row_indices = (void*)&row_indices[0];

  // malloc output result arrays
  int *node_ids          = (int*)malloc(sizeof(int) * top_nodes);
  int *centrality_values = (int*)malloc(sizeof(int) * top_nodes);

  // run topk calculations
  topk_dispatch((struct GunrockGraph*)NULL,
    node_ids,
    centrality_values,
    (const struct GunrockGraph*)graph,
    top_nodes,
    data_type);

  int i;

  for (i = 0; i < top_nodes; ++i)
  {
    printf("Node ID [%d] : CV [%d] \n", node_ids[i], centrality_values[i]);
  }

  free(centrality_values);
  free(node_ids);
  free(graph);
  return 0;
}
