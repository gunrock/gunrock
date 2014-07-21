// test bfs shared library

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
  // define data types
  struct GunrockDataType data_type;
  data_type.VTXID_TYPE = VTXID_INT;
  data_type.SIZET_TYPE = SIZET_UINT;
  data_type.VALUE_TYPE = VALUE_INT;

  // get command line arguments
  struct GunrockConfig  bfs_config;
  bfs_config.source      = 0;     //!< source vertex to begin search
  bfs_config.mark_pred   = false; //!< do not mark predecessors
  bfs_config.idempotence = false;
  bfs_config.queue_size  = 1.0f;

  // define graph
  size_t num_nodes = 7;
  size_t num_edges = 15;

  unsigned int row_offsets[8] = {0,3,6,9,11,14,15,15};
  int col_indices[15] = {1,2,3,0,2,4,3,4,5,5,6,2,5,6,6};

  // build graph as input
  struct GunrockGraph *graph_input =
    (struct GunrockGraph*)malloc(sizeof(struct GunrockGraph));
  graph_input->num_nodes = num_nodes;
  graph_input->num_edges = num_edges;
  graph_input->row_offsets = (void*)&row_offsets[0];
  graph_input->col_indices = (void*)&col_indices[0];

  //CommandLineArgs args(argc, argv);

  // run bfs calculations
  gunrock_bfs(
    (struct GunrockGraph*)NULL,
    (const struct GunrockGraph*)graph_input,
    bfs_config,
    data_type);

  if (graph_input) free(graph_input);
  return 0;
}