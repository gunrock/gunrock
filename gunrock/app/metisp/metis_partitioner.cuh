// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * metis_partitioner.cuh
 *
 * @brief linkage to metis partitioner
 */

#pragma once

#ifdef METIS_FOUND
#include <metis.h>
#endif

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace app {
namespace metisp {

template <
    typename VertexId,
    typename SizeT,
    typename Value/*,
    bool     ENABLE_BACKWARD = false,
    bool     KEEP_ORDER      = false,
    bool     KEEP_NODE_NUM   = false*/>
struct MetisPartitioner : PartitionerBase<VertexId,SizeT,Value/*, 
    ENABLE_BACKWARD, KEEP_ORDER, KEEP_NODE_NUM*/>
{
  typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
  typedef Csr<VertexId, SizeT, Value> GraphT;

  // Members
  float *weitage;

  // Methods
  /*MetisPartitioner()
  {
      weitage=NULL;
  }*/

  MetisPartitioner(const GraphT &graph, int num_gpus, float *weitage = NULL,
                   bool _enable_backward = false, bool _keep_order = false,
                   bool _keep_node_num = false)
      : BasePartitioner(_enable_backward, _keep_order, _keep_node_num) {
    Init2(graph, num_gpus, weitage);
  }

  void Init2(const GraphT &graph, int num_gpus, float *weitage) {
    this->Init(graph, num_gpus);
    this->weitage = new float[num_gpus + 1];
    if (weitage == NULL)
      for (int gpu = 0; gpu < num_gpus; gpu++)
        this->weitage[gpu] = 1.0f / num_gpus;
    else {
      float sum = 0;
      for (int gpu = 0; gpu < num_gpus; gpu++) sum += weitage[gpu];
      for (int gpu = 0; gpu < num_gpus; gpu++)
        this->weitage[gpu] = weitage[gpu] / sum;
    }
    for (int gpu = 0; gpu < num_gpus; gpu++)
      this->weitage[gpu + 1] += this->weitage[gpu];
  }

  ~MetisPartitioner() {
    if (weitage != NULL) {
      delete[] weitage;
      weitage = NULL;
    }
  }

  cudaError_t Partition(GraphT *&sub_graphs, int **&partition_tables,
                        VertexId **&convertion_tables,
                        VertexId **&original_vertexes,
                        // SizeT**    &in_offsets,
                        SizeT **&in_counter, SizeT **&out_offsets,
                        SizeT **&out_counter, SizeT **&backward_offsets,
                        int **&backward_partitions,
                        VertexId **&backward_convertions, float factor = -1,
                        int seed = -1) {
    cudaError_t retval = cudaSuccess;
#ifdef METIS_FOUND
    {
      // typedef idxtype idx_t;
      idx_t nodes = this->graph->nodes;
      idx_t edges = this->graph->edges;
      idx_t ngpus = this->num_gpus;
      idx_t ncons = 1;
      idx_t objval;
      idx_t *tpartition_table = new idx_t[nodes];  //=this->partition_tables[0];
      idx_t *trow_offsets = new idx_t[nodes + 1];
      idx_t *tcolumn_indices = new idx_t[edges];

      for (idx_t node = 0; node <= nodes; node++)
        trow_offsets[node] = this->graph->row_offsets[node];
      for (idx_t edge = 0; edge < edges; edge++)
        tcolumn_indices[edge] = this->graph->column_indices[edge];

      // int Status =
      METIS_PartGraphKway(
          &nodes,           // nvtxs  : the number of vertices in the graph
          &ncons,           // ncon   : the number of balancing constraints
          trow_offsets,     // xadj   : the adjacency structure of the graph
          tcolumn_indices,  // adjncy : the adjacency structure of the graph
          NULL,             // vwgt   : the weights of the vertices
          NULL,             // vsize  : the size of the vertices
          NULL,             // adjwgt : the weights of the edges
          &ngpus,   // nparts : the number of parts to partition the graph
          NULL,     // tpwgts : the desired weight for each partition and
                    // constraint
          NULL,     // ubvec  : the allowed load imbalance tolerance 4 each
                    // constraint
          NULL,     // options: the options
          &objval,  // objval : the returned edge-cut or the total communication
                    // volume
          tpartition_table);  // part   : the returned partition vector of the
                              // graph

      for (SizeT i = 0; i < nodes; i++)
        this->partition_tables[0][i] = tpartition_table[i];
      delete[] tpartition_table;
      tpartition_table = NULL;
      delete[] trow_offsets;
      trow_offsets = NULL;
      delete[] tcolumn_indices;
      tcolumn_indices = NULL;

      retval = this->MakeSubGraph();
      sub_graphs = this->sub_graphs;
      partition_tables = this->partition_tables;
      convertion_tables = this->convertion_tables;
      original_vertexes = this->original_vertexes;
      // in_offsets           = this->in_offsets;
      in_counter = this->in_counter;
      out_offsets = this->out_offsets;
      out_counter = this->out_counter;
      backward_offsets = this->backward_offsets;
      backward_partitions = this->backward_partitions;
      backward_convertions = this->backward_convertions;
    }
#else
    {
      const char *str =
          "Metis was not found during installation, therefore metis "
          "partitioner cannot be used.";
      retval = util::GRError(cudaErrorUnknown, str, __FILE__, __LINE__);
    }  // METIS_FOUND
#endif
    return retval;
  }
};

}  // namespace metisp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
