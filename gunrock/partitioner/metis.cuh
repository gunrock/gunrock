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

#include <gunrock/util/basic_utils.h>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
#include <gunrock/partitioner/partitioner_base.cuh>

namespace gunrock {
namespace partitioner {
namespace metis {

template <typename GraphT, graph::GraphFlag FLAG>
struct GraphTypeSwitch {};

template <typename GraphT>
struct GraphTypeSwitch<GraphT, graph::HAS_CSR> {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  static util::Array1D<SizeT, SizeT> &GetOffsets(GraphT &graph) {
    return graph.CsrT::row_offsets;
  }

  static util::Array1D<SizeT, VertexT> &GetIndices(GraphT &graph) {
    return graph.CsrT::column_indices;
  }
};

template <typename GraphT>
struct GraphTypeSwitch<GraphT, graph::HAS_CSC> {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CscT CscT;

  static util::Array1D<SizeT, SizeT> &GetOffsets(GraphT &graph) {
    return graph.CscT::column_offsets;
  }

  static util::Array1D<SizeT, VertexT> &GetIndices(GraphT &graph) {
    return graph.CscT::row_indices;
  }
};

template <typename GraphT>
cudaError_t Partition_CSR_CSC(GraphT &org_graph, GraphT *&sub_graphs,
                              util::Parameters &parameters,
                              int num_subgraphs = 1,
                              PartitionFlag flag = PARTITION_NONE,
                              util::Location target = util::HOST,
                              float *weitage = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::CscT CscT;
  typedef typename GraphT::GpT GpT;

  cudaError_t retval = cudaSuccess;

#ifdef METIS_FOUND
  {
    auto &partition_table = org_graph.GpT::partition_table;
    // typedef idxtype idx_t;
    idx_t nodes = org_graph.nodes;
    // idx_t       edges  = org_graph.edges;
    idx_t nsubgraphs = num_subgraphs;
    idx_t ncons = 1;
    idx_t objval;
    util::Array1D<SizeT, idx_t> tpartition_table;
    util::Array1D<SizeT, idx_t> trow_offsets;
    util::Array1D<SizeT, idx_t> tcolumn_indices;

    tpartition_table.SetName("partitioner::metis::tpartition_table");
    retval = tpartition_table.Allocate(org_graph.nodes, target);
    if (retval) return retval;

    trow_offsets.SetName("partitioner::metis::trow_offsets");
    retval = trow_offsets.Allocate(org_graph.nodes + 1, target);
    if (retval) return retval;

    tcolumn_indices.SetName("partitioner::metis::tcolumn_indices");
    retval = tcolumn_indices.Allocate(org_graph.edges, target);
    if (retval) return retval;

    typedef GraphTypeSwitch<GraphT,
                            GraphT::FLAG &(graph::HAS_CSR | graph::HAS_CSC)>
        GraphSwitchT;
    retval = trow_offsets.ForEach(
        GraphSwitchT::GetOffsets(org_graph),
        [] __host__ __device__(idx_t & trow_offset, const SizeT &offset) {
          trow_offset = offset;
        },
        org_graph.nodes + 1, target);
    if (retval) return retval;

    retval = tcolumn_indices.ForEach(
        GraphSwitchT::GetIndices(org_graph),
        [] __host__ __device__(idx_t & tcolumn_index, const VertexT &index) {
          tcolumn_index = index;
        },
        org_graph.edges, target);
    if (retval) return retval;

    // int Status =
    METIS_PartGraphKway(
        &nodes,               // nvtxs  : the number of vertices in the graph
        &ncons,               // ncon   : the number of balancing constraints
        trow_offsets + 0,     // xadj   : the adjacency structure of the graph
        tcolumn_indices + 0,  // adjncy : the adjacency structure of the graph
        NULL,                 // vwgt   : the weights of the vertices
        NULL,                 // vsize  : the size of the vertices
        NULL,                 // adjwgt : the weights of the edges
        &nsubgraphs,  // nparts : the number of parts to partition the graph
        NULL,  // tpwgts : the desired weight for each partition and constraint
        NULL,  // ubvec  : the allowed load imbalance tolerance 4 each
               // constraint
        NULL,  // options: the options
        &objval,  // objval : the returned edge-cut or the total communication
                  // volume
        tpartition_table +
            0);  // part   : the returned partition vector of the graph

    retval = partition_table.ForEach(
        tpartition_table,
        [] __host__ __device__(int &partition, const idx_t &tpartition) {
          partition = tpartition;
        },
        org_graph.nodes, target);
    if (retval) return retval;

    if (retval = tpartition_table.Release()) return retval;
    if (retval = trow_offsets.Release()) return retval;
    if (retval = tcolumn_indices.Release()) return retval;
  }
#else
  {
    retval = util::GRError(cudaErrorUnknown,
                           "Metis was not found during installation, "
                           "therefore metis partitioner cannot be used.",
                           __FILE__, __LINE__);

  }  // METIS_FOUND
#endif

  return retval;
}

template <typename GraphT, gunrock::graph::GraphFlag GraphN>
struct GraphT_Switch {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    return util::GRError(cudaErrorUnknown,
                         "Metis dows not work with given graph representation",
                         __FILE__, __LINE__);
  }
};

template <typename GraphT>
struct GraphT_Switch<GraphT, gunrock::graph::HAS_CSR> {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    return Partition_CSR_CSC(org_graph, sub_graphs, parameters, num_subgraphs,
                             flag, target, weitage);
  }
};

template <typename GraphT>
struct GraphT_Switch<GraphT, gunrock::graph::HAS_CSC> {
  static cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                               util::Parameters &parameters,
                               int num_subgraphs = 1,
                               PartitionFlag flag = PARTITION_NONE,
                               util::Location target = util::HOST,
                               float *weitage = NULL) {
    return Partition_CSR_CSC(org_graph, sub_graphs, parameters, num_subgraphs,
                             flag, target, weitage);
  }
};

template <typename GraphT>
cudaError_t Partition(GraphT &org_graph, GraphT *&sub_graphs,
                      util::Parameters &parameters, int num_subgraphs = 1,
                      PartitionFlag flag = PARTITION_NONE,
                      util::Location target = util::HOST,
                      float *weitage = NULL) {
  return GraphT_Switch<GraphT, gunrock::graph::GraphType_Num<
                                   GraphT::FLAG>::VAL>::Partition(org_graph,
                                                                  sub_graphs,
                                                                  parameters,
                                                                  num_subgraphs,
                                                                  flag, target,
                                                                  weitage);
}

}  // namespace metis
}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
