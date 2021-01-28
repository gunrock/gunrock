// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dynamic_graph_weighted.cuh
 *
 * @brief DYN (Dynamic) Graph Data Structure
 */
#pragma once

#include <gunrock/graph/dynamic_graph/dynamic_graph.cuh>
#include <gunrock/graph/dynamic_graph/dynamic_graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief Weighted dynamic graph data structure which uses
 * a per-vertex data structure based on the graph flags.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG,
          unsigned int cudaHostRegisterFlag>
struct Dyn<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag, true, true>
    : DynamicGraphBase<VertexT, SizeT, ValueT, FLAG> {
  template <typename PairT>
  cudaError_t InsertEdgesBatch(util::Array1D<SizeT, PairT> &edges,
                               util::Array1D<SizeT, ValueT> &vals,
                               SizeT batchSize, bool batch_directed = true,
                               util::Location target = util::DEVICE) {
    if (target != util::DEVICE) {
      edges.Move(util::HOST, util::DEVICE);
      vals.Move(util::HOST, util::DEVICE);
    }

    this->dynamicGraph.InsertEdgesBatch(
        edges.GetPointer(util::DEVICE), vals.GetPointer(util::DEVICE),
        batchSize, (!this->is_directed) && batch_directed);

    if (target != util::DEVICE) {
      edges.Release(util::DEVICE);
      vals.Release(util::DEVICE);
    }
    return cudaSuccess;
  }

  template <typename PairT>
  cudaError_t DeleteEdgesBatch(util::Array1D<SizeT, PairT> &edges,
                               SizeT batchSize,
                               util::Location target = util::DEVICE) {
    if (target != util::DEVICE) {
      edges.Move(util::HOST, util::DEVICE);
    }

    this->dynamicGraph.DeleteEdgesBatch(edges.GetPointer(util::DEVICE),
                                        batchSize, !this->is_directed);

    if (target != util::DEVICE) {
      edges.Release(util::DEVICE);
    }
    return cudaSuccess;
  }

  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    this->is_directed = csr.directed;
    this->dynamicGraph.Allocate();
    this->dynamicGraph.BulkBuildFromCsr(
        csr.row_offsets.GetPointer(util::HOST),
        csr.column_indices.GetPointer(util::HOST),
        csr.edge_values.GetPointer(util::HOST), csr.nodes, csr.directed,
        csr.node_values.GetPointer(util::HOST));
    this->nodes = csr.nodes;
    this->edges = csr.edges;
    return cudaSuccess;
  }

  template <typename CsrT_in>
  cudaError_t ToCsr(CsrT_in &csr,
                    util::Location target = util::LOCATION_DEFAULT,
                    cudaStream_t stream = 0, bool quiet = false) {
    this->dynamicGraph.ToCsr(csr.row_offsets.GetPointer(util::DEVICE),
                             csr.column_indices.GetPointer(util::DEVICE),
                             csr.edge_values.GetPointer(util::DEVICE),
                             csr.nodes, csr.edges,
                             csr.node_values.GetPointer(util::DEVICE));
    return cudaSuccess;
  }
};

template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG,
          unsigned int cudaHostRegisterFlag>
struct Dyn<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag, false, true> {
  template <typename PairT>
  cudaError_t InsertEdgesBatch(util::Array1D<SizeT, PairT> src,
                               util::Array1D<SizeT, ValueT> vals,
                               SizeT batchSize, bool batch_directed = true,
                               util::Location target = util::DEVICE) {
    return cudaSuccess;
  }
  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    return cudaSuccess;
  }

  template <typename CsrT_in>
  cudaError_t ToCsr(CsrT_in &csr,
                    util::Location target = util::LOCATION_DEFAULT,
                    cudaStream_t stream = 0, bool quiet = false) {
    return cudaSuccess;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    return cudaSuccess;
  }
};
}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
