//----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csc.cuh
 *
 * @brief CSC (Compressed Sparse Column) Graph Data Structure
 */

#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/graph_base.cuh>
#include <gunrock/graph/coo.cuh>
#include <gunrock/util/binary_search.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief CSC data structure which uses Compressed Sparse Column
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE | HAS_CSC,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
          bool VALID = true>
struct Csc : public GraphBase<_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSC,
                              cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG | HAS_CSC;
  static const util::ArrayFlag ARRAY_FLAG =
      util::If_Val<(FLAG & GRAPH_PINNED) != 0,
                   (FLAG & ARRAY_RESERVE) | util::PINNED,
                   FLAG & ARRAY_RESERVE>::Value;
  typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
      BaseGraph;
  typedef Csc<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag> CscT;
  typedef typename util::If<(FLAG & HAS_EDGE_VALUES) != 0, ValueT,
                            util::NullType>::Type EdgeValueT;
  typedef typename util::If<(FLAG & HAS_NODE_VALUES) != 0, ValueT,
                            util::NullType>::Type NodeValueT;

  // Column indices corresponding to all the
  // non-zero values in the sparse matrix
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag> row_indices;

  // List of indices where each row of the
  // sparse matrix starts
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag> column_offsets;

  // typedef util::Array1D<SizeT, ValueT, ARRAY_FLAG,
  //    cudaHostRegisterFlag> Array_ValueT;
  // typedef util::NullArray<SizeT, ValueT, ARRAY_FLAG,
  //    cudaHostRegisterFlag> Array_NValueT;

  // List of values attached to edges in the graph
  // typename util::If<(FLAG & HAS_EDGE_VALUES) != 0,
  //    Array_ValueT, Array_NValueT >::Type edge_values;
  util::Array1D<SizeT, EdgeValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      edge_values;

  // List of values attached to nodes in the graph
  // typename util::If<(FLAG & HAS_NODE_VALUES) != 0,
  //    Array_ValueT, Array_NValueT >::Type node_values;
  // Array_ValueT node_values;
  util::Array1D<SizeT, NodeValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      node_values;

  /**
   * @brief CSC Constructor
   *
   * @param[in] pinned Use pinned memory for CSC data structure
   * (default: do not use pinned memory)
   */
  Csc() : BaseGraph() {
    column_offsets.SetName("column_offsets");
    row_indices.SetName("row_indices");
    edge_values.SetName("edge_values");
    node_values.SetName("node_values");
  }

  /**
   * @brief CSC destructor
   */
  __host__ __device__ ~Csc() {
    // Release();
  }

  /**
   * @brief Deallocates CSC graph
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(column_offsets.Release(target));
    GUARD_CU(row_indices.Release(target));
    GUARD_CU(node_values.Release(target));
    GUARD_CU(edge_values.Release(target));
    GUARD_CU(BaseGraph ::Release(target));
    return retval;
  }

  /**
   * @brief Allocate memory for CSC graph.
   *
   * @param[in] nodes Number of nodes in COO-format graph
   * @param[in] edges Number of edges in COO-format graph
   */
  cudaError_t Allocate(SizeT nodes, SizeT edges,
                       util::Location target = GRAPH_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseGraph ::Allocate(nodes, edges, target));
    GUARD_CU(column_offsets.Allocate(nodes + 1, target));
    GUARD_CU(row_indices.Allocate(edges, target));
    GUARD_CU(node_values.Allocate(nodes, target));
    GUARD_CU(edge_values.Allocate(edges, target));
    return retval;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    // SizeT invalid_size = util::PreDefinedValues<SizeT>::InvalidValue;
    GUARD_CU(BaseGraph ::Move(source, target, stream));
    GUARD_CU(column_offsets.Move(source, target, this->nodes + 1, 0, stream));
    GUARD_CU(row_indices.Move(source, target, this->edges, 0, stream));
    GUARD_CU(node_values.Move(source, target, this->nodes, 0, stream));
    GUARD_CU(edge_values.Move(source, target, this->edges, 0, stream));
    return retval;
  }

  template <typename VertexT_in, typename SizeT_in, typename ValueT_in,
            GraphFlag FLAG_in, unsigned int cudaHostRegisterFlag_in>
  cudaError_t FromCsc(Csc<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
                          cudaHostRegisterFlag_in> &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target = source.column_offsets.GetSetted() |
               source.column_offsets.GetAllocated();

    // if (retval = BaseGraph::Set(source))
    //    return retval;
    this->nodes = source.nodes;
    this->edges = source.edges;
    this->directed = source.directed;

    GUARD_CU(Allocate(source.nodes, source.edges, target));

    GUARD_CU(column_offsets.Set(source.column_offsets, this->nodes + 1, target,
                                stream));

    GUARD_CU(row_indices.Set(source.row_indices, this->edges, target, stream));

    GUARD_CU(edge_values.Set(source.edge_values, this->edges, target, stream));

    GUARD_CU(node_values.Set(source.node_values, this->nodes, target, stream));

    return retval;
  }

  /**
   * @brief Build CSC graph from COO graph, sorted or unsorted
   *
   * @param[in] output_file Output file to dump the graph topology info
   * @param[in] coo Pointer to COO-format graph
   * @param[in] coo_nodes Number of nodes in COO-format graph
   * @param[in] coo_edges Number of edges in COO-format graph
   * @param[in] ordered_rows Are the rows sorted? If not, sort them.
   * @param[in] undirected Is the graph directed or not?
   * @param[in] reversed Is the graph reversed or not?
   * @param[in] quiet Don't print out anything.
   *
   * Default: Assume rows are not sorted.
   */
  template <typename GraphT>
  cudaError_t FromCoo(GraphT &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0,
                      // bool  ordered_rows = false,
                      // bool  undirected = false,
                      // bool  reversed = false,
                      bool quiet = false) {
    typedef typename GraphT::CooT CooT;
    // typedef Coo<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
    //    cudaHostRegisterFlag_in> CooT;
    util::PrintMsg(
        "Converting " + std::to_string(source.CooT::nodes) + " vertices, " +
            std::to_string(source.CooT::edges) +
            (source.CooT::directed ? " directed" : " undirected") + " edges (" +
            (source.CooT::edge_order == BY_COLUMN_ASCENDING ? " ordered"
                                                            : "unordered") +
            " tuples) to CSC format...",
        !quiet, false);

    time_t mark1 = time(NULL);
    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target = source.CooT::edge_pairs.GetSetted() |
               source.CooT::edge_pairs.GetAllocated();

    /*if (retval = BaseGraph:: template Set<typename CooT::CooT>((typename
       CooT::CooT)source)) return retval;
    */
    this->nodes = source.CooT::nodes;
    this->edges = source.CooT::edges;
    this->directed = source.CooT::directed;

    GUARD_CU(Allocate(source.CooT::nodes, source.CooT::edges, target));

    // Sort COO by row
    GUARD_CU(source.CooT::Order(BY_COLUMN_ASCENDING, target, stream));
    // source.CooT::Display();

    // assign row_indices
    GUARD_CU(row_indices.ForEach(
        source.CooT::edge_pairs,
        [] __host__ __device__(VertexT & row_index,
                               const typename CooT::EdgePairT &edge_pair) {
          row_index = edge_pair.x;
        },
        this->edges, target, stream));

    // assign edge_values
    if (FLAG & HAS_EDGE_VALUES) {
      GUARD_CU(edge_values.ForEach(
          source.CooT::edge_values,
          [] __host__ __device__(EdgeValueT & edge_value,
                                 const typename CooT::ValueT &edge_value_in) {
            edge_value = edge_value_in;
          },
          this->edges, target, stream));
    }

    if (FLAG & HAS_NODE_VALUES) {
      GUARD_CU(node_values.ForEach(
          source.CooT::node_values,
          [] __host__ __device__(NodeValueT & node_value,
                                 const typename CooT::ValueT &node_value_in) {
            node_value = node_value_in;
          },
          this->nodes, target, stream));
    }

    // assign column_offsets
    SizeT edges = this->edges;
    SizeT nodes = this->nodes;
    auto column_edge_compare = [] __host__ __device__(
                                   const typename CooT::EdgePairT &edge_pair,
                                   const VertexT &column) {
      return edge_pair.y < column;
    };
    GUARD_CU(column_offsets.ForAll(
        source.CooT::edge_pairs,
        [nodes, edges, column_edge_compare] __host__ __device__(
            SizeT * column_offsets, const typename CooT::EdgePairT *edge_pairs,
            const VertexT &column) {
          if (column <= edge_pairs[0].y)
            column_offsets[column] = 0;
          else if (column < nodes) {
            auto pos = util::BinarySearch_LeftMost(
                column, edge_pairs, (SizeT)0, edges - 1, column_edge_compare,
                [](const typename CooT::EdgePairT &pair,
                   const VertexT &column) { return (pair.y == column); });
            while (pos < edges && column > edge_pairs[pos].y) pos++;
            column_offsets[column] = pos;
          } else
            column_offsets[column] = edges;
        },
        this->nodes + 1, target, stream));

    time_t mark2 = time(NULL);
    util::PrintMsg("Done (" + std::to_string(mark2 - mark1) + "s).", !quiet);

    return retval;
  }

  template <typename GraphT>
  cudaError_t FromCsr(GraphT &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    typedef typename GraphT::CsrT CsrT;
    typedef Coo<VertexT, SizeT, ValueT, FLAG | HAS_COO, cudaHostRegisterFlag>
        CooT;
    cudaError_t retval = cudaSuccess;

    CooT coo;
    GUARD_CU(coo.FromCsr(source, target, stream, quiet));
    GUARD_CU(FromCoo(coo, target, stream, quiet));
    GUARD_CU(coo.Release());
    return retval;
  }

  /**
   * @brief Display CSC graph to console
   *
   * @param[in] with_edge_value Whether display graph with edge values.
   */
  cudaError_t Display(std::string graph_prefix = "", SizeT nodes_to_show = 40,
                      bool with_edge_values = true) {
    cudaError_t retval = cudaSuccess;
    if (nodes_to_show > this->nodes) nodes_to_show = this->nodes;
    util::PrintMsg(graph_prefix + "Graph containing " +
                   std::to_string(this->nodes) + " vertices, " +
                   std::to_string(this->edges) + " edges, in CSC format." +
                   " Neighbor list of first " + std::to_string(nodes_to_show) +
                   " nodes :");
    for (SizeT node = 0; node < nodes_to_show; node++) {
      std::string str = "";
      for (SizeT edge = column_offsets[node]; edge < column_offsets[node + 1];
           edge++) {
        if (edge - column_offsets[node] > 40) break;
        str = str + "[" + std::to_string(row_indices[edge]);
        if (with_edge_values && (FLAG & HAS_EDGE_VALUES)) {
          str = str + "," + std::to_string(edge_values[edge]);
        }
        if (edge - column_offsets[node] != 40 &&
            edge != column_offsets[node + 1] - 1)
          str = str + "], ";
        else
          str = str + "]";
      }
      if (column_offsets[node + 1] - column_offsets[node] > 40)
        str = str + "...";
      str = str + " : v " + std::to_string(node) + " " +
            std::to_string(column_offsets[node]);
      util::PrintMsg(str);
    }
    return retval;
  }

  __device__ __host__ __forceinline__ SizeT
  GetNeighborListLength(const VertexT &v) const {
    if (util::lessThanZero(v) || v >= this->nodes) return 0;
    return _ldg(column_offsets + (v + 1)) - _ldg(column_offsets + v);
  }

  __device__ __host__ __forceinline__ SizeT
  GetNeighborListOffset(const VertexT &v) const {
    return _ldg(column_offsets + v);
  }

  __device__ __host__ __forceinline__ VertexT GetEdgeSrc(const SizeT &e) const {
    return util::BinarySearch_RightMost(e, column_offsets + 0, (SizeT)0,
                                        this->nodes);
  }

  __device__ __host__ __forceinline__ VertexT
  GetEdgeDest(const SizeT &e) const {
    // return _ldg(row_indices + e);
    return row_indices[e];
  }

  __device__ __host__ __forceinline__ void GetEdgeSrcDest(const SizeT &e,
                                                          VertexT &src,
                                                          VertexT &dest) const {
    src = util::BinarySearch_RightMost(e, column_offsets + 0, (SizeT)0,
                                       this->nodes);
    dest = row_indices[e];
  }
};  // CSC

template <typename VertexT, typename SizeT, typename ValueT, GraphFlag _FLAG,
          unsigned int cudaHostRegisterFlag>
struct Csc<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag, false> {
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    return cudaSuccess;
  }

  template <typename CooT_in>
  cudaError_t FromCoo(CooT_in &coo,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    return cudaSuccess;
  }

  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    return cudaSuccess;
  }

  template <typename CscT_in>
  cudaError_t FromCsc(CscT_in &csc,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    return cudaSuccess;
  }

  __host__ __device__ __forceinline__ SizeT
  GetNeighborListLength(const VertexT &v) const {
    return 0;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  cudaError_t Display(std::string graph_prefix = "", SizeT nodes_to_show = 40,
                      bool with_edge_values = true) {
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
