// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csr.cuh
 *
 * @brief CSR (Compressed Sparse Row) Graph Data Structure
 */

#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/graph_base.cuh>
#include <gunrock/graph/coo.cuh>
#include <gunrock/util/binary_search.cuh>
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
          bool VALID = true>
struct Csr : public GraphBase<_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSR,
                              cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG | HAS_CSR;
  static const util::ArrayFlag ARRAY_FLAG =
      util::If_Val<(FLAG & GRAPH_PINNED) != 0,
                   (FLAG & ARRAY_RESERVE) | util::PINNED,
                   FLAG & ARRAY_RESERVE>::Value;
  typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
      BaseGraph;
  typedef Csr<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag> CsrT;

  // Column indices corresponding to all the
  // non-zero values in the sparse matrix
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      column_indices;

  // List of indices where each row of the
  // sparse matrix starts
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag> row_offsets;

  typedef util::Array1D<SizeT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      Array_ValueT;
  typedef util::NullArray<SizeT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      Array_NValueT;

  // List of values attached to edges in the graph
  typename util::If<(FLAG & HAS_EDGE_VALUES) != 0, Array_ValueT,
                    Array_NValueT>::Type edge_values;

  // List of values attached to nodes in the graph
  typename util::If<(FLAG & HAS_NODE_VALUES) != 0, Array_ValueT,
                    Array_NValueT>::Type node_values;

  /**
   * @brief CSR Constructor
   *
   * @param[in] pinned Use pinned memory for CSR data structure
   * (default: do not use pinned memory)
   */
  Csr() : BaseGraph() {
    column_indices.SetName("column_indices");
    row_offsets.SetName("row_offsets");
    edge_values.SetName("edge_values");
    node_values.SetName("node_values");
  }

  /**
   * @brief CSR destructor
   */
  __host__ __device__ ~Csr() {
    // Release();
  }

  /**
   * @brief Deallocates CSR graph
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(row_offsets.Release(target));
    GUARD_CU(column_indices.Release(target));
    GUARD_CU(node_values.Release(target));
    GUARD_CU(edge_values.Release(target));
    GUARD_CU(BaseGraph ::Release(target));
    return retval;
  }

  /**
   * @brief Allocate memory for CSR graph.
   *
   * @param[in] nodes Number of nodes in COO-format graph
   * @param[in] edges Number of edges in COO-format graph
   */
  cudaError_t Allocate(SizeT nodes, SizeT edges,
                       util::Location target = GRAPH_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseGraph ::Allocate(nodes, edges, target));
    GUARD_CU(row_offsets.Allocate(nodes + 1, target));
    GUARD_CU(column_indices.Allocate(edges, target));
    GUARD_CU(node_values.Allocate(nodes, target));
    GUARD_CU(edge_values.Allocate(edges, target));
    return retval;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    SizeT invalid_size = util::PreDefinedValues<SizeT>::InvalidValue;
    GUARD_CU(BaseGraph ::Move(source, target, stream));
    GUARD_CU(row_offsets.Move(source, target, invalid_size, 0, stream));
    GUARD_CU(column_indices.Move(source, target, invalid_size, 0, stream));
    GUARD_CU(edge_values.Move(source, target, invalid_size, 0, stream));
    GUARD_CU(node_values.Move(source, target, invalid_size, 0, stream));
    return retval;
  }

  template <typename VertexT_in, typename SizeT_in, typename ValueT_in,
            GraphFlag FLAG_in, unsigned int cudaHostRegisterFlag_in>
  cudaError_t FromCsr(Csr<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
                          cudaHostRegisterFlag_in> &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target =
          source.row_offsets.GetSetted() | source.row_offsets.GetAllocated();

    GUARD_CU(BaseGraph::Set(source));
    GUARD_CU(Allocate(source.nodes, source.edges, target));

    GUARD_CU(
        row_offsets.Set(source.row_offsets, this->nodes + 1, target, stream));

    GUARD_CU(
        column_indices.Set(source.column_indices, this->edges, target, stream));

    GUARD_CU(edge_values.Set(source.edge_values, this->edges, target, stream));

    GUARD_CU(node_values.Set(source.node_values, this->nodes, target, stream));

    return retval;
  }

  /**
   * @brief Build CSR graph from COO graph, sorted or unsorted
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
            (source.CooT::edge_order == BY_ROW_ASCENDING ? " ordered"
                                                         : "unordered") +
            " tuples) to CSR format...",
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
    GUARD_CU(source.CooT::Order(BY_ROW_ASCENDING, target, stream));
    // source.CooT::Display();

    // assign column_indices
    GUARD_CU(column_indices.ForEach(
        source.CooT::edge_pairs,
        [] __host__ __device__(VertexT & column_index,
                               const typename CooT::EdgePairT &edge_pair) {
          column_index = edge_pair.y;
        },
        this->edges, target, stream));

    // assign edge_values
    if (FLAG & HAS_EDGE_VALUES) {
      GUARD_CU(edge_values.ForEach(
          source.CooT::edge_values,
          [] __host__ __device__(ValueT & edge_value,
                                 const typename CooT::ValueT &edge_value_in) {
            edge_value = edge_value_in;
          },
          this->edges, target, stream));
    }

    // assign row_offsets
    SizeT edges = this->edges;
    SizeT nodes = this->nodes;
    auto row_edge_compare = [] __host__ __device__(
                                const typename CooT::EdgePairT &edge_pair,
                                const VertexT &row) {
      return edge_pair.x < row;
    };
    GUARD_CU(row_offsets.ForAll(
        source.CooT::edge_pairs,
        [nodes, edges, row_edge_compare] __host__ __device__(
            SizeT * row_offsets, const typename CooT::EdgePairT *edge_pairs,
            const VertexT &row) {
          if (row <= edge_pairs[0].x)
            row_offsets[row] = 0;
          else if (row < nodes) {
            auto pos = util::BinarySearch_LeftMost(
                row, edge_pairs, (SizeT)0, edges - 1, row_edge_compare,
                [](const typename CooT::EdgePairT &pair, const VertexT &row) {
                  return (pair.x == row);
                });
            // if (row > edge_pairs[edges-1].x)
            //    pos = edges;
            // else {
            while (pos < edges && row > edge_pairs[pos].x) pos++;
            //}
            // if (pos > edges || row >= edge_pairs[edges-1].x)
            //    printf("Error row_offsets[%d] = %d\n",
            //        row, pos);
            row_offsets[row] = pos;
          } else
            row_offsets[row] = edges;
        },
        this->nodes + 1, target, stream));

    time_t mark2 = time(NULL);
    util::PrintMsg("Done (" + std::to_string(mark2 - mark1) + "s).", !quiet);

    // for (SizeT v = 0; v < nodes; v++)
    //{
    //    if (row_offsets [v] > row_offsets[v+1])
    //    {
    //        util::PrintMsg("Error: row_offsets["
    //            + std::to_string(v) + "] = " + std::to_string(row_offsets[v])
    //            + " > row_offsets[" + std::to_string(v+1)
    //            + "] = " + std::to_string(row_offsets[v+1]));
    //        continue;
    //    }
    //
    //    if (row_offsets[v] < 0 || row_offsets[v] > edges)
    //    {
    //        util::PrintMsg("Error: row_offsets["
    //            + std::to_string(v) + "] = " + std::to_string(row_offsets[v])
    //            + " > edges = " + std::to_string(edges));
    //        continue;
    //    }
    //
    //    SizeT e_start = row_offsets[v];
    //    SizeT e_end = row_offsets[v+1];
    //    SizeT degree = e_end - e_start;
    //    for (SizeT e = e_start; e < e_end; e++)
    //    {
    //        if (source.CooT::edge_pairs[e].x != v)
    //            util::PrintMsg("Error: edge_pairs[" + std::to_string(e)
    //                + "] = (" + std::to_string(source.CooT::edge_pairs[e].x)
    //                + ", " + std::to_string(source.CooT::edge_pairs[e].y)
    //                + ") != v " + std::to_string(v));
    //    }
    //}
    return retval;
  }

  template <typename GraphT>
  cudaError_t FromCsc(GraphT &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    typedef typename GraphT::CscT CscT;
    typedef Coo<VertexT, SizeT, ValueT, FLAG | HAS_COO, cudaHostRegisterFlag>
        CooT;

    cudaError_t retval = cudaSuccess;
    CooT coo;
    GUARD_CU(coo.FromCsc(source, target, stream, quiet));
    GUARD_CU(FromCoo(coo, target, stream, quiet));
    GUARD_CU(coo.Release());
    return retval;
  }

  /**
   * @brief Display CSR graph to console
   *
   * @param[in] with_edge_value Whether display graph with edge values.
   */
  cudaError_t Display(std::string graph_prefix = "", SizeT nodes_to_show = 40,
                      bool with_edge_values = true) {
    cudaError_t retval = cudaSuccess;
    if (nodes_to_show > this->nodes) nodes_to_show = this->nodes;
    util::PrintMsg(graph_prefix + "Graph containing " +
                   std::to_string(this->nodes) + " vertices, " +
                   std::to_string(this->edges) + " edges, in CSR format." +
                   " Neighbor list of first " + std::to_string(nodes_to_show) +
                   " nodes :");
    for (SizeT node = 0; node < nodes_to_show; node++) {
      std::string str = "v " + std::to_string(node) + " " +
                        std::to_string(row_offsets[node]) + " : ";
      for (SizeT edge = row_offsets[node]; edge < row_offsets[node + 1];
           edge++) {
        if (edge - row_offsets[node] > 40) break;
        str = str + "[" + std::to_string(column_indices[edge]);
        if (with_edge_values && (FLAG & HAS_EDGE_VALUES)) {
          str = str + "," + std::to_string(edge_values[edge]);
        }
        if (edge - row_offsets[node] != 40 && edge != row_offsets[node + 1] - 1)
          str = str + "], ";
        else
          str = str + "]";
      }
      if (row_offsets[node + 1] - row_offsets[node] > 40) str = str + "...";
      util::PrintMsg(str);
    }
    return retval;
  }

  /**
   * @brief Sort CSR graph edges per vertex in ascending order
   *
   */
  cudaError_t Sort() {
    cudaError_t retval = cudaSuccess;
    SizeT num_nodes = this->nodes;
    SizeT num_edges = this->edges;

    typedef std::pair<VertexT, ValueT> EdgeValPairT;
    util::Array1D<SizeT, EdgeValPairT> sorted_neighbors;
    GUARD_CU(sorted_neighbors.Allocate(num_edges, util::HOST));
#pragma omp parallel
    do {
      int thread_num = omp_get_thread_num();
      int num_threads = omp_get_num_threads();
      SizeT node_start = (SizeT)(num_nodes)*thread_num / num_threads;
      SizeT node_end = (SizeT)(num_nodes) * (thread_num + 1) / num_threads;
      node_end = (thread_num == (num_threads - 1)) ? num_nodes : node_end;
      for (SizeT node = node_start; node < node_end; node++) {
        SizeT start_offset = row_offsets[node];
        SizeT end_offset = row_offsets[node + 1];
        for (SizeT off = start_offset; off < end_offset; off++) {
          sorted_neighbors[off] =
              std::make_pair(column_indices[off], edge_values[off]);
        }
        std::sort(sorted_neighbors + start_offset,
                  sorted_neighbors + end_offset,
                  [](const EdgeValPairT &a, const EdgeValPairT &b) -> bool {
                    return a.first < b.first;
                  });
        for (SizeT off = start_offset; off < end_offset; off++) {
          column_indices[off] = sorted_neighbors[off].first;
          edge_values[off] = sorted_neighbors[off].second;
        }
      }
    } while (false);
    GUARD_CU(sorted_neighbors.Release(util::HOST));
    return cudaSuccess;
  }

  __device__ __host__ __forceinline__ SizeT
  GetNeighborListLength(const VertexT &v) const {
    if (util::lessThanZero(v) || v >= this->nodes) return 0;
    return _ldg(row_offsets + (v + 1)) - _ldg(row_offsets + v);
  }

  __device__ __host__ __forceinline__ SizeT
  GetNeighborListOffset(const VertexT &v) const {
    return _ldg(row_offsets + v);
  }

  __device__ __host__ __forceinline__ VertexT GetEdgeSrc(const SizeT &e) const {
    return util::BinarySearch_RightMost(e, row_offsets + 0, (SizeT)0,
                                        this->nodes);
  }

  __device__ __host__ __forceinline__ VertexT
  GetEdgeDest(const SizeT &e) const {
    // return _ldg(column_indices + e);
    return column_indices[e];
  }

  __device__ __host__ __forceinline__ void GetEdgeSrcDest(const SizeT &e,
                                                          VertexT &src,
                                                          VertexT &dest) const {
    src =
        util::BinarySearch_RightMost(e, row_offsets + 0, (SizeT)0, this->nodes);
    dest = column_indices[e];
  }

  /*template <typename Tuple>
  void CsrToCsc(Csr<VertexId, SizeT, Value> &target,
          Csr<VertexId, SizeT, Value> &source)
  {
      target.nodes = source.nodes;
      target.edges = source.edges;
      target.average_degree = source.average_degree;
      target.average_edge_value = source.average_edge_value;
      target.average_node_value = source.average_node_value;
      target.out_nodes = source.out_nodes;
      {
          Tuple *coo = (Tuple*)malloc(sizeof(Tuple) * source.edges);
          int idx = 0;
          for (int i = 0; i < source.nodes; ++i)
          {
              for (int j = source.row_offsets[i]; j < source.row_offsets[i+1];
  ++j)
              {
                  coo[idx].row = source.column_indices[j];
                  coo[idx].col = i;
                  coo[idx++].val = (source.edge_values == NULL) ? 0 :
  source.edge_values[j];
              }
          }
          if (source.edge_values == NULL)
              target.template FromCoo<false>(NULL, coo, nodes, edges);
          else
              target.template FromCoo<true>(NULL, coo, nodes, edges);
          free(coo);
      }
  }*/

  /**
   *
   * @brief Store graph information into a file.
   *
   * @param[in] file_name Original graph file path and name.
   * @param[in] v Number of vertices in input graph.
   * @param[in] e Number of edges in input graph.
   * @param[in] row Row-offsets array store row pointers.
   * @param[in] col Column-indices array store destinations.
   * @param[in] edge_values Per edge weight values associated.
   *
   */
  /*void WriteBinary(
      char  *file_name,
      SizeT v,
      SizeT e,
      SizeT *row,
      VertexId *col,
      Value *edge_values = NULL)
  {
      std::ofstream fout(file_name);
      if (fout.is_open())
      {
          fout.write(reinterpret_cast<const char*>(&v), sizeof(SizeT));
          fout.write(reinterpret_cast<const char*>(&e), sizeof(SizeT));
          fout.write(reinterpret_cast<const char*>(row), (v + 1)*sizeof(SizeT));
          fout.write(reinterpret_cast<const char*>(col), e * sizeof(VertexId));
          if (edge_values != NULL)
          {
              fout.write(reinterpret_cast<const char*>(edge_values),
                         e * sizeof(Value));
          }
          fout.close();
      }
  }*/

  /*
   * @brief Write human-readable CSR arrays into 3 files.
   * Can be easily used for python interface.
   *
   * @param[in] file_name Original graph file path and name.
   * @param[in] v Number of vertices in input graph.
   * @param[in] e Number of edges in input graph.
   * @param[in] row_offsets Row-offsets array store row pointers.
   * @param[in] col_indices Column-indices array store destinations.
   * @param[in] edge_values Per edge weight values associated.
   */
  /*void WriteCSR(
      char *file_name,
      SizeT v, SizeT e,
      SizeT    *row_offsets,
      VertexId *col_indices,
      Value    *edge_values = NULL)
  {
      std::cout << file_name << std::endl;
      char rows[256], cols[256], vals[256];

      sprintf(rows, "%s.rows", file_name);
      sprintf(cols, "%s.cols", file_name);
      sprintf(vals, "%s.vals", file_name);

      std::ofstream rows_output(rows);
      if (rows_output.is_open())
      {
          std::copy(row_offsets, row_offsets + v + 1,
                    std::ostream_iterator<SizeT>(rows_output, "\n"));
          rows_output.close();
      }

      std::ofstream cols_output(cols);
      if (cols_output.is_open())
      {
          std::copy(col_indices, col_indices + e,
                    std::ostream_iterator<VertexId>(cols_output, "\n"));
          cols_output.close();
      }

      if (edge_values != NULL)
      {
          std::ofstream vals_output(vals);
          if (vals_output.is_open())
          {
              std::copy(edge_values, edge_values + e,
                        std::ostream_iterator<Value>(vals_output, "\n"));
              vals_output.close();
          }
      }
  }*/

  /*
   * @brief Write Ligra input CSR arrays into .adj file.
   * Can be easily used for python interface.
   *
   * @param[in] file_name Original graph file path and name.
   * @param[in] v Number of vertices in input graph.
   * @param[in] e Number of edges in input graph.
   * @param[in] row Row-offsets array store row pointers.
   * @param[in] col Column-indices array store destinations.
   * @param[in] edge_values Per edge weight values associated.
   * @param[in] quiet Don't print out anything.
   */
  /*void WriteToLigraFile(
      const char  *file_name,
      SizeT v, SizeT e,
      SizeT *row,
      VertexId *col,
      Value *edge_values = NULL,
      bool quiet = false)
  {
      char adj_name[256];
      sprintf(adj_name, "%s.adj", file_name);
      if (!quiet)
      {
          printf("writing to ligra .adj file.\n");
      }

      std::ofstream fout3(adj_name);
      if (fout3.is_open())
      {
          fout3 << "AdjacencyGraph" << std::endl << v << std::endl << e <<
  std::endl; for (int i = 0; i < v; ++i) fout3 << row[i] << std::endl; for (int
  i = 0; i < e; ++i) fout3 << col[i] << std::endl; if (edge_values != NULL)
          {
              for (int i = 0; i < e; ++i)
                  fout3 << edge_values[i] << std::endl;
          }
          fout3.close();
      }
  }

  void WriteToMtxFile(
      const char  *file_name,
      SizeT v, SizeT e,
      SizeT *row,
      VertexId *col,
      Value *edge_values = NULL,
      bool quiet = false)
  {
      char adj_name[256];
      sprintf(adj_name, "%s.mtx", file_name);
      if (!quiet)
      {
          printf("writing to .mtx file.\n");
      }

      std::ofstream fout3(adj_name);
      if (fout3.is_open())
      {
          fout3 << v << " " << v << " " << e << std::endl;
          for (int i = 0; i < v; ++i) {
              SizeT begin = row[i];
              SizeT end = row[i+1];
              for (int j = begin; j < end; ++j) {
                  fout3 << col[j]+1 << " " << i+1;
                  if (edge_values != NULL)
                  {
                      fout3 << " " << edge_values[j] << std::endl;
                  }
                  else
                  {
                      fout3 << " " << rand() % 64 << std::endl;
                  }
              }
          }
          fout3.close();
      }
  }*/

  /**
   * @brief Read from stored row_offsets, column_indices arrays.
   *
   * @tparam LOAD_EDGE_VALUES Whether or not to load edge values.
   *
   * @param[in] f_in Input file name.
   * @param[in] quiet Don't print out anything.
   */
  /*template <bool LOAD_EDGE_VALUES>
  void FromCsr(char *f_in, bool quiet = false)
  {
      if (!quiet)
      {
          printf("  Reading directly from stored binary CSR arrays ...\n");
      }
      time_t mark1 = time(NULL);

      std::ifstream input(f_in);
      SizeT v, e;
      input.read(reinterpret_cast<char*>(&v), sizeof(SizeT));
      input.read(reinterpret_cast<char*>(&e), sizeof(SizeT));

      FromScratch<LOAD_EDGE_VALUES, false>(v, e);

      input.read(reinterpret_cast<char*>(row_offsets), (v + 1)*sizeof(SizeT));
      input.read(reinterpret_cast<char*>(column_indices), e * sizeof(VertexId));
      if (LOAD_EDGE_VALUES)
      {
          input.read(reinterpret_cast<char*>(edge_values), e * sizeof(Value));
      }

      time_t mark2 = time(NULL);
      if (!quiet)
      {
          printf("Done reading (%ds).\n", (int) (mark2 - mark1));
      }

      // compute out_nodes
      SizeT out_node = 0;
      for (SizeT node = 0; node < nodes; node++)
      {
          if (row_offsets[node + 1] - row_offsets[node] > 0)
          {
              ++out_node;
          }
      }
      out_nodes = out_node;
  }*/

  /**
   * @brief (Specific for SM) Read from stored row_offsets, column_indices
   * arrays.
   *
   * @tparam LOAD_NODE_VALUES Whether or not to load node values.
   *
   * @param[in] f_in Input graph file name.
   * @param[in] f_label Input label file name.
   * @param[in] quiet Don't print out anything.
   */
  /*template <bool LOAD_NODE_VALUES>
  void FromCsr_SM(char *f_in, char *f_label, bool quiet = false)
  {
      if (!quiet)
      {
          printf("  Reading directly from stored binary CSR arrays ...\n");
      if(LOAD_NODE_VALUES)
              printf("  Reading directly from stored binary label arrays
...\n");
      }
      time_t mark1 = time(NULL);

      std::ifstream input(f_in);
      std::ifstream input_label(f_label);

      SizeT v, e;
      input.read(reinterpret_cast<char*>(&v), sizeof(SizeT));
      input.read(reinterpret_cast<char*>(&e), sizeof(SizeT));

      FromScratch<false, LOAD_NODE_VALUES>(v, e);

      input.read(reinterpret_cast<char*>(row_offsets), (v + 1)*sizeof(SizeT));
      input.read(reinterpret_cast<char*>(column_indices), e * sizeof(VertexId));
      if (LOAD_NODE_VALUES)
      {
          input_label.read(reinterpret_cast<char*>(node_values), v *
sizeof(Value));
      }
//      for(int i=0; i<v; i++) printf("%lld ", (long long)node_values[i]);
printf("\n");

      time_t mark2 = time(NULL);
      if (!quiet)
      {
          printf("Done reading (%ds).\n", (int) (mark2 - mark1));
      }

      // compute out_nodes
      SizeT out_node = 0;
      for (SizeT node = 0; node < nodes; node++)
      {
          if (row_offsets[node + 1] - row_offsets[node] > 0)
          {
              ++out_node;
          }
      }
      out_nodes = out_node;
  }*/

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Check values.
   */
  /*bool CheckValue()
  {
      for (SizeT node = 0; node < nodes; ++node)
      {
          for (SizeT edge = row_offsets[node];
                  edge < row_offsets[node + 1];
                  ++edge)
          {
              int src_node = node;
              int dst_node = column_indices[edge];
              int edge_value = edge_values[edge];
              for (SizeT r_edge = row_offsets[dst_node];
                      r_edge < row_offsets[dst_node + 1];
                      ++r_edge)
              {
                  if (column_indices[r_edge] == src_node)
                  {
                      if (edge_values[r_edge] != edge_value)
                          return false;
                  }
              }
          }
      }
      return true;
  }*/

  /**
   * @brief Find node with largest neighbor list
   * @param[in] max_degree Maximum degree in the graph.
   *
   * \return int the source node with highest degree
   */
  /*int GetNodeWithHighestDegree(int& max_degree)
  {
      int degree = 0;
      int src = 0;
      for (SizeT node = 0; node < nodes; node++)
      {
          if (row_offsets[node + 1] - row_offsets[node] > degree)
          {
              degree = row_offsets[node + 1] - row_offsets[node];
              src = node;
          }
      }
      max_degree = degree;
      return src;
  }*/

  /**
   * @brief Display the neighbor list of a given node.
   *
   * @param[in] node Vertex ID to display.
   */
  /*void DisplayNeighborList(VertexId node)
  {
      if (node < 0 || node >= nodes) return;
      for (SizeT edge = row_offsets[node];
              edge < row_offsets[node + 1];
              edge++)
      {
          util::PrintValue(column_indices[edge]);
          printf(", ");
      }
      printf("\n");
  }*/

  /**
   * @brief Get the degrees of all the nodes in graph
   *
   * @param[in] node_degrees node degrees to fill in
   */
  /*void GetNodeDegree(unsigned long long *node_degrees)
  {
  for(SizeT node=0; node < nodes; ++node)
  {
      node_degrees[node] = row_offsets[node+1]-row_offsets[node];
  }
  }*/

  /**
   * @brief Get the average node value in graph
   */
  /*Value GetAverageNodeValue()
  {
      if (abs(average_node_value - 0) < 0.001 && node_values != NULL)
      {
          double mean = 0, count = 0;
          for (SizeT node = 0; node < nodes; ++node)
          {
              if (node_values[node] < UINT_MAX)
              {
                  count += 1;
                  mean += (node_values[node] - mean) / count;
              }
          }
          average_node_value = static_cast<Value>(mean);
      }
      return average_node_value;
  }*/

  /**
   * @brief Get the average edge value in graph
   */
  /*Value GetAverageEdgeValue()
  {
      if (abs(average_edge_value - 0) < 0.001 && edge_values != NULL)
      {
          double mean = 0, count = 0;
          for (SizeT edge = 0; edge < edges; ++edge)
          {
              if (edge_values[edge] < UINT_MAX)
              {
                  count += 1;
                  mean += (edge_values[edge] - mean) / count;
              }
          }
      }
      return average_edge_value;
  }*/

  /**@}*/
};  // CSR

template <typename VertexT, typename SizeT, typename ValueT, GraphFlag _FLAG,
          unsigned int cudaHostRegisterFlag>
struct Csr<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag, false> {
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

  cudaError_t Sort() { return cudaSuccess; }

  __device__ __host__ __forceinline__ SizeT
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
