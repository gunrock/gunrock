// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * coo.cuh
 *
 * @brief Coordinate Format (a.k.a. triplet format) Graph Data Structure
 */

#pragma once

#include <math.h>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/vector_types.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/oprtr/1D_oprtr/sort.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_1D.cuh>
#include <gunrock/graph/graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief COO data structure which uses Coordinate
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE | HAS_COO,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
          bool VALID = true>
struct Coo : public GraphBase<_VertexT, _SizeT, _ValueT, _FLAG | HAS_COO,
                              cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG | HAS_COO;
  static const util::ArrayFlag ARRAY_FLAG =
      util::If_Val<(FLAG & GRAPH_PINNED) != 0,
                   (FLAG & ARRAY_RESERVE) | util::PINNED,
                   FLAG & ARRAY_RESERVE>::Value;
  typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
      BaseGraph;
  typedef Coo<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag> CooT;

  typedef typename util::VectorType<VertexT, 2>::Type EdgePairT;

  // whether the edges are edge_order
  EdgeOrder edge_order;

  // Source (.x) and Destination (.y) of edges
  util::Array1D<SizeT, EdgePairT, ARRAY_FLAG, cudaHostRegisterFlag> edge_pairs;

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
   * @brief COO Constructor
   */
  Coo() : BaseGraph() {
    edge_pairs.SetName("edge_pairs");
    edge_values.SetName("edge_values");
    node_values.SetName("node_values");
    edge_order = UNORDERED;
  }

  /**
   * @brief COO destructor
   */
  __host__ __device__ ~Coo() {
    // Release();
  }

  /**
   * @brief Deallocates CSR graph
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(edge_pairs.Release(target));
    GUARD_CU(node_values.Release(target));
    GUARD_CU(edge_values.Release(target));
    GUARD_CU(BaseGraph ::Release(target));
    return retval;
  }

  /**
   * @brief Allocate memory for COO graph.
   *
   * @param[in] nodes Number of nodes in COO-format graph
   * @param[in] edges Number of edges in COO-format graph
   */
  cudaError_t Allocate(SizeT nodes, SizeT edges,
                       util::Location target = GRAPH_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseGraph ::Allocate(nodes, edges, target));
    GUARD_CU(edge_pairs.Allocate(edges, target));
    GUARD_CU(node_values.Allocate(nodes, target));
    GUARD_CU(edge_values.Allocate(edges, target));
    return retval;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    SizeT invalid_size = util::PreDefinedValues<SizeT>::InvalidValue;
    GUARD_CU(BaseGraph::Move(source, target, stream));
    GUARD_CU(edge_pairs.Move(source, target, invalid_size, 0, stream));
    GUARD_CU(node_values.Move(source, target, invalid_size, 0, stream));
    GUARD_CU(edge_values.Move(source, target, invalid_size, 0, stream));
    return retval;
  }

  cudaError_t Display(std::string graph_prefix = "", SizeT edges_to_show = 40,
                      bool with_edge_values = true) {
    cudaError_t retval = cudaSuccess;
    if (edges_to_show > this->edges) edges_to_show = this->edges;
    util::PrintMsg(
        graph_prefix + "Graph containing " + std::to_string(this->nodes) +
        " vertices, " + std::to_string(this->edges) +
        " edges, in COO format, ordered " + EdgeOrder_to_string(edge_order) +
        ". First " + std::to_string(edges_to_show) + " edges :");
    for (SizeT e = 0; e < edges_to_show; e++)
      util::PrintMsg("e " + std::to_string(e) + " : " +
                     std::to_string(edge_pairs[e].x) + " -> " +
                     std::to_string(edge_pairs[e].y) +
                     (((FLAG & HAS_EDGE_VALUES) && (with_edge_values))
                          ? (" (" + std::to_string(edge_values[e]) + ")")
                          : ""));
    return retval;
  }

  template <typename VertexT_in, typename SizeT_in, typename ValueT_in,
            GraphFlag FLAG_in, unsigned int cudaHostRegisterFlag_in>
  cudaError_t FromCoo(Coo<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
                          cudaHostRegisterFlag_in> &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target = source.edge_pairs.GetSetted() | source.edge_pairs.GetAllocated();

    this->edge_order = source.edge_order;
    GUARD_CU(BaseGraph::Set(source));
    GUARD_CU(Allocate(source.nodes, source.edges, target));

    GUARD_CU(edge_pairs.Set(source.edge_pairs, this->edges, target, stream));

    GUARD_CU(edge_values.Set(source.edge_values, this->edges, target, stream));

    GUARD_CU(node_values.Set(source.node_values, this->nodes, target, stream));

    return retval;
  }

  template <typename GraphT>
  cudaError_t FromCsr(GraphT &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    typedef typename GraphT::CsrT CsrT;
    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target = source.CsrT::row_offsets.GetSetted() |
               source.CsrT::row_offsets.GetAllocated();

    // if (retval = BaseGraph::Set(source))
    //    return retval;
    this->nodes = source.CsrT::nodes;
    this->edges = source.CsrT::edges;
    this->directed = source.CsrT::directed;
    this->edge_order = UNORDERED;

    GUARD_CU(Allocate(source.CsrT::nodes, source.CsrT::edges, target));

    GUARD_CU(source.CsrT::row_offsets.ForAll(
        edge_pairs, source.CsrT::column_indices,
        [] __host__ __device__(
            typename CsrT::SizeT * row_offsets, EdgePairT * edge_pairs,
            typename CsrT::VertexT * column_indices, const VertexT &row) {
          SizeT e_end = row_offsets[row + 1];
          for (SizeT e = row_offsets[row]; e < e_end; e++) {
            edge_pairs[e].x = row;
            edge_pairs[e].y = column_indices[e];
          }
        },
        this->nodes, target, stream));

    GUARD_CU(
        edge_values.Set(source.CsrT::edge_values, this->edges, target, stream));

    GUARD_CU(
        node_values.Set(source.CsrT::node_values, this->nodes, target, stream));
    return retval;
  }

  template <typename GraphT>
  cudaError_t FromCsc(GraphT &source,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false) {
    typedef typename GraphT::CscT CscT;

    cudaError_t retval = cudaSuccess;
    if (target == util::LOCATION_DEFAULT)
      target = source.edge_pairs.GetSetted() | source.edge_pairs.GetAllocated();

    // if (retval = BaseGraph::template Set<typename CscT::CscT>((typename
    // CscT::CscT)source))
    //    return retval;
    this->nodes = source.CscT::nodes;
    this->edges = source.CscT::edges;
    this->directed = source.CscT::directed;
    this->edge_order = UNORDERED;

    GUARD_CU(Allocate(source.CscT::nodes, source.CscT::edges, target));

    // util::PrintMsg("1");
    // for (SizeT v = 0; v<this -> nodes; v++)
    //    printf("O[%d] = %d\t", v, source.column_offsets[v]);
    // printf("\n");
    // fflush(stdout);

    GUARD_CU(source.column_offsets.ForAll(
        edge_pairs, source.row_indices,
        [] __host__ __device__(
            typename CscT::SizeT * column_offsets, EdgePairT * edge_pairs,
            typename CscT::VertexT * row_indices, const VertexT &column) {
          SizeT e_end = column_offsets[column + 1];
          for (SizeT e = column_offsets[column]; e < e_end; e++) {
            edge_pairs[e].x = row_indices[e];
            edge_pairs[e].y = column;
          }
        },
        this->nodes, target, stream));

    // util::PrintMsg("2");
    GUARD_CU(
        edge_values.Set(source.CscT::edge_values, this->edges, target, stream));

    // util::PrintMsg("3");
    GUARD_CU(
        node_values.Set(source.CscT::node_values, this->nodes, target, stream));
    // util::PrintMsg("4");
    return retval;
  }

  cudaError_t Order(EdgeOrder new_order = BY_ROW_ASCENDING,
                    util::Location target = util::LOCATION_DEFAULT,
                    cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;

    if (new_order == edge_order) return retval;

    if (target == util::LOCATION_DEFAULT)
      target = edge_pairs.GetSetted() | edge_pairs.GetAllocated();

    auto row_ascen_order = [] __host__ __device__(const EdgePairT &e1,
                                                  const EdgePairT &e2) {
      if (e1.x > e2.x) return false;
      if (e1.x < e2.x) return true;
      if (e1.y > e2.y) return false;
      return true;
    };

    auto row_decen_order = [] __host__ __device__(const EdgePairT &e1,
                                                  const EdgePairT &e2) {
      if (e1.x < e2.x) return false;
      if (e1.x > e2.x) return true;
      if (e1.y < e2.y) return false;
      return true;
    };

    auto column_ascen_order = [] __host__ __device__(const EdgePairT &e1,
                                                     const EdgePairT &e2) {
      if (e1.y > e2.y) return false;
      if (e1.y < e2.y) return true;
      if (e1.x > e2.x) return false;
      return true;
    };

    auto column_decen_order = [] __host__ __device__(const EdgePairT &e1,
                                                     const EdgePairT &e2) {
      if (e1.y < e2.y) return false;
      if (e1.y > e2.y) return true;
      if (e1.x < e2.x) return false;
      return true;
    };

    // util::PrintMsg("Before Sorting");
    // Display();
    // Sort
    if (FLAG & HAS_EDGE_VALUES) {
      switch (new_order) {
        case BY_ROW_ASCENDING:
          retval = edge_pairs.Sort_by_Key(edge_values, row_ascen_order,
                                          this->edges, 0, target, stream);
          break;
        case BY_ROW_DECENDING:
          retval = edge_pairs.Sort_by_Key(edge_values, row_decen_order,
                                          this->edges, 0, target, stream);
          break;
        case BY_COLUMN_ASCENDING:
          retval = edge_pairs.Sort_by_Key(edge_values, column_ascen_order,
                                          this->edges, 0, target, stream);
          break;
        case BY_COLUMN_DECENDING:
          retval = edge_pairs.Sort_by_Key(edge_values, column_decen_order,
                                          this->edges, 0, target, stream);
          break;
        case UNORDERED:
          break;
      }
      if (retval) return retval;
    } else {  // no edge values
      switch (new_order) {
        case BY_ROW_ASCENDING:
          retval =
              edge_pairs.Sort(row_ascen_order, this->edges, 0, target, stream);
          break;
        case BY_ROW_DECENDING:
          retval =
              edge_pairs.Sort(row_decen_order, this->edges, 0, target, stream);
          break;
        case BY_COLUMN_ASCENDING:
          retval = edge_pairs.Sort(column_ascen_order, this->edges, 0, target,
                                   stream);
          break;
        case BY_COLUMN_DECENDING:
          retval = edge_pairs.Sort(column_decen_order, this->edges, 0, target,
                                   stream);
          break;
        case UNORDERED:
          break;
      }
      if (retval) return retval;
    }

    edge_order = new_order;
    // util::PrintMsg("After sorting");
    // Display();
    return retval;
  }

  cudaError_t GenerateEdgeValues(ValueT edge_value_min, ValueT edge_value_range,
                                 long edge_value_seed, bool quiet = false) {
    cudaError_t retval = cudaSuccess;

    if (!util::isValid(edge_value_seed)) edge_value_seed = time(NULL);
    srand(edge_value_seed);
    util::PrintMsg("  Generating edge values in [" +
                       std::to_string(edge_value_min) + ", " +
                       std::to_string(edge_value_min + edge_value_range) +
                       "), seed = " + std::to_string(edge_value_seed),
                   !quiet);

    for (SizeT e = 0; e < this->edges; e++) {
      auto x = rand();
      double int_x = 0;
      std::modf(x * 1.0 / edge_value_range, &int_x);
      auto val = x - int_x * edge_value_range;
      edge_values[e] = val + edge_value_min;
    }
    return retval;
  }

  cudaError_t EdgeDouble(bool skew = false, bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    auto &e_values = this->edge_values;
    auto edges = this->edges;
    bool has_edge_values = FLAG & graph::HAS_EDGE_VALUES;

    util::PrintMsg("  Edge doubleing: " + std::to_string(edges) + " -> " +
                       std::to_string(edges * 2) + " edges",
                   !quiet);
    GUARD_CU(edge_pairs.EnsureSize(this->edges * 2, true));
    GUARD_CU(edge_values.EnsureSize(this->edges * 2, true));

    GUARD_CU(edge_pairs.ForAll(
        [e_values, skew, has_edge_values, edges] __host__ __device__(
            EdgePairT * pairs, const SizeT &e) {
          pairs[e + edges].x = pairs[e].y;
          pairs[e + edges].y = pairs[e].x;
          if (has_edge_values) {
            e_values[e + edges] = (skew ? (e_values[e] * -1) : e_values[e]);
          }
        },
        edges, util::HOST));
    this->edges *= 2;
    return retval;
  }

  template <typename EdgeCondT>
  cudaError_t RemoveEdges(EdgeCondT edge_cond,
                          EdgeOrder new_order = BY_ROW_ASCENDING,
                          util::Location target = util::LOCATION_DEFAULT,
                          cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;

    if (edge_order == UNORDERED) {
      GUARD_CU(Order(new_order, target, stream));
    }

    SizeT *thread_edges = NULL;
    SizeT edge_counter = 0;
    util::Array1D<SizeT, EdgePairT, ARRAY_FLAG, cudaHostRegisterFlag>
        old_edge_pairs;
    typename util::If<(FLAG & HAS_EDGE_VALUES) != 0, Array_ValueT,
                      Array_NValueT>::Type old_edge_values;
#pragma omp parallel
    {
      int num_threads = omp_get_num_threads();
      int thread_num = omp_get_thread_num();
      SizeT edge_start = this->edges / num_threads * thread_num;
      SizeT edge_end = this->edges / num_threads * (thread_num + 1);

      if (thread_num == 0) edge_start = 0;
      if (thread_num == num_threads - 1) edge_end = this->edges;

#pragma omp single
      { thread_edges = new SizeT[num_threads + 1]; }

      SizeT t_edges = 0;
      for (SizeT e = edge_start; e < edge_end; e++) {
        if (!edge_cond(edge_pairs + 0, e)) continue;
        t_edges++;
      }
      thread_edges[thread_num] = t_edges;

#pragma omp barrier
#pragma omp single
      {
        edge_counter = 0;
        for (int i = 0; i < num_threads; i++) {
          SizeT old_val = edge_counter;
          edge_counter += thread_edges[i];
          thread_edges[i] = old_val;
        }
        thread_edges[num_threads] = edge_counter;
        // util::PrintMsg("Edges : " + std::to_string(this -> edges)
        //    + " -> " + std::to_string(edge_counter));

        if (edge_counter != this->edges) {
          retval = old_edge_pairs.Allocate(this->edges, target);
          if (retval == cudaSuccess)
            retval = old_edge_pairs.ForEach(
                edge_pairs,
                [] __host__ __device__(EdgePairT & old_pair,
                                       const EdgePairT &pair) {
                  old_pair.x = pair.x;
                  old_pair.y = pair.y;
                },
                this->edges, target, stream);
          auto org_allocated = edge_pairs.GetAllocated();
          if (retval == cudaSuccess) retval = edge_pairs.Release();
          if (retval == cudaSuccess)
            retval = edge_pairs.Allocate(edge_counter, org_allocated);

          if (FLAG & HAS_EDGE_VALUES) {
            if (retval == cudaSuccess)
              retval = old_edge_values.Allocate(this->edges, target);
            if (retval == cudaSuccess)
              retval = old_edge_values.ForEach(
                  edge_values,
                  [] __host__ __device__(ValueT & old_val, const ValueT &val) {
                    old_val = val;
                  },
                  this->edges, target, stream);
            org_allocated = edge_values.GetAllocated();
            if (retval == cudaSuccess) retval = edge_values.Release();
            if (retval == cudaSuccess)
              retval = edge_values.Allocate(edge_counter, org_allocated);
          }
        }

        // util::PrintMsg("Allocated");
      }

      if (edge_counter != this->edges && retval == cudaSuccess) {
        t_edges = 0;
        SizeT t_offset = thread_edges[thread_num];
        // util::PrintMsg("Thread " + std::to_string(thread_num)
        //    + ", offset = " + std::to_string(t_offset));
        for (SizeT e = edge_start; e < edge_end; e++) {
          // if ((e%100) == 0)
          // util::PrintMsg(std::to_string(e));
          if (!edge_cond(old_edge_pairs + 0, e)) continue;
          auto &old_pair = old_edge_pairs[e];
          auto &pair = edge_pairs[t_offset + t_edges];
          pair.x = old_pair.x;
          pair.y = old_pair.y;
          if (FLAG & HAS_EDGE_VALUES)
            edge_values[t_offset + t_edges] = old_edge_values[e];
          t_edges++;
        }
      }
    }
    if (retval) return retval;

    if (edge_counter != this->edges) {
      GUARD_CU(old_edge_pairs.Release());
      GUARD_CU(old_edge_values.Release());
      this->edges = edge_counter;
    }
    delete[] thread_edges;
    thread_edges = NULL;

    return retval;
  }

  cudaError_t RemoveSelfLoops(EdgeOrder new_order = BY_ROW_ASCENDING,
                              util::Location target = util::LOCATION_DEFAULT,
                              cudaStream_t stream = 0, bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    SizeT old_num_edges = this->edges;
    GUARD_CU(RemoveEdges(
        [] __host__ __device__(EdgePairT * pairs, const SizeT &edge_id) {
          auto &edge_pair = pairs[edge_id];
          return edge_pair.x != edge_pair.y;
        },
        new_order, target, stream));

    if (old_num_edges != this->edges)
      util::PrintMsg(" Removed " + std::to_string(old_num_edges - this->edges) +
                         " self circles.",
                     !quiet);
    return retval;
  }

  cudaError_t RemoveDuplicateEdges(
      EdgeOrder new_order = BY_ROW_ASCENDING,
      util::Location target = util::LOCATION_DEFAULT, cudaStream_t stream = 0,
      bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    SizeT old_num_edges = this->edges;
    GUARD_CU(RemoveEdges(
        [] __host__ __device__(EdgePairT * pairs, const SizeT &edge_id) {
          if (edge_id == 0) return true;
          auto &edge_pair = pairs[edge_id];
          auto &p_edge_pair = pairs[edge_id - 1];
          if (p_edge_pair.x != edge_pair.x) return true;
          if (p_edge_pair.y != edge_pair.y) return true;
          return false;
        },
        new_order, target, stream));

    if (old_num_edges != this->edges)
      util::PrintMsg("  Removed " +
                         std::to_string(old_num_edges - this->edges) +
                         " duplicate edges.",
                     !quiet);
    return retval;
  }

  cudaError_t RemoveSelfLoops_DuplicateEdges(
      EdgeOrder new_order = BY_ROW_ASCENDING,
      util::Location target = util::LOCATION_DEFAULT, cudaStream_t stream = 0,
      bool quiet = false) {
    cudaError_t retval = cudaSuccess;
    SizeT old_num_edges = this->edges;
    GUARD_CU(RemoveEdges(
        [] __host__ __device__(EdgePairT * pairs, const SizeT &edge_id) {
          auto &edge_pair = pairs[edge_id];
          if (edge_pair.x == edge_pair.y) return false;
          if (edge_id == 0) return true;
          auto &p_edge_pair = pairs[edge_id - 1];
          if (p_edge_pair.x != edge_pair.x) return true;
          if (p_edge_pair.y != edge_pair.y) return true;
          return false;
        },
        new_order, target, stream));

    if (old_num_edges != this->edges)
      util::PrintMsg("  Removed " +
                         std::to_string(old_num_edges - this->edges) +
                         " duplicate edges and self circles.",
                     !quiet);
    return retval;
  }

  __device__ __host__ __forceinline__ SizeT
  GetNeighborListLength(const VertexT &v) const {
    // Not implemented
    return util::PreDefinedValues<SizeT>::InvalidValue;
  }

  __device__ __host__ __forceinline__ VertexT
  GetEdgeDest(const SizeT &e) const {
    auto &pair = edge_pairs[e];
    return e.y;
  }

  __device__ __host__ __forceinline__ VertexT GetEdgeSrc(const SizeT &e) const {
    auto &pair = edge_pairs[e];
    return e.x;
  }

  __device__ __host__ __forceinline__ void GetEdgeSrcDest(const SizeT &e,
                                                          VertexT &src,
                                                          VertexT &dest) const {
    auto &pair = edge_pairs[e];
    src = pair.x;
    dest = pair.y;
  }
};  // Coo

template <typename VertexT, typename SizeT, typename ValueT, GraphFlag _FLAG,
          unsigned int cudaHostRegisterFlag>
struct Coo<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag, false> {
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
