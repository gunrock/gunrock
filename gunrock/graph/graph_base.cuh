// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graph_base.cuh
 *
 * @brief Base Graph Data Structure
 */

#pragma once

#include <gunrock/util/binary_search.cuh>

namespace gunrock {
namespace graph {

//#define ENABLE_GRAPH_DEBUG

/**
 * @brief Predefined flags for graph types
 */
using GraphFlag = uint32_t;
enum : GraphFlag {
  ARRAY_RESERVE = 0x000F,

  GRAPH_NONE = 0x0000,
  HAS_EDGE_VALUES = 0x0010,
  HAS_NODE_VALUES = 0x0020,

  TypeMask = 0x0F00,
  HAS_CSR = 0x0100,
  HAS_CSC = 0x0200,
  HAS_COO = 0x0400,
  HAS_GP = 0x0800,

  GRAPH_PINNED = 0x1000,
};

template <GraphFlag FLAG>
struct GraphType_Num {
  static const GraphFlag VAL =
      ((FLAG & HAS_CSR) != 0)
          ? HAS_CSR
          : (((FLAG & HAS_CSC) != 0) ? HAS_CSC
                                     : (((FLAG & HAS_COO) != 0) ? HAS_COO : 0));
};

static const util::Location GRAPH_DEFAULT_TARGET = util::DEVICE;

/*template <typename T, typename SizeT>
__device__ __host__ __forceinline__
SizeT Binary_Search(
    const T* data, T item_to_find, SizeT lower_bound, SizeT upper_bound)
{
    while (lower_bound < upper_bound)
    {
        SizeT mid_point = (lower_bound + upper_bound) >> 1;
        if (_ldg(data + mid_point) < item_to_find)
            lower_bound = mid_point + 1;
        else
            upper_bound = mid_point;
    }

    SizeT retval = util::PreDefinedValues<SizeT>::InvalidValue;
    if (upper_bound == lower_bound)
    {
        if (item_to_find < _ldg(data + upper_bound))
            retval = upper_bound -1;
        else
            retval = upper_bound;
    } else
        retval = util::PreDefinedValues<SizeT>::InvalidValue;

    return retval;
}*/

/**
 * @brief Enum to show how the edges are ordered
 */
enum EdgeOrder {
  BY_ROW_ASCENDING,
  BY_ROW_DECENDING,
  BY_COLUMN_ASCENDING,
  BY_COLUMN_DECENDING,
  UNORDERED,
};

std::string EdgeOrder_to_string(EdgeOrder order) {
  switch (order) {
    case BY_ROW_ASCENDING:
      return "by row ascending";
    case BY_ROW_DECENDING:
      return "by row decending";
    case BY_COLUMN_ASCENDING:
      return "by column ascending";
    case BY_COLUMN_DECENDING:
      return "by column decending";
    case UNORDERED:
      return "unordered";
  }
  return "unspecified";
}

/**
 * @brief GraphBase data structure to store basic info about a graph.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename _VertexT, typename _SizeT, typename _ValueT, GraphFlag _FLAG,
          unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct GraphBase {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG;
  static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;

  SizeT nodes;    // Number of nodes in the graph
  SizeT edges;    // Number of edges in the graph
  bool directed;  // Whether the graph is directed

  GraphBase() : nodes(0), edges(0), directed(true) {}

  __host__ __device__ ~GraphBase() {
    // Release();
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    nodes = 0;
    edges = 0;
    directed = true;
    return cudaSuccess;
  }

  cudaError_t Allocate(SizeT nodes, SizeT edges,
                       util::Location target = GRAPH_DEFAULT_TARGET) {
    this->nodes = nodes;
    this->edges = edges;
    return cudaSuccess;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  template <typename GraphT_in>
  cudaError_t Set(GraphT_in &source,
                  util::Location target = util::LOCATION_DEFAULT,
                  cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    this->nodes = source.nodes;
    this->edges = source.edges;
    this->directed = source.directed;
    return retval;
  }

  template <typename CooT_in>
  cudaError_t FromCoo(CooT_in &coo) {
    Set(coo);
    return cudaSuccess;
  }

  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr) {
    Set(csr);
    return cudaSuccess;
  }

  template <typename CscT_in>
  cudaError_t FromCsc(CscT_in &csc) {
    Set(csc);
    return cudaSuccess;
  }

  SizeT GetNeighborListLength(const VertexT &v) { return 0; }
};

/**
 * @brief Get the average degree of all the nodes in graph
 */
template <typename GraphT>
double GetAverageDegree(GraphT &graph) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  double mean = 0, count = 0;
  for (VertexT v = 0; v < graph.nodes; ++v) {
    count += 1;
    mean += (graph.GetNeighborListLength(v) - mean) / count;
  }
  return mean;
}

/**
 * @brief Get the Standard Deviation of degrees of all the nodes in graph
 */
template <typename GraphT>
double GetStddevDegree(GraphT &graph) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  auto average_degree = GetAverageDegree(graph);

  float accum = 0.0f;
  for (VertexT v = 0; v < graph.nodes; ++v) {
    float d = graph.GetNeighborListLength(v);
    accum += (d - average_degree) * (d - average_degree);
  }
  return sqrt(accum / (graph.nodes - 1));
}

/**
 * @brief Find log-scale degree histogram of the graph.
 */
template <typename GraphT, typename ArrayT>
cudaError_t GetHistogram(GraphT &graph, ArrayT &histogram,
                         util::Location target = util::HOST,
                         cudaStream_t stream = 0) {
  typedef typename ArrayT::ValueT CountT;
  typedef typename GraphT::SizeT SizeT;

  cudaError_t retval = cudaSuccess;
  auto length = sizeof(SizeT) * 8 + 1;

  GUARD_CU(histogram.EnsureSize_(length, target));

  // Initialize
  GUARD_CU(
      histogram.ForEach([] __host__ __device__(CountT & count) { count = 0; },
                        length, target, stream));

  // Count
  GUARD_CU(histogram.ForAll(
      [graph] __host__ __device__(CountT * counts, SizeT & v) {
        auto num_neighbors = graph.GetNeighborListLength(v);
        int log_length = 0;
        while (num_neighbors >= (1 << log_length)) {
          log_length++;
        }
        _atomicAdd(counts + log_length, (CountT)1);
      },
      graph.nodes, target, stream));

  return retval;
}

template <typename GraphT, typename ArrayT>
cudaError_t PrintHistogram(GraphT &graph, ArrayT &histogram) {
  typedef typename GraphT::SizeT SizeT;
  cudaError_t retval = cudaSuccess;

  util::PrintMsg("Degree Histogram (" + std::to_string(graph.nodes) +
                 " vertices, " + std::to_string(graph.edges) + " edges):");
  int max_log_length = 0;
  for (int i = sizeof(SizeT) * 8; i >= 0; i--) {
    if (i < histogram.GetSize() && histogram[i] > 0) {
      max_log_length = i;
      break;
    }
  }

  for (int i = 0; i <= max_log_length; i++) {
    util::PrintMsg("    Degree " +
                   (i == 0 ? "0" : ("2^" + std::to_string(i - 1))) + ": " +
                   std::to_string(histogram[i]) + " (" +
                   std::to_string(histogram[i] * 100.0 / graph.nodes) + " %)");
  }
  return retval;
}

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
