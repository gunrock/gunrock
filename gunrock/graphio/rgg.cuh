// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * rgg.cuh
 *
 * @brief RGG Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <time.h>
#include <list>
#include <random>
#include <gunrock/graphio/utils.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/util/parameters.h>
#include <gunrock/util/test_utils.h>

namespace gunrock {
namespace graphio {
namespace rgg {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

template <typename T>
inline T SqrtSum(T x, T y) {
  return sqrt(x * x + y * y);
}

template <typename T>
T P2PDistance(T co_x0, T co_y0, T co_x1, T co_y1) {
  return SqrtSum(co_x0 - co_x1, co_y0 - co_y1);
}

class RggPoint {
 public:
  double x, y;
  long long node;

  RggPoint() {}
  RggPoint(double x, double y, long long node) {
    this->x = x;
    this->y = y;
    this->node = node;
  }
};

// inline bool operator< (const RggPoint& lhs, const RggPoint& rhs)
template <typename Point>
bool XFirstPointCompare(Point lhs, Point rhs) {
  if (lhs.x < rhs.x) return true;
  if (lhs.x > rhs.x) return false;
  if (lhs.y < rhs.y) return true;
  return false;
}

template <typename T>
bool PureTwoFactor(T x) {
  if (x < 3) return true;
  while (x > 0) {
    if ((x % 2) != 0) return false;
    x /= 2;
  }
  return true;
}

cudaError_t UseParameters(util::Parameters &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<double>(
      graph_prefix + "rgg-thfactor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.55, "Threshold-factor", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      graph_prefix + "rgg-threshold",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Threshold, default is thfactor * sqrt(log(#nodes) / #nodes)",
      __FILE__, __LINE__));

  return retval;
}

/*
 * @brief Build random geometry graph (RGG).
 *
 * @tparam WITH_VALUES Whether or not associate with per edge weight values.
 * @tparam VertexT Vertex identifier.
 * @tparam Value Value type.
 * @tparam SizeT Graph size type.
 *
 * @param[in] nodes
 * @param[in] graph
 * @param[in] threshould
 * @param[in] undirected
 * @param[in] value_multipiler
 * @param[in] value_min
 * @param[in] seed
 */
template <typename GraphT>
cudaError_t Build(util::Parameters &parameters, GraphT &graph,
                  std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CsrT CsrT;

  bool quiet = parameters.Get<bool>("quiet");
  std::string dataset = "rgg_";
  // bool undirected = !parameters.Get<bool>(graph_prefix + "directed");
  SizeT scale = parameters.Get<SizeT>(graph_prefix + "graph-scale");
  SizeT num_nodes = 0;
  if (!parameters.UseDefault(graph_prefix + "graph-nodes")) {
    num_nodes = parameters.Get<SizeT>(graph_prefix + "graph-nodes");
    dataset = dataset + std::to_string(num_nodes) + "_";
  } else {
    num_nodes = 1 << scale;
    dataset = dataset + "n" + std::to_string(scale) + "_";
  }
  double thfactor = parameters.Get<double>(graph_prefix + "rgg-thfactor");
  double threshold = 0;
  if (!parameters.UseDefault(graph_prefix + "rgg-threshold")) {
    threshold = parameters.Get<double>(graph_prefix + "rgg-threshold");
    dataset = dataset + "t" + std::to_string(threshold);
  } else {
    threshold = thfactor * sqrt(log(num_nodes) / num_nodes);
    dataset = dataset + std::to_string(threshold);
  }
  if (parameters.UseDefault("dataset"))
    parameters.Set<std::string>("dataset", dataset);

  bool random_edge_values =
      parameters.Get<bool>(graph_prefix + "random-edge-values");
  double edge_value_range =
      parameters.Get<double>(graph_prefix + "edge-value-range");
  double edge_value_min =
      parameters.Get<double>(graph_prefix + "edge-value-min");

  int seed = time(NULL);
  if (parameters.UseDefault(graph_prefix + "graph-seed"))
    seed = parameters.Get<int>(graph_prefix + "graph-seed");

  util::PrintMsg("Generating RGG " + graph_prefix +
                     "graph, threshold = " + std::to_string(threshold) +
                     ", seed = " + std::to_string(seed) + "...",
                 !quiet);
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  int reserved_size = 50;
  util::Location target = util::HOST;
  SizeT num_edges = 0;
  long long row_length = 1.0 / threshold + 1;
  long long reserved_factor2 = 8;
  long long initial_length =
      reserved_factor2 * num_nodes / row_length / row_length;

  util::Array1D<SizeT, RggPoint> points;
  util::Array1D<SizeT, SizeT> row_offsets;
  util::Array1D<SizeT, VertexT> col_index_;
  util::Array1D<SizeT, ValueT> values_;
  util::Array1D<SizeT, SizeT> offsets;
  util::Array1D<SizeT, VertexT *> blocks;
  util::Array1D<SizeT, SizeT> block_size;
  util::Array1D<SizeT, SizeT> block_length;

  points.SetName("graphio::rgg::points");
  row_offsets.SetName("graphio::rgg::row_offsets");
  col_index_.SetName("graphio::rgg::col_index_");
  values_.SetName("graphio::rgg::values_");
  offsets.SetName("graphio::rgg::offsets");
  blocks.SetName("graphio::rgg::blocks");
  block_size.SetName("graphio::rgg::block_size");
  block_length.SetName("graphio::rgg::block_length");

  GUARD_CU(points.Allocate(num_nodes + 1, target));
  GUARD_CU(row_offsets.Allocate(num_nodes + 1, target));
  GUARD_CU(col_index_.Allocate(reserved_size * num_nodes, target));
  if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
    GUARD_CU(values_.Allocate(reserved_size * num_nodes, target));
  SizeT tLength = row_length * row_length + 1;
  GUARD_CU(blocks.Allocate(tLength, target));
  GUARD_CU(block_size.Allocate(tLength, target));
  GUARD_CU(block_length.Allocate(tLength, target));

  if (initial_length < 4) initial_length = 4;

  GUARD_CU(block_size.ForEach(
      block_length, blocks,
      [] __host__ __device__(SizeT & size, SizeT & length, VertexT * &block) {
        size = 0;
        length = 0;
        block = NULL;
      },
      tLength, target));

#pragma omp parallel
  do {
    int thread_num = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    SizeT node_start = (long long)(num_nodes)*thread_num / num_threads;
    SizeT node_end = (long long)(num_nodes) * (thread_num + 1) / num_threads;
    SizeT counter = 0;
    VertexT *col_index = col_index_ + reserved_size * node_start;
    ValueT *values = (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                         ? values_ + reserved_size * node_start
                         : NULL;
    unsigned int seed_ = seed + 805 * thread_num;
    Engine engine(seed_);
    Distribution distribution(0.0, 1.0);

#pragma omp single
    { retval = offsets.Allocate(num_threads + 1, target); }
    if (retval) break;

    for (VertexT node = node_start; node < node_end; node++) {
      points[node].x = distribution(engine);
      points[node].y = distribution(engine);
      points[node].node = node;
    }

#pragma omp barrier
#pragma omp single
    {
      std::stable_sort(points + 0, points + num_nodes,
                       XFirstPointCompare<RggPoint>);
    }

    for (VertexT node = node_start; node < node_end; node++) {
      SizeT x_index = points[node].x / threshold;
      SizeT y_index = points[node].y / threshold;
      SizeT block_index = x_index * row_length + y_index;
#pragma omp atomic
      block_size[block_index]++;
    }

#pragma omp barrier
#pragma omp single
    {
      for (SizeT i = 0; i < row_length * row_length; i++)
        if (block_size[i] != 0) blocks[i] = new VertexT[block_size[i]];
    }

    for (VertexT node = node_start; node < node_end; node++) {
      double co_x0 = points[node].x;  // co_x[node];
      double co_y0 = points[node].y;  // co_y[node];
      // RggPoint point(co_x0, co_y0, node);
      SizeT x_index = co_x0 / threshold;
      SizeT y_index = co_y0 / threshold;
      SizeT block_index = x_index * row_length + y_index;
      SizeT pos = 0;

#pragma omp atomic capture
      {
        pos = block_length[block_index];
        block_length[block_index] += 1;
      }
      blocks[block_index][pos] = node;
    }

#pragma omp barrier

    for (VertexT node = node_start; node < node_end; node++) {
      row_offsets[node] = counter;
      double co_x0 = points[node].x;
      double co_y0 = points[node].y;
      SizeT x_index = co_x0 / threshold;
      SizeT y_index = co_y0 / threshold;
      SizeT x_start = x_index < 2 ? 0 : x_index - 2;
      SizeT y_start = y_index < 2 ? 0 : y_index - 2;

      for (SizeT x1 = x_start; x1 <= x_index + 2; x1++)
        for (SizeT y1 = y_start; y1 <= y_index + 2; y1++) {
          if (x1 >= row_length || y1 >= row_length) continue;

          SizeT block_index = x1 * row_length + y1;
          VertexT *block = blocks[block_index];
          for (SizeT i = 0; i < block_length[block_index]; i++) {
            VertexT peer = block[i];
            if (node >= peer) continue;
            double co_x1 = points[peer].x;  // co_x[peer];
            double co_y1 = points[peer].y;  // co_y[peer];
            double dis_x = co_x0 - co_x1;
            double dis_y = co_y0 - co_y1;
            if (fabs(dis_x) > threshold || fabs(dis_y) > threshold) continue;
            double dis = SqrtSum(dis_x, dis_y);
            if (dis > threshold) continue;

            col_index[counter] = peer;
            if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
              if (random_edge_values) {
                values[counter] =
                    distribution(engine) * edge_value_range + edge_value_min;
              } else {
                values[counter] = 1;
              }
            }
            counter++;
          }
        }
    }
    offsets[thread_num + 1] = counter;

#pragma omp barrier
#pragma omp single
    {
      offsets[0] = 0;
      for (int i = 0; i < num_threads; i++) offsets[i + 1] += offsets[i];
      num_edges = offsets[num_threads];
      retval = graph.CsrT::Allocate(num_nodes, num_edges, target);
      // coo = (EdgeTupleType*) malloc (sizeof(EdgeTupleType) * edges);
    }
    if (retval) break;

    SizeT offset = offsets[thread_num];
    for (VertexT node = node_start; node < node_end; node++) {
      SizeT end_edge = (node != node_end - 1 ? row_offsets[node + 1] : counter);
      graph.CsrT::row_offsets[node] = offset + row_offsets[node];
      for (SizeT edge = row_offsets[node]; edge < end_edge; edge++) {
        // VertexT peer = col_index[edge];
        graph.CsrT::column_indices[offset + edge] = col_index[edge];
        if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
          graph.CsrT::edge_values[offset + edge] = values[edge];
      }
    }

    col_index = NULL;
    values = NULL;
  } while (false);
  if (retval) return retval;
  graph.CsrT::row_offsets[num_nodes] = num_edges;

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  util::PrintMsg("RGG generated in " + std::to_string(elapsed) + " ms.",
                 !quiet);

  SizeT counter = 0;
  for (SizeT i = 0; i < row_length * row_length; i++)
    if (block_size[i] != 0) {
      counter += block_length[i];
      delete[] blocks[i];
      blocks[i] = NULL;
    }

  GUARD_CU(row_offsets.Release());
  GUARD_CU(offsets.Release());
  GUARD_CU(points.Release());
  GUARD_CU(blocks.Release());
  GUARD_CU(block_size.Release());
  GUARD_CU(block_length.Release());
  GUARD_CU(col_index_.Release());
  GUARD_CU(values_.Release());

  return retval;
}

template <typename GraphT, bool CSR_SWITCH>
struct CsrSwitch {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(Build(parameters, graph, graph_prefix));
    GUARD_CU(graph.FromCsr(graph, util::HOST, 0, parameters.Get<bool>("quiet"),
                           true));
    return retval;
  }
};

template <typename GraphT>
struct CsrSwitch<GraphT, false> {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    typedef graph::Csr<typename GraphT::VertexT, typename GraphT::SizeT,
                       typename GraphT::ValueT, GraphT::FLAG | graph::HAS_CSR,
                       GraphT::cudaHostRegisterFlag>
        CsrT;
    cudaError_t retval = cudaSuccess;

    CsrT csr;
    GUARD_CU(Build(parameters, csr, graph_prefix));
    GUARD_CU(graph.FromCsr(csr, util::HOST, 0, parameters.Get<bool>("quiet"),
                           false));
    GUARD_CU(csr.Release());
    return retval;
  }
};

template <typename GraphT>
cudaError_t Load(util::Parameters &parameters, GraphT &graph_,
                 std::string graph_prefix = "") {
  return CsrSwitch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSR) != 0>::Load(
      parameters, graph_, graph_prefix);
}

}  // namespace rgg
}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
