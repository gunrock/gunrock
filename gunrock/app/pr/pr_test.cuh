// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_test.cu
 *
 * @brief Test related functions for PageRank
 */

#pragma once

// boost includes
#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/page_rank.hpp>

namespace gunrock {
namespace app {
namespace pr {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the PageRank result
 *
 * @param[in] vertices Vertex Ids
 * @param[in] ranks PageRank values for the vertices
 * @param[in] num_vertices Number of vertices to display
 */
template <typename VertexT, typename SizeT, typename ValueT>
void DisplaySolution(VertexT *vertices, ValueT *ranks, SizeT num_vertices) {
  // at most top 10 ranked vertices
  SizeT top = (num_vertices < 10) ? num_vertices : 10;
  util::PrintMsg("Top " + std::to_string(top) +
                 " Ranked vertices and PageRanks:");
  for (SizeT i = 0; i < top; ++i) {
    util::PrintMsg("Vertex ID: " + std::to_string(vertices[i]) +
                   ", PageRank: " + std::to_string(ranks[i]));
  }
}

template <typename GraphT>
cudaError_t Compensate_ZeroDegrees(GraphT &graph, bool quiet = false) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CooT CooT;
  typedef typename CooT::EdgePairT EdgePairT;
  cudaError_t retval = cudaSuccess;
  auto &graph_coo = graph.coo();

  util::Array1D<SizeT, int> vertex_markers;
  GUARD_CU(vertex_markers.Allocate(graph_coo.nodes, util::HOST));
  GUARD_CU(vertex_markers.ForEach(
      [] __host__ __device__(int &marker) { marker = 0; }, graph_coo.nodes,
      util::HOST));

  GUARD_CU(graph_coo.edge_pairs.ForEach(
      [vertex_markers] __host__ __device__(const EdgePairT &pair) {
        vertex_markers[pair.x] = 1;
      },
      graph_coo.edges, util::HOST));

  util::Array1D<SizeT, VertexT> zeroDegree_vertices;
  GUARD_CU(zeroDegree_vertices.Allocate(graph.nodes, util::HOST));
  SizeT counter = 0;
  for (VertexT v = 0; v < graph_coo.nodes; v++)
    if (vertex_markers[v] == 0) {
      zeroDegree_vertices[counter] = v;
      counter++;
    }
  GUARD_CU(vertex_markers.Release());

  if (counter == 0) {
    GUARD_CU(zeroDegree_vertices.Release());
    return retval;
  }

  if (!quiet)
    util::PrintMsg("Adding 1 vertex and " +
                   std::to_string(counter + graph_coo.nodes) +
                   " edges to compensate 0 degree vertices");

  CooT new_coo;
  GUARD_CU(new_coo.Allocate(graph_coo.nodes + 1,
                            graph_coo.edges + counter + graph_coo.nodes,
                            util::HOST));
  GUARD_CU(new_coo.edge_pairs.ForEach(
      graph_coo.edge_pairs,
      [] __host__ __device__(EdgePairT & new_pair, const EdgePairT &old_pair) {
        new_pair.x = old_pair.x;
        new_pair.y = old_pair.y;
      },
      graph_coo.edges, util::HOST));
  if (CooT::FLAG & gunrock::graph::HAS_EDGE_VALUES) {
    GUARD_CU(new_coo.edge_values.ForEach(
        graph_coo.edge_values,
        [] __host__ __device__(ValueT & new_value, const ValueT &old_value) {
          new_value = old_value;
        },
        graph_coo.edges, util::HOST));
  }
  if (CooT::FLAG & gunrock::graph::HAS_NODE_VALUES) {
    GUARD_CU(new_coo.node_values.ForEach(
        graph_coo.node_values,
        [] __host__ __device__(ValueT & new_value, const ValueT &old_value) {
          new_value = old_value;
        },
        graph_coo.nodes, util::HOST));
  }

  SizeT edge_counter = graph_coo.edges;
  for (SizeT i = 0; i < counter; i++) {
    auto &edgePair = new_coo.edge_pairs[edge_counter + i];
    edgePair.x = zeroDegree_vertices[i];
    edgePair.y = graph_coo.nodes;
    if (CooT::FLAG & gunrock::graph::HAS_EDGE_VALUES)
      new_coo.edge_values[edge_counter + i] =
          util::PreDefinedValues<ValueT>::InvalidValue;
  }
  edge_counter += counter;
  for (VertexT v = 0; v < graph_coo.nodes; v++) {
    auto &edgePair = new_coo.edge_pairs[edge_counter + v];
    edgePair.x = graph_coo.nodes;
    edgePair.y = v;
    if (CooT::FLAG & gunrock::graph::HAS_EDGE_VALUES)
      new_coo.edge_values[edge_counter + v] =
          util::PreDefinedValues<ValueT>::InvalidValue;
  }
  if (CooT::FLAG & gunrock::graph::HAS_NODE_VALUES)
    new_coo.node_values[graph_coo.nodes] =
        util::PreDefinedValues<ValueT>::InvalidValue;

  GUARD_CU(zeroDegree_vertices.Release());
  new_coo.edge_order = gunrock::graph::UNORDERED;
  GUARD_CU(graph.FromCoo(new_coo, util::LOCATION_DEFAULT, 0, quiet));
  GUARD_CU(new_coo.Release());

  return retval;
}

/******************************************************************************
 * PageRank Testing Routines
 *****************************************************************************/

template <typename VertexT, typename ValueT>
struct RankPair {
  VertexT vertex_id;
  ValueT page_rank;

  RankPair(VertexT vertex_id, ValueT page_rank)
      : vertex_id(vertex_id), page_rank(page_rank) {}
};

template <typename RankPairT>
bool PRCompare(const RankPairT &elem1, const RankPairT &elem2) {
  return elem1.page_rank > elem2.page_rank;
}

/**
 * @brief Simple CPU-based reference PageRank implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   parameters    Running parameters
 * @param[in]   graph         Input graph
 * @param[in]   src           Source vertex for personalized PageRank (if any)
 * @param[out]  node_ids      Top ranking vertices
 * @param[out]  ranks         Ranking values
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CPU_Reference(util::Parameters &parameters, const GraphT &graph,
                     typename GraphT::VertexT src,
                     typename GraphT::VertexT *node_ids, ValueT *ranks) {
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  bool quiet = parameters.Get<bool>("quiet");
  bool normalize = parameters.Get<bool>("normalize");
  bool scale = parameters.Get<bool>("scale");
  bool undirected = parameters.Get<bool>("undirected");
  VertexT max_iter = parameters.Get<VertexT>("max-iter");
  ValueT threshold = parameters.Get<ValueT>("threshold");
  ValueT delta = parameters.Get<ValueT>("delta");

#ifdef BOOST_FOUND
  if (!normalize && undirected) {
    using namespace boost;

    // preparation
    typedef adjacency_list<vecS, vecS, bidirectionalS, no_property,
                           property<edge_index_t, VertexT> >
        BGraphT;

    BGraphT g;

    for (SizeT e = 0; e < graph.edges; ++e) {
      auto &pair = graph.CooT::edge_pairs[e];
      BGraphT::edge_descriptor edge = add_edge(pair.x, pair.y, g).first;
      put(edge_index, g, edge, pair.x);
    }

    // compute PageRank
    CpuTimer cpu_timer;
    cpu_timer.Start();

    std::vector<ValueT> ranks_vec(num_vertices(g));
    page_rank(g,
              make_iterator_property_map(ranks_vec.begin(),
                                         get(boost::vertex_index, g)),
              boost::graph::n_iterations(max_iter));

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    // Sort the top ranked vertices
    typedef RankPair<VertexT, ValueT> RankPairT;
    RankPairT *pr_list =
        (RankPairT *)malloc(sizeof(RankPairT) * num_vertices(g));
    for (VertexT v = 0; v < num_vertices(g); ++v) {
      pr_list[v].vertex_id = v;
      pr_list[v].page_rank = ranks_vec[v];
    }
    std::stable_sort(pr_list, pr_list + num_vertices(g), PRCompare<RankPairT>);

    for (VertexT v = 0; v < num_vertices(g); ++v) {
      node_ids[v] = pr_list[v].vertex_id;
      ranks[v] = pr_list[v].page_rank;
    }
    free(pr_list);
    pr_list = NULL;
    return elapsed;
  }
#endif

  SizeT nodes = graph.nodes;
  SizeT edges = graph.edges;
  ValueT *rank_current = (ValueT *)malloc(sizeof(ValueT) * nodes);
  ValueT *rank_next = (ValueT *)malloc(sizeof(ValueT) * nodes);
  ValueT *rev_degrees = (ValueT *)malloc(sizeof(ValueT) * nodes);
  bool to_continue = true;
  SizeT iteration = 0;
  ValueT init_value =
      normalize ? (scale ? 1.0 : (1.0 / (ValueT)nodes)) : (1.0 - delta);
  ValueT reset_value =
      normalize ? (scale ? (1.0 - delta) : ((1.0 - delta) / (ValueT)nodes))
                : (1.0 - delta);
  util::CpuTimer cpu_timer;

#pragma omp parallel for
  for (VertexT v = 0; v < nodes; v++) {
    rank_current[v] = init_value;
    rank_next[v] = normalize ? 0.0 : (1.0 - delta);
    rev_degrees[v] = 0;
  }

#pragma omp parallel for
  for (SizeT e = 0; e < edges; e++) {
    VertexT v = graph.CooT::edge_pairs[e].x;
#pragma omp atomic
    rev_degrees[v] += 1;
  }
#pragma omp parallel for
  for (VertexT v = 0; v < nodes; v++) {
    if (rev_degrees[v] > 1e-6) rev_degrees[v] = 1 / rev_degrees[v];
  }

  cpu_timer.Start();
  while (to_continue) {
#pragma omp parallel for
    for (SizeT e = 0; e < graph.edges; e++) {
      auto &edge_pair = graph.CooT::edge_pairs[e];
      auto &src = edge_pair.x;
      auto &dest = edge_pair.y;
      ValueT dist_rank = rank_current[src] * rev_degrees[src];
      if (!isfinite(dist_rank)) continue;
#pragma omp atomic
      rank_next[dest] += dist_rank;
    }
    to_continue = false;
    iteration++;

#pragma omp parallel for
    for (VertexT v = 0; v < nodes; v++) {
      ValueT rank_new = delta * rank_next[v];
      if (!isfinite(rank_new)) rank_new = 0;
      rank_new = rank_new + reset_value;
      if (iteration <= max_iter &&
          fabs(rank_new - rank_current[v]) > threshold * rank_current[v]) {
        to_continue = true;
      }
      rank_current[v] = rank_new;
      rank_next[v] = 0;
    }
    if (iteration >= max_iter) to_continue = false;
  }
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  typedef RankPair<VertexT, ValueT> RankPairT;
  // Sort the top ranked vertices
  RankPairT *pr_list = (RankPairT *)malloc(sizeof(RankPairT) * nodes);

#pragma omp parallel for
  for (VertexT v = 0; v < nodes; ++v) {
    pr_list[v].vertex_id = v;
    pr_list[v].page_rank = rank_current[v];
  }

  std::stable_sort(pr_list, pr_list + nodes, PRCompare<RankPairT>);

#pragma omp parallel for
  for (VertexT v = 0; v < nodes; ++v) {
    node_ids[v] = pr_list[v].vertex_id;
    ranks[v] = (scale & normalize) ? (pr_list[v].page_rank / (ValueT)nodes)
                                   : pr_list[v].page_rank;
  }

  free(pr_list);
  pr_list = NULL;
  free(rank_current);
  rank_current = NULL;
  free(rank_next);
  rank_next = NULL;

  return elapsed;
}

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph, typename GraphT::VertexT src,
    typename GraphT::VertexT *h_node_ids, ValueT *h_ranks,
    typename GraphT::VertexT *ref_node_ids = NULL, ValueT *ref_ranks = NULL,
    bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  /*if (!quiet_mode)
  {
      double total_pr = 0;
      for (SizeT i = 0; i < graph->nodes; ++i)
      {
          total_pr += h_rank[i];
      }
      printf("Total rank : %.10lf\n", total_pr);
  }*/

  bool quiet = parameters.Get<bool>("quiet");
  ValueT threshold = parameters.Get<ValueT>("threshold");
  SizeT nodes = graph.nodes;

  // Verify the result
  util::PrintMsg("Rank Validity: ", !quiet, false);
  ValueT *unorder_ranks = (ValueT *)malloc(sizeof(ValueT) * nodes);
  SizeT *v_counts = (SizeT *)malloc(sizeof(SizeT) * nodes);
  SizeT error_count = 0;
  for (VertexT v = 0; v < nodes; v++) v_counts[v] = 0;

  for (VertexT v_ = 0; v_ < nodes; v_++) {
    VertexT v = h_node_ids[v_];
    if (util::lessThanZero(v) || v >= nodes) {
      util::PrintMsg("FAIL: node_id[" + std::to_string(v_) + "] (" +
                         std::to_string(v) + ") is out of bound.",
                     error_count == 0 && !quiet);
      error_count++;
      continue;
    }
    if (v_counts[v] > 0) {
      util::PrintMsg("FAIL: node_id[" + std::to_string(v_) + "] (" +
                         std::to_string(v) + ") appears more than once.",
                     error_count == 0 && !quiet);
      error_count++;
      continue;
    }
    v_counts[v]++;
    unorder_ranks[v] = h_ranks[v_];
  }
  for (VertexT v = 0; v < nodes; v++)
    if (v_counts[v] == 0) {
      util::PrintMsg(
          "FAIL: vertex " + std::to_string(v) + " does not appear in result.",
          error_count == 0 && !quiet);
      error_count++;
    }
  free(v_counts);
  v_counts = NULL;

  if (ref_node_ids != NULL && ref_ranks != NULL) {
    double ref_total_rank = 0;
    double max_diff = 0;
    VertexT max_diff_pos = nodes;
    double max_rdiff = 0;
    VertexT max_rdiff_pos = nodes;
    for (VertexT v_ = 0; v_ < nodes; v_++) {
      VertexT v = ref_node_ids[v_];
      if (util::lessThanZero(v) || v >= nodes) {
        util::PrintMsg("FAIL: ref_node_id[" + std::to_string(v_) + "] (" +
                           std::to_string(v) + ") is out of bound.",
                       error_count == 0 && !quiet);
        error_count++;
        continue;
      }

      auto &ref_rank = ref_ranks[v_];
      ref_total_rank += ref_rank;
      ValueT diff = fabs(ref_rank - unorder_ranks[v]);
      if ((ref_rank > 1e-12 && diff > threshold * ref_rank) ||
          (ref_rank <= 1e-12 && diff > threshold)) {
        util::PrintMsg("FAIL: rank[" + std::to_string(v) + "] (" +
                           std::to_string(unorder_ranks[v]) +
                           ") != " + std::to_string(ref_rank),
                       error_count == 0 && !quiet);
        error_count++;
      }
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_pos = v_;
      }
      if (ref_rank > 1e-12) {
        ValueT rdiff = diff / ref_rank;
        if (rdiff > max_rdiff) {
          max_rdiff = rdiff;
          max_rdiff_pos = v_;
        }
      }
    }
    if (error_count == 0)
      util::PrintMsg("PASS", !quiet);
    else
      util::PrintMsg("number of errors : " + std::to_string(error_count),
                     !quiet);

    util::PrintMsg("Reference total rank : " + std::to_string(ref_total_rank),
                   !quiet);

    util::PrintMsg("Maximum difference : ", !quiet, false);
    util::PrintMsg(
        "rank[" + std::to_string(ref_node_ids[max_diff_pos]) + "] " +
            std::to_string(unorder_ranks[ref_node_ids[max_diff_pos]]) +
            " vs. " + std::to_string(ref_ranks[max_diff_pos]) + ", ",
        max_diff_pos < nodes && !quiet, false);
    util::PrintMsg(std::to_string(max_diff), !quiet);

    util::PrintMsg("Maximum relative difference :", !quiet, false);
    util::PrintMsg(
        "rank[" + std::to_string(ref_node_ids[max_rdiff_pos]) + "] " +
            std::to_string(unorder_ranks[ref_node_ids[max_rdiff_pos]]) +
            " vs. " + std::to_string(ref_ranks[max_rdiff_pos]) + ", ",
        max_rdiff_pos < nodes && !quiet, false);
    util::PrintMsg(std::to_string(max_rdiff * 100), !quiet);
  } else {
    if (error_count == 0)
      util::PrintMsg("PASS", !quiet);
    else
      util::PrintMsg("number of errors : " + std::to_string(error_count),
                     !quiet);
  }

  util::PrintMsg("Order Validity: ", !quiet, false);
  error_count = 0;
  for (VertexT v = 0; v < nodes - 1; v++)
    if (h_ranks[v] < h_ranks[v + 1]) {
      util::PrintMsg("FAIL: rank[" + std::to_string(h_node_ids[v]) + "] (" +
                         std::to_string(h_ranks[v]) + "), place " +
                         std::to_string(v) + " < rank[" +
                         std::to_string(h_node_ids[v + 1]) + "] (" +
                         std::to_string(h_ranks[v + 1]) + "), place " +
                         std::to_string(v + 1),
                     error_count == 0 & !quiet);
      error_count++;
    }
  if (error_count == 0)
    util::PrintMsg("PASS", !quiet);
  else
    util::PrintMsg("number of errors : " + std::to_string(error_count), !quiet);
  free(unorder_ranks);
  unorder_ranks = NULL;

  return error_count;
}

}  // namespace pr
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
