// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_test.cu
 *
 * @brief Test related functions for hits
 */

#pragma once

namespace gunrock {
namespace app {
namespace hits {

/******************************************************************************
 * HITS Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference hits ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(const GraphT &graph, typename GraphT::ValueT *ref_hrank,
                     typename GraphT::ValueT *ref_arank,
                     typename GraphT::SizeT max_iter, bool quiet) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // Allocate space for next ranks
  ValueT *curr_hrank = new ValueT[graph.nodes];
  ValueT *curr_arank = new ValueT[graph.nodes];
  ValueT *next_hrank = new ValueT[graph.nodes];
  ValueT *next_arank = new ValueT[graph.nodes];

  // Set next scores to 1 and 0
  for (SizeT v = 0; v < graph.nodes; v++) {
    curr_hrank[v] = 1.0;
    curr_arank[v] = 1.0;
  }

  for (SizeT iterCount = 0; iterCount < max_iter; iterCount++) {
    // Set next scores to 1 and 0
    for (SizeT v = 0; v < graph.nodes; v++) {
      next_hrank[v] = 0.0;
      next_arank[v] = 0.0;
    }

    // Iterate through graph to add hub and auth scores
    for (SizeT link = 0; link < graph.edges; link++) {
      VertexT src = graph.edge_pairs[link].x;
      VertexT dest = graph.edge_pairs[link].y;

      next_hrank[src] += curr_arank[dest];
      next_arank[dest] += curr_hrank[src];
    }

    // Normalize
    ValueT h_norm = 0.0;
    ValueT a_norm = 0.0;

    for (SizeT v = 0; v < graph.nodes; v++) {
      h_norm += pow(next_hrank[v], 2.0);
      a_norm += pow(next_arank[v], 2.0);
    }

    h_norm = sqrt(h_norm);
    a_norm = sqrt(a_norm);

    for (SizeT v = 0; v < graph.nodes; v++) {
      next_hrank[v] /= h_norm;
      next_arank[v] /= a_norm;
    }

    // Swap current and next
    auto curr_hrank_temp = curr_hrank;
    curr_hrank = next_hrank;
    next_hrank = curr_hrank_temp;

    auto curr_arank_temp = curr_arank;
    curr_arank = next_arank;
    next_arank = curr_arank_temp;
  }

  // Copy to ref
  for (SizeT v = 0; v < graph.nodes; v++) {
    ref_hrank[v] = curr_hrank[v];
    ref_arank[v] = curr_arank[v];
  }

  delete[] curr_hrank;
  delete[] curr_arank;
  delete[] next_hrank;
  delete[] next_arank;

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  return elapsed;
}

// Structs and functions to rank hubs and authorities
template <typename RankPairT>
bool HITSCompare(const RankPairT &elem1, const RankPairT &elem2) {
  return elem1.rank > elem2.rank;
}

template <typename GraphT>
struct RankPair {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;

  VertexT vertex_id;
  ValueT rank;
};

template <typename GraphT>
struct RankList {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;

  typedef RankPair<GraphT> RankPairT;
  RankPairT *rankPairs;
  SizeT num_nodes;

  RankList(ValueT *ranks, SizeT num_nodes) {
    rankPairs = new RankPairT[num_nodes];

    this->num_nodes = num_nodes;

    for (SizeT v = 0; v < this->num_nodes; v++) {
      rankPairs[v].vertex_id = v;
      rankPairs[v].rank = ranks[v];
    }

    std::stable_sort(rankPairs, rankPairs + num_nodes, HITSCompare<RankPairT>);
  }

  ~RankList() {
    delete[] rankPairs;
    rankPairs = NULL;
  }
};

/**
 * @brief Displays the hits result
 *
 * @param[in] vertices Vertex Ids
 * @param[in] hrank HITS hub scores (sorted)
 * @param[in] arank HITS authority scores (sorted)
 * @param[in] num_vertices Number of vertices to display
 */
template <typename GraphT>
void DisplaySolution(typename GraphT::ValueT *hrank,
                     typename GraphT::ValueT *arank,
                     typename GraphT::SizeT num_vertices) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;

  typedef RankList<GraphT> RankListT;
  RankListT hlist(hrank, num_vertices);
  RankListT alist(arank, num_vertices);

  // At most top 10 ranked vertices
  SizeT top = (num_vertices < 10) ? num_vertices : 10;

  util::PrintMsg("Top " + std::to_string(top) + " Ranks:");

  util::PrintMsg("Hub Ranks:");

  for (SizeT i = 0; i < top; ++i) {
    util::PrintMsg(
        "Vertex ID: " + std::to_string(hlist.rankPairs[i].vertex_id) +
        ", Hub Rank: " + std::to_string(hlist.rankPairs[i].rank));
  }

  util::PrintMsg("Authority Ranks:");
  for (SizeT i = 0; i < top; ++i) {
    util::PrintMsg(
        "Vertex ID: " + std::to_string(alist.rankPairs[i].vertex_id) +
        ", Authority Rank: " + std::to_string(alist.rankPairs[i].rank));
  }

  return;
}

/**
 * @brief Validation of hits results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::ValueT *h_hrank, typename GraphT::ValueT *h_arank,
    typename GraphT::ValueT *ref_hrank, typename GraphT::ValueT *ref_arank,
    typename GraphT::ValueT tol, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::VertexT ValueT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");

  printf("Tol: %f\n", tol);

  if (!quick) {
    for (SizeT v = 0; v < graph.nodes; v++) {
      if (fabs(ref_hrank[v] - h_hrank[v]) > tol) num_errors++;
      if (fabs(ref_arank[v] - h_arank[v]) > tol) num_errors++;
    }

    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
  }

  return num_errors;
}

}  // namespace hits
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
