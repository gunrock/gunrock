// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * louvain_test.cu
 *
 * @brief Test related functions for Louvain
 */

#pragma once

#include <map>
#include <unordered_map>
#include <set>

namespace gunrock {
namespace app {
namespace louvain {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

cudaError_t UseParameters_test(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<uint32_t>(
      "omp-threads",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "Number of threads for parallel omp louvain implementation; 0 for "
      "default.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "omp-runs",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1, "Number of runs for parallel omp louvain implementation.", __FILE__,
      __LINE__));

  return retval;
}

/**
 * @brief Displays the community detection result (i.e. communities of vertices)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] community for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length) {
  if (length > 40) length = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < length; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(array[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * Louvain Testing Routines
 *****************************************************************************/

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
ValueT Get_Modularity(const GraphT &graph,
                      typename GraphT::VertexT *communities = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT nodes = graph.nodes;
  ValueT *w_v2 = new ValueT[nodes];
  ValueT *w_c = new ValueT[nodes];
  ValueT m2 = 0;
  ValueT w_in = 0;

  for (VertexT v = 0; v < nodes; v++) {
    w_v2[v] = 0;
    w_c[v] = 0;
  }

  //#pragma omp parallel for //reduction(+:m)
  for (VertexT v = 0; v < nodes; v++) {
    SizeT start_e = graph.GetNeighborListOffset(v);
    SizeT degree = graph.GetNeighborListLength(v);
    VertexT c_v = (communities == NULL) ? v : communities[v];

    for (SizeT k = 0; k < degree; k++) {
      SizeT e = start_e + k;
      VertexT u = graph.GetEdgeDest(e);
      ValueT w = graph.edge_values[e];
      w_v2[v] += w;
      w_c[c_v] += w;

      VertexT c_u = (communities == NULL) ? u : communities[u];
      if (c_v != c_u) continue;
      w_in += w;
    }
  }

  ValueT q = 0;
  for (VertexT v = 0; v < nodes; v++) {
    m2 += w_v2[v];
    if (w_c[v] != 0) q += w_c[v] * w_c[v];
  }

  delete[] w_v2;
  w_v2 = NULL;
  delete[] w_c;
  w_c = NULL;
  return (w_in - q / m2) / m2;

  /*
  q = 0;
  ValueT w1 = 0, w2 = 0;
  //#pragma omp parallel for //reduction(+:q)
  for (VertexT v = 0; v < nodes; v++)
  {
      VertexT comm_v = (communities == NULL) ? v : communities[v];
      q_v[v] = 0;
      SizeT start_e = graph.GetNeighborListOffset(v);
      SizeT degree  = graph.GetNeighborListLength(v);
      ValueT w_v2_v = w_v2[v];
      std::unordered_map<VertexT, ValueT> w_v2v;

      for (SizeT k = 0; k < degree; k++)
      {
          SizeT   e = start_e + k;
          VertexT u = graph.GetEdgeDest(e);
          if (comm_v != ((communities == NULL) ? u : communities[u]))
              continue;
          ValueT  w = graph.edge_values[e];
          auto it = w_v2v.find(u);
          if (it == w_v2v.end())
              w_v2v[u] = w;
          else
              it -> second += w;
      }

      auto &comm = comms[comm_v];
      for (auto it = comm.begin(); it != comm.end(); it++)
      {
          VertexT u = *it;
          auto it2 = w_v2v.find(u);
          ValueT  w = 0;
          if (it2 != w_v2v.end())
              w = w_v2v[u];
          w1 += w;
          w2 += w_v2_v * w_v2[u];
          q_v[v] += (w - w_v2_v * w_v2[u] / m);
      }
      //q += q_v;
      w_v2v.clear();
  }

  for (VertexT v = 0; v < nodes; v++)
      q += q_v[v];
  util::PrintMsg("w1 = " + std::to_string(w1) +
      + ", w2 = " + std::to_string(w2)
      + ", w_c^2 / m = " + std::to_string(w1 - q));
  q /= m;

  delete[] q_v ; q_v  = NULL;
  delete[] w_2v; w_2v = NULL;
  delete[] w_v2; w_v2 = NULL;
  return q;
  */
}

/**
 * @brief Simple CPU-based reference Louvain Community Detection implementation
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   parameters    Input parameters
 * @param[in]   graph         Input graph
 * @param[out]  communities   Community IDs for each vertex
 * \return      double        Time taken for the Louvain implementation
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CPU_Reference(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::VertexT *communities,
                     std::vector<std::vector<typename GraphT::VertexT> *>
                         *pass_communities = NULL,
                     std::vector<GraphT *> *pass_graphs = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  VertexT max_passes = parameters.Get<VertexT>("max-passes");
  VertexT max_iters = parameters.Get<VertexT>("max-iters");
  bool pass_stats = parameters.Get<bool>("pass-stats");
  bool iter_stats = parameters.Get<bool>("iter-stats");
  ValueT pass_gain_threshold = parameters.Get<ValueT>("pass-th");
  ValueT iter_gain_threshold = parameters.Get<ValueT>("iter-th");

  bool has_pass_communities = false;
  if (pass_communities != NULL)
    has_pass_communities = true;
  else
    pass_communities = new std::vector<std::vector<VertexT> *>;
  pass_communities->clear();
  bool has_pass_graphs = false;
  if (pass_graphs != NULL) has_pass_graphs = true;

  ValueT q = Get_Modularity(graph);
  ValueT *w_v2c = new ValueT[graph.nodes];
  VertexT *neighbor_comms = new VertexT[graph.nodes];
  VertexT *comm_convert = new VertexT[graph.nodes];
  ValueT *w_v2self = new ValueT[graph.nodes];
  ValueT *w_v2 = new ValueT[graph.nodes];
  ValueT *w_c2 = new ValueT[graph.nodes];
  typedef std::pair<VertexT, ValueT> pairT;
  std::vector<pairT> *w_c2c = new std::vector<pairT>[graph.nodes];

  GraphT temp_graph;
  temp_graph.Allocate(graph.nodes, graph.edges, util::HOST);
  auto &temp_row_offsets = temp_graph.CsrT::row_offsets;
  auto &temp_column_indices = temp_graph.CsrT::column_indices;
  auto &temp_edge_values = temp_graph.CsrT::edge_values;

  auto c_graph = &graph;
  auto n_graph = c_graph;
  n_graph = NULL;

  ValueT m2 = 0;
  for (SizeT e = 0; e < graph.edges; e++) m2 += graph.CsrT::edge_values[e];
  for (SizeT v = 0; v < graph.nodes; v++)
    w_v2c[v] = util::PreDefinedValues<ValueT>::InvalidValue;

  util::CpuTimer cpu_timer, pass_timer, iter_timer;
  cpu_timer.Start();

  int pass_num = 0;
  while (pass_num < max_passes) {
    // Pass initialization
    if (pass_stats) pass_timer.Start();
    if (iter_stats) iter_timer.Start();

    auto &current_graph = *c_graph;
    SizeT nodes = current_graph.nodes;
    auto c_communities = new std::vector<VertexT>;
    auto &current_communities = *c_communities;
    current_communities.reserve(nodes);

    for (VertexT v = 0; v < nodes; v++) {
      current_communities[v] = v;
      w_v2[v] = 0;
      w_v2self[v] = 0;
      SizeT start_e = current_graph.GetNeighborListOffset(v);
      SizeT degree = current_graph.GetNeighborListLength(v);

      for (SizeT k = 0; k < degree; k++) {
        SizeT e = start_e + k;
        VertexT u = current_graph.GetEdgeDest(e);
        ValueT w = current_graph.edge_values[e];
        w_v2[v] += w;
        if (u == v) w_v2self[v] += w;
      }
      w_c2[v] = w_v2[v];
    }
    if (iter_stats) iter_timer.Stop();
    util::PrintMsg("pass " + std::to_string(pass_num) +
                       ", pre-iter, elapsed = " +
                       std::to_string(iter_timer.ElapsedMillis()),
                   iter_stats);

    // Modulation Optimization
    int iter_num = 0;
    ValueT pass_gain = 0;
    while (iter_num < max_iters) {
      if (iter_stats) iter_timer.Start();
      ValueT iter_gain = 0;
      for (VertexT v = 0; v < nodes; v++) {
        SizeT start_e = current_graph.GetNeighborListOffset(v);
        SizeT degree = current_graph.GetNeighborListLength(v);

        VertexT num_neighbor_comms = 0;
        for (SizeT k = 0; k < degree; k++) {
          SizeT e = start_e + k;
          VertexT u = current_graph.GetEdgeDest(e);
          ValueT w = current_graph.edge_values[e];
          VertexT c = current_communities[u];

          if (!util::isValid(w_v2c[c])) {
            w_v2c[c] = w;
            neighbor_comms[num_neighbor_comms] = c;
            num_neighbor_comms++;
          } else
            w_v2c[c] += w;
        }

        VertexT org_comm = current_communities[v];
        VertexT new_comm = org_comm;
        ValueT w_v2c_org = 0;
        if (util::isValid(w_v2c[org_comm])) w_v2c_org = w_v2c[org_comm];
        ValueT w_v2_v = w_v2[v];
        ValueT gain_base =
            w_v2self[v] - w_v2c_org - (w_v2[v] - w_c2[org_comm]) * w_v2_v / m2;
        ValueT max_gain = 0;

        for (VertexT i = 0; i < num_neighbor_comms; i++) {
          VertexT c = neighbor_comms[i];
          if (c == org_comm) {
            w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
            continue;
          }
          ValueT w_v2c_c = w_v2c[c];
          w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
          ValueT gain = gain_base + w_v2c_c - w_c2[c] * w_v2_v / m2;

          if (gain > max_gain) {
            max_gain = gain;
            new_comm = c;
          }
        }

        if (max_gain > 0 && new_comm != current_communities[v]) {
          iter_gain += max_gain;
          current_communities[v] = new_comm;
          w_c2[new_comm] += w_v2[v];
          w_c2[org_comm] -= w_v2[v];
        }
      }

      iter_num++;
      iter_gain *= 2;
      iter_gain /= m2;
      q += iter_gain;
      pass_gain += iter_gain;
      if (iter_stats) iter_timer.Stop();
      util::PrintMsg(
          "pass " + std::to_string(pass_num) + ", iter " +
              std::to_string(iter_num) + ", q = " + std::to_string(q) +
              ", iter_gain = " + std::to_string(iter_gain) +
              ", pass_gain = " + std::to_string(pass_gain) +
              ", elapsed = " + std::to_string(iter_timer.ElapsedMillis()),
          iter_stats);
      if (iter_gain < iter_gain_threshold) break;
    }

    // Community Aggregation
    if (iter_stats) iter_timer.Start();
    VertexT num_comms = 0;
    for (VertexT v = 0; v < nodes; v++) comm_convert[v] = 0;
    for (VertexT v = 0; v < nodes; v++)
      comm_convert[current_communities[v]] = 1;
    for (VertexT v = 0; v < nodes; v++) {
      if (comm_convert[v] == 0) continue;
      comm_convert[v] = num_comms;
      num_comms++;
    }

    for (VertexT v = 0; v < nodes; v++)
      current_communities[v] = comm_convert[current_communities[v]];
    pass_communities->push_back(c_communities);

    for (VertexT v = 0; v < nodes; v++) {
      SizeT start_e = current_graph.GetNeighborListOffset(v);
      SizeT degree = current_graph.GetNeighborListLength(v);
      VertexT comm_v = current_communities[v];
      auto &w_c2c_c = w_c2c[comm_v];

      for (SizeT k = 0; k < degree; k++) {
        SizeT e = start_e + k;
        VertexT u = current_graph.GetEdgeDest(e);
        ValueT w = current_graph.edge_values[e];
        VertexT comm_u = current_communities[u];

        w_c2c_c.push_back(std::make_pair(comm_u, w));
      }
    }

    SizeT num_edges = 0;
    auto &w_2c = w_v2c;
    temp_row_offsets[0] = 0;
    for (VertexT c = 0; c < num_comms; c++) {
      auto &w_c2c_c = w_c2c[c];
      VertexT num_neighbor_comms = 0;
      for (auto it = w_c2c_c.begin(); it != w_c2c_c.end(); it++) {
        VertexT u_c = it->first;
        ValueT w = it->second;
        if (!util::isValid(w_2c[u_c])) {
          w_2c[u_c] = w;
          neighbor_comms[num_neighbor_comms] = u_c;
          num_neighbor_comms++;
        } else
          w_2c[u_c] += w;
      }
      w_c2c_c.clear();

      for (VertexT i = 0; i < num_neighbor_comms; i++) {
        VertexT u_c = neighbor_comms[i];
        ValueT w = w_2c[u_c];
        temp_column_indices[num_edges + i] = u_c;
        temp_edge_values[num_edges + i] = w;
        w_2c[u_c] = util::PreDefinedValues<ValueT>::InvalidValue;
      }
      num_edges += num_neighbor_comms;
      temp_row_offsets[c + 1] = num_edges;
    }

    n_graph = new GraphT;
    auto &next_graph = *n_graph;
    if (has_pass_graphs) pass_graphs->push_back(n_graph);
    next_graph.Allocate(num_comms, num_edges, util::HOST);
    memcpy(next_graph.CsrT::row_offsets + 0, temp_row_offsets + 0,
           sizeof(SizeT) * (num_comms + 1));
    memcpy(next_graph.CsrT::column_indices + 0, temp_column_indices + 0,
           sizeof(VertexT) * num_edges);
    memcpy(next_graph.CsrT::edge_values + 0, temp_edge_values + 0,
           sizeof(ValueT) * num_edges);

    if (iter_stats) iter_timer.Stop();
    util::PrintMsg("pass " + std::to_string(pass_num) +
                       ", graph compaction, elapsed = " +
                       std::to_string(iter_timer.ElapsedMillis()),
                   iter_stats);

    if (pass_stats) pass_timer.Stop();
    util::PrintMsg(
        "pass " + std::to_string(pass_num) + ", #v = " + std::to_string(nodes) +
            " -> " + std::to_string(num_comms) +
            ", #e = " + std::to_string(current_graph.edges) + " -> " +
            std::to_string(num_edges) + ", #iter = " +
            std::to_string(iter_num) + ", q = " + std::to_string(q) +
            ", pass_gain = " + std::to_string(pass_gain) +
            ", elapsed = " + std::to_string(pass_timer.ElapsedMillis()),
        pass_stats);

    if (pass_num != 0 && !has_pass_graphs) {
      current_graph.Release(util::HOST);
      delete c_graph;
    }
    c_graph = n_graph;
    n_graph = NULL;

    pass_num++;
    if (pass_gain < pass_gain_threshold) break;
  }

  // Assining communities to vertices in original graph
  for (VertexT v = 0; v < graph.nodes; v++) communities[v] = v;
  pass_num = 0;
  for (auto it = pass_communities->begin(); it != pass_communities->end();
       it++) {
    auto &v2c = *(*it);
    for (VertexT v = 0; v < graph.nodes; v++) {
      communities[v] = v2c[communities[v]];
    }
    pass_num++;
  }
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  // Clearn-up
  if (!has_pass_communities) {
    for (auto it = pass_communities->begin(); it != pass_communities->end();
         it++) {
      (*it)->clear();
      delete *it;
    }
    pass_communities->clear();
    delete pass_communities;
    pass_communities = NULL;
  }

  temp_graph.Release();
  delete[] comm_convert;
  comm_convert = NULL;
  delete[] w_c2c;
  w_c2c = NULL;
  delete[] w_v2self;
  w_v2self = NULL;
  delete[] w_v2;
  w_v2 = NULL;
  delete[] w_c2;
  w_c2 = NULL;
  delete[] w_v2c;
  w_v2c = NULL;
  delete[] neighbor_comms;
  neighbor_comms = NULL;
  return elapsed;
}

/**
 * @brief Simple CPU-based reference Louvain Community Detection implementation
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   parameters    Input parameters
 * @param[in]   graph         Input graph
 * @param[out]  communities   Community IDs for each vertex
 * \return      double        Time taken for the Louvain implementation
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double OMP_Reference(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::VertexT *communities,
                     std::vector<std::vector<typename GraphT::VertexT> *>
                         *pass_communities = NULL,
                     std::vector<GraphT *> *pass_graphs = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  VertexT max_passes = parameters.Get<VertexT>("max-passes");
  VertexT max_iters = parameters.Get<VertexT>("max-iters");
  bool pass_stats = parameters.Get<bool>("pass-stats");
  bool iter_stats = parameters.Get<bool>("iter-stats");
  ValueT pass_gain_threshold = parameters.Get<ValueT>("pass-th");
  ValueT iter_gain_threshold = parameters.Get<ValueT>("iter-th");
  int num_threads = parameters.Get<int>("omp-threads");
  ValueT first_threshold = parameters.Get<ValueT>("1st-th");

#pragma omp parallel
  {
    if (num_threads == 0) num_threads = omp_get_num_threads();
  }
  util::PrintMsg("#threads = " + std::to_string(num_threads) +
                 ", 1st-th = " + std::to_string(first_threshold));

  bool has_pass_communities = false;
  if (pass_communities != NULL)
    has_pass_communities = true;
  else
    pass_communities = new std::vector<std::vector<VertexT> *>;
  pass_communities->clear();
  bool has_pass_graphs = false;
  if (pass_graphs != NULL) has_pass_graphs = true;

  ValueT q = Get_Modularity(graph);
  ValueT **w_v2cs = new ValueT *[num_threads];
  VertexT **neighbor_commss = new VertexT *[num_threads];
  ValueT *iter_gains = new ValueT[num_threads];
  VertexT *comm_convert = new VertexT[graph.nodes];
  ValueT *w_v2self = new ValueT[graph.nodes];
  ValueT *w_v2 = new ValueT[graph.nodes];
  ValueT *w_c2 = new ValueT[graph.nodes];
  VertexT *comm_counts = new VertexT[num_threads];

  typedef std::pair<VertexT, ValueT> pairT;
  std::vector<pairT> **w_c2cs = new std::vector<pairT> *[num_threads];
#pragma omp parallel num_threads(num_threads)
  {
    int thread_num = omp_get_thread_num();
    w_v2cs[thread_num] = new ValueT[graph.nodes];
    auto &w_v2c = w_v2cs[thread_num];
    for (SizeT v = 0; v < graph.nodes; v++)
      w_v2c[v] = util::PreDefinedValues<ValueT>::InvalidValue;

    neighbor_commss[thread_num] = new VertexT[graph.nodes];
    w_c2cs[thread_num] = new std::vector<pairT>[graph.nodes];
  }

  // GraphT temp_graph;
  // temp_graph.Allocate(graph.nodes, graph.edges, util::HOST);
  // auto &temp_row_offsets    = temp_graph.CsrT::row_offsets;
  // auto &temp_column_indices = temp_graph.CsrT::column_indices;
  // auto &temp_edge_values    = temp_graph.CsrT::edge_values;
  SizeT *temp_row_offsets = new SizeT[graph.nodes + 1];
  std::vector<SizeT> *temp_column_indicess =
      new std::vector<SizeT>[num_threads];
  std::vector<ValueT> *temp_edge_valuess = new std::vector<ValueT>[num_threads];

  auto c_graph = &graph;
  auto n_graph = c_graph;
  n_graph = NULL;

  ValueT m2 = 0;
  //#pragma omp parallel for num_threads(num_threads) reduction(+:m2)
  for (SizeT e = 0; e < graph.edges; e++) m2 += graph.CsrT::edge_values[e];

  util::CpuTimer cpu_timer, pass_timer, iter_timer;
  cpu_timer.Start();

  int pass_num = 0;
  while (pass_num < max_passes) {
    // if (pass_num > 1)
    //    num_threads = 1;

    // Pass initialization
    if (pass_stats) pass_timer.Start();
    if (iter_stats) iter_timer.Start();

    auto &current_graph = *c_graph;
    SizeT nodes = current_graph.nodes;
    auto c_communities = new std::vector<VertexT>;
    auto &current_communities = *c_communities;
    current_communities.reserve(nodes);

#pragma omp parallel for num_threads(num_threads)
    for (VertexT v = 0; v < nodes; v++) {
      current_communities[v] = v;
      w_v2[v] = 0;
      w_v2self[v] = 0;
      SizeT start_e = current_graph.GetNeighborListOffset(v);
      SizeT degree = current_graph.GetNeighborListLength(v);

      for (SizeT k = 0; k < degree; k++) {
        SizeT e = start_e + k;
        VertexT u = current_graph.GetEdgeDest(e);
        ValueT w = current_graph.edge_values[e];
        w_v2[v] += w;
        if (u == v) w_v2self[v] += w;
      }
      w_c2[v] = w_v2[v];
    }
    if (iter_stats) iter_timer.Stop();
    util::PrintMsg("pass " + std::to_string(pass_num) +
                       ", pre-iter, elapsed = " +
                       std::to_string(iter_timer.ElapsedMillis()),
                   iter_stats);

    // Modulation Optimization
    int iter_num = 0;
    ValueT pass_gain = 0;
    ValueT iter_gain = 0;
    bool to_continue = true;

    //#pragma omp parallel num_threads(num_threads)
    // while (iter_num < max_iters)
    {
      while (to_continue) {
        // int thread_num = omp_get_thread_num();
        //#pragma omp single
        {
          if (iter_stats) iter_timer.Start();
          iter_gain = 0;
        }
        for (int t = 0; t < num_threads; t++) iter_gains[t] = 0;
          // iter_gains[thread_num] = 0;
          // auto &w_v2c = w_v2cs[thread_num];
          // auto &neighbor_comms = neighbor_commss[thread_num];
          // auto &iter_gain = iter_gains[thread_num];

#pragma omp parallel for num_threads(num_threads)
        //#pragma omp for
        for (VertexT v = 0; v < nodes; v++)
        //#pragma omp parallel num_threads(num_threads)
        {
          int thread_num = omp_get_thread_num();
          // iter_gains[thread_num] = 0;
          // VertexT start_v = nodes / num_threads * thread_num;
          // VertexT end_v   = nodes / num_threads * (thread_num + 1);
          auto &w_v2c = w_v2cs[thread_num];
          auto &neighbor_comms = neighbor_commss[thread_num];
          // auto &iter_gain = iter_gains[thread_num];
          // if (thread_num == 0)
          //    start_v = 0;
          // if (thread_num == num_threads - 1)
          //    end_v = nodes;

          // for (VertexT v = start_v; v < end_v; v++)
          {
            SizeT start_e = current_graph.GetNeighborListOffset(v);
            SizeT degree = current_graph.GetNeighborListLength(v);

            VertexT num_neighbor_comms = 0;
            for (SizeT k = 0; k < degree; k++) {
              SizeT e = start_e + k;
              VertexT u = current_graph.GetEdgeDest(e);
              ValueT w = current_graph.edge_values[e];
              VertexT c = current_communities[u];

              if (!util::isValid(w_v2c[c])) {
                w_v2c[c] = w;
                neighbor_comms[num_neighbor_comms] = c;
                num_neighbor_comms++;
              } else
                w_v2c[c] += w;
            }

            VertexT org_comm = current_communities[v];
            VertexT new_comm = org_comm;
            ValueT w_v2c_org = 0;
            if (util::isValid(w_v2c[org_comm])) w_v2c_org = w_v2c[org_comm];
            ValueT w_v2_v = w_v2[v];
            ValueT gain_base = w_v2self[v] - w_v2c_org -
                               (w_v2_v - w_c2[org_comm]) * w_v2_v / m2;
            //- w_v2_v * w_v2_v / m2;
            ValueT max_gain = 0;
            ValueT max_w_v2c_c = 0;

            for (VertexT i = 0; i < num_neighbor_comms; i++) {
              VertexT c = neighbor_comms[i];
              if (c == org_comm) {
                w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
                continue;
              }
              ValueT w_v2c_c = w_v2c[c];
              w_v2c[c] = util::PreDefinedValues<ValueT>::InvalidValue;
              ValueT gain = gain_base + w_v2c_c - w_c2[c] * w_v2_v / m2;
              //- (w_c2[c] - w_c2[org_comm]) * w_v2_v / m2;

              if (gain > max_gain) {
                max_gain = gain;
                new_comm = c;
                max_w_v2c_c = w_v2c_c;
              }
            }

            if (new_comm != current_communities[v]) {
              // max_gain = w_v2self[v] - w_v2c_org + max_w_v2c_c
              //    - (w_v2_v - w_c2[org_comm] + w_c2[new_comm]) * w_v2_v / m2;

              if (max_gain > 0) {
                current_communities[v] = new_comm;
#pragma omp atomic
                w_c2[new_comm] += w_v2[v];

#pragma omp atomic
                w_c2[org_comm] -= w_v2[v];

                iter_gains[thread_num] += max_gain;
              }
            }
          }
        }

        //#pragma omp barrier
        //#pragma omp single
        {
          iter_gain = 0;
          for (int t = 0; t < num_threads; t++) {
            iter_gain += iter_gains[t];
          }

          iter_gain *= 2;
          iter_gain /= m2;
          q += iter_gain;
          pass_gain += iter_gain;
          if (iter_stats) {
            iter_timer.Stop();
            util::PrintMsg(
                "pass " + std::to_string(pass_num) + ", iter " +
                    std::to_string(iter_num) + ", q = " + std::to_string(q) +
                    ", iter_gain = " + std::to_string(iter_gain) +
                    ", pass_gain = " + std::to_string(pass_gain) +
                    ", elapsed = " + std::to_string(iter_timer.ElapsedMillis()),
                iter_stats);
          }
          iter_num++;
          if ((pass_num != 0 && iter_gain < iter_gain_threshold) ||
              (pass_num == 0 && iter_gain < first_threshold) ||
              iter_num >= max_iters)
            to_continue = false;
        }
      }  // end of while (to_continue)
    }    // end of omp parallel

    // Community Aggregation
    if (iter_stats) iter_timer.Start();

    // util::CpuTimer timer1, timer2, timer3, timer4;

    // timer1.Start();
    VertexT num_comms = 0;
    SizeT num_edges = 0;
#pragma omp parallel num_threads(num_threads)
    {
      int thread_num = omp_get_thread_num();
      VertexT start_v = nodes / num_threads * thread_num;
      VertexT end_v = nodes / num_threads * (thread_num + 1);
      if (thread_num == 0) start_v = 0;
      if (thread_num == num_threads - 1) end_v = nodes;

#pragma omp for
      for (VertexT v = 0; v < nodes; v++)
        // for (VertexT v = start_v; v < end_v; v++)
        comm_convert[v] = 0;
#pragma omp barrier
        // util::PrintMsg(std::to_string(thread_num) + " 0");

#pragma omp for
      for (VertexT v = 0; v < nodes; v++)
        // for (VertexT v = start_v; v < end_v; v++)
        comm_convert[current_communities[v]] = 1;
#pragma omp barrier
      // util::PrintMsg(std::to_string(thread_num) + " 1");

      auto &comm_count = comm_counts[thread_num];
      comm_count = 1;
      for (VertexT v = start_v; v < end_v; v++) {
        if (comm_convert[v] == 0) continue;

        comm_convert[v] = comm_count;
        comm_count++;
      }
#pragma omp barrier
      // util::PrintMsg(std::to_string(thread_num) + " 2");

#pragma omp single
      {
        num_comms = 0;
        for (int t = 0; t < num_threads; t++) {
          VertexT temp = comm_counts[t] - 1;
          comm_counts[t] = num_comms;
          num_comms += temp;
        }
      }

      //#pragma omp for
      // for (VertexT v = 0; v < nodes; v++)
      for (VertexT v = start_v; v < end_v; v++) {
        if (comm_convert[v] != 0) {
          comm_convert[v]--;
          comm_convert[v] += comm_counts[thread_num];
        }
      }
#pragma omp barrier
      // util::PrintMsg(std::to_string(thread_num) + " 3");

#pragma omp for
      for (VertexT v = 0; v < nodes; v++)
        // for (VertexT v = start_v; v < end_v; v++)
        current_communities[v] = comm_convert[current_communities[v]];
        //}

#pragma omp single
      {
        pass_communities->push_back(c_communities);
        // timer1.Stop();

        // timer2.Start();
      }

      //#pragma omp parallel for num_threads(num_threads)
      ////reduction(+:iter_gain) for (VertexT v = 0; v < nodes; v++)
      //{
      // int thread_num = omp_get_thread_num();
      // VertexT start_v = nodes / num_threads * thread_num;
      // VertexT end_v   = nodes / num_threads * (thread_num + 1);
      // if (thread_num == 0)
      //    start_v = 0;
      // if (thread_num == num_threads - 1)
      //    end_v = nodes;

#pragma omp for
      for (VertexT v = 0; v < nodes; v++)
      // for (VertexT v = start_v; v < end_v; v++)
      {
        int thread_num = omp_get_thread_num();
        auto &w_c2c = w_c2cs[thread_num];
        SizeT start_e = current_graph.GetNeighborListOffset(v);
        SizeT degree = current_graph.GetNeighborListLength(v);
        VertexT comm_v = current_communities[v];
        auto &w_c2c_c = w_c2c[comm_v];

        for (SizeT k = 0; k < degree; k++) {
          SizeT e = start_e + k;
          VertexT u = current_graph.GetEdgeDest(e);
          ValueT w = current_graph.edge_values[e];
          VertexT comm_u = current_communities[u];

          w_c2c_c.push_back(std::make_pair(comm_u, w));
        }
      }
      //}

      //#pragma omp single
      //{
      //    timer2.Stop();
      //    timer3.Start();
      //}
      // temp_row_offsets[0] = 0;
      //#pragma omp parallel num_threads(num_threads)
      //{
      // int thread_num = omp_get_thread_num();
      VertexT start_c = num_comms / num_threads * thread_num;
      VertexT end_c = num_comms / num_threads * (thread_num + 1);
      if (thread_num == 0) start_c = 0;
      if (thread_num == num_threads - 1) end_c = num_comms;

      auto &temp_column_indices = temp_column_indicess[thread_num];
      auto &temp_edge_values = temp_edge_valuess[thread_num];
      auto &w_2c = w_v2cs[thread_num];
      auto &neighbor_comms = neighbor_commss[thread_num];

      for (VertexT c = start_c; c < end_c; c++) {
        VertexT num_neighbor_comms = 0;
        for (int t = 0; t < num_threads; t++) {
          auto &w_c2c_c = w_c2cs[t][c];
          for (auto it = w_c2c_c.begin(); it != w_c2c_c.end(); it++) {
            VertexT u_c = it->first;
            ValueT w = it->second;
            if (!util::isValid(w_2c[u_c])) {
              w_2c[u_c] = w;
              neighbor_comms[num_neighbor_comms] = u_c;
              num_neighbor_comms++;
            } else
              w_2c[u_c] += w;
          }
          w_c2c_c.clear();
        }

        for (VertexT i = 0; i < num_neighbor_comms; i++) {
          VertexT u_c = neighbor_comms[i];
          ValueT w = w_2c[u_c];
          temp_column_indices.push_back(u_c);
          temp_edge_values.push_back(w);
          w_2c[u_c] = util::PreDefinedValues<ValueT>::InvalidValue;
        }
        temp_row_offsets[c] = num_neighbor_comms;
      }

#pragma omp barrier
#pragma omp single
      {
        num_edges = 0;
        for (VertexT c = 0; c < num_comms; c++) {
          SizeT temp = temp_row_offsets[c];
          temp_row_offsets[c] = num_edges;
          num_edges += temp;
        }
        temp_row_offsets[num_comms] = num_edges;

        n_graph = new GraphT;
        // auto &next_graph = *n_graph;
        if (has_pass_graphs) pass_graphs->push_back(n_graph);
        n_graph->Allocate(num_comms, num_edges, util::HOST);
      }

      memcpy(n_graph->CsrT::column_indices + temp_row_offsets[start_c],
             temp_column_indices.data(),
             temp_column_indices.size() * sizeof(VertexT));
      memcpy(n_graph->CsrT::edge_values + temp_row_offsets[start_c],
             temp_edge_values.data(), temp_edge_values.size() * sizeof(ValueT));
      memcpy(n_graph->CsrT::row_offsets + start_c, temp_row_offsets + start_c,
             sizeof(SizeT) *
                 (end_c - start_c + ((thread_num == num_threads - 1) ? 1 : 0)));
      temp_column_indices.clear();
      temp_edge_values.clear();
    }
    // timer3.Stop();

    // timer4.Start();
    // timer4.Stop();

    // util::PrintMsg("Timers = "
    //    + std::to_string(timer1.ElapsedMillis()) + " "
    //    + std::to_string(timer2.ElapsedMillis()) + " "
    //    + std::to_string(timer3.ElapsedMillis()) + " "
    //    + std::to_string(timer4.ElapsedMillis()));

    if (iter_stats) {
      iter_timer.Stop();
      util::PrintMsg("pass " + std::to_string(pass_num) +
                     ", graph compaction, elapsed = " +
                     std::to_string(iter_timer.ElapsedMillis()));
    }

    if (pass_stats) {
      pass_timer.Stop();
      util::PrintMsg(
          "pass " + std::to_string(pass_num) + ", #v = " +
          std::to_string(nodes) + " -> " + std::to_string(num_comms) +
          ", #e = " + std::to_string(current_graph.edges) + " -> " +
          std::to_string(num_edges) + ", #iter = " + std::to_string(iter_num) +
          ", q = " + std::to_string(q) +
          ", pass_gain = " + std::to_string(pass_gain) +
          ", elapsed = " + std::to_string(pass_timer.ElapsedMillis()));
    }

    if (pass_num != 0 && !has_pass_graphs) {
      current_graph.Release(util::HOST);
      delete c_graph;
    }
    c_graph = n_graph;
    n_graph = NULL;

    pass_num++;
    if (pass_gain < pass_gain_threshold) break;
  }

  // Assining communities to vertices in original graph
  for (VertexT v = 0; v < graph.nodes; v++) communities[v] = v;
  pass_num = 0;
  for (auto it = pass_communities->begin(); it != pass_communities->end();
       it++) {
    auto &v2c = *(*it);
    for (VertexT v = 0; v < graph.nodes; v++) {
      communities[v] = v2c[communities[v]];
    }
    pass_num++;
  }
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  // Clearn-up
  if (!has_pass_communities) {
    for (auto it = pass_communities->begin(); it != pass_communities->end();
         it++) {
      (*it)->clear();
      delete *it;
    }
    pass_communities->clear();
    delete pass_communities;
    pass_communities = NULL;
  }

  // temp_graph.Release();
  delete[] temp_row_offsets;
  temp_row_offsets = NULL;
  delete[] temp_column_indicess;
  temp_column_indicess = NULL;
  delete[] temp_edge_valuess;
  temp_edge_valuess = NULL;
  delete[] comm_convert;
  comm_convert = NULL;
  delete[] w_v2self;
  w_v2self = NULL;
  delete[] w_v2;
  w_v2 = NULL;
  delete[] w_c2;
  w_c2 = NULL;
#pragma omp parallel num_threads(num_threads)
  {
    int thread_num = omp_get_thread_num();
    delete[] w_v2cs[thread_num];
    w_v2cs[thread_num] = NULL;

    delete[] neighbor_commss[thread_num];
    neighbor_commss[thread_num] = NULL;

    delete[] w_c2cs[thread_num];
    w_c2cs[thread_num] = 0;
  }
  delete[] w_c2cs;
  w_c2cs = NULL;
  delete[] w_v2cs;
  w_v2cs = NULL;
  delete[] neighbor_commss;
  neighbor_commss = NULL;
  return elapsed;
}

/**
 * @brief Validation of Louvain results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::VertexT *communities,
    typename GraphT::VertexT *ref_communities = NULL, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  char *comm_markers = new char[graph.nodes];
  for (VertexT v = 0; v < graph.nodes; v++) comm_markers[v] = 0;

  util::PrintMsg("Community Validity: ", !quiet, false);
  // Verify the result
  for (VertexT v = 0; v < graph.nodes; v++) {
    auto c = communities[v];
    if (c < 0 || c >= graph.nodes) {
      util::PrintMsg("FAIL: communties[" + std::to_string(v) +
                         "] = " + std::to_string(c) + " out of bound",
                     (!quiet) && (num_errors == 0));
      num_errors++;
      continue;
    }
    comm_markers[c] = 1;
  }
  if (num_errors != 0) {
    delete[] comm_markers;
    comm_markers = NULL;
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    return num_errors;
  }
  util::PrintMsg("PASS", !quiet);

  ValueT num_comms = 0;
  for (VertexT v = 0; v < graph.nodes; v++) {
    if (comm_markers[v] != 0) {
      num_comms++;
      comm_markers[v] = 0;
    }
  }
  ValueT q = Get_Modularity(graph, communities);
  util::PrintMsg("Computed: #communities = " + std::to_string(num_comms) +
                 ", modularity = " + std::to_string(q));

  if (ref_communities == NULL) {
    delete[] comm_markers;
    comm_markers = NULL;
    return num_errors;
  }

  for (VertexT v = 0; v < graph.nodes; v++) {
    auto c = ref_communities[v];
    if (c < 0 || c >= graph.nodes) {
      num_errors++;
      continue;
    }
    comm_markers[c] = 1;
  }
  if (num_errors != 0) {
    util::PrintMsg("Reference: " + std::to_string(num_errors) +
                       " vertices have communities out of bound.",
                   !quiet);
    delete[] comm_markers;
    comm_markers = NULL;
    return 0;
  }

  num_comms = 0;
  for (VertexT v = 0; v < graph.nodes; v++)
    if (comm_markers[v] != 0) num_comms++;

  q = Get_Modularity(graph, ref_communities);
  util::PrintMsg("Reference: #communities = " + std::to_string(num_comms) +
                 ", modularity = " + std::to_string(q));

  delete[] comm_markers;
  comm_markers = NULL;
  return num_errors;
}

}  // namespace louvain
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
