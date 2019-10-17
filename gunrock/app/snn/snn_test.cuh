// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * snn_test.cuh
 *
 * @brief Test related functions for snn
 */

#pragma once

// #define SNN_DEBUG 1

#ifdef SNN_DEBUG
#define debug(a...) fprintf(stderr, a)
#else
#define debug(a...)
#endif

#include <set>
#include <vector>

#include <gunrock/app/knn/knn_test.cuh>

namespace gunrock {
namespace app {
namespace snn {

/******************************************************************************
 * SNN Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference snn ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT, 
         typename VertexT = typename GraphT::VertexT,
         typename SizeT = typename GraphT::SizeT>
double CPU_Reference(
    const GraphT &graph,
    SizeT k,        // number of nearest neighbor
    SizeT eps,      // min number of SNN to increase snn-density
    SizeT min_pts,  // mininum snn-density to be core point
    VertexT point_x,// index of reference point
    VertexT point_y,// index of reference point
    SizeT *cluster, // cluster id
    SizeT *core_point_counter, 
    SizeT *cluster_counter, 
    bool quiet) {
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CsrT CsrT;

  struct Point {
    VertexT x;
    SizeT e_id;
    ValueT dist;

    Point() {}
    Point(VertexT X, SizeT E_id, ValueT Dist) : x(X), e_id(E_id), dist(Dist) {}
  };

  struct comp {
    inline bool operator()(const Point &p1, const Point &p2) {
      return (p1.dist < p2.dist);
    }
  };

  auto nodes = graph.nodes;
  auto edges = graph.edges;

  // util::PrintMsg("#threads: " + std::to_string(omp_get_num_threads()));

#pragma omp parallel for
  for (auto x = 0; x < nodes; ++x) {
    cluster[x] = x;
  }

  SizeT *knns = (SizeT*)malloc(sizeof(SizeT) * graph.nodes * k);    // knns
  std::set<SizeT> core_points;
  std::vector<std::set<SizeT>> adj;
  adj.resize(nodes);

#pragma omp parallel for
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    auto x_end = x_start + num;
    for (auto y = x_start; y < x_end; ++y) {
      auto neighbor = graph.CsrT::GetEdgeDest(y);
      adj[x].insert(neighbor);
    }
  }

  Point *distance = (Point *)malloc(sizeof(Point) * edges);
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

// implement CPU reference implementation
#pragma omp parallel for
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    auto x_end = x_start + num;
    for (SizeT i = x_start; i < x_end; ++i) {
      VertexT y = graph.column_indices[i];
      ValueT d1 = (ValueT)(x - point_x);
      ValueT d2 = (ValueT)(y - point_y);
      ValueT dist = d1*d1 + d2*d2;
      distance[i] = Point(x, i, dist);
    }
  }

// Sort distances for each adjacency list
#pragma omp parallel for
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    auto x_end = x_start + num;
    std::sort(distance + x_start, distance + x_end, comp());
  }

  /*
  //debug only
  for (int tested_node = 0; tested_node < nodes; ++tested_node){
  //    auto tested_node = 62734;
      auto e_start = graph.CsrT::GetNeighborListOffset(tested_node);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(tested_node);
      auto e_end = e_start + num_neighbors;
      printf("sorted neighbors of thread %d\n", tested_node);
      for (int x = e_start; x < e_end; ++x) {
          printf("%d(%lld) ", graph.column_indices[distance[x].e_id], distance[x].dist);
      }
      printf("\n");
  }
  */

  // Debug
#if SNN_DEBUG
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    auto x_end = x_start + num;
    debug("sorted neighbors of %d\n", x);
    for (int neighbor = x_start; neighbor < x_end; ++neighbor) {
      debug("(%d, %d)(%d) ", distance[neighbor].x,
            graph.column_indices[distance[neighbor].e_id],
            distance[neighbor].dist);
    }
    debug("\n");
  }
#endif

// Find k nearest neighbors
#pragma omp parallel for
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    if (num < k) continue;
    auto x_end = x_start + num;
    debug("%d nearest neighbors\n", k);
    int i = 0;
    for (int neighbor = x_start; neighbor < x_end && i < k; ++neighbor, ++i) {
      // k nearest points to (point_x, pointy)
      knns[x * k + i] = graph.CsrT::GetEdgeDest(distance[neighbor].e_id);
      debug("%d ", graph.CsrT::GetEdgeDest(distance[neighbor].e_id));
    }
    debug("\n");
  }

  // Find snn-density
  for (SizeT x = 0; x < nodes; ++x) {
    auto x_start = graph.CsrT::GetNeighborListOffset(x);
    auto num = graph.CsrT::GetNeighborListLength(x);
    if (num < k) continue;
    auto x_end = x_start + num;
    int snn_density = 0;
#pragma omp parallel for reduction(+ : snn_density)
    for (int i = 0; i < k; ++i) {
      auto near_neighbor = knns[x * k + i];
      int counter = 0;
      auto y_start = graph.CsrT::GetNeighborListOffset(near_neighbor);
      auto y_num = graph.CsrT::GetNeighborListLength(near_neighbor);
      for (int z = y_start; z < y_start + y_num; ++z) {
        auto y = graph.CsrT::GetEdgeDest(z);
        if (adj[x].find(y) != adj[x].end()) {
          ++counter;
        }
      }
      if (counter >= eps) snn_density += 1;
      debug("density of %d is %d\n", x, snn_density);
    }
    if (snn_density >= min_pts) {
      core_points.insert(x);
    }
  }

#if SNN_DEBUG
  debug("core points: ");
  for (auto cpb = core_points.begin(); cpb != core_points.end(); ++cpb) {
    debug("%d ", *cpb);
  }
  debug("\n");
#endif
  *core_point_counter = core_points.size();

  for (SizeT x = 0; x < nodes; ++x) {
    if (core_points.find(x) != core_points.end()) {
      for (SizeT y = 0; y < nodes; ++y) {
        if (x != y && core_points.find(y) != core_points.end()) {
          int counter = 0;
          auto y_start = graph.CsrT::GetNeighborListOffset(y);
          auto y_num = graph.CsrT::GetNeighborListLength(y);
          //#pragma omp parallel for reduction(+:counter)
          for (int z = y_start; z < y_start + y_num; ++z) {
            auto m = graph.CsrT::GetEdgeDest(z);
            if (adj[x].find(m) != adj[x].end()) {
              counter += 1;
            }
          }
          if (counter >= eps) {
            // Merge x and y core points
            auto m = min(cluster[x], cluster[y]);
            cluster[x] = m;
            cluster[y] = m;
          }
        }
      }
    }
  }

#pragma omp parallel for
  for (int i = 0; i < nodes; ++i) {
    // only non-core points
    if (core_points.find(i) == core_points.end()) {
      auto num_neighbors = graph.CsrT::GetNeighborListLength(i);
      // only non-noise points
      if (num_neighbors >= k) {  // was k
        auto e_start = graph.CsrT::GetNeighborListOffset(i);
        for (auto e = e_start; e < e_start + num_neighbors; ++e) {
          auto m = graph.CsrT::GetEdgeDest(distance[e].e_id);
          if (core_points.find(m) != core_points.end()) {
            cluster[i] = cluster[m];
            break;
          }
        }
      }
    }
  }

#if SNN_DEBUG
  for (int i = 0; i < nodes; ++i) printf("cluster[%d] = %d\n", i, cluster[i]);
#endif

  std::set<SizeT> cluster_set; 
#if SNN_DEBUG
  printf("cpu clusters: ");
#endif
  for (int i = 0; i < nodes; ++i){
      
      if (cluster_set.find(cluster[i]) == cluster_set.end()){
          cluster_set.insert(cluster[i]);
#if SNN_DEBUG
          printf("%d ", cluster[i]);
#endif
      }
  }
#if SNN_DEBUG
  printf("\n");
#endif
  *cluster_counter = cluster_set.size();

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  delete[] distance;
  return elapsed;
}

/**
 * @brief Validation of snn results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename SizeT = typename GraphT::SizeT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph,
                                        SizeT *h_cluster,
                                        SizeT *h_core_point_counter,
                                        SizeT *h_cluster_counter,
                                        SizeT *ref_cluster,
                                        SizeT *ref_core_point_counter,
                                        SizeT *ref_cluster_counter,
                                        //SizeT *h_knns,
                                        //SizeT *ref_knns,
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");
  bool snn = parameters.Get<bool>("snn");
  SizeT k = parameters.Get<int>("k");

  if (quick) return num_errors;

  printf("Validate results start, num_errors so far %d\n", num_errors);

  if (*ref_core_point_counter != *h_core_point_counter){
      ++num_errors;
      printf("cpu core point counter %d, gpu core point counter %d\n",
              *ref_core_point_counter, *h_core_point_counter);
  }

  if (*ref_cluster_counter != *h_cluster_counter){
      ++num_errors;
      printf("cpu cluster counter %d, gpu cluster counter %d\n",
              *ref_cluster_counter, *h_cluster_counter);
  }
/*
  for (SizeT v = 0; v < graph.nodes; ++v) {
    auto v_start = graph.CsrT::GetNeighborListOffset(v);
    auto num = graph.CsrT::GetNeighborListLength(v);
    if (num < k) continue;
    auto v_end = v_start + num;
    int i = 0;
    for (SizeT neighbor = v_start; neighbor < v_end && i < k; ++neighbor, ++i) {
      if (h_knns[v * k + i] != ref_knns[v * k + i]) {
         // if (v*k + i < 100)
          //    printf("knns[%d] %d != %d\n", v*k+i, h_knns[v*k+i], ref_knns[v*k+i]);
        ++num_errors;
      }
    }
  }
  printf("Validate results stop, num_errors so far %d\n", num_errors);

  if (num_errors > 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred in KNN.",
                   !quiet);
  } else {
    util::PrintMsg("PASSED KNN", !quiet);
  }

  SizeT knn_errors = num_errors;
  */
  num_errors = 0;

  for (SizeT i = 0; i < graph.nodes; ++i) {
    if (snn && (h_cluster[i] != ref_cluster[i])) {
      printf("[%d] %d != %d\n", i, h_cluster[i], ref_cluster[i]);
      ++num_errors;
    }
  }

  if (num_errors > 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred in SNN.",
                   !quiet);
  } else {
    util::PrintMsg("PASSED SNN", !quiet);
  }

  return num_errors;
      //+ knn_errors;
}

}  // namespace snn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
