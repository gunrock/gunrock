// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_test.cu
 *
 * @brief Test related functions for knn
 */

#pragma once
/*
//#define KNN_DEBUG 1

#ifdef KNN_DEBUG
#define debug(a...) fprintf(stderr, a)
#else
#define debug(a...)
#endif
*/
namespace gunrock {
namespace app {
namespace knn {

/******************************************************************************
 * KNN Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference knn ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    typename GraphT::SizeT k,          // number of nearest neighbor
    typename GraphT::VertexT point_x,  // index of reference point
    typename GraphT::VertexT point_y,  // index of reference point
    typename GraphT::SizeT
        *k_nearest_neighbors,  // edge indecies of k nearest neighbors
    bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::VertexT VertexT;

  struct Point {
    VertexT x;
    SizeT e_id;
    VertexT dist;

    Point() {}
    Point(VertexT X, SizeT E_id, VertexT Dist) : x(X), e_id(E_id), dist(Dist) {}
  };

  struct comp {
    inline bool operator()(const Point &p1, const Point &p2) {
      return (p1.dist < p2.dist);
    }
  };

  Point *distance = (Point *)malloc(sizeof(Point) * graph.edges);
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // implement CPU reference implementation
  for (VertexT x = 0; x < graph.nodes; ++x) {
    for (SizeT i = graph.row_offsets[x]; i < graph.row_offsets[x + 1]; ++i) {
      VertexT y = graph.column_indices[i];
      VertexT dist =
          (x - point_x) * (x - point_x) + (y - point_y) * (y - point_y);
      debug("distance[%d](%d, %d) from (%d, %d) is %d\n", i, x, y, point_x,
            point_y, dist);
      distance[i] = Point(x, i, dist);
    }
  }

  std::sort(distance, distance + graph.edges, comp());

  printf("%d nearest neighbors\n", k);
  for (int i = 0; i < k; ++i) {
    // k nearest points to (point_x, pointy)
    k_nearest_neighbors[i] = distance[i].e_id;
    printf("(%d, %d), ", distance[i].x, graph.column_indices[distance[i].e_id]);
  }
  printf("\n");

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  delete[] distance;
  distance = NULL;
  return elapsed;
}

/**
 * @brief Validation of knn results
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
    util::Parameters &parameters, GraphT &graph, typename GraphT::SizeT k,
    typename GraphT::SizeT *h_k_nearest_neighbors,
    typename GraphT::SizeT *ref_k_nearest_neighbors, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  for (SizeT i = 0; i < k; ++i) {
    if (h_k_nearest_neighbors[i] != ref_k_nearest_neighbors[i]) {
      debug("[%d/%d] %d != %d\n", i, k, h_k_nearest_neighbors[i],
            ref_k_nearest_neighbors[i]);
      ++num_errors;
    }
  }

  if (num_errors == 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
  }

  return num_errors;
}

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
