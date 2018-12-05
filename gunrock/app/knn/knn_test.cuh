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

//#define KNN_DEBUG 1

#ifdef KNN_DEBUG
#define debug(a...) fprintf(stderr, a)
#else
#define debug(a...)
#endif

#include <set>
#include <vector>

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
    typename GraphT::SizeT eps,        // mininum number of shared neighbors to increase snn-density
    typename GraphT::SizeT min_pts,    // mininum snn-density to be core point
    typename GraphT::VertexT point_x,  // index of reference point
    typename GraphT::VertexT point_y,  // index of reference point
    typename GraphT::SizeT* cluster,    // cluster id
    bool quiet) {

    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::CsrT CsrT;

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

    auto nodes = graph.nodes;
    auto edges = graph.edges;

//#pragma omp parallel for
    for (auto x = 0; x < nodes; ++x)
        cluster[x] = x;

    std::set<SizeT> core_points;
    std::vector<std::set<SizeT>> adj; adj.resize(nodes);

//#pragma omp parallel for
    for (SizeT x = 0; x < nodes; ++x){
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start+num;
        for (auto y = x_start; y < x_end; ++y){
            auto neighbor = graph.CsrT::GetEdgeDest(y);
            adj[x].insert(neighbor);
        }
    }
    
    Point *distance = (Point *)malloc(sizeof(Point) * edges);
    Point **knns     = (Point **)malloc(sizeof(Point*) * nodes * k);
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    // implement CPU reference implementation
//#pragma omp parallel for
    for (SizeT x = 0; x < graph.nodes; ++x) {
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start+num;
        for (SizeT i = x_start; i < x_end; ++i) {
            VertexT y = graph.column_indices[i];
            VertexT dist = (x - point_x) * (x - point_x) + (y - point_y) * (y - point_y);
//            debug("distance[%d](%d, %d) from (%d, %d) is %d\n", i, x, y, point_x, point_y, dist);
            distance[i] = Point(x, i, dist);
        }
    }

    // Sort distances for each adjacency list
//#pragma omp parallel for
    for (SizeT x = 0; x < graph.nodes; ++x){
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start+num;
        std::sort(distance + x_start, distance + x_end, comp());
    }

    // Debug
    for (SizeT x = 0; x < graph.nodes; ++x){
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start+num;
        debug("sorted neighbors of %d\n", x);
        for (int neighbor = x_start; neighbor < x_end; ++neighbor) {
            debug("(%d, %d)(%d) ", distance[neighbor].x, graph.column_indices[distance[neighbor].e_id], distance[neighbor].dist);
        }
        debug("\n");
    }

    // Find k nearest neighbors
//#pragma omp parallel for
    for (SizeT x = 0; x < graph.nodes; ++x){
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start + num;
        if (num < k)
            continue;
        debug("%d nearest neighbors\n", k);
        int i = 0;
        for (int neighbor = x_start; neighbor < x_end && i < k; ++neighbor, ++i) {
            // k nearest points to (point_x, pointy)
            knns[x * k + i] = &distance[neighbor];
            debug("(%d, %d) ", distance[neighbor].x, 
                    graph.CsrT::GetEdgeDest(distance[neighbor].e_id)); 
        }
        debug("\n");
    }

    // Find snn-density
//#pragma omp parallel for
    for (SizeT x = 0; x < graph.nodes; ++x){
        auto x_start = graph.CsrT::GetNeighborListOffset(x);
        auto num = graph.CsrT::GetNeighborListLength(x);
        auto x_end = x_start+num;
        if (num < k)
            continue;
        int snn_density = 0;
        for (int i = 0; i < k; ++i) {
            auto near_neighbor = knns[x * k + i];
            int counter = 0;
            auto y_start = graph.CsrT::GetNeighborListOffset(near_neighbor->x);
            auto y_num = graph.CsrT::GetNeighborListLength(near_neighbor->x);
            for (int z = y_start; z < y_start + y_num; ++z){
                auto y = graph.CsrT::GetEdgeDest(z);
                if (adj[x].find(y) != adj[x].end()){
                    ++counter;
                }
            }
            if (counter >= eps)
                ++snn_density;
        }
        if (snn_density >= min_pts){
            core_points.insert(x);
        }
    }

    debug("core points: ");
    for (auto cpb = core_points.begin(); cpb != core_points.end(); ++cpb){
        debug("%d ", *cpb);
    }
    debug("\n");

    for (SizeT x = 0; x < nodes; ++x){
        if (core_points.find(x) != core_points.end()){
            for (SizeT y = 0; y < nodes; ++y){
                if (core_points.find(y) != core_points.end()){
                    int counter = 0;
                    auto y_start = graph.CsrT::GetNeighborListOffset(y);
                    auto y_num = graph.CsrT::GetNeighborListLength(y);
                    for (int z = y_start; z < y_start + y_num; ++z){
                        auto m = graph.CsrT::GetEdgeDest(z);
                        if (adj[x].find(m) != adj[x].end()){
                            ++counter;
                        }
                    }
                    if (counter >= eps){
                        // Merge x and y core points
                        auto m = min(cluster[x], cluster[y]);
                        cluster[x] = m;
                        cluster[y] = m;
                    }
                }
            }
        }
    }
    for (int i=0; i<nodes; ++i)
        debug("cluster[%d] = %d\n", i, cluster[i]);

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    delete[] distance;
    delete[] knns;
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
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::SizeT *h_cluster,
    typename GraphT::SizeT *ref_cluster, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  for (SizeT i = 0; i < graph.nodes; ++i) {
    if (h_cluster[i] != ref_cluster[i]) {
      debug("[%d] %d != %d\n", i, h_cluster[i], ref_cluster[i]);
      ++num_errors;
    }
  }

  if (num_errors > 0) {
        util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
  }else{
      util::PrintMsg("PASS", !quiet);
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
