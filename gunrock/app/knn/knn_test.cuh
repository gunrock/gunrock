// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_test.cuh
 *
 * @brief Test related functions for knn
 */

#pragma once

#include <set>
#include <map>
#include <vector>
#include <algorithm>

namespace gunrock {
namespace app {
namespace knn {

template<typename SizeT, typename ValueT>
std::pair<ValueT, SizeT> flip_pair(const std::pair<SizeT, ValueT> &p){
    return std::pair<ValueT, SizeT>(p.second, p.first);
}

template<typename SizeT, typename ValueT, typename C>
std::multimap<ValueT, SizeT, C> flip_map(const std::map<SizeT, ValueT, C>& map){
    std::multimap<ValueT, SizeT, C> result;
    std::transform(map.begin(), map.end(), std::inserter(result, result.begin()),
            flip_pair<SizeT, ValueT>);
    return result;
}

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
template <
        typename VertexT,
        typename SizeT,
        typename ValueT>
double CPU_Reference(
        util::Array1D<SizeT, ValueT> &points,    // points
        SizeT n,           // number of points
        SizeT dim,         // number of dimension
        SizeT k,           // number of nearest neighbor
        SizeT *knns,       // knns
        bool quiet) {

    struct comp {
        inline bool operator()(const ValueT &dist1, const ValueT &dist2) {
            return (dist1 < dist2);
        }
    };

    std::vector<std::map<SizeT, ValueT, comp>> distance0;
    distance0.resize(n);
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    // CPU reference implementation
    //#pragma omp parallel for
    for (SizeT p1 = 0; p1 < n; ++p1){
        for (SizeT p2 = p1+1; p2 < n; ++p2){
            ValueT d = euclidean_distance(dim, points, p1, p2);
            distance0[p1][p2] = d;
            distance0[p2][p1] = d;
        }
    }
    
    std::vector<std::multimap<ValueT, SizeT, comp>> distance;
    distance.resize(n);
    for (SizeT p=0; p<n; ++p){
        distance[p] = flip_map(distance0[p]);
    }
   
    for (SizeT p = 0; p < n; ++p) {
        int i = 0;
        for (auto& dd :distance[p]){
            knns[p * k + i] = dd.second;
            ++i;
            if (i == k)
                break;
        }
    }

#ifdef KNN_DEBUG
    //debug of knns
    debug("nearest neighbors\n");
    for (SizeT p = 0; p < n; ++p) {
        debug("%d: ", (int)p);
        for (SizeT i = 0; i < k; ++i){
            debug("%.lf ", knns[p * k + i]);;
        }
        debug("\n");
    }
#endif

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of KNN results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  h_knns        KNN computed on GPU
 * @param[in]  ref_knns      KNN computed on CPU
 * @param[in]  points        List of points 
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename SizeT, typename ValueT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                    GraphT &graph,
                    SizeT *h_knns,
                    SizeT *ref_knns,
                    util::Array1D<SizeT, ValueT> points,
                    bool verbose = true) {

  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");
  SizeT k = parameters.Get<SizeT>("k");
  SizeT num_points = parameters.Get<SizeT>("n");
  SizeT dim = parameters.Get<SizeT>("dim");

  if (quick) return num_errors;

#ifdef KNN_DEBUG
  // Debug write out KNNs of GPU and CPU:
  debug("gpu knns and cpu knns:\n");
  for (SizeT v = 0; v < num_points; ++v){
      debug("%2d: ", (int)v);
      for (SizeT i = 0; i < k; ++i){
          debug("%2d ", (int)h_knns[v * k + i]);
      }
      debug("\t\t");
      for (SizeT i = 0; i < k; ++i){
          debug("%2d ", (int)ref_knns[v * k + i]);
      }
      debug("\n");
  }
#endif

  for (SizeT v = 0; v < num_points; ++v){
      bool notyet = true;
      for (SizeT i = 0; i < k; ++i){
          SizeT w1 =   h_knns[v * k + i];
          SizeT w2 = ref_knns[v * k + i];
          if (w1 != w2){
              ValueT dist1 = euclidean_distance(dim, points, v, w1);
              ValueT dist2 = euclidean_distance(dim, points, v, w2);
              if (dist1 != dist2){
                  if (notyet){
                      util::PrintMsg("[" + std::to_string(v) + "] ", !quiet);
                      notyet = false;
                  }
                  util::PrintMsg(
                          "dist(" + 
                          std::to_string(v) + ", " + std::to_string(w1) + ") = " + 
                          std::to_string(dist1) + " != " + 
                          std::to_string(dist2) + " dist(" + 
                          std::to_string(v) + ", " + std::to_string(w2) + "  ", 
                          !quiet);
                  ++num_errors;
              }
          }
      }
  }

  if (num_errors > 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred in KNN.",
                   !quiet);
  } else {
    util::PrintMsg("PASSED KNN", !quiet);
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
