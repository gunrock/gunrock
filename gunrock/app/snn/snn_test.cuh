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

//#define SNN_DEBUG 1

#ifdef SNN_DEBUG
    #define debug(a...) fprintf(stderr, a)
#else
    #define debug(a...)
#endif

#include <iostream>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cassert>

#include <gunrock/app/knn/knn_test.cuh>
#include <gunrock/app/snn/snn_helpers.cuh>

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
    SizeT num_points,          // number of points
    SizeT k,                   // number of nearest neighbor
    SizeT eps,                 // min number of SNN to increase snn-density
    SizeT min_pts,             // mininum snn-density to be core point
    SizeT *knns,               // k nearest neighbor array
    SizeT *cluster,            // cluster id
    SizeT *core_point_counter, // counter of core points
    SizeT *cluster_counter,    // counter of clusters
    bool quiet) {
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::CsrT CsrT;

  //util::PrintMsg("#threads: " + std::to_string(omp_get_num_threads()));

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

#pragma omp parallel for
  for (auto x = 0; x < num_points; ++x) {
    cluster[x] = util::PreDefinedValues<SizeT>::InvalidValue;
  }

  // For each point make a binary search tree of its k nearest neighbors 
  std::vector<std::set<SizeT>> knns_set; knns_set.resize(num_points);
  for (SizeT x = 0; x < num_points; ++x){
    knns_set[x] = std::set<SizeT>(knns + (x*k), knns + ((x+1)*k));
  }

#ifdef SNN_DEBUG
  for (auto x = 0; x < num_points; ++x){
      debug("knns[%d]: ", x);
      for (auto y : knns_set[x]){
          debug("%d ", y);
      }
      debug("\n");
  }
#endif

  // Table of sets of shared nn for each point
  std::vector<std::set<SizeT>> snns; snns.resize(num_points);

  // Set of core points
  std::set<SizeT> core_points;
  // Intersection of two knns sets             
  std::vector<SizeT> common_knns; common_knns.resize(k);

  debug("Looking for snns\n");
  for (SizeT x = 0; x < num_points; ++x){
      int snn_density = 0;
      //for each q in kNN(x)
      debug("Snn of %d\n", x);
      for (SizeT i = 0; i < k; ++i){
          SizeT q = knns[x * k + i];
          debug("%d - knn[%d]\t", q, x);
          //if x is in kNN(q)
          /*if (snns[q].find(x) != snns[q].end()){
              debug("\n");
              continue;
          }*/
          if (knns_set[q].find(x) != knns_set[q].end()){
              debug("%d - knn[%d]\t", x, q);
              // checking size of set the common 
              auto it = std::set_intersection(knns_set[x].begin(), 
                      knns_set[x].end(), knns_set[q].begin(), 
                      knns_set[q].end(), common_knns.begin());
              common_knns.resize(it-common_knns.begin());
              debug("they shared %d neighbors\t", common_knns.size());
              if (common_knns.size() >= eps){
                  snns[x].insert(q);
                  snns[q].insert(x);
                  debug("%d %d - snn\n", x, q);
              }else{
                  debug("\n");
              }
          }else{
              debug("\n");
          }
      }
  }

  // Find core points:
  for (SizeT x = 0; x < num_points; ++x){
      if (snns[x].size() > 0){
          debug("density[%d] = %d\t", x, snns[x].size());
          debug("snns: ");
          for (auto ss :snns[x]){
            debug("%d ", ss);
          }
          debug("\n");
      }
      if (snns[x].size() >= min_pts){
          core_points.insert(x);
          debug("%d is core_point\n", x);
      }else
          debug("%d is not core_point\n", x);
  }
  
  // Set core points counter
  *core_point_counter = core_points.size();

#if SNN_DEBUG
  debug("core points (%d): ", core_points.size());
  for (auto cpb = core_points.begin(); cpb != core_points.end(); ++cpb) {
    debug("%d ", *cpb);
  }
  debug("\n");
#endif

  // Create empty clusters:
  DisjointSet<SizeT> clusters(num_points);

  // Find clusters. Union core points
  for (auto c1 : core_points){
    for (auto c2 : core_points){
      if (snns[c1].find(c2) != snns[c1].end()){
        clusters.Union(c1, c2);
        cluster[c1] = cluster[c2] = clusters.Find(c1);
      }
    }
  }

#if SNN_DEBUG
  debug("clusters after union core points:\n");
  for (int i = 0; i < num_points; ++i) 
    debug("cluster[%d] = %d\n", i, cluster[i]);
#endif

  debug("assign non core points\n");
  // Assign non core points
  for (SizeT x = 0; x < num_points; ++x){
    if (core_points.find(x) == core_points.end()){
      // x is non core point
      debug("%d - non core point\n", x);
      SizeT nearest_core_point = util::PreDefinedValues<SizeT>::InvalidValue;
      SizeT similarity_to_nearest_core_point = 0;
      for (auto q : knns_set[x]){
        debug("%d, knn of %d\n", q);
        if (core_points.find(q) != core_points.end()){
          debug("\t%d is core point\n", q);
          // q is core point
          auto it = std::set_intersection(knns_set[x].begin(), knns_set[x].end(),
                  knns_set[q].begin(), knns_set[q].end(), common_knns.begin());
          common_knns.resize(it-common_knns.begin());
          if (!util::isValid(nearest_core_point) || 
                  common_knns.size() > similarity_to_nearest_core_point){
            similarity_to_nearest_core_point = common_knns.size();
            nearest_core_point = q;
          }
        }
      }
      if (util::isValid(nearest_core_point) && 
              similarity_to_nearest_core_point >= eps){
        // x is not a noise point
        clusters.Union(x, nearest_core_point);
        cluster[x] = cluster[nearest_core_point] = clusters.Find(nearest_core_point);
      }
    }
  }

#if SNN_DEBUG
  debug("clusters after assigne non core points\n");
  for (int i = 0; i < num_points; ++i) 
    debug("cluster[%d] = %d\n", i, cluster[i]);
#endif

  std::unordered_set<SizeT> cluster_set; 
  for (SizeT x = 0; x < num_points; ++x){
    if (util::isValid(cluster[x])){
      cluster_set.insert(clusters.Find(x)); // have to be clusters.Find(x) because array stores not updated cluster numbers
    }
  }


#if SNN_DEBUG
  debug("cpu clusters: ");
  for (SizeT x = 0; x < num_points; ++x){
    if (!util::isValid(cluster[x])){
      debug("%x does not have cluster, it is noise point", x);
    }else{
      debug("cluster of %d is %d\n", x, clusters.Find(x));
    }
  }
#endif

 
  // Set cluster counter
  *cluster_counter = cluster_set.size();

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
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
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");
  SizeT num_points = parameters.Get<SizeT>("n");
  SizeT k = parameters.Get<int>("k");
  SizeT eps = parameters.Get<int>("eps");
  SizeT min_pts = parameters.Get<int>("min-pts");

  if (quick) return num_errors;

  printf("Validate results start, num_errors so far %d\n", num_errors);

  printf("number of points: %d\n", num_points);
  printf("cpu core point counter %d, gpu core point counter %d\n",
              *ref_core_point_counter, *h_core_point_counter);

  if (*ref_core_point_counter != *h_core_point_counter){
      ++num_errors;
      printf("error\n");
  }

  printf("cpu cluster counter %d, gpu cluster counter %d\n",
              *ref_cluster_counter, *h_cluster_counter);
  
  if (*ref_cluster_counter != *h_cluster_counter){
      ++num_errors;
      printf("error\n");
  }

  std::vector<bool> unvisited_cluster_of; unvisited_cluster_of.resize(num_points, true);
  for (SizeT i = 0; i < num_points; ++i) {
    if (unvisited_cluster_of[i]){
      unvisited_cluster_of[i] = false;
      auto h_cluster_of_i = h_cluster[i];
      auto ref_cluster_of_i = ref_cluster[i];
      for (SizeT j = 0; j < num_points; ++j){
        if (not unvisited_cluster_of[j]) continue;
        if (ref_cluster[j] == ref_cluster_of_i){
          if (h_cluster[j] != h_cluster_of_i){
            printf("error: gpu %d and %d supposed to be in one cluster but are: %d != %d\n", i, j, h_cluster[i], h_cluster[j]);
            ++num_errors;
          }
          unvisited_cluster_of[j] = false;
        }
      }
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
