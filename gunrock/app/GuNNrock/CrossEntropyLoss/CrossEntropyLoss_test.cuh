// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtc_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once
#include <gunrock/app/test_base.cuh>


#include <random>
#include <chrono>

#ifdef BOOST_FOUND
// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#else
#include <queue>
#include <vector>
#include <utility>
#include <cstring>
#endif

namespace gunrock {
namespace app {
namespace CrossEntropyLoss {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/


/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

void CPU_Reference(const int num_nodes, const int num_classes, const double *tmp_logits,
    const int *ground_truth, double *grad, double &total_loss) {
  total_loss = 0;
//  gunrock::util::cpu_mt::PrintCPUArray("tmp_logits: ", tmp_logits, num_nodes * num_classes);
  double *logits = new double[num_nodes * num_classes];
  double *max_logits = new double[num_nodes];
  std::memcpy(logits, tmp_logits, sizeof(double) * num_classes * num_nodes);

  int count = 0;
  for (int i = 0; i < num_nodes * num_classes; i++) {
    grad[i] = 0;
  }
//#pragma omp parallel for schedule(static) reduction(+:total_loss) reduction(+:count)
  for (int i = 0; i < num_nodes; i++) {
    if (ground_truth[i] < 0) continue;
    count++;
    double *logit = &logits[i * num_classes];
    double max_logit = -1e30, sum_exp = 0;
//#ifdef SIMD
//#pragma omp simd reduction(max:max_logit)
//#endif
    for (int j = 0; j < num_classes; j++)
      max_logit = std::max(max_logit, logit[j]);
//#ifdef SIMD
//#pragma omp simd reduction(+:sum_exp)
//#endif
    max_logits[i] = max_logit;
    for (int j = 0; j < num_classes; j++) {
      logit[j] -= max_logit;
      sum_exp += exp(logit[j]);
    }
    total_loss += log(sum_exp) - logit[ground_truth[i]];

//#ifdef SIMD
//#pragma omp simd
//#endif
    for (int j = 0; j < num_classes; j++) {
      double prob = exp(logit[j]) / sum_exp;
      grad[i * num_classes + j] = prob;
    }
    grad[i * num_classes + ground_truth[i]] -= 1.0;
  }
  total_loss /= count;
//  gunrock::util::cpu_mt::PrintCPUArray("logits: ", logits, num_nodes * num_classes);
//  gunrock::util::cpu_mt::PrintCPUArray("ref_max_logits: ", max_logits, num_nodes);
//#ifdef SIMD
//#pragma omp parallel for simd schedule(static)
//#else
//#pragma omp parallel for schedule(static)
//#endif
  for (int i = 0; i < num_classes * num_nodes; i++)
    grad[i] /= count;
}

void rand_truth(int num_clases, int num_nodes, int *arr) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  arr[0] = 0;
  for(int i = 1; i < num_nodes; i++)
    arr[i] = rng() % (num_clases + 1) - 1;
}

template <typename T>
void rand_logits(int len, T *arr) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  T range = 1000;
  for(int i = 0; i < len; i++) {
    arr[i] = (T(rng()) / rng.max() - 0.5) * range * 2;
//    printf("{%d, %lf} ", i, arr[i]);
  }
//  gunrock::util::cpu_mt::PrintCPUArray("arr: ", arr, len);
}


}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
