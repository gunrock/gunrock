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

//#define debug(a...) printf(a)

namespace gunrock {
namespace app {
namespace knn {

/**
 * @brief Speciflying parameters for KNN Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_test(util::Parameters &parameters) {
    cudaError_t retval = cudaSuccess;

    GUARD_CU(parameters.Use<uint32_t>(
        "omp-threads",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
        "Number of threads for parallel omp knn implementation; 0 for "
        "default.",
        __FILE__, __LINE__));
    return retval;
}

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
double CPU_Reference(util::Parameters &parameters,
        util::Array1D<SizeT, ValueT> &points,    // points
        SizeT n,           // number of points
        SizeT dim,         // number of dimension
        SizeT k,           // number of nearest neighbor
        SizeT *knns,       // knns
        bool quiet) {
    cudaError_t retval = cudaSuccess;

    struct comp {
        inline bool operator()(const ValueT &dist1, const ValueT &dist2) {
            return (dist1 < dist2);
        }
    };

    int num_threads = parameters.Get<int>("omp-threads");
    #pragma omp parallel
    {
        if (num_threads == 0) num_threads = omp_get_num_threads();
    }

    util::PrintMsg("#threads = " + std::to_string(num_threads), !quiet);

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    util::Array1D<SizeT, ValueT> distance;
    util::Array1D<SizeT, SizeT>  keys;
    
    GUARD_CU(distance   .Allocate(n*n, util::DEVICE));
    GUARD_CU(keys       .Allocate(n*n, util::DEVICE));

    GUARD_CU(points.Move(util::HOST, util::DEVICE, n*dim));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    GUARD_CU(distance.ForAll(
        [n, dim, points, keys, k] 
        __host__ __device__ (ValueT* d, const SizeT &src){
            auto pos = src / n;
            auto i = src % n;
            // auto dist = euclidean_distance(dim, points, pos, i);
            ValueT dist = 0;
            if (pos == i) {
                dist = util::PreDefinedValues<ValueT>::MaxValue;
            } else {
                dist = euclidean_distance(dim, points, pos, i);
            }
            d[src] = dist;
            keys[src] = i;
        },
        n*n, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    util::Array1D<uint64_t, char> temp_storage;
    util::Array1D<SizeT, ValueT> distance_out;
    util::Array1D<SizeT, SizeT>  keys_out;
    util::Array1D<SizeT, SizeT>  offsets;
    
    GUARD_CU(distance_out   .Allocate(n*n, util::DEVICE));
    GUARD_CU(keys_out       .Allocate(n*n, util::DEVICE));
    GUARD_CU(offsets        .Allocate(n+1, util::DEVICE));

    GUARD_CU(offsets.ForAll(
        [n] __host__ __device__ (SizeT *ro, const SizeT &pos){
            ro[pos] = pos*(n);
        }, n+1, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    // Sort all the distance using CUB
    GUARD_CU(util::cubSegmentedSortPairs(temp_storage, 
        distance, distance_out, keys, keys_out, n*n,
        n, offsets));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    keys.Release(util::DEVICE);
    distance.Release(util::DEVICE);
    distance_out.Release(util::DEVICE);

    util::Array1D<SizeT, SizeT>  knns_d;
    GUARD_CU(knns_d   .Allocate(n*k, util::DEVICE));

    // Choose k nearest neighbors for each node
    GUARD_CU(knns_d.ForAll(
        [n, k, keys_out] 
        __host__ __device__(SizeT* knns_, const SizeT &src){
            auto pos = src / k;
            auto i = src % k;
            knns_[pos * k + i] = keys_out[pos * n + i];
        },
        n*k, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    keys_out.Release(util::DEVICE);

    knns_d.SetPointer(knns, n*k, util::HOST);
    knns_d.Move(util::DEVICE, util::HOST);

    // std::vector<std::map<SizeT, ValueT, comp>> distance0;
    // distance0.resize(n);

    // // CPU reference implementation
    // #pragma omp parallel for num_threads(num_threads)
    // for (SizeT p1 = 0; p1 < n; ++p1){
    //     for (SizeT p2 = p1+1; p2 < n; ++p2){
    //         ValueT d = euclidean_distance(dim, points, p1, p2);
    //         distance0[p1][p2] = d;
    //         distance0[p2][p1] = d;
    //     }
    // }
    // debug("insertion to distance0 is done\n");
    // #pragma omp barrier
    
    // std::vector<std::multimap<ValueT, SizeT, comp>> distance;
    // distance.resize(n);
    // #pragma omp parallel for num_threads(num_threads)
    // for (SizeT p=0; p<n; ++p){
    //     distance[p] = flip_map(distance0[p]);
    // }
    
    // #pragma omp barrier
    // debug("insertion to distance is done\n");
   
    // #pragma omp parallel for num_threads(num_threads)
    // for (SizeT p = 0; p < n; ++p) {
    //     int i = 0;
    //     for (auto& dd :distance[p]){
    //         knns[p * k + i] = dd.second;
    //         ++i;
    //         if (i == k)
    //             break;
    //     }
    // }
    // debug("insertion to knns is done\n");
    // #pragma omp barrier
    
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

  for (SizeT v = 0; v < num_points; ++v) {
      for (SizeT i = 0; i < k; ++i) {
          SizeT w1 = h_knns[v * k + i];
          SizeT w2 = ref_knns[v * k + i];
          if (w1 != w2) {
              ValueT dist1 = euclidean_distance(dim, points, v, w1);
              ValueT dist2 = euclidean_distance(dim, points, v, w2);
              if (dist1 != dist2){
                  util::PrintMsg(
                    "point::nearest-neighbor = [gpu]" + 
                    std::to_string(v) + "::" + std::to_string(h_knns[v * k + i]) + 
                    " !=  [cpu]" + 
                    std::to_string(v) + "::" + std::to_string(ref_knns[v * k + i]), 
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
