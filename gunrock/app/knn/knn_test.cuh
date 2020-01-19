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

//#define KNN_TEST_DEBUG

#ifdef KNN_TEST_DEBUG
    #define debug(a...) printf(a)
#else
    #define debug(a...)
#endif

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

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    GUARD_CU(points.Move(util::HOST, util::DEVICE, n*dim));

    util::Array1D<SizeT, ValueT, util::UNIFIED> data;
    GUARD_CU(data   .Allocate(n*dim, util::DEVICE));

    GUARD_CU(data.ForAll(
        [points] __host__ __device__ (ValueT *d, const SizeT &pos){
            d[pos] = points[pos];
        }, n*dim, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    int num_devices;
    int MAX_DATA = 10;
    cudaError_t retvals[40];

    cudaGetDeviceCount(&num_devices);
    cudaStream_t stream[1024];
    cudaEvent_t  event[1024];
     
    util::Array1D<SizeT, util::Array1D<SizeT, ValueT>> distance;
    util::Array1D<SizeT, util::Array1D<SizeT, SizeT>>  keys;

    GUARD_CU(distance   .Allocate(num_devices*MAX_DATA, util::HOST));
    GUARD_CU(keys       .Allocate(num_devices*MAX_DATA, util::HOST));

    
    for(int dev = 0; dev < num_devices; dev++) {
        GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");

        for(int peer = 0; peer < num_devices; peer++) {
            int peer_access_avail = 0;
            GUARD_CU2(cudaDeviceCanAccessPeer(&peer_access_avail, dev, peer),
              "cudaDeviceCanAccessPeer failed");

          if (peer_access_avail) {
            GUARD_CU2(cudaDeviceEnablePeerAccess(peer, 0),
                                   "cudaDeviceEnablePeerAccess failed");
          }
        }

        for(int d = 0; d < MAX_DATA; d++) {
            GUARD_CU2(cudaStreamCreateWithFlags(&stream[(dev*MAX_DATA) + d], 
                cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed.");
            GUARD_CU2(cudaEventCreate(&event[(dev*MAX_DATA) + d]), 
                "cudaEventCreate failed.");
            GUARD_CU2(cudaEventRecord(event[(dev*MAX_DATA) + d], stream[(dev*MAX_DATA) + d]), 
                "cudaEventRecord failed.");

            GUARD_CU(distance[(dev*MAX_DATA) + d]    .Allocate(n, util::DEVICE));
            GUARD_CU(keys[(dev*MAX_DATA) + d]        .Allocate(n, util::DEVICE));
        }
        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }
    
    util::Array1D<SizeT, SizeT, util::UNIFIED>  knns_d;
    cudaSetDevice(0);
    GUARD_CU(knns_d   .Allocate(n*k, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    // Find K nearest neighbors for each point in the dataset
    // Can use multi-gpu to speed up the computation
    for (SizeT m = 0; m < n; m+(num_devices*MAX_DATA)) {
        // #pragma omp parallel for 
        for (int dev = 0; dev < num_devices; dev++) {            
            util::GRError(cudaSetDevice(dev), "cudaSetDevice failed.");
            
            auto &error = retvals[dev];
            
            /*
             ***************************************
             * [TODO] Fix illegal memory access.
             ***************************************
             */

            // Calculate N distances for each point
            auto distance_op = [n, dim, data, keys, k, m, distance, MAX_DATA, dev] 
                __host__ __device__ (const SizeT &i) {
                    auto pos = i % MAX_DATA;        // position in data'th array
                    auto id = i % n;
                    auto row = (dev*MAX_DATA)+pos;
                    auto v = m+row;
                    ValueT dist = 0;
                    if (i == m) {
                        dist = util::PreDefinedValues<ValueT>::MaxValue;
                    } else {
                        dist = euclidean_distance(dim, data.GetPointer(util::DEVICE), v, id);
                    }
                    distance[row][id] = dist;
                    keys[row][id] = id;
                };

            error = oprtr::For(distance_op, n*MAX_DATA, util::DEVICE, stream[dev]);

            #pragma omp parallel for
            for(int x = 0; x < MAX_DATA; x++) {
                error = util::GRError(cudaStreamWaitEvent(stream[(dev*MAX_DATA) + x], event[dev], 0),
                    "cudaStreamWaitEvent failed", __FILE__, __LINE__);

                util::CUBRadixSort(true, n, 
                        distance[(dev*MAX_DATA) + x].GetPointer(util::DEVICE),
                        keys[(dev*MAX_DATA) + x].GetPointer(util::DEVICE), 
                        (ValueT*)NULL, (SizeT*)NULL, 
                        (void*)NULL, (size_t)0, stream[(dev*MAX_DATA) + x]);
                        
                error = util::GRError(cudaStreamWaitEvent(stream[dev], event[(dev*MAX_DATA) + x], 0),
                    "cudaStreamWaitEvent failed", __FILE__, __LINE__);
            }

            // Choose k nearest neighbors for each node
            auto knns_op = [m, k, knns_d, keys, MAX_DATA, dev]
                __host__ __device__ (const SizeT &i){                
                    auto pos = i % MAX_DATA;        // position in data'th array
                    auto id = i % k;
                    auto row = (dev*MAX_DATA)+pos;
                    auto v = m+row;

                    knns_d[(v*k) + id] = keys[row][id];
                };
                
            error = oprtr::For(knns_op, k*MAX_DATA, util::DEVICE, stream[dev]);

            error = util::GRError(cudaStreamSynchronize(stream[dev]),
                        "cudaStreamSynchronize failed", __FILE__, __LINE__); 
        }
    }

    // Move data to CPU
    knns_d.SetPointer(knns, n*k, util::HOST);
    knns_d.Move(util::DEVICE, util::HOST);

    // Clean-up
    for (int dev = 0; dev < num_devices; dev++) {
        GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");
        for(int x = 0; x < MAX_DATA; x++) {
            GUARD_CU(keys[(dev*MAX_DATA) + x].Release(util::DEVICE));
            GUARD_CU(distance[(dev*MAX_DATA) + x].Release(util::DEVICE));
            GUARD_CU2(cudaStreamDestroy(stream[(dev*MAX_DATA) + x]), "cudaStreamDestroy failed.");
        }
    }
    
    GUARD_CU2(cudaSetDevice(0), "cudaSetDevice failed.");
    GUARD_CU(knns_d.Release(util::DEVICE));

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
              ValueT dist1 = euclidean_distance(dim, points.GetPointer(util::HOST), v, w1);
              ValueT dist2 = euclidean_distance(dim, points.GetPointer(util::HOST), v, w2);
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
