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
    bool transpose = parameters.Get<bool>("transpose");

    util::Array1D<SizeT, ValueT, util::UNIFIED> data;
    GUARD_CU(data   .Allocate(n*dim, util::DEVICE));

    GUARD_CU(data.ForAll(
        [points] __host__ __device__ (ValueT *d, const SizeT &pos){
            d[pos] = points[pos];
        }, n*dim, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    int num_devices; // = 3;
    cudaGetDeviceCount(&num_devices);

    /*
     ***************************************
     *  [TODO] Consider boundary conditions*
     ***************************************
     */
    int MAX_DATA = 2;
    int CHUNK = MAX_DATA*num_devices;

    cudaError_t retvals[CHUNK];
    cudaStream_t stream[CHUNK];
    cudaEvent_t  event[CHUNK];
     
    util::Array1D<SizeT, util::Array1D<SizeT, ValueT>> distance;
    util::Array1D<SizeT, util::Array1D<SizeT, SizeT>>  keys;

    GUARD_CU(distance   .Allocate(CHUNK, util::HOST));
    GUARD_CU(keys       .Allocate(CHUNK, util::HOST));

    // CUB Temporary arrays
    util::Array1D<SizeT, util::Array1D<SizeT, ValueT>> cub_distance;
    util::Array1D<SizeT, util::Array1D<SizeT, SizeT>>  cub_keys;

    GUARD_CU(cub_distance   .Allocate(CHUNK, util::HOST));
    GUARD_CU(cub_keys       .Allocate(CHUNK, util::HOST));
    
    util::Array1D<SizeT, util::Array1D<SizeT, SizeT>>  knns_d;
    GUARD_CU(knns_d   .Allocate(n, util::HOST));

    util::PrintMsg("Host Allocation done.");

    for (int p = 0; p < ((n+(CHUNK-1))/CHUNK); p++) {
        for (int dev = 0; dev < num_devices; dev++) {
            GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");
            for(int d = 0; d < MAX_DATA; d++) {
                auto jump = (p*CHUNK);
                auto index =  jump + ((dev*MAX_DATA) + d);
                if (index >= n) break;
                GUARD_CU(knns_d[index]  .Allocate(k, util::HOST | util::DEVICE));
            }
            GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        }
    }

    util::PrintMsg("KNNs Allocation done.");

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
            auto row = (dev*MAX_DATA) + d;
            if (row >= CHUNK) break;
            GUARD_CU2(cudaStreamCreateWithFlags(&stream[row], 
                cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed.");
            GUARD_CU2(cudaEventCreate(&event[row]), 
                "cudaEventCreate failed.");
            GUARD_CU2(cudaEventRecord(event[row], stream[row]), 
                "cudaEventRecord failed.");

            GUARD_CU(distance[row]    .Allocate(n, util::DEVICE));
            GUARD_CU(keys[row]        .Allocate(n, util::DEVICE));

            GUARD_CU(cub_distance[row]    .Allocate(n, util::DEVICE));
            GUARD_CU(cub_keys[row]        .Allocate(n, util::DEVICE));
        }

        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    util::PrintMsg("Distance/Keys and Device management done.");
   
 
    // Find K nearest neighbors for each point in the dataset
    // Can use multi-gpu to speed up the computation
    for (SizeT m = 0; m < ((n+(CHUNK-1))/(CHUNK)); m++) {
        //#pragma omp parallel for
        for (int dev = 0; dev < num_devices; dev++) {
            util::GRError(cudaSetDevice(dev), "cudaSetDevice failed.");
           // #pragma omp parallel for
            for(int x = 0; x < MAX_DATA; x++) {
                auto row = (dev*MAX_DATA) + x;
                auto v = (m*CHUNK) + row;
                if (v < n && row < CHUNK){
                    auto &error = retvals[row];
                    auto &ith_distances = distance[row];
                    auto &ith_keys = keys[row];

                    // Calculate N distances for each point
                    auto distance_op = [n, dim, data, ith_keys, ith_distances, row, v, transpose] 
                        __host__ __device__ (const SizeT &i) {
                            ValueT dist = 0;
                            if (i == v) {
                                dist = util::PreDefinedValues<ValueT>::MaxValue;
                            } else {
                                dist = euclidean_distance(dim, n, data.GetPointer(util::DEVICE), v, i, transpose);
                            }
                            ith_distances[i] = dist;
                            ith_keys[i] = i;
                        };

                    error = oprtr::For(distance_op, n, util::DEVICE, stream[row]);

                    error = util::GRError(cudaStreamSynchronize(stream[row]),
                        "cudaStreamSynchronize failed", __FILE__, __LINE__);

                    util::CUBRadixSort(true, n, 
                            ith_distances.GetPointer(util::DEVICE),
                            ith_keys.GetPointer(util::DEVICE), 
                            cub_distance[row].GetPointer(util::DEVICE),
                            cub_keys[row].GetPointer(util::DEVICE),
                            (void*)NULL, (size_t)0, stream[row]);
                            
                    error = util::GRError(cudaStreamSynchronize(stream[row]),
                        "cudaStreamSynchronize failed", __FILE__, __LINE__);

                    // Choose k nearest neighbors for each node
                    auto &ith_knns = knns_d[v];
                    auto knns_op = [m, k, ith_knns, ith_keys, row, v]
                        __host__ __device__ (const SizeT &i) {     
                            ith_knns[i] = ith_keys[i];
                        };
                        
                    error = oprtr::For(knns_op, k, util::DEVICE, stream[row]);

                    error = util::GRError(cudaStreamSynchronize(stream[row]),
                                "cudaStreamSynchronize failed", __FILE__, __LINE__);
                }
            }
        }
    }

    util::PrintMsg("Main algroithm loop done.");

    for (int dev = 0; dev < num_devices; dev++) {
        util::GRError(cudaSetDevice(dev), "cudaSetDevice failed.");
        retvals[dev] = util::GRError(cudaDeviceSynchronize(),
                            "cudaDeviceSynchronize failed", __FILE__, __LINE__);
    }

    util::PrintMsg("Synchronized all GPUs.");

    for (int p = 0; p < ((n+(CHUNK-1))/(CHUNK)); p++) {
        for (int dev = 0; dev < num_devices; dev++) {
            GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");
            for(int d = 0; d < MAX_DATA; d++) {
                auto jump = (p*CHUNK);
                auto index =  jump + ((dev*MAX_DATA) + d);
                if (index >= n) break;
                GUARD_CU(knns_d[index]  .Move(util::DEVICE, util::HOST));
            }
        }
    }

    for (int dev = 0; dev < num_devices; dev++) {
        util::GRError(cudaSetDevice(dev), "cudaSetDevice failed.");
        retvals[dev] = util::GRError(cudaDeviceSynchronize(),
                            "cudaDeviceSynchronize failed", __FILE__, __LINE__);
    }

    util::PrintMsg("Moved KNNs from DEVICE to HOST.");

    // #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        for (int l = 0; l < k; l++) {
            auto h_knns = knns_d[t].GetPointer(util::HOST);
            knns[(t*k) + l] = h_knns[l];
        }
    }

    util::PrintMsg("Copy KNNs device result to output KNNs array.");

    // Clean-up
    for (int dev = 0; dev < num_devices; dev++) {
        GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");
        for(int x = 0; x < MAX_DATA; x++) {
            auto row = (dev*MAX_DATA) + x;
            if (row >= CHUNK) break;
            GUARD_CU(keys[row].Release(util::DEVICE));
            GUARD_CU(distance[row].Release(util::DEVICE));
            
            GUARD_CU(cub_keys[row].Release(util::DEVICE));
            GUARD_CU(cub_distance[row].Release(util::DEVICE));

            GUARD_CU2(cudaStreamDestroy(stream[row]), "cudaStreamDestroy failed.");
        }

        util::PrintMsg("Clean-up GPU Arrays:: keys, distance done.");
    }

    GUARD_CU(keys.Release(util::HOST));
    GUARD_CU(distance.Release(util::HOST));
    
    GUARD_CU(cub_keys.Release(util::HOST));
    GUARD_CU(cub_distance.Release(util::HOST));

    util::PrintMsg("Clean-up CPU Arrays:: keys, distance done.");
    
    for (int p = 0; p < ((n+(CHUNK-1))/(CHUNK)); p++) {
        for (int dev = 0; dev < num_devices; dev++) {
            GUARD_CU2(cudaSetDevice(dev), "cudaSetDevice failed.");
            for(int d = 0; d < MAX_DATA; d++) {
                auto jump = (p*CHUNK);
                auto index =  jump + ((dev*MAX_DATA) + d);
                if (index >= n) break;
                GUARD_CU(knns_d[index]  .Release(util::DEVICE | util::HOST));
            }
        }
    }

    util::PrintMsg("Clean-up CPU Arrays:: knns_d[*] done.");

    GUARD_CU(knns_d  .Release(util::HOST));

    util::PrintMsg("Clean-up done.");

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
  bool transpose = parameters.Get<bool>("transpose");

  if (quick) return num_errors;

  for (SizeT v = 0; v < num_points; ++v) {
      for (SizeT i = 0; i < k; ++i) {
          SizeT w1 = h_knns[v * k + i];
          SizeT w2 = ref_knns[v * k + i];
          if (w1 != w2) {
              ValueT dist1 = euclidean_distance(dim, num_points, points.GetPointer(util::HOST), v, w1, transpose);
              ValueT dist2 = euclidean_distance(dim, num_points, points.GetPointer(util::HOST), v, w2, transpose);
              if (dist1 != dist2){
                  util::PrintMsg(
                    "point::nearest-neighbor, gpu_knn(" + 
                    std::to_string(v) + ")=" + std::to_string(h_knns[v * k + i]) + 
                    " !=  cpu_knn(" + 
                    std::to_string(v) + ")=" + std::to_string(ref_knns[v * k + i]), 
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
