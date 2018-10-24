// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_problem.cuh
 *
 * @brief GPU Storage management Structure for Geo Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace geo {

/**
 * @brief Speciflying parameters for Geo Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Template Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef typename util::Array1D<SizeT, ValueT> ArrayT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // Device arrays to store latitudes and longitudes
    util::Array1D<SizeT, ValueT> latitude;
    util::Array1D<SizeT, ValueT> longitude;

    // Use for Stop_Condition for a complete Geo run
    util::Array1D<SizeT, SizeT> active;
    SizeT active_;

    // Store inverse of Haversine Distances
    util::Array1D<SizeT, ValueT> Dinv;

    // Run as many iterations as possible to do a
    // complete geolocation -> uses atomics()
    bool geo_complete;

    // Number of iterations for geolocation app
    int geo_iter;

    // Number of iterations for a spatial median kernel
    int spatial_iter;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      latitude.SetName("latitude");
      longitude.SetName("longitude");
      active.SetName("active");
      Dinv.SetName("Dinv");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(latitude.Release(target));
      GUARD_CU(longitude.Release(target));
      GUARD_CU(active.Release(target));
      GUARD_CU(Dinv.Release(target));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     ProblemFlag flag, bool geo_complete_,
                     util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      SizeT nodes = this->sub_graph->nodes;
      SizeT edges = this->sub_graph->edges + 1;

      printf("Number of nodes for allocation: %u\n", nodes);

      geo_complete = geo_complete_;

      GUARD_CU(latitude.Allocate(nodes, target));
      GUARD_CU(longitude.Allocate(nodes, target));
      GUARD_CU(active.Allocate(1, util::HOST | target));
      GUARD_CU(Dinv.Allocate(edges, target));

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }
      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(ValueT *h_latitude, ValueT *h_longitude, int _geo_iter,
                      int _spatial_iter, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;
      SizeT edges = this->sub_graph->edges + 1;

      // Ensure data are allocated
      GUARD_CU(latitude.EnsureSize_(nodes, target));
      GUARD_CU(longitude.EnsureSize_(nodes, target));
      GUARD_CU(active.EnsureSize_(1, util::HOST | target));
      GUARD_CU(Dinv.EnsureSize_(edges, target));

      this->geo_iter = _geo_iter;
      this->spatial_iter = _spatial_iter;

      // Reset data

      // Using spatial center we can determine the invalid predicted locations.
      GUARD_CU(active.ForAll(
          [] __host__ __device__(SizeT * x, const VertexT &pos) { x[pos] = 0; },
          1, target, this->stream));

      this->active_ = 0;

      // Assumes that all vertices have invalid positions, in reality
      // a preprocessing step is needed to assign nodes that do have
      // positions to have proper positions already.
      GUARD_CU(latitude.SetPointer(h_latitude, nodes, util::HOST));
      GUARD_CU(latitude.Move(util::HOST, util::DEVICE));

      GUARD_CU(longitude.SetPointer(h_longitude, nodes, util::HOST));
      GUARD_CU(longitude.Move(util::HOST, util::DEVICE));

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  bool geo_complete;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief geolocation default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    geo_complete = _parameters.Get<bool>("geo-complete");
  }

  /**
   * @brief geolocation default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_predicted_lat, ValueT *h_predicted_lon,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(
            data_slice.latitude.SetPointer(h_predicted_lat, nodes, util::HOST));
        GUARD_CU(data_slice.latitude.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.longitude.SetPointer(h_predicted_lon, nodes,
                                                 util::HOST));
        GUARD_CU(data_slice.longitude.Move(util::DEVICE, util::HOST));

      } else if (target == util::HOST) {
        GUARD_CU(data_slice.latitude.ForEach(
            h_predicted_lat,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.longitude.ForEach(
            h_predicted_lon,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));
      }
    } else {  // num_gpus != 1

      // ============ INCOMPLETE TEMPLATE - MULTIGPU ============

      // // TODO: extract the results from multiple GPUs, e.g.:
      // // util::Array1D<SizeT, ValueT *> th_distances;
      // // th_distances.SetName("bfs::Problem::Extract::th_distances");
      // // GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));

      // for (int gpu = 0; gpu < this->num_gpus; gpu++)
      // {
      //     auto &data_slice = data_slices[gpu][0];
      //     if (target == util::DEVICE)
      //     {
      //         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      //         // GUARD_CU(data_slice.distances.Move(util::DEVICE,
      //         util::HOST));
      //     }
      //     // th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
      // } //end for(gpu)

      // for (VertexT v = 0; v < nodes; v++)
      // {
      //     int gpu = this -> org_graph -> GpT::partition_table[v];
      //     VertexT v_ = v;
      //     if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
      //         v_ = this -> org_graph -> GpT::convertion_table[v];

      //     // h_distances[v] = th_distances[gpu][v_];
      // }

      // // GUARD_CU(th_distances.Release());
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], this->flag,
                               this->geo_complete, target));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(ValueT *h_latitude, ValueT *h_longitude, int _geo_iter,
                    int _spatial_iter, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(h_latitude, h_longitude, _geo_iter,
                                       _spatial_iter, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace geo
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
