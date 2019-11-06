// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_problem.cuh
 *
 * @brief GPU Storage management Structure for RW Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace rw {

/**
 * @brief Speciflying parameters for RW Problem
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

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // problem specific storage arrays:
    util::Array1D<SizeT, VertexT> walks;
    util::Array1D<SizeT, float> rand;
    util::Array1D<SizeT, uint64_t> neighbors_seen;
    util::Array1D<SizeT, uint64_t> steps_taken;
    int walk_length;
    int walks_per_node;
    int walk_mode;
    bool store_walks;
    curandGenerator_t gen;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      walks.SetName("walks");
      rand.SetName("rand");
      neighbors_seen.SetName("neighbors_seen");
      steps_taken.SetName("steps_taken");
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

      GUARD_CU(walks.Release(target));
      GUARD_CU(rand.Release(target));
      GUARD_CU(neighbors_seen.Release(target));
      GUARD_CU(steps_taken.Release(target));

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
                     util::Location target, ProblemFlag flag, int walk_length_,
                     int walks_per_node_, int walk_mode_, bool store_walks_,
                     int seed) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      walk_length = walk_length_;
      walks_per_node = walks_per_node_;
      walk_mode = walk_mode_;
      store_walks = store_walks_;

      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      if (store_walks_) {
        GUARD_CU(walks.Allocate(sub_graph.nodes * walk_length * walks_per_node,
                                target));
      } else {
        GUARD_CU(walks.Allocate(1, target));  // Dummy allocation
      }
      GUARD_CU(rand.Allocate(sub_graph.nodes * walks_per_node, target));
      GUARD_CU(
          neighbors_seen.Allocate(sub_graph.nodes * walks_per_node, target));
      GUARD_CU(steps_taken.Allocate(sub_graph.nodes * walks_per_node, target));

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
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;
      int walks_per_node = this->walks_per_node;
      int walk_length = this->walk_length;

      // Ensure data are allocated
      if (this->store_walks) {
        GUARD_CU(
            walks.EnsureSize_(nodes * walk_length * walks_per_node, target));
      } else {
        GUARD_CU(walks.EnsureSize_(1, target));
      }
      GUARD_CU(rand.EnsureSize_(nodes * walks_per_node, target));
      GUARD_CU(neighbors_seen.EnsureSize_(nodes * walks_per_node, target));
      GUARD_CU(steps_taken.EnsureSize_(nodes * walks_per_node, target));

      // Reset data
      if (this->store_walks) {
        GUARD_CU(walks.ForEach(
            [] __host__ __device__(VertexT & x) {
              x = util::PreDefinedValues<VertexT>::InvalidValue;
            },
            nodes * walk_length * walks_per_node, target, this->stream));
      } else {
        GUARD_CU(walks.ForEach(
            [] __host__ __device__(VertexT & x) {
              x = util::PreDefinedValues<VertexT>::InvalidValue;
            },
            1, target, this->stream));
      }

      GUARD_CU(
          rand.ForEach([] __host__ __device__(float &x) { x = (float)0.0; },
                       nodes * walks_per_node, target, this->stream));

      GUARD_CU(neighbors_seen.ForEach(
          [] __host__ __device__(uint64_t & x) { x = (uint64_t)0; },
          nodes * walks_per_node, target, this->stream));
      GUARD_CU(steps_taken.ForEach(
          [] __host__ __device__(uint64_t & x) { x = (uint64_t)0; },
          nodes * walks_per_node, target, this->stream));

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  int walk_length;
  int walks_per_node;
  int walk_mode;
  bool store_walks;
  int seed;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief RW default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    walk_length = _parameters.Get<int>("walk-length");
    walks_per_node = _parameters.Get<int>("walks-per-node");
    walk_mode = _parameters.Get<int>("walk-mode");
    store_walks = _parameters.Get<bool>("store-walks");
    seed = _parameters.Get<int>("seed");
  }

  /**
   * @brief RW default destructor
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
  cudaError_t Extract(VertexT *h_walks, uint64_t *h_neighbors_seen,
                      uint64_t *h_steps_taken,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      int walk_length = this->walk_length;
      int walks_per_node = this->walks_per_node;

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        if (this->store_walks) {
          GUARD_CU(data_slice.walks.SetPointer(
              h_walks, nodes * walk_length * walks_per_node, util::HOST));
          GUARD_CU(data_slice.walks.Move(util::DEVICE, util::HOST));
        }
        GUARD_CU(data_slice.neighbors_seen.SetPointer(
            h_neighbors_seen, nodes * walks_per_node, util::HOST));
        GUARD_CU(data_slice.neighbors_seen.Move(util::DEVICE, util::HOST));
        GUARD_CU(data_slice.steps_taken.SetPointer(
            h_steps_taken, nodes * walks_per_node, util::HOST));
        GUARD_CU(data_slice.steps_taken.Move(util::DEVICE, util::HOST));

      } else if (target == util::HOST) {
        if (this->store_walks) {
          GUARD_CU(data_slice.walks.ForEach(
              h_walks,
              [] __host__ __device__(const VertexT &device_val,
                                     VertexT &host_val) {
                host_val = device_val;
              },
              nodes * walk_length * walks_per_node, util::HOST));
        }

        GUARD_CU(data_slice.neighbors_seen.ForEach(
            h_neighbors_seen,
            [] __host__ __device__(const uint64_t &device_val,
                                   uint64_t &host_val) {
              host_val = device_val;
            },
            nodes * walks_per_node, util::HOST));

        GUARD_CU(data_slice.steps_taken.ForEach(
            h_steps_taken,
            [] __host__ __device__(const uint64_t &device_val,
                                   uint64_t &host_val) {
              host_val = device_val;
            },
            nodes * walks_per_node, util::HOST));
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
                               this->gpu_idx[gpu], target, this->flag,
                               this->walk_length, this->walks_per_node,
                               this->walk_mode, this->store_walks, this->seed));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace rw
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
