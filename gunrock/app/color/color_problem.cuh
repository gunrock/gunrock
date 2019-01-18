// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * color_problem.cuh
 *
 * @brief GPU Storage management Structure for color Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

#include <curand.h>
#include <curand_kernel.h>

namespace gunrock {
namespace app {
namespace color {

/**
 * @brief Speciflying parameters for color Problem
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
    util::Array1D<SizeT, VertexT> colors;
    util::Array1D<SizeT, ValueT> color_temp;
    util::Array1D<SizeT, ValueT> color_temp2;
    util::Array1D<SizeT, ValueT> color_predicate;
    util::Array1D<SizeT, float> rand;
    util::Array1D<SizeT, VertexT> prohibit;
    util::Array1D<SizeT, bool> visited;

    curandGenerator_t gen;
    bool color_balance;
    bool use_jpl;
    bool test_run;
    int no_conflict;
    int user_iter;
    bool min_color;
    int prohibit_size;

    util::Array1D<SizeT, SizeT> colored;
    SizeT colored_;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      prohibit.SetName("prohibit");
      visited.SetName("visited");
      colors.SetName("colors");
      color_temp.SetName("color_temp");
      color_temp2.SetName("color_temp2");
      color_predicate.SetName("color_predicate");
      rand.SetName("rand");
      colored.SetName("colored");
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
      if (prohibit_size != 0) {
        GUARD_CU(visited.Release(target));
        GUARD_CU(prohibit.Release(target));
      }
      if (color_balance) {
        GUARD_CU(color_temp.Release(target));
        GUARD_CU(color_temp2.Release(target));
        GUARD_CU(color_predicate.Release(target));
      }
      GUARD_CU(colors.Release(target));
      GUARD_CU(rand.Release(target));
      GUARD_CU(colored.Release(target));

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
                     util::Location target, ProblemFlag flag,
                     bool color_balance_, int seed, int user_iter_,
                     bool min_color_, bool test_run_, bool use_jpl_,
                     int no_conflict_, int prohibit_size_) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      color_balance = color_balance_;
      user_iter = user_iter_;
      min_color = min_color_;
      test_run = test_run_;
      use_jpl = use_jpl_;
      no_conflict = no_conflict_;
      prohibit_size = prohibit_size_;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(gen, seed);

      if (prohibit_size != 0) {
        GUARD_CU(visited.Allocate(sub_graph.nodes, target));
        GUARD_CU(prohibit.Allocate(sub_graph.nodes * prohibit_size, target));
      }
      if (color_balance) {
        printf("DEBUG: allocating for advance \n");
        GUARD_CU(color_temp.Allocate(sub_graph.edges, target));
        GUARD_CU(color_temp2.Allocate(sub_graph.edges, target));
        GUARD_CU(color_predicate.Allocate(sub_graph.nodes, target));
      }
      GUARD_CU(colors.Allocate(sub_graph.nodes, target));
      GUARD_CU(rand.Allocate(sub_graph.nodes, target));
      GUARD_CU(colored.Allocate(1, util::HOST | target));

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
      SizeT edges = this->sub_graph->edges;
      // Ensure data are allocated
      if (prohibit_size != 0) {
        GUARD_CU(visited.EnsureSize_(nodes, target));
        GUARD_CU(prohibit.EnsureSize_(nodes * prohibit_size, target));
      }
      if (color_balance) {
        GUARD_CU(color_temp.EnsureSize_(edges, target));
        GUARD_CU(color_temp2.EnsureSize_(edges, target));
        GUARD_CU(color_predicate.EnsureSize_(nodes, target));
      }
      GUARD_CU(colors.EnsureSize_(nodes, target));
      GUARD_CU(rand.EnsureSize_(nodes, target));
      GUARD_CU(colored.EnsureSize_(1, util::HOST | target));

      // Reset data
      if (prohibit_size != 0) {
        GUARD_CU(visited.ForEach([] __host__ __device__(bool &x) { x = false; },
                                 nodes, target, this->stream));

        GUARD_CU(prohibit.ForEach(
            [] __host__ __device__(VertexT & x) {
              x = util::PreDefinedValues<VertexT>::InvalidValue;
            },
            nodes, target, this->stream));
      }
      if (color_balance) {
        GUARD_CU(color_temp.ForEach(
            [] __host__ __device__(ValueT & x) {
              x = util::PreDefinedValues<ValueT>::InvalidValue;
            },
            edges, target, this->stream));
        GUARD_CU(color_temp2.ForEach(
            [] __host__ __device__(ValueT & x) {
              x = util::PreDefinedValues<ValueT>::InvalidValue;
            },
            edges, target, this->stream));
        GUARD_CU(color_predicate.ForEach(
            [] __host__ __device__(ValueT & x) {
              x = util::PreDefinedValues<ValueT>::InvalidValue;
            },
            nodes, target, this->stream));
      }

      GUARD_CU(colors.ForEach(
          [] __host__ __device__(VertexT & x) {
            x = util::PreDefinedValues<VertexT>::InvalidValue;
          },
          nodes, target, this->stream));

      GUARD_CU(
          rand.ForEach([] __host__ __device__(float &x) { x = (float)0.0f; },
                       nodes, target, this->stream));

      curandGenerateUniform(gen, rand.GetPointer(util::DEVICE), nodes);

      GUARD_CU(colored.ForAll(
          [] __host__ __device__(SizeT * x, const VertexT &pos) { x[pos] = 0; },
          1, util::HOST | target, this->stream));

      this->colored_ = 0;

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  int seed;
  int user_iter;
  bool min_color;
  bool test_run;
  bool use_jpl;
  bool color_balance;
  int no_conflict;
  int prohibit_size;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief color default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    seed = _parameters.Get<int>("seed");
    color_balance = _parameters.Get<bool>("LBCOLOR");
    min_color = _parameters.Get<bool>("min-color");
    user_iter = _parameters.Get<int>("user-iter");
    test_run = _parameters.Get<bool>("test-run");
    use_jpl = _parameters.Get<bool>("JPL");
    no_conflict = _parameters.Get<int>("no-conflict");
    prohibit_size = _parameters.Get<int>("prohibit-size");
  }

  /**
   * @brief color default destructor
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
  cudaError_t Extract(VertexT *h_colors, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.colors.SetPointer(h_colors, nodes, util::HOST));
        GUARD_CU(data_slice.colors.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {
        GUARD_CU(data_slice.colors.ForEach(
            h_colors,
            [] __host__ __device__(const VertexT &device_val,
                                   VertexT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));
      }
    } else {  // num_gpus != 1

      // ============ INCOMPLETE TEMPLATE - MULTIGPU ============

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

    // if (this -> parameters.template Get<bool>("mark-pred"))
    //    this -> flag = this -> flag | Mark_Predecessors;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag,
                               this->color_balance, this->seed, this->user_iter,
                               this->min_color, this->test_run, this->use_jpl,
                               this->no_conflict, this->prohibit_size));
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

}  // namespace color
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
