// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_problem.cuh
 *
 * @brief GPU Storage Management Structure for k-Core Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace kcore {

/**
 * @brief Specifying parameters for k-core Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  // <TODO> Add problem specific command-line parameter usages here, e.g.:
  // GUARD_CU(parameters.Use<bool>(
  //    "mark-pred",
  //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
  //    false,
  //    "Whether to mark predecessor info.",
  //    __FILE__, __LINE__));
  // </TODO>

  return retval;
}

/**
 * @brief k-Core Problem structure.
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
    // <TODO> add problem specific storage arrays:
    util::Array1D<SizeT, int> degrees;
    util::Array1D<SizeT, SizeT> k_cores;
    util::Array1D<SizeT, VertexT> initial_frontier;
    util::Array1D<SizeT, SizeT> empty_flag;
    util::Array1D<unsigned int, unsigned int> delete_bitmap;
    util::Array1D<unsigned int, unsigned int> to_be_deleted_bitmap;
    // </TODO>

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      // <TODO> name of the problem specific arrays:
      degrees.SetName("degrees");
      k_cores.SetName("k cores");
      initial_frontier.SetName("array to reset frontier");
      empty_flag.SetName("flag for whether graph is empty");
      delete_bitmap.SetName("bitmap of vertex deleted status");
      to_be_deleted_bitmap.SetName("bitmap of vertices marked for deletion");
      // </TODO>
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

      // <TODO> Release problem specific data, e.g.:
      GUARD_CU(degrees.Release(target));
      GUARD_CU(k_cores.Release(target));
      GUARD_CU(initial_frontier.Release(target));
      GUARD_CU(empty_flag.Release(target));
      GUARD_CU(delete_bitmap.Release(target));
      GUARD_CU(to_be_deleted_bitmap.Release(target));
      // </TODO>

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing k-core-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     util::Location target, ProblemFlag flag) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      // <TODO> allocate problem specific data here, e.g.:
      GUARD_CU(degrees.Allocate(sub_graph.nodes, target));
      GUARD_CU(k_cores.Allocate(sub_graph.nodes, target));
      GUARD_CU(initial_frontier.Allocate(sub_graph.nodes, target));
      GUARD_CU(empty_flag.Allocate(1, target));
      GUARD_CU(delete_bitmap.Allocate(((sub_graph.nodes / 32) + 1), target));
      GUARD_CU(to_be_deleted_bitmap.Allocate(((sub_graph.nodes / 32) + 1), target));
      // </TODO>

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

      if (target & util::DEVICE) {
        // <TODO> move sub-graph used by the problem onto GPU,
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
        // </TODO>
      }

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

      GUARD_CU(initial_frontier.ForAll(
            [sub_graph] __host__ __device__(VertexT * initial_frontier, const SizeT &pos) {
              initial_frontier[pos] = pos;
            },
            sub_graph.nodes, target, this->stream));

      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(GraphT &sub_graph, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      // Ensure data are allocated
      // <TODO> ensure size of problem specific data:
      GUARD_CU(degrees.EnsureSize_(nodes, target));
      GUARD_CU(k_cores.EnsureSize_(nodes, target));
      GUARD_CU(initial_frontier.EnsureSize_(nodes, target));
      GUARD_CU(empty_flag.EnsureSize_(1, target));
      GUARD_CU(delete_bitmap.EnsureSize_(((nodes / 32) + 1), target));
      GUARD_CU(to_be_deleted_bitmap.EnsureSize_(((nodes / 32) + 1), target));
      // </TODO>

      // Reset data
      // <TODO> reset problem specific data, e.g.:

      GUARD_CU(degrees.ForAll(
            [sub_graph] __host__ __device__(int * degrees, const SizeT &pos) {
              degrees[pos] = sub_graph.GetNeighborListLength(pos);
            },
            nodes, target, this->stream));

      GUARD_CU(k_cores.ForEach([] __host__ __device__(SizeT & x) { x = (SizeT)0; },
            nodes, target, this->stream));

      GUARD_CU(empty_flag.ForAll(
          [] __host__ __device__(SizeT * x, const VertexT &pos) { x[pos] = 0; },
          1, target, this->stream));

      GUARD_CU(to_be_deleted_bitmap.ForEach([] __host__ __device__(unsigned int & x) { x = 0u; },
            ((nodes / 32) + 1), target, this->stream));

      // </TODO>

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief k-core default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief k-core default destructor
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
  cudaError_t Extract(
      // <TODO> problem specific data to extract
      SizeT *h_k_cores,
      // </TODO>
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        // <TODO> extract the results from single GPU, e.g.:
        GUARD_CU(data_slice.k_cores.SetPointer(h_k_cores, nodes, util::HOST));
        GUARD_CU(data_slice.k_cores.Move(util::DEVICE, util::HOST));
        // </TODO>
      } else if (target == util::HOST) {
        // <TODO> extract the results from single CPU, e.g.:
        GUARD_CU(data_slice.k_cores.ForEach(
            h_k_cores,
            [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        // </TODO>
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
   * @param     graph       The graph to compute k-cores for
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    // <TODO> get problem specific flags from parameters, e.g.:
    // if (this -> parameters.template Get<bool>("mark-pred"))
    //    this -> flag = this -> flag | Mark_Predecessors;
    // </TODO>

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(
      // <TODO> problem specific data if necessary, eg
      // VertexT src,
      GraphT &graph,
      // </TODO>
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(graph, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // <TODO> Additional problem specific initialization
    // </TODO>

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
