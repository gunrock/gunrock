// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Speciflying parameters for SSSP Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  GUARD_CU(parameters.Use<bool>(
      "mark-pred",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to mark predecessor info.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing SSSP-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // sssp-specific storage arrays
    util::Array1D<SizeT, ValueT> distances;  // source distance
    util::Array1D<SizeT, LabelT>
        labels;  // labels to mark latest iteration the vertex been visited
    util::Array1D<SizeT, VertexT> preds;       // predecessors of vertices
    util::Array1D<SizeT, VertexT> temp_preds;  // predecessors of vertices

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      distances.SetName("distances");
      labels.SetName("labels");
      preds.SetName("preds");
      temp_preds.SetName("temp_preds");
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

      GUARD_CU(distances.Release(target));
      GUARD_CU(labels.Release(target));
      GUARD_CU(preds.Release(target));
      GUARD_CU(temp_preds.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      GUARD_CU(distances.Allocate(sub_graph.nodes, target));
      GUARD_CU(labels.Allocate(sub_graph.nodes, target));
      if (flag & Mark_Predecessors) {
        GUARD_CU(preds.Allocate(sub_graph.nodes, target));
        GUARD_CU(temp_preds.Allocate(sub_graph.nodes, target));
      }

      /*if (target & util::DEVICE)
      {
          GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
      }*/
      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      // Ensure data are allocated
      GUARD_CU(distances.EnsureSize_(nodes, target));
      GUARD_CU(labels.EnsureSize_(nodes, target));
      if (this->flag & Mark_Predecessors) {
        GUARD_CU(preds.EnsureSize_(nodes, target));
        GUARD_CU(temp_preds.EnsureSize_(nodes, target));
      }

      // Reset data
      GUARD_CU(distances.ForEach(
          [] __host__ __device__(ValueT & distance) {
            distance = util::PreDefinedValues<ValueT>::MaxValue;
          },
          nodes, target, this->stream));

      GUARD_CU(labels.ForEach(
          [] __host__ __device__(LabelT & label) {
            label = util::PreDefinedValues<LabelT>::InvalidValue;
          },
          nodes, target, this->stream));

      if (this->flag & Mark_Predecessors) {
        GUARD_CU(preds.ForAll(
            [] __host__ __device__(VertexT * preds_, const SizeT &pos) {
              preds_[pos] = pos;
            },
            nodes, target, this->stream));

        GUARD_CU(temp_preds.ForAll(
            [] __host__ __device__(VertexT * preds_, const SizeT &pos) {
              preds_[pos] = pos;
            },
            nodes, target, this->stream));
      }

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief SSSPProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief SSSPProblem default destructor
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
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
   * @param[out] h_distances Host array to store computed vertex distances from
   * the source.
   * @param[out] h_preds     Host array to store computed vertex predecessors.
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_distances, VertexT *h_preds = NULL,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(
            data_slice.distances.SetPointer(h_distances, nodes, util::HOST));
        GUARD_CU(data_slice.distances.Move(util::DEVICE, util::HOST));

        if ((this->flag & Mark_Predecessors) == 0) return retval;
        GUARD_CU(data_slice.preds.SetPointer(h_preds, nodes, util::HOST));
        GUARD_CU(data_slice.preds.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {
        GUARD_CU(data_slice.distances.ForEach(
            h_distances,
            [] __host__ __device__(const ValueT &distance, ValueT &h_distance) {
              h_distance = distance;
            },
            nodes, util::HOST));

        if (this->flag & Mark_Predecessors)
          GUARD_CU(data_slice.preds.ForEach(
              h_preds,
              [] __host__ __device__(const VertexT &pred, VertexT &h_pred) {
                h_pred = pred;
              },
              nodes, util::HOST));
      }
    } else {  // num_gpus != 1
      util::Array1D<SizeT, ValueT *> th_distances;
      util::Array1D<SizeT, VertexT *> th_preds;
      th_distances.SetName("bfs::Problem::Extract::th_distances");
      th_preds.SetName("bfs::Problem::Extract::th_preds");
      GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));
      GUARD_CU(th_preds.Allocate(this->num_gpus, util::HOST));

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        auto &data_slice = data_slices[gpu][0];
        if (target == util::DEVICE) {
          GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
          GUARD_CU(data_slice.distances.Move(util::DEVICE, util::HOST));
          if (this->flag & Mark_Predecessors)
            GUARD_CU(data_slice.preds.Move(util::DEVICE, util::HOST));
        }
        th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
        th_preds[gpu] = data_slice.preds.GetPointer(util::HOST);
      }  // end for(gpu)

      for (VertexT v = 0; v < nodes; v++) {
        int gpu = this->org_graph->GpT::partition_table[v];
        VertexT v_ = v;
        if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) == 0)
          v_ = this->org_graph->GpT::convertion_table[v];

        h_distances[v] = th_distances[gpu][v_];
        if (this->flag & Mark_Predecessors) h_preds[v] = th_preds[gpu][v_];
      }

      GUARD_CU(th_distances.Release());
      GUARD_CU(th_preds.Release());
    }  // end if

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

    if (this->parameters.template Get<bool>("mark-pred"))
      this->flag = this->flag | Mark_Predecessors;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // Fillin the initial input_queue for SSSP problem
    int gpu;
    VertexT src_;
    if (this->num_gpus <= 1) {
      gpu = 0;
      src_ = src;
    } else {
      gpu = this->org_graph->partition_table[src];
      if (this->flag & partitioner::Keep_Node_Num)
        src_ = src;
      else
        src_ = this->org_graph->GpT::convertion_table[src];
    }
    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    ValueT src_distance = 0;
    if (target & util::HOST) {
      data_slices[gpu]->distances[src_] = src_distance;
      if (this->flag & Mark_Predecessors)
        data_slices[gpu]->preds[src_] =
            util::PreDefinedValues<VertexT>::InvalidValue;
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaMemcpy(
                    data_slices[gpu]->distances.GetPointer(util::DEVICE) + src_,
                    &src_distance, sizeof(ValueT), cudaMemcpyHostToDevice),
                "SSSPProblem cudaMemcpy distances failed");

      if (this->flag & Mark_Predecessors) {
        VertexT src_pred = util::PreDefinedValues<VertexT>::InvalidValue;
        GUARD_CU2(
            cudaMemcpy(data_slices[gpu]->preds.GetPointer(util::DEVICE) + src_,
                       &src_pred, sizeof(VertexT), cudaMemcpyHostToDevice),
            "SSSPProblem cudaMemcpy preds failed");
      }
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  /** @} */
};

}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
