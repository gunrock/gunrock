// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * graphsum_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace CrossEntropyLoss {
/**
 * @brief Speciflying parameters for graphsum Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

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

  typedef util::Array1D<SizeT, ValueT> Array;
  // Helper structures

  struct DataSlice : BaseDataSlice {
    util::Array1D<SizeT, ValueT> logits;
    util::Array1D<SizeT, ValueT> grad;
    util::Array1D<SizeT, ValueT> loss;
    util::Array1D<SizeT, int> ground_truth;
    int num_nodes, num_classes;
    bool training;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      logits.SetName("logits");
      grad.SetName("grad");
      ground_truth.SetName("ground_truth");
      loss.SetName("loss");
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

      GUARD_CU(logits.Release(target))
      GUARD_CU(ground_truth.Release(target))

      if (training) {
        GUARD_CU(grad.Release(target))
      }
      GUARD_CU(BaseDataSlice ::Release(target))
      return retval;
    }

    /**
     * @brief initializing graphsum-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, const int num_nodes, const int num_classes,
                     const bool training = 1,
                     int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;

      this->num_nodes = num_nodes;
      this->num_classes = num_classes;
      this->training = training;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag))
      GUARD_CU(logits.Allocate(num_nodes * num_classes, target | util::HOST))
      GUARD_CU(ground_truth.Allocate(num_nodes, target | util::HOST))
      GUARD_CU(loss.Allocate(1, target | util::HOST))

      if (training) {
        util::PrintMsg("target: " + util::Location_to_string(target));
        GUARD_CU(grad.Init(num_nodes * num_classes, target))
//        util::PrintMsg("Allocated on: " + util::Location_to_string(grad.GetAllocated()));
      }

//      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(bool train, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      this->training = train;

//      GUARD_CU(ground_truth.ForEach(
//          []__host__ __device__(int &x) {
//            x = -1;
//          }
//      ))

      GUARD_CU(loss.ForEach(
          []__host__ __device__(ValueT &x) {
            x = 0;
          }, loss.GetSize(), util::DEVICE
      ))

      if (training) {
        GUARD_CU(grad.ForEach(
            []__host__ __device__(ValueT &x) {
              x = 0;
            }, grad.GetSize(), util::DEVICE
        ))
      }
      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief graphsum default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief graphsum default destructor
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


  cudaError_t Extract(ValueT *grad_out, ValueT *loss,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]))

//        util::PrintMsg(util::Location_to_string(data_slice.grad.GetSetted()));
//        util::PrintMsg(util::Location_to_string(data_slice.grad.GetAllocated()));

        GUARD_CU(
            data_slice.grad.SetPointer(grad_out,
                data_slice.num_nodes * data_slice.num_classes, util::HOST))
        GUARD_CU(data_slice.grad.Move(util::DEVICE, util::HOST))

        GUARD_CU(data_slice.loss.SetPointer(loss, 1, util::HOST))
        GUARD_CU(data_slice.loss.Move(util::DEVICE, util::HOST))
      }
    }

    return retval;
  }

  cudaError_t Extract(ValueT *loss,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]))

        GUARD_CU(data_slice.loss.SetPointer(loss, 1, util::HOST))
        GUARD_CU(data_slice.loss.Move(util::DEVICE, util::HOST))
      }
    }

    return retval;
  }

  /**
   * @brief      initialization function.
   *
   * @param      graph   The graph that SSSP processes on
   * @param[in]  dim     The dimension of the feature vector
   * @param[in]  target  The target
   * @param[in]  Location  Memory location to work on
   *
   * @return     cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, const int num_nodes, const int num_classes, ValueT *p_logits,
      int *truth, bool training = 1, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], num_nodes,
        num_classes, training, this->num_gpus, this->gpu_idx[gpu], target, this->flag))

      GUARD_CU(data_slice.logits.SetPointer(p_logits, num_nodes * num_classes, util::HOST))
      GUARD_CU(data_slice.logits.Move(util::HOST, util::DEVICE))

      GUARD_CU(data_slice.ground_truth.SetPointer(truth, num_nodes, util::HOST))
      GUARD_CU(data_slice.ground_truth.Move(util::HOST, util::DEVICE))
//      data_slice.ground_truth.Print();
    }  // end for (gpu)

    return retval;
  }

  cudaError_t Init(GraphT &graph, const int num_nodes, const int num_classes, Array &logits, Array &_grad,
      util::Array1D<SizeT, int> &truth, bool training = 1, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], num_nodes,
        num_classes, training, this->num_gpus, this->gpu_idx[gpu], target, this->flag))

      data_slice.logits = logits;
      data_slice.ground_truth = truth;
      data_slice.grad = _grad;

      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
//      data_slice.ground_truth.Print();
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(bool train, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(train, target));
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  /** @} */
};

}  // namespace graphsum
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
