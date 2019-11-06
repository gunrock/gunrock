// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * template_problem.cuh
 *
 * @brief GPU Storage management Structure for Template Problem Data
 */

#pragma once

#include <iostream>
#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief  Speciflying parameters for BC Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief  BetweennessCentrality Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _ValueT  Type of BC values, usually float or double
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef _ValueT ValueT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing BC-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // device storage arrays
    util::Array1D<SizeT, ValueT> bc_values;  // Final BC values for each vertex
    util::Array1D<SizeT, ValueT>
        sigmas;  // Accumulated sigma values for each vertex
    util::Array1D<SizeT, ValueT>
        deltas;        // Accumulated delta values for each vertex
    VertexT src_node;  // Source vertex ID
    util::Array1D<SizeT, VertexT>
        *forward_output;  // Output vertex IDs by the forward pass
    std::vector<SizeT> *forward_queue_offsets;
    util::Array1D<SizeT, VertexT> original_vertex;
    util::Array1D<int, unsigned char> *barrier_markers;
    util::Array1D<SizeT, bool> first_backward_incoming;
    util::Array1D<SizeT, VertexT> local_vertices;
    util::Array1D<SizeT, bool> middle_event_set;
    util::Array1D<SizeT, cudaEvent_t> middle_events;
    VertexT middle_iteration;
    bool middle_finish;

    util::Array1D<SizeT, VertexT> preds;   // predecessors of vertices
    util::Array1D<SizeT, VertexT> labels;  // Source distance

    /*
     * @brief Default constructor
     */
    DataSlice()
        : BaseDataSlice(),
          src_node(0),
          middle_iteration(0),
          middle_finish(false),
          forward_output(NULL),
          forward_queue_offsets(NULL),
          barrier_markers(NULL) {
      bc_values.SetName("bc_values");
      sigmas.SetName("sigmas");
      deltas.SetName("deltas");
      original_vertex.SetName("original_vertex");
      first_backward_incoming.SetName("first_backward_incoming");
      local_vertices.SetName("local_vertices");
      middle_event_set.SetName("middle_event_set");
      middle_events.SetName("middle_events");
      preds.SetName("preds");
      labels.SetName("labels");
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

      GUARD_CU(labels.Release(target));
      GUARD_CU(preds.Release(target));
      GUARD_CU(bc_values.Release(target));
      GUARD_CU(sigmas.Release(target));
      GUARD_CU(deltas.Release(target));
      GUARD_CU(original_vertex.Release(target));
      GUARD_CU(first_backward_incoming.Release(target));
      GUARD_CU(local_vertices.Release(target));
      GUARD_CU(middle_events.Release(target));
      GUARD_CU(middle_event_set.Release(target));

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        GUARD_CU(forward_output[gpu].Release(target));
        forward_queue_offsets[gpu].resize(0);
      }
      if (forward_output != NULL) {
        delete[] forward_output;
        forward_output = NULL;
      }
      if (forward_queue_offsets != NULL) {
        delete[] forward_queue_offsets;
        forward_queue_offsets = NULL;
      }
      barrier_markers = NULL;

      GUARD_CU(BaseDataSlice::Release(target));
      return retval;
    }

    /**
     * @brief initializing bc-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
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

      GUARD_CU(labels.Allocate(sub_graph.nodes, target | util::HOST));
      GUARD_CU(preds.Allocate(sub_graph.nodes, target));
      GUARD_CU(bc_values.Allocate(sub_graph.nodes, target));
      GUARD_CU(sigmas.Allocate(sub_graph.nodes, target | util::HOST));
      GUARD_CU(deltas.Allocate(sub_graph.nodes, target));

      GUARD_CU(bc_values.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; },
          sub_graph.nodes, target, this->stream));

      forward_queue_offsets = new std::vector<SizeT>[num_gpus];

      forward_output = new util::Array1D<SizeT, VertexT>[num_gpus];
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        forward_queue_offsets[gpu].reserve(sub_graph.nodes);
        forward_queue_offsets[gpu].push_back(0);
        forward_output[gpu].SetName("forward_output[]");
        GUARD_CU(forward_output[gpu].Allocate(sub_graph.nodes, target));
      }

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }

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
      GUARD_CU(labels.EnsureSize_(nodes, target));
      GUARD_CU(preds.EnsureSize_(nodes, target));
      GUARD_CU(bc_values.EnsureSize_(nodes, target));
      GUARD_CU(sigmas.EnsureSize_(nodes, target));
      GUARD_CU(deltas.EnsureSize_(nodes, target));

      // Reset data
      GUARD_CU(labels.ForEach(
          [] __host__ __device__(VertexT & x) {
            x = util::PreDefinedValues<VertexT>::InvalidValue;  //(VertexT)-1;
          },
          nodes, target, this->stream));

      GUARD_CU(preds.ForEach(
          [] __host__ __device__(VertexT & x) {
            x = util::PreDefinedValues<VertexT>::InvalidValue;  //(VertexT)-2;
          },
          nodes, target, this->stream));

      // ?? Do I actually want to be resetting this?
      GUARD_CU(bc_values.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, nodes,
          target, this->stream));

      GUARD_CU(deltas.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, nodes,
          target, this->stream));

      GUARD_CU(sigmas.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, nodes,
          target, this->stream));

      // ?? Reset `src_node YC: in problem::Reset()`

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        GUARD_CU(forward_output[gpu].EnsureSize_(nodes, util::DEVICE));

        forward_queue_offsets[gpu].clear();
        forward_queue_offsets[gpu].reserve(nodes);
        forward_queue_offsets[gpu].push_back(0);

        if (this->num_gpus > 1) middle_event_set[gpu] = false;
      }
      middle_iteration = util::PreDefinedValues<VertexT>::InvalidValue;
      middle_finish = false;

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief BCProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief BCProblem default destructor
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
    for (int gpu = 0; gpu < this->num_gpus; gpu++)
      GUARD_CU(data_slices[gpu].Release(target));

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
   * the source. \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_bc_values, ValueT *h_sigmas, VertexT *h_labels,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(
            data_slice.bc_values.SetPointer(h_bc_values, nodes, util::HOST));
        GUARD_CU(data_slice.bc_values.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.sigmas.SetPointer(h_sigmas, nodes, util::HOST));
        GUARD_CU(data_slice.sigmas.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.labels.SetPointer(h_labels, nodes, util::HOST));
        GUARD_CU(data_slice.labels.Move(util::DEVICE, util::HOST));

      } else if (target == util::HOST) {
        GUARD_CU(data_slice.bc_values.ForEach(
            h_bc_values,
            [] __host__ __device__(const ValueT &x, ValueT &h_x) { h_x = x; },
            nodes, util::HOST));

        GUARD_CU(data_slice.sigmas.ForEach(
            h_sigmas,
            [] __host__ __device__(const ValueT &x, ValueT &h_x) { h_x = x; },
            nodes, util::HOST));

        GUARD_CU(data_slice.labels.ForEach(
            h_labels,
            [] __host__ __device__(const VertexT &x, VertexT &h_x) { h_x = x; },
            nodes, util::HOST));
      }
    } else {
      // TODO: extract the results from multiple GPUs, e.g.:
    }

    // Scale final results by 0.5
    // YC: ?
    for (VertexT v = 0; v < nodes; ++v) {
      h_bc_values[v] *= (ValueT)0.5;
    }

    // Logging
    // for(VertexT v = 0; v < nodes; ++v) {
    //    std::cout
    //        << "v=" << v
    //        << " | h_bc_values[v]="   << h_bc_values[v]
    //    << std::endl;
    //}

    // for(VertexT v = 0; v < nodes; ++v) {
    //    std::cout
    //        << "v=" << v
    //        << " | h_sigmas[v]="   << h_sigmas[v]
    //    << std::endl;
    //}

    // for(VertexT v = 0; v < nodes; ++v) {
    //    std::cout
    //        << "v=" << v
    //        << " | h_labels[v]="   << h_labels[v]
    //    << std::endl;
    //}

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that BC processes on
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
                               this->gpu_idx[gpu], target, this->flag));
    }  // end for (gpu)

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (target & util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
        GUARD_CU2(cudaStreamSynchronize(data_slices[gpu]->stream),
                  "cudaStreamSynchronize failed");
      }
    }
    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    std::cout << "Problem->Reset(" << src << ")" << std::endl;
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    int gpu;
    VertexT tsrc;
    if (this->num_gpus <= 1) {
      gpu = 0;
      tsrc = src;
    } else {
      // TODO
    }

    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    VertexT src_label = 0;
    VertexT src_pred = util::PreDefinedValues<VertexT>::InvalidValue;
    ValueT src_sigma = 1.0;
    data_slices[gpu]->src_node = tsrc;

    if (target & util::HOST) {
      data_slices[gpu]->labels[tsrc] = src_label;
      data_slices[gpu]->preds[tsrc] = src_pred;
      data_slices[gpu]->sigmas[tsrc] = src_sigma;
    }
    if (target & util::DEVICE) {
      GUARD_CU2(
          cudaMemcpy(data_slices[gpu]->labels.GetPointer(util::DEVICE) + tsrc,
                     &src_label, sizeof(VertexT), cudaMemcpyHostToDevice),
          "BCProblem cudaMemcpy labels failed");

      GUARD_CU2(
          cudaMemcpy(data_slices[gpu]->preds.GetPointer(util::DEVICE) + tsrc,
                     &src_pred, sizeof(VertexT), cudaMemcpyHostToDevice),
          "BCProblem cudaMemcpy preds failed");

      GUARD_CU2(
          cudaMemcpy(data_slices[gpu]->sigmas.GetPointer(util::DEVICE) + tsrc,
                     &src_sigma, sizeof(ValueT), cudaMemcpyHostToDevice),
          "BCProblem cudaMemcpy sigmas failed");
    }
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    return retval;
  }

  /** @} */
};

}  // namespace bc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
