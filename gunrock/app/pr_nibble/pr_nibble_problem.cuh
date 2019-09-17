// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_nibble_problem.cuh
 *
 * @brief GPU Storage management Structure for pr_nibble Problem Data
 */

#pragma once

#include <iostream>
#include <math.h>
#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace pr_nibble {

/**
 * @brief Speciflying parameters for pr_nibble Problem
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
    util::Array1D<SizeT, ValueT> values;  // Output values

    util::Array1D<SizeT, ValueT> grad;  // Gradient values
    util::Array1D<SizeT, ValueT> y;     // Intermediate quantity
    util::Array1D<SizeT, ValueT> z;     // Intermediate quantity
    util::Array1D<SizeT, ValueT> q;     // Truncated z-values
    util::Array1D<SizeT, int> touched;  // Keep track of

    VertexT src;        // Node to start local PR from
    VertexT src_neib;   // Neighbor of reference node
    int num_ref_nodes;  // Number of source nodes (hardcoded to 1 for now)

    int *d_grad_scale;  // Gradient magnitude for convergence check
    int *h_grad_scale;  // Gradient magnitude for convergence check
    ValueT *d_grad_scale_value;
    ValueT *h_grad_scale_value;

    ValueT eps;    // Tolerance for convergence
    ValueT alpha;  // Parameterizes conductance/size of output cluster
    ValueT rho;    // Parameterizes conductance/size of output cluster
    int max_iter;  // Maximum number of iterations

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      values.SetName("values");
      grad.SetName("grad");
      q.SetName("q");
      y.SetName("y");
      z.SetName("z");
      touched.SetName("touched");
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

      GUARD_CU(values.Release(target));
      GUARD_CU(grad.Release(target));
      GUARD_CU(q.Release(target));
      GUARD_CU(y.Release(target));
      GUARD_CU(z.Release(target));
      GUARD_CU(touched.Release(target));

      GUARD_CU(BaseDataSlice::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * @param[in] _eps        Convergence criteria
     * @param[in] _alpha
     * @param[in] _rho
     * @param[in] _max_iter   Max number of iterations

     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     util::Location target, ProblemFlag flag, ValueT _eps,
                     ValueT _alpha, ValueT _rho, int _max_iter) {
      cudaError_t retval = cudaSuccess;

      eps = _eps;
      alpha = _alpha;
      rho = _rho;
      max_iter = _max_iter;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU(values.Allocate(sub_graph.nodes, target));
      GUARD_CU(grad.Allocate(sub_graph.nodes, target));
      GUARD_CU(q.Allocate(sub_graph.nodes, target));
      GUARD_CU(y.Allocate(sub_graph.nodes, target));
      GUARD_CU(z.Allocate(sub_graph.nodes, target));
      GUARD_CU(touched.Allocate(sub_graph.nodes, target));

      h_grad_scale = (int *)malloc(sizeof(int) * 1);
      GUARD_CU(cudaMalloc((void **)&d_grad_scale, 1 * sizeof(int)));

      h_grad_scale_value = (ValueT *)malloc(sizeof(ValueT) * 1);
      GUARD_CU(cudaMalloc((void **)&d_grad_scale_value, 1 * sizeof(ValueT)));

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }
      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * @param[in] _src             Source node
     * @param[in] _src_neib        Neighbor of source node (!! HACK)
     * @param[in] _num_ref_nodes   Number of source nodes (HARDCODED to 1
     * elsewhere) \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(VertexT _src, VertexT _src_neib, int _num_ref_nodes,
                      util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      src = _src;
      src_neib = _src_neib;
      num_ref_nodes = _num_ref_nodes;

      // Ensure data are allocated
      GUARD_CU(values.EnsureSize_(nodes, target));
      GUARD_CU(grad.EnsureSize_(nodes, target));
      GUARD_CU(q.EnsureSize_(nodes, target));
      GUARD_CU(y.EnsureSize_(nodes, target));
      GUARD_CU(z.EnsureSize_(nodes, target));
      GUARD_CU(touched.EnsureSize_(nodes, target));

      // Reset data
      GUARD_CU(
          values.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                         nodes, target, this->stream));

      GUARD_CU(
          grad.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                       nodes, target, this->stream));

      GUARD_CU(q.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                         nodes, target, this->stream));

      GUARD_CU(y.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                         nodes, target, this->stream));

      GUARD_CU(z.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                         nodes, target, this->stream));

      GUARD_CU(touched.ForEach([] __host__ __device__(int &x) { x = 0; }, nodes,
                               target, this->stream));

      return retval;
    }
  };  // DataSlice

  // Problem attributes
  util::Array1D<SizeT, DataSlice> *data_slices;

  ValueT phi;
  ValueT vol;
  int max_iter;
  ValueT eps;

  ValueT alpha;
  ValueT rho;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief pr_nibble default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    // Load command line parameters
    phi = _parameters.Get<ValueT>("phi");
    max_iter = _parameters.Get<int>("max-iter");
    eps = _parameters.Get<ValueT>("eps");
    vol = _parameters.Get<ValueT>("vol");
    if (vol == 0.0) {
      vol = 1.0;
    }
  }

  /**
   * @brief pr_nibble default destructor
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
   * @param[in] h_values      Host array for PR values
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_values, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    ValueT *h_grad = new ValueT[nodes];
    ValueT *h_y = new ValueT[nodes];
    ValueT *h_z = new ValueT[nodes];
    ValueT *h_q = new ValueT[nodes];

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.grad.SetPointer(h_grad, nodes, util::HOST));
        GUARD_CU(data_slice.grad.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.y.SetPointer(h_y, nodes, util::HOST));
        GUARD_CU(data_slice.y.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.z.SetPointer(h_z, nodes, util::HOST));
        GUARD_CU(data_slice.z.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.q.SetPointer(h_q, nodes, util::HOST));
        GUARD_CU(data_slice.q.Move(util::DEVICE, util::HOST));

      } else if (target == util::HOST) {
        GUARD_CU(data_slice.grad.ForEach(
            h_grad,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.y.ForEach(
            h_y,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.z.ForEach(
            h_z,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.q.ForEach(
            h_q,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));
      }

      for (SizeT i = 0; i < nodes; ++i) {
        SizeT d = this->org_graph->GetNeighborListLength(i);
        double d_sqrt = sqrt((double)d);
        h_values[i] = abs(h_q[i] * d_sqrt);
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

    ValueT num_edges = (ValueT)graph.edges / 2.0;
    ValueT log_num_edges = log2(num_edges);

    // alpha
    this->alpha = pow(this->phi, 2) / (225.0 * log(100.0 * sqrt(num_edges)));

    // rho
    if (1.0f + log2((ValueT)this->vol) > log_num_edges) {
      this->rho = log_num_edges;
    } else {
      this->rho = 1.0f + log2((ValueT)this->vol);
    }
    this->rho = pow(2.0f, this->rho);
    this->rho = 1.0 / this->rho;
    this->rho *= 1.0 / (48.0 * log_num_edges);

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(
          this->sub_graphs[gpu], this->num_gpus, this->gpu_idx[gpu], target,
          this->flag, this->eps, this->alpha, this->rho, this->max_iter));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src       Source vertex
   * @param[in] src_neib  Source vertex neighbor (!! HACK)
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src, VertexT src_neib,
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    int num_ref_nodes = 1;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(src, src_neib, num_ref_nodes, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    int gpu;
    VertexT src_;
    if (this->num_gpus <= 1) {
      gpu = 0;
      src_ = src;
    } else {
      // TODO -- MULTIGPU
    }

    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    SizeT src_d = this->org_graph->GetNeighborListLength(src_);
    ValueT src_d_sqrt = sqrt((ValueT)src_d);
    ValueT src_dn_sqrt = 1.0 / src_d_sqrt;
    ValueT src_grad = -1.0 * this->alpha * src_dn_sqrt / (double)num_ref_nodes;

    ValueT thresh = this->rho * this->alpha * src_d_sqrt;
    if (-src_grad < thresh) {
      printf("pr_nibble::Problem: `-1 * src_grad < thresh` -> breaking");
      return retval;
    }

    if (target & util::HOST) {
      data_slices[gpu]->grad[src_] = src_grad;
    }

    if (target & util::DEVICE) {
      GUARD_CU2(
          cudaMemcpy(data_slices[gpu]->grad.GetPointer(util::DEVICE) + src_,
                     &src_grad, sizeof(ValueT), cudaMemcpyHostToDevice),
          "PRNibble cudaMemcpy distances failed");

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace pr_nibble
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
