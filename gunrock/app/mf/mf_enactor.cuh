// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_enactor.cuh
 *
 * @brief Max Flow Problem Enactor
 */

#pragma once
#include <gunrock/util/sort_device.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>

#include <gunrock/app/mf/mf_helpers.cuh>
#include <gunrock/app/mf/mf_problem.cuh>

#include <gunrock/oprtr/1D_oprtr/for.cuh>
#include <gunrock/oprtr/oprtr.cuh>

// uncoment for debug output
// #define MF_DEBUG 1

#if MF_DEBUG
#define debug_aml(a...) printf(a);
#else
#define debug_aml(a...)
#endif

namespace gunrock {
namespace app {
namespace mf {

/**
 * @brief Speciflying parameters for MF Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 *		      info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}
/**
 * @brief defination of MF iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct MFIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT GraphT;
  typedef typename GraphT::CsrT CsrT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  MFIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of mf, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    auto enactor = this->enactor;
    auto gpu_num = this->gpu_num;
    auto num_gpus = enactor->num_gpus;
    auto gpu_offset = num_gpus * gpu_num;
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    auto source = data_slice.source;
    auto sink = data_slice.sink;
    auto num_repeats = data_slice.num_repeats;
    auto &capacity = graph.edge_values;
    auto &reverse = data_slice.reverse;
    auto &flow = data_slice.flow;
    auto &residuals = data_slice.residuals;
    auto &excess = data_slice.excess;
    auto &height = data_slice.height;
    auto &lowest_neighbor = data_slice.lowest_neighbor;
    auto &local_vertices = data_slice.local_vertices;
    auto &active = data_slice.active;
    auto null_ptr = &local_vertices;
    null_ptr = NULL;
    auto &mark = data_slice.mark;
    auto &queue = data_slice.queue0;
    auto &queue0 = data_slice.queue0;
    auto &queue1 = data_slice.queue1;

    auto advance_preflow_op =
        [capacity, flow, excess, height, reverse, source, residuals] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   const SizeT &output_pos) -> bool {
      if (!util::isValid(dest) or !util::isValid(src)) return false;
      if (dest != source) residuals[edge_id] = capacity[edge_id];
      if (src != source) return false;
      auto c = capacity[edge_id];
      residuals[edge_id] = 0;
      residuals[reverse[edge_id]] = capacity[reverse[edge_id]] + c;
      atomicAdd(excess + dest, c);
      return true;
    };

    auto par_global_relabeling_op =
        [graph, source, sink, height, reverse, queue0, queue1,
         residuals] __host__
        __device__(VertexT * v_q, const SizeT &pos) {
          VertexT first = 0, last = 0;
          auto &queue = (pos == 0 ? queue0 : queue1);
          auto &start = (pos == 0 ? source : sink);
          height[start] = (pos == 0 ? graph.nodes : 0);
          auto H = height[start];
          queue[last++] = start;
          while (first < last) {
            auto v = queue[first++];
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            ++H;
            for (auto e = e_start; e < e_end; ++e) {
              auto neighbor = graph.CsrT::GetEdgeDest(e);
              if (residuals[reverse[e]] < MF_EPSILON) continue;
              if (height[neighbor] > H + 1) {
                height[neighbor] = H + 1;
                queue[last++] = neighbor;
              }
            }
          }
        };

    auto global_relabeling_op =
        [graph, source, sink, height, reverse, queue, residuals] __host__
        __device__(VertexT * v_q, const SizeT &pos) {
          VertexT first = 0, last = 0;
          queue[last++] = sink;
          auto H = (VertexT)0;
          height[sink] = 0;
          while (first < last) {
            auto v = queue[first++];
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            ++H;
            for (auto e = e_start; e < e_end; ++e) {
              auto neighbor = graph.CsrT::GetEdgeDest(e);
              if (residuals[reverse[e]] < MF_EPSILON) continue;
              if (height[neighbor] > H + 1) {
                height[neighbor] = H + 1;
                queue[last++] = neighbor;
              }
            }
          }
          height[source] = graph.nodes;
          first = 0;
          last = 0;
          queue[last++] = source;
          H = (VertexT)graph.nodes;
          while (first < last) {
            auto v = queue[first++];
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            ++H;
            for (auto e = e_start; e < e_end; ++e) {
              auto neighbor = graph.CsrT::GetEdgeDest(e);
              if (residuals[reverse[e]] < MF_EPSILON) continue;
              if (height[neighbor] > H + 1) {
                height[neighbor] = H + 1;
                queue[last++] = neighbor;
              }
            }
          }
        };

    auto compute_lockfree_op =
        [graph, excess, residuals, reverse, height, iteration, source, sink,
         active] __host__
        __device__(const int &counter, const VertexT &v) {
          // v - current vertex
          if (v == 0) active[(counter + 1) % 2] = 0;

          // if not a valid vertex, do not apply compute:
          if (!util::isValid(v) || v == source || v == sink) return;

          VertexT neighbor_num = graph.CsrT::GetNeighborListLength(v);
          ValueT excess_v = excess[v];
          if (excess_v < MF_EPSILON || neighbor_num == 0) return;

          // turn off vertices which relabeling drop out from graph

          // else, try push-relable:
          VertexT e_start = graph.CsrT::GetNeighborListOffset(v);
          VertexT e_end = e_start + neighbor_num;

          VertexT lowest_id = util::PreDefinedValues<VertexT>::InvalidValue;
          VertexT lowest_h = util::PreDefinedValues<VertexT>::MaxValue;
          ValueT lowest_r = 0;
          VertexT lowest_n = 0;
          // look for lowest height among neighbors
          for (VertexT e_id = e_start; e_id < e_end; ++e_id) {
            ValueT r = residuals[e_id];  // capacity[e_id] - flow[e_id];
            if (r < MF_EPSILON) continue;
            VertexT n = graph.CsrT::GetEdgeDest(e_id);
            VertexT h = height[n];
            if (h < lowest_h) {
              lowest_id = e_id;
              lowest_h = h;
              lowest_r = r;
              lowest_n = n;
            }
          }

          // if a valid lowest h was found:
          if (!util::isValid(lowest_id)) return;
          active[counter % 2] = 1;
          if (lowest_h < height[v]) {
            // push
            ValueT f = fminf(lowest_r, excess_v);
            atomicAdd(excess + v, -f);
            atomicAdd(excess + lowest_n, f);
            residuals[lowest_id] -= f;
            residuals[reverse[lowest_id]] += f;
            debug_aml("push, %lf, %d->%d\n", f, v, lowest_id);
          } else {
            // relabel
            height[v] = lowest_h + 1;
            debug_aml("relabel, %d new height %d\n", v, lowest_h + 1);
          }
        };

    if (iteration == 0) {
      // ADVANCE_PREFLOW_OP
      oprtr_parameters.advance_mode = "ALL_EDGES";
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), &local_vertices, null_ptr, oprtr_parameters,
          advance_preflow_op));

      // GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      // GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
      //        "cudaStreamSynchronize failed.");
    }

    // Global relabeling
    // Height reset
    //    if (iteration == 0){
    //    fprintf(stderr, "global relabeling in iteration %d\n", iteration);
    GUARD_CU(height.ForAll(
        [graph] __host__ __device__(VertexT * h, const VertexT &pos) {
          h[pos] = 2 * graph.nodes + 1;
        },
        graph.nodes, util::DEVICE, oprtr_parameters.stream));

    // Serial relabeling on the GPU (ignores moves)
    GUARD_CU(frontier.V_Q()->ForAll(global_relabeling_op, 1, util::DEVICE,
                                    //      GUARD_CU(frontier.V_Q()->ForAll(par_global_relabeling_op,
                                    //      2, util::DEVICE,
                                    oprtr_parameters.stream));
    //   }
    debug_aml("[%d]frontier que length before compute op is %d\n", iteration,
              frontier.queue_length);

    // Run Lockfree Push-Relable
    // GUARD_CU(frontier.V_Q()->ForAll(compute_lockfree_op,
    //            graph.nodes, util::DEVICE, oprtr_parameters.stream));

    SizeT loop_size = graph.nodes;
    gunrock::oprtr::RepeatFor(
        compute_lockfree_op,                       /* lambda */
        num_repeats,                               /* num_repeats (int) */
        graph.nodes,                               /* ForIterT loop_size */
        util::DEVICE,                              /* target */
        oprtr_parameters.stream,                   /* stream */
        util::PreDefinedValues<int>::InvalidValue, /* grid_size */
        util::PreDefinedValues<int>::InvalidValue, /* block_size */
        2 /* mode: stacked kernels */);

    debug_aml("[%d]frontier que length after compute op is %d\n", iteration,
              frontier.queue_length);

    // GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
    //        "cudaStreamSynchronize failed");

    active.Move(util::DEVICE, util::HOST, 2, 0, oprtr_parameters.stream);

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed");

    if (active[0] == 0 && active[1] == 0) {
      GUARD_CU(oprtr::For(
          [residuals, capacity, flow] __host__ __device__(const SizeT &e) {
            flow[e] = capacity[e] - residuals[e];
          },
          graph.edges, util::DEVICE, oprtr_parameters.stream));
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                "cudaStreamSynchronize failed");
    }
    return retval;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES  Number of data associated with each
   *				      transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES  Number of data associated with each
                                    transmition item, typed ValueT
   * @param[in] received_length     The number of transmition items received
   * @param[in] peer_		      which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    auto &enactor = this->enactor;
    auto &problem = enactor->problem;
    auto gpu_num = this->gpu_num;
    auto gpu_offset = gpu_num * enactor->num_gpus;
    auto &data_slice = problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;

    auto &capacity = data_slice.sub_graph[0].edge_values;
    auto &flow = data_slice.flow;
    auto &excess = data_slice.excess;
    auto &height = data_slice.height;

    /*	for key " +
                        std::to_string(key) + " and for in_pos " +
                        std::to_string(in_pos) + " and for vertex ass ins " +
                        std::to_string(vertex_associate_ins[in_pos]) +
                        " and for value ass ins " +
                        std::to_string(value__associate_ins[in_pos]));*/

    auto expand_op = [capacity, flow, excess, height] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto enactor = this->enactor;
    int num_gpus = enactor->num_gpus;
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[0];
    auto &retval = enactor_slice.enactor_stats.retval;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;

    if (retval != cudaSuccess) {
      printf("(CUDA error %d @ GPU %d: %s\n", retval, 0 % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    // if (enactor_slice.enactor_stats.iteration > 1)
    //    return true;
    if (data_slice.active[0] > 0 || data_slice.active[1] > 0) return false;
    return true;
  }

};  // end of MFIteration

/* MF enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::VertexT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::GraphT GraphT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;

  typedef MFIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief MFEnactor constructor
   */
  Enactor() : BaseEnactor("mf"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief MFEnactor destructor
   */
  virtual ~Enactor() {
    // Release();
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));

    auto num_gpus = this->num_gpus;

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto gpu_offset = gpu * num_gpus;
      auto &enactor_slice = this->enactor_slices[gpu_offset + 0];
      auto &graph = problem.sub_graphs[gpu];
      auto nodes = graph.nodes;
      auto edges = graph.edges;
      GUARD_CU(
          enactor_slice.frontier.Allocate(nodes, edges, this->queue_factors));
    }

    iterations = new IterationT[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of mf, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    debug_aml("Run enact\n");
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);

    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(const VertexT &src, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    debug_aml("Enactor Reset, src %d\n", src);

    typedef typename GraphT::GpT GpT;

    GUARD_CU(BaseEnactor::Reset(target));

    SizeT nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? nodes : 0;
          if (peer_ == 0) {
            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(nodes, target | util::HOST);
            for (SizeT i = 0; i < nodes; ++i) {
              tmp[i] = (VertexT)i % nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                nodes, target, 0));

            tmp.Release();
          }
        }
      } else {
        // MULTIGPU INCOMPLETE
      }
    }
    debug_aml("Enactor Reset end\n");
    GUARD_CU(BaseEnactor::Sync())
    return retval;
  }

  /**
   * @brief Enacts a MF computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    debug_aml("enact\n");
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU MF Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace mf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
