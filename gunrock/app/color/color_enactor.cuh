// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * color_enactor.cuh
 *
 * @brief color Problem Enactor
 */

#pragma once

#include <gunrock/util/sort_device.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>

#include <gunrock/app/color/color_problem.cuh>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#include <gunrock/oprtr/1D_oprtr/for.cuh>
#endif

#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace color {

/**
 * @brief Speciflying parameters for color Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of color iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct ColorIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  ColorIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of color, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    auto &colors = data_slice.colors;
    auto &visited = data_slice.visited;
    auto &rand = data_slice.rand;
    auto &color_predicate = data_slice.color_predicate;
    auto &color_temp = data_slice.color_temp;
    auto &color_temp2 = data_slice.color_temp2;
    auto &prohibit = data_slice.prohibit;
    auto &color_balance = data_slice.color_balance;
    auto &colored = data_slice.colored;
    auto &use_jpl = data_slice.use_jpl;
    auto &no_conflict = data_slice.no_conflict;
    auto &prohibit_size = data_slice.prohibit_size;
    auto &test_run = data_slice.test_run;
    auto &min_color = data_slice.min_color;
    auto stream = oprtr_parameters.stream;
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    auto null_ptr = null_frontier;
    auto user_iter = data_slice.user_iter;
    auto gen = data_slice.gen;

    //======================================================================//
    // Jones-Plassman-Luby Graph Coloring: Compute Operator                 //
    //======================================================================//
    if (use_jpl) {
      if (!color_balance) {
        if (iteration % 2)
          curandGenerateUniform(gen, rand.GetPointer(util::DEVICE),
                                graph.nodes);

        auto jpl_color_op =
            [graph, colors, rand, iteration, min_color, colored] __host__
            __device__(
                // #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                //                const int &counter, const VertexT &v) {
                // #else
                VertexT * v_q, const SizeT &pos) {
              VertexT v = pos;               // v_q[pos];
                                             // #endif
              if (pos == 0) colored[0] = 0;  // reset colored ahead-of-time
              if (util::isValid(colors[v])) return;

              SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
              SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

              bool colormax = true;
              bool colormin = true;
              int color = iteration * 2;

              for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
                VertexT u = graph.CsrT::GetEdgeDest(e);

                if ((util::isValid(colors[u])) && (colors[u] != color + 1) &&
                        (colors[u] != color + 2) ||
                    (v == u))
                  continue;
                if (rand[v] <= rand[u]) colormax = false;
                if (min_color) {
                  if (rand[v] >= rand[u]) colormin = false;
                }
              }

              if (colormax) colors[v] = color + 1;
              if (min_color) {
                if (colormin) colors[v] = color + 2;
              }
            };

#if 0  // (RepeatFor Implementation)
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        SizeT loop_size = frontier.queue_length;
        gunrock::oprtr::RepeatFor(
            jpl_color_op,                              /* lambda */
            user_iter,                                 /* num_repeats (int) */
            loop_size,                                 /* ForIterT loop_size */
            util::DEVICE,                              /* target */
            stream,                                    /* stream */
            util::PreDefinedValues<int>::InvalidValue, /* grid_size */
            util::PreDefinedValues<int>::InvalidValue, /* block_size */
            2 /* mode: stacked kernels */);
// #else
#endif

        GUARD_CU(frontier.V_Q()->ForAll(jpl_color_op, frontier.queue_length,
                                        util::DEVICE, stream));
        // #endif

      }

      else {
        //======================================================================//
        // Jones-Plassman-Luby Graph Coloring: NeighborReduce + Compute Op //
        //======================================================================//
        auto advance_op = [graph, iteration, colors, rand] __host__ __device__(
                              const VertexT &src, VertexT &dest,
                              const SizeT &edge_id, const VertexT &input_item,
                              const SizeT &input_pos,
                              SizeT &output_pos) -> ValueT {
          if (util::isValid(colors[dest])) return (ValueT)-1;
          return rand[dest];
        };

        auto reduce_op = [rand, colors, iteration] __host__ __device__(
                             const ValueT &a, const ValueT &b) -> ValueT {
          return (a < b) ? b : a;
        };

        oprtr_parameters.reduce_values_out = &color_predicate;
        oprtr_parameters.reduce_values_temp = &color_temp;
        oprtr_parameters.reduce_values_temp2 = &color_temp2;
        oprtr_parameters.reduce_reset = true;
        oprtr_parameters.advance_mode = "ALL_EDGES";

        frontier.queue_length = graph.nodes;
        frontier.queue_reset = true;
        static ValueT Identity = util::PreDefinedValues<ValueT>::MinValue;

        GUARD_CU(oprtr::NeighborReduce<oprtr::OprtrType_V2V |
                                       oprtr::OprtrMode_REDUCE_TO_SRC |
                                       oprtr::ReduceOp_Max>(
            graph.csr(), null_ptr, null_ptr, oprtr_parameters, advance_op,
            reduce_op, Identity));

        auto reduce_color_op =
            [graph, rand, colors, color_predicate, iteration, colored] __host__
            __device__(VertexT * v_q, const SizeT &pos) {
              if (pos == 0) colored[0] = 0;  // reset colored ahead-of-time
              VertexT v = v_q[pos];
              if (util::isValid(colors[v])) return;

              if (color_predicate[v] < rand[v]) colors[v] = iteration;

              return;
            };

        GUARD_CU(frontier.V_Q()->ForAll(reduce_color_op, graph.nodes,
                                        util::DEVICE, stream));
      }
    }

    //======================================================================//
    // Min-Max Hash Graph Coloring: Compute Operator                        //
    //======================================================================//
    else {
      auto color_op =
          [graph, colors, rand, iteration, prohibit_size, visited, prohibit,
           colored] __host__
          __device__(VertexT * v_q, const SizeT &pos) {
            if (pos == 0) colored[0] = 0;  // reset colored ahead-of-time
            VertexT v = v_q[pos];
            SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto temp = rand[v];

            VertexT max = v;  // active max vertex
            VertexT min = v;  // active min vertex

            for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
              VertexT u = graph.CsrT::GetEdgeDest(e);
              if ((rand[u] > temp) && !util::isValid(colors[u])) max = u;

              if ((rand[u] < temp) && !util::isValid(colors[u])) min = u;

              temp = rand[u];  // compare against e-1
            }

            // hash coloring
            auto max_color = iteration * 2 + 1;
            auto max_neighbors = graph.CsrT::GetNeighborListLength(max);
            auto min_color = iteration * 2 + 2;
            auto min_neighbors = graph.CsrT::GetNeighborListLength(min);
            auto max_offset = max * prohibit_size;
            auto min_offset = min * prohibit_size;
            int hash_color = -1;

            if (prohibit_size != 0) {
              for (int c_max = 0; 2 * c_max + 1 <= max_color &&
                                  !util::isValid(colors[max]) && !visited[max];
                   c_max++) {
                for (int i = 0; (i < prohibit_size) || (i < max_neighbors);
                     i++) {
                  if (prohibit[max_offset + i] == 2 * c_max + 1) {
                    hash_color = -1;  // if any element in prohibit list
                                      // conflict, reset to -1
                    continue;
                  } else
                    hash_color = c_max;
                }
                if (hash_color != -1) {
                  colors[max] = hash_color;
                  break;
                }
              }

              for (int c_min = 0; 2 * c_min + 2 <= min_color &&
                                  !util::isValid(colors[min]) && !visited[min];
                   c_min++) {
                for (int i = 0; (i < prohibit_size) || (i < min_neighbors);
                     i++) {
                  if (prohibit[min_offset + i] == 2 * c_min + 2) {
                    hash_color = -1;  // if any element in prohibit list
                                      // conflict, reset to -1
                    continue;
                  } else
                    hash_color = c_min;
                }
                if (hash_color != -1) {
                  colors[min] = hash_color;
                  break;
                }
              }
            }
            // if hash c loring fail because not enough space, fall back to
            // color by iteration
            if (!util::isValid(colors[max])) colors[max] = max_color;

            if (!util::isValid(colors[min])) colors[min] = min_color;
          };
      // color by max and min independent set, non-exactsolution
      GUARD_CU(frontier.V_Q()->ForAll(color_op, frontier.queue_length,
                                      util::DEVICE, stream));
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

      // optinal coloring by hash function n * prohibit_size (non-exact)
      if (prohibit_size != 0) {
        auto gen_op = [graph, colors, prohibit_size] __host__ __device__(
                          VertexT * prohibit_, const SizeT &pos) {
          VertexT v = pos / prohibit_size;
          SizeT a_idx = pos % prohibit_size;
          SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
          if ((a_idx < num_neighbors)) {
            SizeT e = graph.CsrT::GetNeighborListOffset(v) + a_idx;
            VertexT u = graph.CsrT::GetEdgeDest(e);
            if (util::isValid(colors[u])) prohibit_[pos] = colors[u];
          }
        };
        GUARD_CU(prohibit.ForAll(gen_op, graph.nodes * prohibit_size,
                                 util::DEVICE, stream));
        GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");
      }

      auto resolve_op =
          [graph, rand, no_conflict, colors, visited, prohibit_size] __host__
          __device__(VertexT * v_q, const SizeT &pos) {
            VertexT v = v_q[pos];
            SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

            if (util::isValid(colors[v])) {
              for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
                VertexT u = graph.CsrT::GetEdgeDest(e);
                if ((colors[u] == colors[v]) && (rand[u] >= rand[v])) {
                  if (prohibit_size != 0) visited[v] = true;
                  colors[v] = util::PreDefinedValues<VertexT>::InvalidValue;
                  // colors[v] = v + graph.nodes;
                  break;
                }
              }
            }
          };

      GUARD_CU(frontier.V_Q()->ForAll(resolve_op, frontier.queue_length,
                                      util::DEVICE, stream));
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    }

    if (test_run) {
      // reset atomic count
      GUARD_CU(colors.ForAll(
          [colored] __host__ __device__(VertexT * x, const VertexT &pos) {
            if (util::isValid(x[pos])) atomicAdd(&colored[0], 1);
          },
          graph.nodes, util::DEVICE, stream));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
      GUARD_CU(colored.Move(util::DEVICE, util::HOST));
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    }

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;
    auto user_iter = data_slice.user_iter;
    auto &graph = data_slice.sub_graph[0];
    auto test_run = data_slice.test_run;
    auto frontier = enactor_slices[0].frontier;

    // atomic based stop condition
    if (test_run && (data_slice.colored[0] >= graph.nodes)) {
      printf("Max iteration: %d\n", iter);
      return true;
    }

    // user defined stop condition
    if (!test_run && (iter == user_iter)) return true;

    return false;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
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
};  // end of colorIteration

/**
 * @brief Color enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor : public EnactorBase<typename _Problem::GraphT,
                                   typename _Problem::GraphT::VertexT,
                                   // types used for the operators,
                                   // e.g.: typename
                                   // _Problem::LabelT,
                                   typename _Problem::GraphT::ValueT,
                                   // types used for inter GPU
                                   // communication, e.g.: typename
                                   // _Problem::ValueT,
                                   ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef ColorIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief color constructor
   */
  Enactor() : BaseEnactor("Color"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief color destructor
   */
  virtual ~Enactor() { /*Release();*/
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
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of color, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        //       per element in the inter-GPU sub-frontiers
        0, 1, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    SizeT num_nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    // In this case, we add a single `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = num_nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_nodes : 0;
          if (peer_ == 0) {
            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(num_nodes, target | util::HOST);
            for (SizeT i = 0; i < num_nodes; ++i) {
              tmp[i] = (VertexT)i % num_nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                num_nodes, target, 0));

            tmp.Release();
          }
        }
      } else {
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a color computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Color Done.", this->flag & Debug);
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
