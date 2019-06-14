// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_enactor.cuh
 *
 * @brief RW Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/rw/rw_problem.cuh>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

namespace gunrock {
namespace app {
namespace rw {

/**
 * @brief Speciflying parameters for RW Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of RW iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct RWIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  RWIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of RW, one iteration
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

    // problem specific data alias:
    auto &walks = data_slice.walks;
    auto &rand = data_slice.rand;
    auto &walk_length = data_slice.walk_length;
    auto &walks_per_node = data_slice.walks_per_node;
    auto &walk_mode = data_slice.walk_mode;
    auto &store_walks = data_slice.store_walks;
    auto &gen = data_slice.gen;
    auto &neighbors_seen = data_slice.neighbors_seen;
    auto &steps_taken = data_slice.steps_taken;

    if (walk_mode == 0) {  // uniform random walk

      auto uniform_rw_op =
          [graph, walks, rand, iteration, walk_length, store_walks,
           neighbors_seen, steps_taken] __host__
          __device__(VertexT * v, const SizeT &i) {
            SizeT write_idx =
                (i * walk_length) + iteration;  // Write location in RW array
            if (store_walks) {
              walks[write_idx] = v[i];  // record current position in walk
            }

            if (!util::isValid(v[i])) {
              return;
            }

            if (iteration < walk_length - 1) {
              SizeT num_neighbors = graph.GetNeighborListLength(v[i]);
              if (num_neighbors == 0) {
                v[i] = util::PreDefinedValues<VertexT>::InvalidValue;
                return;
              }

              // Randomly sample neighbor
              SizeT neighbor_list_offset = graph.GetNeighborListOffset(v[i]);
              SizeT rand_offset =
                  (SizeT)round(0.5 + num_neighbors * rand[i]) - 1;
              VertexT neighbor =
                  graph.GetEdgeDest(neighbor_list_offset + rand_offset);

              v[i] = neighbor;  // Replace vertex w/ neighbor in queue
              steps_taken[i]++;
              neighbors_seen[i] += (uint64_t)
                  num_neighbors;  // Record number of neighbors we've seen
            }
          };

      curandSetStream(gen, oprtr_parameters.stream);
      curandGenerateUniform(gen, rand.GetPointer(util::DEVICE),
                            graph.nodes * walks_per_node);
      GUARD_CU(frontier.V_Q()->ForAll(uniform_rw_op, frontier.queue_length,
                                      util::DEVICE, oprtr_parameters.stream));

    } else if (walk_mode ==
               1) {  // greedy: walk to neighbor w/ maximum node value
      auto greedy_rw_op =
          [graph, walks, iteration, walk_length, store_walks, neighbors_seen,
           steps_taken] __host__
          __device__(VertexT * v, const SizeT &i) {
            SizeT write_idx =
                (i * walk_length) + iteration;  // Write location in RW array
            if (store_walks) {
              walks[write_idx] = v[i];  // record current position in walk
            }

            if (!util::isValid(v[i])) {
              return;
            }

            if (iteration < walk_length - 1) {
              SizeT num_neighbors = graph.GetNeighborListLength(v[i]);
              if (num_neighbors == 0) {
                v[i] = util::PreDefinedValues<VertexT>::InvalidValue;
                return;
              }

              SizeT neighbor_list_offset = graph.GetNeighborListOffset(v[i]);

              // Find neighbor with max value
              VertexT max_neighbor_id =
                  graph.GetEdgeDest(neighbor_list_offset + 0);
              ValueT max_neighbor_val = graph.node_values[max_neighbor_id];
              for (SizeT offset = 1; offset < num_neighbors; offset++) {
                VertexT neighbor =
                    graph.GetEdgeDest(neighbor_list_offset + offset);
                ValueT neighbor_val = graph.node_values[neighbor];
                if (neighbor_val > max_neighbor_val) {
                  max_neighbor_id = neighbor;
                  max_neighbor_val = neighbor_val;
                }
              }
              v[i] = max_neighbor_id;  // Replace vertex w/ neighbor in queue
              steps_taken[i]++;
              neighbors_seen[i] += (uint64_t)
                  num_neighbors;  // Record number of neighbors we've seen
            }
          };

      GUARD_CU(frontier.V_Q()->ForAll(greedy_rw_op, frontier.queue_length,
                                      util::DEVICE, oprtr_parameters.stream));

    } else if (walk_mode == 2) {
      curandGenerateUniform(gen, rand.GetPointer(util::DEVICE),
                            graph.nodes * walks_per_node);

      auto stochastic_greedy_rw_op =
          [graph, walks, rand, iteration, walk_length, store_walks,
           neighbors_seen, steps_taken] __host__
          __device__(VertexT * v, const SizeT &i) {
            SizeT write_idx =
                (i * walk_length) + iteration;  // Write location in RW array
            if (store_walks) {
              walks[write_idx] = v[i];  // record current position in walk
            }

            if (!util::isValid(v[i])) {
              return;
            }

            if (iteration < walk_length - 1) {
              SizeT num_neighbors = graph.GetNeighborListLength(v[i]);
              if (num_neighbors == 0) {
                v[i] = util::PreDefinedValues<VertexT>::InvalidValue;
                return;
              }

              SizeT neighbor_list_offset = graph.GetNeighborListOffset(v[i]);

              VertexT neighbor, next_neighbor;

              ValueT sum_neighbor_scores = 0;
              for (SizeT offset = 0; offset < num_neighbors; offset++) {
                neighbor = graph.GetEdgeDest(neighbor_list_offset + offset);
                sum_neighbor_scores += graph.node_values[neighbor];
              }

              ValueT r = rand[i] * sum_neighbor_scores;

              ValueT acc = 0;
              for (SizeT offset = 0; offset < num_neighbors; offset++) {
                neighbor = graph.GetEdgeDest(neighbor_list_offset + offset);
                ValueT neighbor_score = graph.node_values[neighbor];
                acc += neighbor_score;
                if (r < acc) {
                  next_neighbor = neighbor;
                  break;
                }
              }
              v[i] = next_neighbor;  // Replace vertex w/ neighbor in queue
              steps_taken[i]++;
              neighbors_seen[i] += (uint64_t)
                  num_neighbors;  // Record number of neighbors we've seen
            }
          };

      GUARD_CU(frontier.V_Q()->ForAll(stochastic_greedy_rw_op,
                                      frontier.queue_length, util::DEVICE,
                                      oprtr_parameters.stream));

    } else {
      printf("ERROR: unknown walk_mode=%d\n", walk_mode);
    }

    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    return cudaSuccess;  // No need to load balance or get output size
  }
  cudaError_t Check_Queue_Size(int peer_) {
    return cudaSuccess;  // no need to check queue size for RW
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;

    return iter == data_slice.walk_length;
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
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
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
};  // end of RWIteration

/**
 * @brief Template enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
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
  typedef RWIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief RW constructor
   */
  Enactor() : BaseEnactor("RW"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief RW destructor
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
    // !! POSSIBLE BUG: @sgpyc suggested changing the 2 to 1, but that causes
    //  strange behavior, where V_Q does not get initialized properly.
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
   * @brief one run of RW, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 0, IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(int walks_per_node, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    SizeT num_nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = num_nodes * walks_per_node;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_nodes * walks_per_node : 0;
          if (peer_ == 0) {
            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(num_nodes * walks_per_node, target | util::HOST);
            for (SizeT i = 0; i < num_nodes * walks_per_node; ++i) {
              tmp[i] = (VertexT)i % num_nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                num_nodes * walks_per_node, target, 0));

            tmp.Release();
          }
        }
      } else {
        // MULTIGPU INCOMPLETE
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a RW computing on the specified graph.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU RW Done.", this->flag & Debug);
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
