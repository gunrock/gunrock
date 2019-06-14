// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sm_enactor.cuh
 *
 * @brief SM Problem Enactor
 */

#pragma once

#include <gunrock/util/select_device.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief Speciflying parameters for Subgraph Matching Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of SM iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SMIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;

  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  SMIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of sm, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data sn works on
    auto &enactor = this->enactor[0];
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &subgraphs = data_slice.subgraphs;
    auto &constrain = data_slice.constrain;
    auto &isValid = data_slice.isValid;
    auto &NG = data_slice.NG;
    auto &NG_src = data_slice.NG_src;
    auto &NG_dest = data_slice.NG_dest;
    auto &query_ro = data_slice.query_ro;
    auto &query_ci = data_slice.query_ci;
    auto &counter = data_slice.counter;
    auto &flags = data_slice.flags;
    auto &indices = data_slice.indices;
    auto &partial = data_slice.partial;
    auto &num_subs = data_slice.num_subs;
    auto &results = data_slice.results;
    auto &row_offsets = graph.CsrT::row_offsets;
    auto &col_indices = graph.CsrT::column_indices;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = oprtr_parameters.stream;
    auto target = util::DEVICE;
    size_t nodes_query = data_slice.nodes_query;
    size_t nodes_data = graph.nodes;

    auto print_frontier = [] __host__ __device__(VertexT * v_q,
                                                 const SizeT &pos) {
      VertexT v = v_q[pos];
      printf("Frontier @ %u = %u\n", pos, v);
      return;
    };

    //	GUARD_CU(frontier.V_Q()->ForAll(print_frontier , frontier.queue_length,
    //util::DEVICE, stream));
    //	GUARD_CU(frontier.Next_V_Q()->ForAll(print_frontier ,
    //frontier.queue_length, util::DEVICE, stream));

    // First add degrees to subgraph matching
    GUARD_CU(subgraphs.ForAll(
        [row_offsets] __host__ __device__(VertexT * subgraphs_,
                                          const SizeT &v) {
          subgraphs_[v] = row_offsets[v + 1] - row_offsets[v];
        },
        graph.nodes, target, stream));
    // advance to filter out data graph nodes which don't satisfy constrain
    auto advance_op = [subgraphs, constrain, isValid] __host__ __device__(
                          const VertexT &src, VertexT &dest,
                          const SizeT &edge_id, const VertexT &input_item,
                          const SizeT &input_pos, SizeT &output_pos) -> bool {
      if (isValid[src]) {
        if (subgraphs[src] >= constrain[0]) {
          return true;
        } else {
          isValid[src] = false;
          atomicAdd(subgraphs + dest, -1);
        }
      }
      return false;
    };
    auto filter_op = [subgraphs, nodes_query] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos, SizeT &output_pos) -> bool {
      if (!util::isValid(dest)) {
        return false;
      }
      return true;
    };

    auto distribute_op =
        [subgraphs, isValid, NG, query_ro, query_ci, flags, counter, NG_src,
         NG_dest, results, nodes_data] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   SizeT &output_pos) -> bool {
      if (src < 0 || src >= nodes_data) return false;
      if ((!isValid[src]) || (!isValid[dest])) {
        return false;
      }
      // NG has query node id sequence in its even pos; min degree of neighbors
      // in odd pos
      VertexT query_id = NG[counter[0] * 2];
      // special init for first iteration
      if (counter[0] == 0) {
        if (subgraphs[src] < (query_ro[query_id + 1] - query_ro[query_id]))
          return false;
        // 1 way look ahead
        if (subgraphs[dest] < NG[counter[0] * 2 + 1]) {
          return false;
        }
        flags[src] = true;
        return true;
      } else {
        // check if src belongs to partial results
        if (!flags[src]) return false;
        if (subgraphs[dest] < (query_ro[query_id + 1] - query_ro[query_id]))
          return false;
        // 1 way look ahead to be done in filter
        results[dest] = 1;
        return true;
      }
    };
    auto look_ahead_op = [isValid, flags, nodes_data] __host__ __device__(
                             const VertexT &src, VertexT &dest,
                             const SizeT &edge_id, const VertexT &input_item,
                             const SizeT &input_pos,
                             SizeT &output_pos) -> bool {
      if (src < 0 || src >= nodes_data) return false;
      if ((!isValid[src]) || (!isValid[dest])) {
        return false;
      }
      flags[src] = true;
      return true;
    };
    // Compute number of triangles for each edge and atomicly add the count to
    // each node, then divided by 2 The intersection operation
    auto intersect_op = [subgraphs] __host__ __device__(
                            VertexT & comm_node, VertexT & edge) -> bool {
      atomicAdd(subgraphs + comm_node, 1);
      return true;
    };

    frontier.queue_length = graph.edges;
    frontier.queue_reset = true;
    size_t pointer_head = 0;
    int num_init = nodes_query;
    for (int iter = 0; iter < num_init; ++iter) {
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          advance_op));
    }

    for (int iter = 0; iter < nodes_query; ++iter) {
      GUARD_CU(counter.ForAll(
          [iter] __host__ __device__(VertexT * counter_, const SizeT &v) {
            counter_[v] = iter;
          },
          1, target, stream));
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          distribute_op));

      if (iter == 0) {
      } else {
        // Initialize flagss to be composed of 0
        GUARD_CU(flags.ForAll(
            [] __device__(bool *flags_, const SizeT &v) { flags_[v] = false; },
            graph.nodes, target, stream));

        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.Next_V_Q(), frontier.V_Q(), oprtr_parameters,
            look_ahead_op));
      }
      GUARD_CU(util::CUBSelect_flagged(
          indices.GetPointer(util::DEVICE), flags.GetPointer(util::DEVICE),
          partial.GetPointer(util::DEVICE) + pointer_head,
          num_subs.GetPointer(util::DEVICE), nodes_data));
      GUARD_CU(num_subs.Move(util::DEVICE, util::HOST));
      pointer_head += num_subs.GetPointer(util::HOST)[0];
    }

    return retval;
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
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool { return true; };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    return cudaSuccess;  // No need to load balance or get output size
  }
  cudaError_t Check_Queue_Size(int peer_) {
    return cudaSuccess;  // no need to check queue size for RW
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &enactor_slice = this->enactor->enactor_slices[0];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &iter = enactor_stats.iteration;
    if (iter == 1)
      return true;
    else
      return false;
  }
};  // end of SMIteration

/**
 * @brief SM enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::LabelT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::LabelT LabelT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef SMIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief SMEnactor constructor
   */
  Enactor() : BaseEnactor("sm"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SMEnactor destructor
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
   * @brief Initialize the enactor.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

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
   * @brief one run of ss, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // TODO: change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 1, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(SizeT num_srcs, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    typedef typename GraphT::CsrT CsrT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1))
      //|| (gpu == this->problem->org_graph->GpT::partition_table[src]))
      {
        this->thread_slices[gpu].init_size = num_srcs;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_srcs : 0;
          if (peer_ == 0) {
            auto &graph = this->problem->sub_graphs[gpu];
            util::Array1D<SizeT, VertexT> tmp_srcs;
            tmp_srcs.Allocate(num_srcs, target | util::HOST);
            int pos = 0;
            for (SizeT i = 0; i < graph.nodes; ++i) {
              for (SizeT j = graph.CsrT::row_offsets[i];
                   j < graph.CsrT::row_offsets[i + 1]; ++j) {
                tmp_srcs[pos++] = i;
              }
            }
            GUARD_CU(tmp_srcs.Move(util::HOST, target));
            GUARD_CU(frontier.V_Q()->EnsureSize_(graph.edges, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp_srcs,
                [] __host__ __device__(VertexT & v, VertexT & src) { v = src; },
                num_srcs, target, 0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a SM computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU SM Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
