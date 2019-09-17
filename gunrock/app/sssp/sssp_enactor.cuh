// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of SSSP iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SSSPIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push |
                                             (((EnactorT::Problem::FLAG &
                                                Mark_Predecessors) != 0)
                                                  ? Update_Predecessors
                                                  : 0x0)> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push |
                                          (((EnactorT::Problem::FLAG &
                                             Mark_Predecessors) != 0)
                                               ? Update_Predecessors
                                               : 0x0)>
      BaseIterationLoop;

  SSSPIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of sssp, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data sssp that works on
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &distances = data_slice.distances;
    auto &labels = data_slice.labels;
    auto &preds = data_slice.preds;
    // auto         &row_offsets        =   graph.CsrT::row_offsets;
    auto &weights = graph.CsrT::edge_values;
    auto &original_vertex = graph.GpT::original_vertex;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    // auto         &stream             =   enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;

    // The advance operation
    auto advance_op =
        [distances, weights, original_vertex, preds] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      ValueT src_distance = Load<cub::LOAD_CG>(distances + src);
      ValueT edge_weight = Load<cub::LOAD_CS>(weights + edge_id);
      ValueT new_distance = src_distance + edge_weight;

      // Check if the destination node has been claimed as someone's child
      ValueT old_distance = atomicMin(distances + dest, new_distance);

      if (new_distance < old_distance) {
        if (!preds.isEmpty()) {
          VertexT pred = src;
          if (!original_vertex.isEmpty()) pred = original_vertex[src];
          Store(preds + dest, pred);
        }
        return true;
      }
      return false;
    };

    // The filter operation
    auto filter_op = [labels, iteration] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos, SizeT &output_pos) -> bool {
      if (!util::isValid(dest)) return false;
      if (labels[dest] == iteration) return false;
      labels[dest] = iteration;
      return true;
    };

    oprtr_parameters.label = iteration + 1;
    // Call the advance operator, using the advance operation
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op, filter_op));

    if (oprtr_parameters.advance_mode != "LB_CULL" &&
        oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
      frontier.queue_reset = false;
      // Call the filter operator, using the filter operation
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          filter_op));
    }

    // Get back the resulted frontier length
    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));

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
    auto iteration = enactor_slice.enactor_stats.iteration;
    auto &distances = data_slice.distances;
    auto &labels = data_slice.labels;
    auto &preds = data_slice.preds;
    auto label = this->enactor->mgpu_slices[this->gpu_num]
                     .in_iteration[iteration % 2][peer_];

    auto expand_op = [distances, labels, label, preds] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      ValueT in_val = value__associate_ins[in_pos];
      ValueT old_val = atomicMin(distances + key, in_val);
      if (old_val <= in_val) return false;
      if (labels[key] == label) return false;
      labels[key] = label;
      if (!preds.isEmpty()) preds[key] = vertex_associate_ins[in_pos];
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of SSSPIteration

/**
 * @brief SSSP enactor class.
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
  typedef SSSPIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief SSSPEnactor constructor
   */
  Enactor() : BaseEnactor("sssp"), problem(NULL) {
    this->max_num_vertex_associates =
        (Problem::FLAG & Mark_Predecessors) != 0 ? 1 : 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SSSPEnactor destructor
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

      for (int peer = 0; peer < this->num_gpus; peer++) {
        this->enactor_slices[gpu * this->num_gpus + peer]
            .oprtr_parameters.labels = &(problem.data_slices[gpu]->labels);
      }
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
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [src] __host__ __device__(VertexT & v) { v = src; }, 1, target,
                0));
          }
        }
      }

      else {
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
   * @brief one run of sssp, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0, 1,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a SSSP computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU SSSP Done.", this->flag & Debug);
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
