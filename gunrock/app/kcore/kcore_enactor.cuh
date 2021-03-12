//-----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_enactor.cuh
 *
 * @brief k-core Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/kcore/kcore_problem.cuh>

//#define PRINT_DEBUG(FLAG, ID, K) (FLAG && (ID == 3366) && (K <= 129))
#define PRINT_DEBUG(FLAG, ID, K) false

namespace gunrock {
namespace app {
namespace kcore {


/**
 * @brief Specifying parameters for k-core Enactor
 * @param parameters The util::Parameter<...> structure holding all parameters
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}


/**
 * @brief definition of k-core iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct kcoreIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push>
      BaseIterationLoop;

  kcoreIterationLoop() : BaseIterationLoop() {}

  __device__ void markToDelete(util::Array1D<unsigned int, unsigned int, 0U, 0U> bitmap, VertexT vertex) {
    int intIndex = vertex / 32;   //which int contains the bit of interest?
    int bitIndex = vertex % 32;   //index of bit of interest within the int
    unsigned int mask = 1 << (31 - bitIndex);
    unsigned int orResult = atomicOr(&bitmap[intIndex], mask);
  }

  __device__ bool checkVertexRemains(util::Array1D<unsigned int, unsigned int, 0U, 0U> bitmap, VertexT vertex) {
    int intIndex = vertex / 32;   //which int contains the bit of interest?
    int bitIndex = vertex % 32;   //index of bit of interest within the int
    unsigned int mask = 1 << (31 - bitIndex);
    bool andResult = (bitmap[intIndex] & mask) == 0;
    return andResult;
  }

  /**
   * @brief Core computation of k-core, one iteration
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

    util::Parameters &parameters = this->enactor->problem->parameters;
    bool quiet = parameters.Get<bool>("quiet");

    auto &degrees = data_slice.degrees;
    auto &k_cores = data_slice.k_cores;
    auto &initial_frontier = data_slice.initial_frontier;
    auto &delete_bitmap = data_slice.delete_bitmap;
    auto &to_be_deleted_bitmap = data_slice.to_be_deleted_bitmap;

    auto k = iteration + 1;

    auto &empty = data_slice.empty_flag;

    frontier.queue_reset = true;

    // --
    // Define operations

      auto advance_op = [degrees, k_cores, k, quiet, delete_bitmap, to_be_deleted_bitmap, this
      ] __device__(const VertexT &src, VertexT &dest,
                            const SizeT &edge_id, const VertexT &input_item,
                            const SizeT &input_pos, SizeT &output_pos) -> bool {

        if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
          printf("advance: src %u, dest %u, edgeID %u, input item %u, input pos %u, output pos %u\n, degrees %u, kcores %u\n", src, dest, edge_id, input_item, input_pos, output_pos, degrees[src], k_cores[src]);
        }

        if (!checkVertexRemains(delete_bitmap, src)) {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("vertex %u deleted\n", src);
          }
            return false;
        }

        if(degrees[src] > k) {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("deg(vertex) = %u > k (%u) for %u.\n", degrees[src], k, src);
          }
          return false;
        }

        else {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("deg(vertex) = %u < k (%u) for %u.\n", degrees[src], k, src);
          }
          k_cores[src] = k;
          markToDelete(to_be_deleted_bitmap, src);
	  if(!checkVertexRemains(delete_bitmap, dest)){
	    return false;
	  }
          return true;
        }

      };

      auto filter_op = [degrees, k_cores, k, quiet, delete_bitmap, this
      ] __device__(const VertexT &src, VertexT &dest,
                            const SizeT &edge_id, const VertexT &input_item,
                            const SizeT &input_pos, SizeT &output_pos) -> bool {

        if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
          printf("filter: src %u, dest %u, edgeID %u, input item %u, input pos %u, output pos %u\n, degrees %u, kcores %u\n", src, dest, edge_id, input_item, input_pos, output_pos, degrees[dest], k_cores[dest]);
        }

        if (!checkVertexRemains(delete_bitmap, dest)) {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("vertex %u has been removed\n", dest);
          }
          return false;
        }

        int oldDegrees = atomicSub(degrees + dest, 1);

        if (oldDegrees != (k + 1)) {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("vertex %u degree = %u\n", dest, (oldDegrees - 1));
          }
          return false;
        }

        else {
          if (PRINT_DEBUG(!quiet, src, k) || PRINT_DEBUG(!quiet, dest, k)) {
            printf("vertex %u degree = %u\n", dest, (oldDegrees - 1));
          }
          return true;
        }

      };

    // --
    // Run

    //Fill frontier with all vertices in graph for next iteration
    SizeT num_nodes = graph.nodes;
    frontier.queue_length = num_nodes;
    GUARD_CU(frontier.V_Q()->ForEach(
      initial_frontier,
      [] __host__ __device__(VertexT & v, VertexT & i) { v = i; 
      }, num_nodes, util::DEVICE, 0));

    while(true) {

      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
      if (!quiet) {
        printf("frontier queue length: %u\n", frontier.queue_length);
        printf("advancing...\n");
      }

      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op));

      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

      frontier.queue_reset = true;
      GUARD_CU(frontier.work_progress.GetQueueLength(
               frontier.queue_index, frontier.queue_length, false,
               oprtr_parameters.stream, true));

      //Mark vertices as deleted
      GUARD_CU(delete_bitmap.ForAll(
        [delete_bitmap, to_be_deleted_bitmap] __host__ __device__(unsigned int * x, const SizeT &pos) {
          delete_bitmap[pos] = delete_bitmap[pos] | to_be_deleted_bitmap[pos]; },
          (graph.nodes/32 + 1), util::DEVICE, oprtr_parameters.stream));
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

      if (!quiet) {
        printf("frontier queue length: %u\n", frontier.queue_length);
        printf("filtering...\n");
      }

      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        filter_op));

      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

      frontier.queue_reset = true;
      GUARD_CU(frontier.work_progress.GetQueueLength(
               frontier.queue_index, frontier.queue_length, false,
               oprtr_parameters.stream, true));
      if(frontier.queue_length == 0) break;
 
    }

    if (!quiet) {
      printf("after iteration %u, frontier queue length: %u\n", iteration, frontier.queue_length);
    }

    //Compute number of vertices remaining in graph
    GUARD_CU(empty.ForAll(
        [] __host__ __device__(SizeT * x, const VertexT &pos) { x[pos] = 1; },
        1, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

    GUARD_CU(delete_bitmap.ForAll(
      [delete_bitmap, empty] __host__ __device__(SizeT * x, const SizeT &pos) {
        if(delete_bitmap[pos] != UINT_MAX){
          empty[0] = 0;
        }
      },
      (graph.nodes/32 + 1), util::DEVICE, oprtr_parameters.stream));

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

    return retval;
  } //end of core

  bool Stop_Condition(int gpu_num = 0) {

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &empty = data_slice.empty_flag;

    util::Parameters &parameters = this->enactor->problem->parameters;
    bool quiet = parameters.Get<bool>("quiet");

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
    GUARD_CU(empty.Move(util::DEVICE, util::HOST));
    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

    if(empty[0] == 1) {
      if (!quiet) printf("all vertices have been deleted.\n");
      return true;
    }

    if(!quiet) printf("Vertices remaining. Continue.\n");

    /*auto &iteration = enactor_stats.iteration;
    if(!quiet) {
      printf("Vertices remaining. Continue.\n");
      auto &delete_bitmap = data_slice.delete_bitmap;
      GUARD_CU(delete_bitmap.Move(util::DEVICE, util::HOST));
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
      auto &k_cores = data_slice.k_cores;
      GUARD_CU(k_cores.Move(util::DEVICE, util::HOST));
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
      auto &degrees = data_slice.degrees;
      GUARD_CU(degrees.Move(util::DEVICE, util::HOST));
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
      for (SizeT i = 0; i < (graph.nodes/32) + 1; i++) {
        if(delete_bitmap[i] != UINT_MAX){
          printf("bitmap[%u]: %X\n", i, delete_bitmap[i]);
	  for (SizeT vertex = (i * 32); vertex < ((i + 1) * 32); vertex++){
	    printf("k-core[%llu]: %llu\t", vertex, k_cores[vertex]);
	    printf("degree[%llu]: %llu\t", vertex, degrees[vertex]);
	  }
	  printf("\n");
        }
      }
    }*/

    frontier.queue_length = graph.nodes;

    return false;
  } //end of stop condition

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

    auto expand_op = [
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of kcoreIteration

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
  typedef kcoreIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief k-core constructor
   */
  Enactor() : BaseEnactor("k-core"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief k-core destructor
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
    GUARD_CU(BaseEnactor::Init(
        problem, Enactor_None,
        2, NULL,
        target, false));
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
   * @brief one run of k-core, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        0, 1,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
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

    //Initialize frontier:
    auto &data_slice = this->problem->data_slices[0][0];
    auto &initial_frontier = data_slice.initial_frontier;
    auto &bitmap = data_slice.delete_bitmap;
    auto &degrees = data_slice.degrees;
    SizeT num_nodes = this->problem->sub_graphs[0].nodes;
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1)) {
        this->thread_slices[gpu].init_size = num_nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_nodes : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                initial_frontier,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; 
                }, num_nodes, target, 0));
          }
        }
      } else {
        printf("No multi-GPU implementation.\n");
      }
    }

    //Initialize deleted flags
    unsigned int overflowBitmask = UINT_MAX >> (num_nodes % 32);
    GUARD_CU(bitmap.ForAll(
      [degrees, overflowBitmask, num_nodes] __host__ __device__(SizeT * x, const VertexT &pos) { 
        unsigned int mask = 0u;
        for(int i = 0; i < 32; i++) {
          unsigned int vertexID = (32 * pos) + i;
          if(vertexID < num_nodes) {
            if(degrees[(32 * pos) + i] == 0) {
              mask |= (1 << (31 - i));
            }
          }
        }
	if(pos == num_nodes/32) mask |= overflowBitmask;
        x[pos] = mask; },
      ((num_nodes/32) + 1), util::DEVICE, 0));

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a k-core computation on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Template Done.", this->flag & Debug);
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
