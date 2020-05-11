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
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &subgraphs = data_slice.subgraphs;
    auto &constrain = data_slice.constrain;
    auto &isValid = data_slice.isValid;
    auto &NS = data_slice.NS;
    auto &NN = data_slice.NN;
    auto &NT = data_slice.NT;
    auto &NT_offset = data_slice.NT_offset;
    auto &query_ro = data_slice.query_ro;
    auto &query_ci = data_slice.query_ci;
    auto &counter = data_slice.counter;
    auto &temp_count = data_slice.temp_count;
    auto &indices = data_slice.indices;
    auto &results = data_slice.results;
    auto &flags_write = data_slice.flags_write;
    auto &row_offsets = graph.CsrT::row_offsets;
    auto &col_indices = graph.CsrT::column_indices;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = oprtr_parameters.stream;
    auto target = util::DEVICE;
    size_t nodes_query = data_slice.nodes_query;
    unsigned long nodes_data = graph.nodes;
    unsigned long edges_data = graph.edges;
    unsigned long mem_limit = nodes_data * nodes_data;

    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    auto complete_graph = null_frontier;

    // Store data graph degrees to subgraphs
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
    };  // advance_op
    auto prune_op =
        [subgraphs, row_offsets, col_indices, isValid, NS, NN, NT, NT_offset,
         query_ro, query_ci, flags_write, counter, results, temp_count,
         nodes_data, nodes_query, mem_limit] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   SizeT &output_pos) -> bool {
      if (src < 0 || src >= nodes_data) return false;
      if ((!isValid[src]) || (!isValid[dest])) return false;
      // NS has query node id sequence from pos 0 to pos nodes_query - 1; min
      // degree of neighbors from pos nodes_query to end
      VertexT query_id = NS[counter[0]];  // node id of current query node
      SizeT min_degree = NS[counter[0] + nodes_query];
      int nn = NN[counter[0]];  // pos of previously visited neighbor in NS
      unsigned long n = nodes_data;
      // first iteration (counter[0] = 0), src nodes are candidates
      if (nn == -1) {
        // check candidates' degrees
        if (subgraphs[src] < (query_ro[query_id + 1] - query_ro[query_id]))
          return false;
        // 1 way look ahead
        if (subgraphs[dest] < min_degree) {
          return false;
        }
        flags_write[src] = true;
        return true;
      }
      // flags represent all possible node combinations in each iteration
      // with no consideration of repeating nodes

      // each flag's pos represent a combination of nodes
      // we calculate the node ids based on the value of pos
      // The largest pos is n ^ counter - 1

      // Check NN
      int total = 1;
      for (int i = 0; i < counter[0]; ++i) {
        total = total * n;
      }
      int stride_src = 1;
      for (int i = nn + 1; i < counter[0]; ++i) {
        stride_src = stride_src * n;
      }
      int combination[50];  // 50 is the largest nodes_query value
      for (unsigned long i = 0; i < temp_count[0]; ++i) {
        // src is not at nn pos of current combination
        if (src != (results[i] / stride_src) % n) continue;
        // src satisfies, later iterations counter[0] > 0, dest nodes are
        // candidates check candidates' degrees
        if (subgraphs[dest] < (query_ro[query_id + 1] - query_ro[query_id]))
          continue;
        // printf("src: %d, dest:%d, results:%d, dest degree: %d, query
        // degree:%d\n", src, dest, results[i], subgraphs[dest],
        // (query_ro[query_id + 1] - query_ro[query_id]));

        // Fill combination with current result[i]'s representation
        int stride = total;
        int temp = results[i];
        int j = 0;
        for (j = 0; j < counter[0]; ++j) {
          stride = stride / n;
          combination[j] = temp / stride;
          temp = temp - combination[j] * stride;
        }
        // First check: check if dest is duplicated with any of the member
        for (j = 0; j < counter[0]; ++j) {
          if (dest == combination[j]) break;
        }
        if (j < counter[0]) continue;  // dest is a duplicate, aborted

        // Second check: check if dest has any matched NT
        int k = 0;
        for (k = NT_offset[counter[0]]; k < NT_offset[counter[0] + 1]; ++k) {
          int nt = NT[k];  // non-tree edge's other node pos in NS
          // check if dest is connected to nt's node in combination
          int nt_node = combination[nt];
          int offset = 0;
          for (offset = row_offsets[dest]; offset < row_offsets[dest + 1];
               ++offset) {
            if (nt_node == col_indices[offset]) break;
          }
          if (offset >=
              row_offsets[dest + 1]) {  // dest has no neighbor nt_node
            break;  // dest doesn't satisfy nt node connections
          }
        }
        if (k <
            NT_offset[counter[0] + 1]) {  // dest doesn't satisfy all NT edges
          continue;
        }
        // Checks finished, add dest to combination and write to new flags pos
        unsigned long pos = (unsigned long)i * nodes_data + (unsigned long)dest;
        if (pos >= mem_limit) {
          continue;
        }
        flags_write[pos] = true;
      }
      return false;
    };  // prune_op

    // first iteration, filter by basic constrain, and update valid degree,
    // could run multiple iterations to do more filter
    for (int iter = 0; iter < 1; ++iter) {
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), complete_graph, complete_graph, oprtr_parameters,
          advance_op));
    }
    unsigned long total = 1;
    for (int iter = 0; iter < nodes_query; ++iter) {
      // set counter to be equal to iter
      GUARD_CU(counter.ForAll(
          [iter] __host__ __device__(VertexT * counter_, const SizeT &v) {
            counter_[v] = iter;
          },
          1, target, stream));
      // First iteration
      if (iter == 0) {
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), complete_graph, complete_graph, oprtr_parameters,
            prune_op));
      } else {
        // total is the largest combination value this iteration could have
        total = total * nodes_data;
        GUARD_CU(flags_write.ForAll(
            [] __device__(bool *x, const unsigned long &pos) {
              x[pos] = false;
            },
            mem_limit, target, stream));

        // Second and later iterations
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), complete_graph, complete_graph, oprtr_parameters,
            prune_op));
        GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");

        GUARD_CU(temp_count.Move(util::DEVICE, util::HOST));
        // Update indices and reset results for compression
        unsigned long size = min(temp_count[0] * nodes_data, mem_limit);
        GUARD_CU(indices.ForAll(
            [results, nodes_data] __device__(unsigned long *x,
                                             const unsigned long &pos) {
              x[pos] =
                  results[pos / nodes_data] * nodes_data + (pos % nodes_data);
            },
            size, target, stream));
        GUARD_CU(results.ForAll(
            [] __host__ __device__(unsigned long *x, const unsigned long &pos) {
              x[pos] = 0;
            },
            mem_limit, target, stream));
      }

      GUARD_CU(util::CUBSelect_flagged(indices.GetPointer(util::DEVICE),
                                       flags_write.GetPointer(util::DEVICE),
                                       results.GetPointer(util::DEVICE),
                                       temp_count.GetPointer(util::DEVICE),
                                       mem_limit));
      // counter.Print();
      // indices.Print();
      // flags_write.Print();
      // results.Print();
      // temp_count.Print();
    }  // results and temp_count contains final results

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
  cudaError_t Reset(util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;

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
        // MULTI_GPU INCOMPLETE
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