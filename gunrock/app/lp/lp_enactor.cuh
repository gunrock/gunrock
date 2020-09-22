// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * lp_enactor.cuh
 *
 * @brief LP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/lp/lp_problem.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/kernel_segmode.hxx>
#include <gunrock/util/scan_device.cuh>
#include <gunrock/util/track_utils.cuh>


namespace gunrock {
namespace app {
namespace lp {

/**
 * @brief Speciflying parameters for LP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief definition of LP iteration loop
 * @tparam EnactorT Type of enactor
 */

template <typename EnactorT>
struct LPIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT::CsrT CsrT;
  typedef typename ProblemT::GraphT::GpT GpT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;

  typedef IterationLoopBase<EnactorT,
                            Use_FullQ | Push>
      BaseIterationLoop;

  LPIterationLoop() : BaseIterationLoop() {}



  cudaError_t Gather(int peer_) {
    cudaError_t retval = cudaSuccess;

    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    cudaError_t retval = cudaSuccess;
    bool over_sized = false;
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &frontier = enactor_slice.frontier;
    auto &stream = enactor_slice.stream;
    auto &graph = this->enactor->problem->sub_graphs[this->gpu_num];

    if (((this->enactor->flag & Size_Check) == 0 &&
         ((this->flag & Skip_PreScan) != 0)))
    {
      frontier.output_length[0] = 0;
    }

    else {
      // printf("Size check runs\n");
      retval = CheckSize<SizeT, SizeT>(
          this->enactor->flag & Size_Check, "scanned_edges",
          frontier.queue_length + 2, &frontier.output_offsets, over_sized, -1,
          -1, -1, false);

      if (retval) return retval;

      GUARD_CU(oprtr::ComputeOutputLength<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), enactor_slice.oprtr_parameters));

      GUARD_CU(
          frontier.output_length.Move(util::DEVICE, util::HOST, 1, 0, stream));
    }
    return retval;
  }

  cudaError_t Check_Queue_Size(int peer_) {
    // enactor slice for this and peer_
    int k = this->gpu_num * this->enactor->num_gpus + peer_;
    auto &enactor_slice = this->enactor->enactor_slices[k];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;
    auto request_length = frontier.output_length[0] + 2;
    auto iteration = enactor_stats.iteration;
    auto &retval = enactor_stats.retval;
    auto &graph = this->enactor->problem->sub_graphs[this->gpu_num];
    bool over_sized = false;
    if (this->enactor->flag & Debug) {
      util::PrintMsg(
          "queue_size = " + std::to_string(frontier.Next_V_Q()->GetSize()) +
              "output_length = " + std::to_string(request_length),
          this->gpu_num, iteration, peer_);
    }
    if ((this->enactor->flag & Size_Check) == 0 &&
        (this->flag & Skip_PreScan) != 0) {
      frontier.output_length[0] = 0;
    } else { 
      retval = CheckSize<SizeT, VertexT>(
          true, "queue3", request_length, frontier.Next_V_Q(), over_sized,
          this->gpu_num, iteration, peer_, false);
      if (retval) return retval;
      retval = CheckSize<SizeT, VertexT>(true, "queue3", graph.nodes + 2,
                                         frontier.V_Q(), over_sized,
                                         this->gpu_num, iteration, peer_, true);
      if (retval) return retval;
    } 
    return retval;
  }

  /**
   * @brief Core computation of lp, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    static const int LOG_THREADS = 9;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &labels = data_slice.labels;
    auto &cub_temp_storage = data_slice.cub_temp_storage;
    auto &old_labels = data_slice.old_labels;
    auto &neighbour_labels = data_slice.neighbour_labels;
    auto &segments_temp = data_slice.segments_temp;
    auto &segments = data_slice.segments;
    auto &neighbour_labels_size = data_slice.neighbour_labels_size;
    auto &segments_size = data_slice.segments_size;

    // information related to the partitioned graph,
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    bool debug = ((this->enactor->flag & Debug) != 0);
    auto target = util::DEVICE;
    auto &gpu_num = this->gpu_num;

    auto &visited = data_slice.visited;
    
#if TO_TRACK
    util::PrintMsg(
        "Core queue_length = " + std::to_string(frontier.queue_length), gpu_num,
        iteration, peer_);
#endif
#ifdef RECORD_PER_ITERATION_STATS
    GpuTimer gpu_timer;
#endif
    
      frontier.queue_reset = true;
      enactor_stats.nodes_queued[0] += frontier.queue_length;

      if (debug)
        util::PrintMsg("Forward Advance begin", gpu_num, iteration, peer_);

      util::Array1D<SizeT, VertexT> *null_frontier = NULL;
      frontier.queue_length = graph.nodes;
      frontier.queue_reset = true;
  
      auto frontier_elements = frontier.V_Q();

      auto compute_op = [segments_temp, segments_size, graph] __host__
      __device__(VertexT * v, const SizeT &i) {
        
            segments_temp[i] = graph.CsrT::GetNeighborListLength(v[i]);
            atomicAdd(&segments_size[0], 1);

      };

      GUARD_CU(frontier.V_Q()->ForAll(compute_op, frontier.queue_length, target, stream));

      GUARD_CU(frontier.V_Q()->Print("Frontier: ",
                    frontier.queue_length,
                    util::DEVICE,
                    stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed.");

      GUARD_CU(util::cubExclusiveSum(cub_temp_storage, segments_temp,
        segments, frontier.queue_length , stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed.");
        
    
      auto elements = frontier.V_Q();
   
      GUARD_CU(frontier.V_Q()->ForAll(
        [segments, neighbour_labels, segments_size, neighbour_labels_size, labels, graph] __host__ __device__(
        const VertexT *v, const SizeT &index) {
          VertexT idx = v[index];
          SizeT start_edge = graph.CsrT::GetNeighborListOffset(idx);
          SizeT num_neighbors = graph.CsrT::GetNeighborListLength(idx);
          int offset = segments[index];
      
          for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {

            VertexT u = graph.CsrT::GetEdgeDest(e);   
         
            neighbour_labels[offset++] = labels[u];
            atomicAdd(&neighbour_labels_size[0], 1);

          };
        },
      frontier.queue_length, util::DEVICE, oprtr_parameters.stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed.");

      neighbour_labels_size.Move(util::DEVICE, util::HOST, 1, 0 , stream);

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
             "cudaStreamSynchronize failed.");

      int* modes = util::segmented_mode(((int*)(neighbour_labels.GetPointer(util::DEVICE))), 
                            ((int*)neighbour_labels_size.GetPointer(util::HOST))[0],
                            ((int*)(segments.GetPointer(util::DEVICE))), 
                            (int)frontier.queue_length, 
                            mgpu::less_t<int>(), 
                            *oprtr_parameters.context);
      
     
      segments.SetPointer(modes,
        (SizeT)frontier.queue_length,
        util::HOST);

      segments.Move(util::HOST, util::DEVICE, frontier.queue_length, 0 , stream);

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
             "cudaStreamSynchronize failed.");
            
      GUARD_CU(frontier.V_Q()->ForAll(
      [segments, labels, graph, old_labels] __host__ __device__(
      const VertexT *v, const SizeT &index) {         

        VertexT idx = v[index];
        
        if (segments[index] > -1){
          old_labels[idx] = labels[idx];
          labels[idx] = segments[index];
        }
      
        else {
          old_labels[idx] = labels[idx];
          labels[idx] = labels[idx];
        }
      },
      frontier.queue_length, util::DEVICE, oprtr_parameters.stream));

      auto filter_op =
          [old_labels, labels, visited] __host__ __device__(
              const VertexT &src, VertexT &dest, const SizeT &edge_id,
              const VertexT &input_item, const SizeT &input_pos,
              SizeT &output_pos) -> bool {

        if (old_labels[dest] == labels[dest]){
          return false;
        }
        else {
          bool already_added = atomicMax(visited + dest, 1) == 1;
          return !already_added;
        }
      };

      auto advance_op =
          [visited] __host__
          __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool {
                      // intentional no-op
                      return true;
                    };
      
#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Start();
#endif

      auto &work_progress = frontier.work_progress;
      auto queue_index = frontier.queue_index;

      GUARD_CU(oprtr::For(
          [work_progress, queue_index] __host__ __device__(SizeT i) {
            SizeT *counter = work_progress.GetQueueCounter(queue_index + 1);
            counter[0] = 0;
          },
          1, util::DEVICE, oprtr_parameters.stream, 1, 1));
      
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op, filter_op));
      
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters, 
        filter_op));

#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Stop();
      float elapsed = gpu_timer.ElapsedMillis();
      float mteps = frontier.output_length[0] / (elapsed * 1000);
      enactor_stats.per_iteration_advance_time.push_back(elapsed);
      enactor_stats.per_iteration_advance_mteps.push_back(mteps);
      enactor_stats.per_iteration_advance_input_edges.push_back(
          frontier.queue_length);
      enactor_stats.per_iteration_advance_output_edges.push_back(
          frontier.output_length[0]);
      enactor_stats.per_iteration_advance_direction.push_back(true);
#endif
      if (debug)
        util::PrintMsg("Forward Advance end", gpu_num, iteration, peer_);

      printf("Frontier queue length before work progress: %d\n", frontier.queue_length);

      GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));

      printf("Frontier queue length: %d\n", frontier.queue_length);
          // end of forward

  }

  cudaError_t UpdatePreds(SizeT num_elements) { return cudaSuccess; }

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
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;
    auto &labels = data_slice.labels;
    auto &masks = data_slice.visited_masks;

    auto expand_op =
        [labels, masks] __host__
        __device__(VertexT & key, const SizeT &in_pos,
                   VertexT *vertex_associate_ins,
                   ValueT *value__associate_ins) -> bool {
      SizeT mask_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      MaskT mask_bit = util::PreDefinedValues<MaskT>::InvalidValue;

      if (_ldg(labels + key) != util::PreDefinedValues<LabelT>::MaxValue)
        return false;
      
      // else
      return true;
    };

    retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of LPIteration

/**
 * @brief LP enactor class.
 * @tparam _Problem             LP problem type
 * @tparam ARRAY_FLAG           Flags for util::Array1D used in the enactor
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
  typedef LPIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief LPEnactor constructor
   */
  Enactor() : BaseEnactor("lp"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 0;
  }

  /**
   * @brief LPEnactor destructor
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
    util::Parameters &parameters = problem.parameters;

    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      // for each gpu the slice is initialised to the starting point (peer = 0)
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

    SizeT num_nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ?  num_nodes: 0;
          if (peer_ == 0) {
           GUARD_CU(frontier.V_Q()->ForAll(
            [] __host__ __device__ (VertexT * v, const SizeT &i) {
              v[i] = i;
            }, frontier.queue_length, target, 0));
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
   * @brief one run of lp, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    // so each iteration loop gets 
    // thread data 
    gunrock::app::Iteration_Loop<
        (Enactor::Problem::FLAG != 0) ? 1 : 0, 0,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a LP computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU LP Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};  // end of enactor

}  // namespace lp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
