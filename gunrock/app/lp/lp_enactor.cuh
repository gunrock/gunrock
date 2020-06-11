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
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/lp/lp_problem.cuh>
// #include <gunrock/app/lp/bfs_app.cu>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/track_utils.cuh>
#include <gunrock/app/lp/lp_kernel.cuh>
#include <gunrock/util/scan_device.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/array_utils.cuh>
namespace gunrock {
namespace app {
namespace lp {

/**
 * @brief Speciflying parameters for BFS Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  // both push and pull label propagation doesn't care about multiple operations being carried out
  // but in push we will be writing a lot of times and that means its a lot of 
  // atomic writes
  // thus we can disable this initially and then enable it for pull communications
  GUARD_CU(parameters.Use<bool>(
      "idempotence",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to enable idempotence optimization", __FILE__, __LINE__));

    
  // optional argument as user can choose to not supply
  // however its a required field, and by default we do pull?
  // makes sense to do push as we will be moving from push to pull
  GUARD_CU(parameters.Use<bool>(
      "pull",
      util::OPTIONAL_ARGUMENT | util::REQUIRED_PARAMETER,
      false, "Whether to enable direction optimizing BFS", __FILE__, __LINE__));

  // will be needing a few variables to decide between quality - time tradeoff?
  // to switch from pull to push

  GUARD_CU(parameters.Use<float>(
      "do-a",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.001, "Threshold to switch from forward-push to backward-pull in DOBFS",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "do-b",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.200, "Threshold to switch from backward-pull to forward-push in DOBFS",
      __FILE__, __LINE__));

  return retval;
}

/**
 * @brief defination of BFS iteration loop
 * @tparam EnactorT Type of enactor
 */

 // does it not have to be Push/Pull flag here?
template <typename EnactorT>
struct LPIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push | 0x0> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT::CsrT CsrT;
  typedef typename ProblemT::GraphT::GpT GpT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;

  typedef IterationLoopBase<EnactorT,
                            Use_FullQ | Push |
                                (((ProblemT::FLAG & Mark_Predecessors) != 0)
                                     ? Update_Predecessors
                                     : 0x0)>
      BaseIterationLoop;

  LPIterationLoop() : BaseIterationLoop() {}



  cudaError_t Gather(int peer_) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &graph = data_slice.sub_graph[0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;

    data_slice.num_visited_vertices += frontier.queue_length;
    data_slice.num_unvisited_vertices =
        graph.nodes - data_slice.num_visited_vertices;
    float predicted_backward_visits =
        (data_slice.num_visited_vertices == 0)
            ? (std::numeric_limits<float>::infinity())
            : (data_slice.num_unvisited_vertices * 1.0 * graph.nodes /
               data_slice.num_visited_vertices);
    float predicted_forward_visits =
        frontier.queue_length * 1.0 * graph.edges / graph.nodes;

    auto iteration_ = enactor_stats.iteration % 4;
    if (this->enactor->direction_optimized) {
      if (data_slice.previous_direction == FORWARD) {
        if (predicted_forward_visits >
                predicted_backward_visits * this->enactor->do_a &&
            !data_slice.been_in_backward)
          data_slice.direction_votes[iteration_] = BACKWARD;
        else
          data_slice.direction_votes[iteration_] = FORWARD;
      } else {
        data_slice.been_in_backward = true;
        if (predicted_forward_visits >
            predicted_backward_visits * this->enactor->do_b)
          data_slice.direction_votes[iteration_] = BACKWARD;
        else
          data_slice.direction_votes[iteration_] = FORWARD;
      }
    } else
      data_slice.direction_votes[iteration_] = FORWARD;
    // data_slice.direction_votes[(iteration_ + 1) % 4] = UNDECIDED;
    data_slice.direction_votes[(iteration_ + 1) % 4] = FORWARD;
    if (this->enactor->num_gpus > 1 && enactor_stats.iteration != 0 &&
        this->enactor->direction_optimized) {
      // while ( this->enactor->problem->data_slices[0]->direction_votes[iteration_] ==
      //     UNDECIDED) 
      while (
          this->enactor->problem->data_slices[0]->direction_votes[iteration_] ==
          FORWARD) 
      {
        sleep(0);
      }
      data_slice.current_direction =
          this->enactor->problem->data_slices[0]->direction_votes[iteration_];
    } else if (enactor_stats.iteration == 0)
      data_slice.direction_votes[iteration_] = FORWARD;
    else {
      data_slice.current_direction = data_slice.direction_votes[iteration_];
    }

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
         ((this->flag & Skip_PreScan) != 0)) ||
        (this->enactor->problem->data_slices[0]->current_direction == BACKWARD))
    //(AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_FORWARD ||
    // AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_BACKWARD))
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
    } else{  // if
            // (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
    //(AdvanceKernelPolicy::ADVANCE_MODE != gunrock::oprtr::advance::LB_CULL)
    // {
      retval = CheckSize<SizeT, VertexT>(
          true, "queue3", request_length, frontier.Next_V_Q(), over_sized,
          this->gpu_num, iteration, peer_, false);
      if (retval) return retval;
      retval = CheckSize<SizeT, VertexT>(true, "queue3", graph.nodes + 2,
                                         frontier.V_Q(), over_sized,
                                         this->gpu_num, iteration, peer_, true);
      if (retval) return retval;
     
      // if (enactor->problem->use_double_buffer)
      //{
      //    if (enactor_stats->retval = Check_Size<SizeT, Value>(
      //        true, "queue3", request_length,
      //        &frontier_queue->values[selector ^ 1], over_sized, thread_num,
      //        iteration, peer_, false))
      //        return;
      //    if (enactor_stats->retval = Check_Size<SizeT, Value>(
      //        true, "queue3", graph_slice->nodes + 2,
      //        &frontier_queue->values[selector], over_sized, thread_num,
      //        iteration, peer_, true))
      //        return;
      //}
    }  // else {
    //}

    return retval;
  }

  /**
   * @brief Core computation of bfs, one iteration
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
    auto &data = data_slice.data;
    auto &segments_temp = data_slice.segments_temp;
    auto &segments = data_slice.segments;
    auto &data_size = data_slice.data_size;
    auto &segments_size = data_slice.segments_size;

    // do we need predecessors?
    // we dont but can be used for shortcutting however lets not bother about this
  
    auto &preds = data_slice.preds;
    // @Achal this is information related to the partitioned graph,
    // see if its required in our context
    auto &original_vertex = graph.GpT::original_vertex;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    bool debug = ((this->enactor->flag & Debug) != 0);
    // @Achal mark_preds needs to be enabled here too
    // bool mark_preds = ((this->enactor->problem->flag & Mark_Predecessors) != 0);
    // idempotence depends on whether we are doing push/pull
    // so might as well change how we set this flag
    //idempotence=Pull
    bool idempotence =
        ((this->enactor->problem->flag & Enable_Idempotence) != 0);
    auto target = util::DEVICE;
    // so we have the peer and the current gpu
    // do all enactors have this two way shells?
    auto &gpu_num = this->gpu_num;

#if TO_TRACK
    util::PrintMsg(
        "Core queue_length = " + std::to_string(frontier.queue_length), gpu_num,
        iteration, peer_);
#endif
#ifdef RECORD_PER_ITERATION_STATS
    GpuTimer gpu_timer;
#endif
    
    // data_slice is common for all 
    // is current direction a field thats specifically for bfs?
    if (data_slice.current_direction == FORWARD) {
      frontier.queue_reset = true;
      enactor_stats.nodes_queued[0] += frontier.queue_length;

      if (debug)
        util::PrintMsg("Forward Advance begin", gpu_num, iteration, peer_);

      // LabelT label = iteration + 1;
      util::Array1D<SizeT, VertexT> *null_frontier = NULL;
      frontier.queue_length = graph.nodes;
      frontier.queue_reset = true;
  
      // segments_size[0] = 0;
      // data_size = 0;
    
      auto frontier_elements = frontier.V_Q();

      auto compute_op = [segments_temp, segments_size, graph] __host__
      __device__(VertexT * v, const SizeT &i) {
            // data[data_size++];
            segments_temp[i] = graph.CsrT::GetNeighborListLength(v[i]);
            atomicAdd(&segments_size[0], 1);

      };

      GUARD_CU(frontier.V_Q()->ForAll(compute_op, frontier.queue_length, target, stream));

      GUARD_CU(frontier.V_Q()->Print("Frontier: ",
                    frontier.queue_length,
                    util::DEVICE,
                    stream));
      // check segments size
      // typecast or add template param
      // this gave an error because I was trying to access the host_pointer
      // GUARD_CU(util::cubInclusiveSum(cub_temp_storage, segments_temp,
      //   segments, (SizeT)segments_size.d_pointer, stream));
              GUARD_CU(util::cubExclusiveSum(cub_temp_storage, segments_temp,
        segments, frontier.queue_length , stream));

        
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
             "cudaStreamSynchronize failed.");
      // auto update_segment_op = [segments] __host__ __device__(int &v) {
      //   if(segments != v){ // first element
      //     *v = segments[]
      //   }
      //   };
      // segments->ForEach(convergence_op, frontier.queue_length,
      //   util::DEVICE, oprtr_parameters.stream)
      
      // GUARD_CU(frontier.V_Q()->ForEach(
      //   [segments] __host__ __device__(VertexT & v, int index) { 
      //     if(index != 0){
      //       segments[index] += segments[index-1]
      //     }
      //   }, frontier.queue_length, target,
      //   0));
      
      // TODO
      // Uncomment the ForEach_Index below 
    auto elements = frontier.V_Q();
    // frontier.V_Q()->GetPointer(util::DEVICE)

    // TODO use advance operator
    // as its better for nested for loops as the parallelism can have load balancing issues

    GUARD_CU(frontier.V_Q()->ForAll(
      [segments, data, segments_size, data_size, labels, graph] __host__ __device__(
        const VertexT *v, const SizeT &index) {
      VertexT idx = v[index];
      SizeT start_edge = graph.CsrT::GetNeighborListOffset(idx);
      SizeT num_neighbors = graph.CsrT::GetNeighborListLength(idx);

      int start_fill = segments[index];
      int i = 0;
      for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
        
        VertexT u = graph.CsrT::GetEdgeDest(e);
        data[start_fill + (i++)] = labels[u];
      // for each neighbour in the neighbour list
      // populate data
      // data[segments[index]+neighbour_index]= neighbour.GetLabel();
        
      };
      },
      frontier.queue_length, util::DEVICE, oprtr_parameters.stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
             "cudaStreamSynchronize failed.");
    // auto apply_op =  [segments, data, segments_size, data_size, labels, graph] __host__ __device__(VertexT & v, int index) { 
    //   // get the neighbours 
    //   // start populating the data array with the information 
    //   // from the segments array

    //   SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
    //   SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

    //   int start_fill = segments[index];
    //   int i = 0;
    //   for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
        
    //     VertexT u = graph.CsrT::GetEdgeDest(e);
    //     data[start_fill + (i++)] = labels[u];
    //   // for each neighbour in the neighbour list
    //   // populate data
    //   // data[segments[index]+neighbour_index]= neighbour.GetLabel();
        
    //   }};

      // TODO
      // BUG Use CUDA kernel as everything is in internal memory anyway
      // use FORALL
      // cuda version wont be an issue
    // #pragma omp parallel for
    // for (SizeT i = 0; i < frontier.queue_length; i++) apply_op(elements[0][i], i);

      // GUARD_CU(frontier.V_Q()->ForEach_index([segments, data, segments_size, data_size, labels, graph] __host__ __device__(VertexT & v, int index) { 
      //     // get the neighbours 
      //     // start populating the data array with the information 
      //     // from the segments array

      //     SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
      //     SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
 
      //     int start_fill = segments[index];
      //     int i = 0;
      //     for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
            
      //       VertexT u = graph.CsrT::GetEdgeDest(e);
      //       data[start_fill + (i++)] = labels[u];
      //     // for each neighbour in the neighbour list
      //     // populate data
      //     // data[segments[index]+neighbour_index]= neighbour.GetLabel();
            
      //     }
      //   }, frontier.queue_length, target,
      //   0));

      // // so now we iterate through the neighbours of this and start 
      // auto compute_op = [const VertexT &src, const VertexT &dest, segments, data, segments_size, data_size] __host__
      //                 __device__(VertexT * v, const SizeT &i) {
      //   data[data_size++] = labels[dest];

      // };
      
      // TODO incorporate the segmented mode here
      // auto segmented_mode = ;//
      // make the segmented mode work such that the output of the operation is just the labels
      // in the same order as the original segments
      // this should be easy given that the operation is a one to one mapping from the segments.count_best array 
      // to the output array
      // extr

      // run a segmented sort
      // get max for each segment
      // run the vertex frontier again
      // set the new labels
      // reuse the segments data structure to store the new labels
      // TODO uncomment the foreach_index below

      // TODO
      // FORALL
      GUARD_CU(frontier.V_Q()->ForAll(
        [segments, labels, old_labels] __host__ __device__(
          const VertexT *v, const SizeT &index) {

          old_labels[index] = labels[index];
          labels[index] = segments[index];
  
        },
        frontier.queue_length, util::DEVICE, oprtr_parameters.stream));


        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
             "cudaStreamSynchronize failed.");
      // auto apply_op2 = [segments, labels, old_labels] __host__ __device__(VertexT & v, int index) { 
         
      //     old_labels[index] = labels[v];
      //     labels[v] = segments[index];
          
            
      //     };
      // #pragma omp parallel for
      // for (SizeT i = 0; i < frontier.queue_length; i++) apply_op2(elements[0][i], i);

      // GUARD_CU(frontier.V_Q()->ForEach_index([segments, labels, old_labels] __host__ __device__(VertexT & v, int index) { 
         
      //     old_labels[index] = labels[v];
      //     labels[v] = segments[index];
          
            
      //     }
      //   , frontier.queue_length, target,
      //   0));

      // how to map a vertex to the segments
      // will they be in order?

      // what do the vertex v and i in the () parameter list signify
      // is that the data we can extract from the compute_op

      // use functors instead of compute if applicable
      // auto compute_op = [const VertexT &src, segments] __host__
      // __device__(VertexT * v, const SizeT &i) {
      //     labels[src] = segments[frontier index];
        
      // };

      auto filter_op =
          [old_labels, labels] __host__ __device__(
              const VertexT &src, VertexT &dest, const SizeT &edge_id,
              const VertexT &input_item, const SizeT &input_pos,
              SizeT &output_pos) -> bool {
                // this somehow uses the 
        // TODO is this just for internal checks
        // why would a user care for isValid?
        // what does this check?
        return (old_labels[input_pos] != labels[input_pos]);

        // if (idempotence && mark_preds) {
        //   VertexT pred = src;
        //   if (original_vertex + 0 != NULL) pred = original_vertex[src];
        //   Store(preds + dest, pred);
        // }
        // return true;
      };
      // we would want to advance on a vertex if its label changes
      // so create a new int label to store the old label
      // -1 initially
      auto advance_op =
          // [idempotence, labels, label, mark_preds, preds,
          [] __host__
          __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool {

            printf("Source is %d, and destination is %d", src, dest);
        // if (!idempotence) {
        //   // Check if the destination node has been claimed as someone's child

        //   // @Achal fact check this
        //   // so if we are not doing idempotence it will mean that we
        //   // are on less writing computes that means pull (as it involves race-free reading )
        //   // so keep that in mind


        
        //   // ensures that the label is minimum 
        //   // this is an atomic operation and hence idempotence has been used 
        //   // to get rid of it
        //   // however
        //   // for us its an atomic add
        //   // which is still alright
        //   // we cannot ignore the value, so we cannot choose to do something like
        //   // if this then false
        //   // for example right now I am still going with min_reduce array of label frequency counts
        //   // hence lets 

        //   // label is outside
        //   // but why do we need an edge frontier?
        //   // is this always the case?
        //   // I think so!
        //   // need to push or pull values accordingly
        //   // if need to push
        //   // I need to send src label to child label 

        //   LabelT old_label = _atomicMin(labels + dest, label);
        //   if (label >= old_label) return false;

        //   // set predecessors
        //   // if (mark_preds) {
        //   //   VertexT pred = src;
        //   //   if (original_vertex + 0 != NULL) pred = original_vertex[src];
        //   //   Store(preds + dest, pred);
        //   // }
        // }
        // as we need to return true everytime
        // scope for two optimisations
        // 1. if we can count all the neighbouring vertices with the same label, we can just update an integer and return false
        // 2. is it possible there can be a point when we know for sure that a neighbour is not going to affect a vertex's label

        return true;
      };
      
      // I don't quite understand the importance of the filter operation as we already have an advance operation
      // is there additional information available here?
      // is this faster, slower, smarter than advance?
      // auto filter_op =
      //     [idempotence, mark_preds, preds, original_vertex] __host__ __device__(
      //         const VertexT &src, VertexT &dest, const SizeT &edge_id,
      //         const VertexT &input_item, const SizeT &input_pos,
      //         SizeT &output_pos) -> bool {
      //           // this somehow uses the 
      //   // TODO is this just for internal checks
      //   // why would a user care for isValid?
      //   // what does this check?
      //   if (!util::isValid(dest)) return false;

      //   // if (idempotence && mark_preds) {
      //   //   VertexT pred = src;
      //   //   if (original_vertex + 0 != NULL) pred = original_vertex[src];
      //   //   Store(preds + dest, pred);
      //   // }
      //   return true;
      // };

// Edge Map
#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Start();
#endif

      // TODO
      auto &work_progress = frontier.work_progress;
      auto queue_index = frontier.queue_index;
      // how does the for operation work?
      //
      GUARD_CU(oprtr::For(
        // these are the two variables that need to be present in the threads during the for op
        // there is the sizeT i which is the loop variable
        // and that is a function arg to the op
          [work_progress, queue_index] __host__ __device__(SizeT i) {
            SizeT *counter = work_progress.GetQueueCounter(queue_index + 1);
            counter[0] = 0;
          },
          1, util::DEVICE, oprtr_parameters.stream, 1, 1));
      
      
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters, 
        filter_op));

      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters, 
        advance_op));

      

      // Call Filter first
      // Then call advance
          // TODO
          // can we optimise this by fusing the two
          // GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          // graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          // advance_op, filter_op));

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

      frontier.queue_reset = false;
      // if (gunrock::oprtr::advance::hasPreScan<
      //      AdvanceKernelPolicy::ADVANCE_MODE>())
      //{
      //    enactor_stats.edges_queued[0] +=
      //        frontier_attribute.output_length[0];
      //} else {
      //    enactor_stats.AccumulateEdges(
      //    work_progress->template GetQueueLengthPointer<unsigned int>(
      //        frontier.queue_index), oprtr_parameters.tream);
      //}
      
      if (oprtr_parameters.advance_mode != "LB_CULL" &&
          oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
        // Filter
        if (debug) util::PrintMsg("Filter begin.", gpu_num, iteration, peer_);

        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
            filter_op));
        if (debug) util::PrintMsg("Filter end.", gpu_num, iteration, peer_);
      }

      // Get back the resulted frontier length
      GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));
      // end of forward
  };

//     else {  // backward
//       SizeT num_blocks = 0;
//       if (data_slice.previous_direction == FORWARD) {
//         GUARD_CU(data_slice.split_lengths.ForEach(
//             [] __host__ __device__(SizeT & length) { length = 0; }, 1, target,
//             stream));

//         if (this->enactor->num_gpus == 1) {
//           if (idempotence) {
//             num_blocks =
//                 (graph.nodes >> (2 + sizeof(MaskT))) / (1 << LOG_THREADS) + 1;
//             if (num_blocks > 480) num_blocks = 480;
//             From_Unvisited_Queue_IDEM<ProblemT, LOG_THREADS>
//                 <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
//                     graph.nodes,
//                     data_slice.split_lengths.GetPointer(util::DEVICE),
//                     data_slice.unvisited_vertices[frontier.queue_index % 2]
//                         .GetPointer(util::DEVICE),
//                     data_slice.visited_masks.GetPointer(util::DEVICE),
//                     data_slice.labels.GetPointer(util::DEVICE));
//           }

//           else {
//             num_blocks = graph.nodes / (1 << LOG_THREADS) + 1;
//             if (num_blocks > 480) num_blocks = 480;
//             From_Unvisited_Queue<ProblemT, LOG_THREADS>
//                 <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
//                     graph.nodes,
//                     data_slice.split_lengths.GetPointer(util::DEVICE),
//                     data_slice.unvisited_vertices[frontier.queue_index % 2]
//                         .GetPointer(util::DEVICE),
//                     data_slice.labels.GetPointer(util::DEVICE));
//           }
//         }  // end of num_gpus == 1

//         else {  // num_gpus != 1
//           num_blocks =
//               data_slice.local_vertices.GetSize() / (1 << LOG_THREADS) + 1;
//           if (num_blocks > 480) num_blocks = 480;
//           if (idempotence) {
//             From_Unvisited_Queue_Local_IDEM<ProblemT, LOG_THREADS>
//                 <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
//                     data_slice.local_vertices.GetSize(),
//                     data_slice.local_vertices.GetPointer(util::DEVICE),
//                     data_slice.split_lengths.GetPointer(util::DEVICE),
//                     data_slice.unvisited_vertices[frontier.queue_index % 2]
//                         .GetPointer(util::DEVICE),
//                     data_slice.visited_masks.GetPointer(util::DEVICE),
//                     data_slice.labels.GetPointer(util::DEVICE));
//           }

//           else {
//             From_Unvisited_Queue_Local<ProblemT, LOG_THREADS>
//                 <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
//                     data_slice.local_vertices.GetSize(),
//                     data_slice.local_vertices.GetPointer(util::DEVICE),
//                     data_slice.split_lengths.GetPointer(util::DEVICE),
//                     data_slice.unvisited_vertices[frontier.queue_index % 2]
//                         .GetPointer(util::DEVICE),
//                     data_slice.labels.GetPointer(util::DEVICE));
//           }
//         }

//         GUARD_CU(data_slice.split_lengths.Move(util::DEVICE, util::HOST, 1, 0,
//                                                stream));
//         GUARD_CU2(cudaStreamSynchronize(stream),
//                   "cudaStreamSynchronize failed");
//         data_slice.num_unvisited_vertices = data_slice.split_lengths[0];
//         data_slice.num_visited_vertices =
//             graph.nodes - data_slice.num_unvisited_vertices;
//         if (debug)
//           util::PrintMsg(
//               "Changing from forward to backward, #unvisited_vertices =" +
//               std::to_string(data_slice.num_unvisited_vertices));

//       } else {
//         data_slice.num_unvisited_vertices = data_slice.split_lengths[0];
//       }

//       GUARD_CU(data_slice.split_lengths.ForEach(
//           [] __host__ __device__(SizeT & length) { length = 0; }, 2, target,
//           stream));
//       enactor_stats.nodes_queued[0] += data_slice.num_unvisited_vertices;
//       num_blocks = data_slice.num_unvisited_vertices / (1 << LOG_THREADS) + 1;
//       if (num_blocks > 480) num_blocks = 480;

// #ifdef RECORD_PER_ITERATION_STATS
//       gpu_timer.Start();
// #endif

//       Inverse_Expand<ProblemT, LOG_THREADS>
//           <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
//               graph, data_slice.num_unvisited_vertices,
//               enactor_stats.iteration + 1, idempotence,
//               data_slice.unvisited_vertices[frontier.queue_index % 2]
//                   .GetPointer(util::DEVICE),
//               data_slice.split_lengths.GetPointer(util::DEVICE),
//               data_slice.unvisited_vertices[(frontier.queue_index + 1) % 2]
//                   .GetPointer(util::DEVICE),
//               frontier.Next_V_Q()->GetPointer(util::DEVICE),
//               data_slice.visited_masks.GetPointer(util::DEVICE),
//               data_slice.labels.GetPointer(util::DEVICE),
//               data_slice.preds.GetPointer(util::DEVICE));

// #ifdef RECORD_PER_ITERATION_STATS
//       gpu_timer.Stop();
//       float elapsed = gpu_timer.ElapsedMillis();
//       enactor_stats.per_iteration_advance_time.push_back(elapsed);
//       enactor_stats.per_iteration_advance_mteps.push_back(-1.0f);
//       enactor_stats.per_iteration_advance_input_edges.push_back(-1.0f);
//       enactor_stats.per_iteration_advance_output_edges.push_back(-1.0f);
//       enactor_stats.per_iteration_advance_direction.push_back(false);
// #endif

//       data_slice.split_lengths.Move(target, util::HOST, 2, 0, stream);
//       GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
//       if (debug)
//         util::PrintMsg("#unvisited v = " +
//                        std::to_string(data_slice.num_unvisited_vertices) +
//                        ", #newly visited = " +
//                        std::to_string(data_slice.split_lengths[1]) +
//                        ", #still unvisited = " +
//                        std::to_string(data_slice.split_lengths[0]));

//       frontier.queue_length = data_slice.split_lengths[1];
//       data_slice.num_visited_vertices =
//           graph.nodes - data_slice.num_unvisited_vertices;
//       enactor_stats.edges_queued[0] += frontier.output_length[0];
//       frontier.queue_reset = false;
//       frontier.queue_index++;
//     }  // end of backward

  //   data_slice.previous_direction = data_slice.current_direction;

  //   return retval;
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
    auto &preds = data_slice.preds;
    auto label = this->enactor->mgpu_slices[this->gpu_num]
                     .in_iteration[iteration % 2][peer_];
    bool idempotence =
        ((this->enactor->problem->flag & Enable_Idempotence) != 0);
    bool mark_preds = ((ProblemT::FLAG & Mark_Predecessors) != 0);

    auto expand_op =
        [labels, label, preds, masks, idempotence, mark_preds] __host__
        __device__(VertexT & key, const SizeT &in_pos,
                   VertexT *vertex_associate_ins,
                   ValueT *value__associate_ins) -> bool {
      SizeT mask_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      MaskT mask_bit = util::PreDefinedValues<MaskT>::InvalidValue;

      if (idempotence) {
        mask_pos = (key  //& (~(1ULL<<(sizeof(VertexT)*8-2)))
                         // LB_CULL::KernelPolicy::ELEMENT_ID_MASK
                    ) >>
                   (2 + sizeof(MaskT));
        MaskT mask_byte = _ldg(masks + mask_pos);
        mask_bit = 1 << (key & ((1 << (2 + sizeof(MaskT))) - 1));
        if ((mask_bit & mask_byte) != 0) return false;
      }

      if (_ldg(labels + key) != util::PreDefinedValues<LabelT>::MaxValue)
        return false;

      labels[key] = label;
      if (idempotence) masks[mask_pos] |= mask_bit;

      if (mark_preds) preds[key] = vertex_associate_ins[in_pos];
      return true;
    };

    retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of BFSIteration

/**
 * @brief BFS enactor class.
 * @tparam _Problem             BFS problem type
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

  bool direction_optimized;
  float do_a, do_b;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief BFSEnactor constructor
   */
  Enactor() : BaseEnactor("bfs"), problem(NULL) {
    this->max_num_vertex_associates =
        (Problem::FLAG & Mark_Predecessors) != 0 ? 1 : 0;
    this->max_num_value__associates = 0;
  }

  /**
   * @brief BFSEnactor destructor
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
    // direction_optimized 
    // direction_optimized = parameters.Get<bool>("direction-optimized");
    do_a = parameters.Get<float>("do-a");
    do_b = parameters.Get<float>("do-b");

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      // for each gpu the slice is initialised to the starting point (peer = 0)
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      
      // TODO
      // does this allocate function duplicate data on each gpu?
      // no it is agnostic to that as the partition should have handled that 
      // and anyway we call this once for each gpu not for each enactor slice
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
      
      // this is where we will have the enactor slice data allocated
      // not allocated but pointer referenced

      // each enactor slice's operator parameter labels have data slices belonging to the specific gpu
      // this is strange?
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
   * @brief one run of bfs, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    // so each iteration loop gets 
    // thread data 
    gunrock::app::Iteration_Loop<
        ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0, 0,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a BFS computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU BFS Done.", this->flag & Debug);
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
