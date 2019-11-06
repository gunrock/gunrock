// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_loop.cuh
 *
 * @brief Base Iteration Loop
 */

#pragma once

#include <chrono>
#include <thread>

#include <gunrock/app/enactor_kernel.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/util/latency_utils.cuh>
#include <moderngpu.cuh>

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/*
 * @brief Iteration loop.
 *
 * @tparam EnactorT
 * @tparam IterationT
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] thread_data
 */
template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES,
          typename IterationT>
void Iteration_Loop(ThreadSlice &thread_data, IterationT &iteration) {
  typedef typename IterationT::Enactor Enactor;
  typedef typename Enactor::Problem Problem;
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::VertexT VertexT;
  typedef typename Enactor::ValueT ValueT;
  typedef typename Problem::DataSlice DataSlice;
  typedef typename Problem::GraphT GraphT;

  Enactor &enactor = ((Enactor *)thread_data.enactor)[0];
  auto &problem = enactor.problem[0];
  int num_gpus = enactor.num_gpus;
  int gpu_num = thread_data.thread_num;
  auto &data_slice = problem.data_slices[gpu_num][0];
  auto &mgpu_slice = enactor.mgpu_slices[gpu_num];
  // auto    &data_slices =  problem.data_slices;
  auto &mgpu_slices = enactor.mgpu_slices;
  auto &graph = problem.sub_graphs[gpu_num];
  auto enactor_slices = enactor.enactor_slices + gpu_num * num_gpus;
  // auto    &streams     =  data_slice.streams;
  auto &stages = mgpu_slice.stages;
  auto &to_shows = mgpu_slice.to_show;

  std::string mssg = "";
  SizeT total_length = 0;
  SizeT received_length = 0;
  SizeT communicate_latency = enactor.communicate_latency;
  // float         communicate_multipy  =   enactor.communicate_multipy;
  SizeT expand_latency = enactor.expand_latency;
  SizeT subqueue_latency = enactor.subqueue_latency;
  SizeT fullqueue_latency = enactor.fullqueue_latency;
  SizeT makeout_latency = enactor.makeout_latency;

  auto &frontier0 = enactor_slices[0].frontier;
  auto &enactor_stats0 = enactor_slices[0].enactor_stats;
  auto &stream0 = enactor_slices[0].stream;

#ifdef ENABLE_PERFORMANCE_PROFILING
  util::CpuTimer cpu_timer;
  std::vector<double> &iter_full_queue_time =
      enactor.iter_full_queue_time[gpu_num].back();
  std::vector<double> &iter_sub_queue_time =
      enactor.iter_sub_queue_time[gpu_num].back();
  std::vector<double> &iter_total_time =
      enactor.iter_total_time[gpu_num].back();
  std::vector<SizeT> &iter_full_queue_nodes_queued =
      enactor.iter_full_queue_nodes_queued[gpu_num].back();
  std::vector<SizeT> &iter_full_queue_edges_queued =
      enactor.iter_full_queue_edges_queued[gpu_num].back();

  cpu_timer.Start();
  double iter_start_time = cpu_timer.MillisSinceStart();
  double iter_stop_time = 0;
  double subqueue_finish_time = 0;

  SizeT h_edges_queued[16];
  SizeT h_nodes_queued[16];
  SizeT previous_edges_queued[16];
  SizeT previous_nodes_queued[16];
  SizeT h_full_queue_edges_queued = 0;
  SizeT h_full_queue_nodes_queued = 0;
  SizeT previous_full_queue_edges_queued = 0;
  SizeT previous_full_queue_nodes_queued = 0;

  for (int peer_ = 0; peer_ < num_gpus; peer_++) {
    h_edges_queued[peer_] = 0;
    h_nodes_queued[peer_] = 0;
    previous_nodes_queued[peer_] = 0;
    previous_edges_queued[peer_] = 0;
  }
#endif

  util::PrintMsg("Iteration entered", enactor.flag & Debug);
  while (!iteration.Stop_Condition(gpu_num)) {
    total_length = 0;
    received_length = frontier0.queue_length;
    mgpu_slice.wait_counter = 0;
    // tretval                  = cudaSuccess;
    // frontier0.queue_offset   = 0;
    frontier0.queue_reset = true;
    if (num_gpus > 1) {
      if (enactor_stats0.iteration != 0)
        for (int i = 1; i < num_gpus; i++) {
          auto &frontier = enactor_slices[i].frontier;
          // frontier.selector     = frontier0.selector;
          // frontier.advance_type = frontier0.advance_type;
          // frontier.queue_offset = 0;
          frontier.queue_reset = true;
          frontier.queue_index = frontier0.queue_index;
          // frontier.current_label= frontier0.current_label;
          enactor_slices[i].enactor_stats.iteration =
              enactor_slices[0].enactor_stats.iteration;
        }

      if (IterationT::FLAG & Unified_Receive) {
        // printf("%d, %d : start_received_length = %d\n",
        //    gpu_num, enactor_stats0.iteration, received_length);
        mgpu_slice.in_length_out[0] = received_length;
        mgpu_slice.in_length_out.Move(util::HOST, util::DEVICE, 1, 0, stream0);
        if (enactor_stats0.retval = util::GRError(
                cudaStreamSynchronize(stream0), "cudaStreamSynchronize failed",
                __FILE__, __LINE__))
          break;
      }
    } else {
      // auto &frontier = enactor_slices[0].frontier;
      // frontier.queue_reset  = true;
      // frontier.queue_offset = 0;
      mgpu_slice.in_length_out[0] = received_length;
    }

    for (int peer = 0; peer < num_gpus; peer++) {
      stages[peer] = 0;
      stages[peer + num_gpus] = 0;
      to_shows[peer] = true;
      to_shows[peer + num_gpus] = true;
      for (int i = 0; i < mgpu_slice.num_stages; i++)
        mgpu_slice.events_set[enactor_stats0.iteration % 4][peer][i] = false;
    }
    // util::cpu_mt::PrintGPUArray<SizeT, VertexId>(
    //    "labels", data_slice.labels.GetPointer(util::DEVICE), graph.nodes,
    //    gpu_num, iteration, -1, streams[0]);

    while (mgpu_slice.wait_counter < num_gpus * 2 &&
           (!iteration.Stop_Condition(gpu_num))) {
      // util::cpu_mt::PrintCPUArray<int, int>("stages", stages, num_gpus * 2,
      // thread_num, iteration);
      for (int peer__ = 0; peer__ < num_gpus * 2; peer__++) {
        auto peer_ = (peer__ % num_gpus);
        auto peer = peer_ <= gpu_num ? peer_ - 1 : peer_;
        auto gpu_ = peer < gpu_num ? gpu_num : gpu_num + 1;
        auto &enactor_slice = enactor_slices[peer_];
        auto &enactor_stats = enactor_slice.enactor_stats;
        auto &frontier = enactor_slice.frontier;
        auto iteration_num = enactor_stats.iteration;
        auto iteration_num_ = iteration_num % 4;
        auto pre_stage = stages[peer__];
        auto &stage = stages[peer__];
        auto &stream =
            (peer__ <= num_gpus) ? enactor_slice.stream : enactor_slice.stream2;
        auto &to_show = to_shows[peer__];
        auto &retval = enactor_stats.retval;
        // selector            = frontier_attribute[peer_].selector;
        // scanned_edges_      = &(data_slice->scanned_edges  [peer_]);
        // frontier_attribute_ = &(frontier_attribute         [peer_]);
        // work_progress_      = &(work_progress              [peer_]);

        if ((enactor.flag & Debug) != 0 && to_show) {
          // mssg=" ";mssg[0]='0'+data_slice->wait_counter;
          mssg = std::to_string(mgpu_slice.wait_counter);
          ShowDebugInfo(enactor, gpu_num, peer__, mssg, stream);
        }
        to_show = true;

        switch (stage) {
          case 0:  // Assign marker & Scan
            if (peer__ == 0) {
              if (frontier.queue_length == 0) {  // empty local queue
                // SetRecord(mgpu_slice, iteration_num, peer__, 3, stream);
                stage = 4;
                break;
              } else {
                if (IterationT::FLAG & Use_SubQ) {
                  stage = 1;
                  break;
                } else {
                  SetRecord(mgpu_slice, iteration_num, peer__, 2, stream);
                  stage = 3;
                  break;
                }
              }
            }

            if (peer__ < num_gpus) {  // wait and expand incoming
              if (!(mgpu_slices[peer]
                        .events_set[iteration_num_][gpu_ + num_gpus][0])) {
                to_show = false;
                break;
              }

              frontier.queue_length =
                  mgpu_slice.in_length[iteration_num % 2][peer_];
#ifdef ENABLE_PERFORMANCE_PROFILING
              enactor_stats.iter_in_length.back().push_back(
                  mgpu_slice.in_length[iteration_num % 2][peer_]);
#endif
              if (frontier.queue_length != 0) {
                if (retval = util::GRError(
                        cudaStreamWaitEvent(
                            stream,
                            mgpu_slices[peer]
                                .events[iteration_num_][gpu_ + num_gpus][0],
                            0),
                        "cudaStreamWaitEvent failed", __FILE__, __LINE__))
                  break;
              }
              mgpu_slice.in_length[iteration_num % 2][peer_] = 0;
              mgpu_slices[peer].events_set[iteration_num_][gpu_ + num_gpus][0] =
                  false;

              if (frontier.queue_length == 0) {
                // SetRecord(mgpu_slice, iteration_num, peer__, 3, stream);
                // printf(" %d\t %d\t %d\t Expand and subQ skipped\n",
                //    gpu_num, iteration_num, peer__);
                stage = 4;
                break;
              }

              if (expand_latency != 0)
                util::latency::Insert_Latency(
                    expand_latency, frontier.queue_length, stream,
                    mgpu_slice.latency_data.GetPointer(util::DEVICE));

              iteration.template ExpandIncoming<NUM_VERTEX_ASSOCIATES,
                                                NUM_VALUE__ASSOCIATES>(
                  received_length, peer_);
              // printf("%d, Expand, selector = %d, keys = %p\n",
              //    thread_num, selector^1,
              //    frontier.keys[selector^1] .GetPointer(util::DEVICE));

              // frontier.selector ^= 1;
              // frontier.queue_index++;
              if ((IterationT::FLAG & Use_SubQ) == 0) {
                if (IterationT::FLAG & Unified_Receive) {
                  // SetRecord(mgpu_slice, iteration_num, peer__, 3, stream);
                  stage = 4;
                } else {
                  SetRecord(mgpu_slice, iteration_num, peer__, 2, stream);
                  stage = 3;
                }
              } else {
                SetRecord(mgpu_slice, iteration_num, peer__, stage, stream);
                stage = 1;
              }
              break;
            }

            if (peer__ == num_gpus) {  // out-going to local, not in use
              stage = 4;
              break;
            }

            if (peer__ > num_gpus) {
              if (iteration_num == 0) {  // first iteration, nothing to send
                SetRecord(mgpu_slice, iteration_num, peer__, 0, stream);
                stage = 4;
                break;
              }

              // Push Neighbor
              if (communicate_latency != 0)
                util::latency::Insert_Latency(
                    communicate_latency, mgpu_slice.out_length[peer_], stream,
                    mgpu_slice.latency_data.GetPointer(util::DEVICE));

              PushNeighbor<Enactor, NUM_VERTEX_ASSOCIATES,
                           NUM_VALUE__ASSOCIATES>(enactor, gpu_num, peer);
              SetRecord(mgpu_slice, iteration_num, peer__, stage, stream);
              stage = 4;
              break;
            }
            break;

          case 1:  // Comp Length
            if (peer_ != 0) {
              if (retval = CheckRecord(mgpu_slice, iteration_num, peer_,
                                       stage - 1, stage, to_show))
                break;
              if (!to_show) break;
              frontier.queue_length = mgpu_slice.in_length_out[peer_];
            }
            if (retval = iteration.Compute_OutputLength(peer_)) break;

            // TODO: Verify this
            // if (enactor -> size_check ||
            //    (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
            //        != oprtr::advance::TWC_FORWARD &&
            //     Iteration::AdvanceKernelPolicy::ADVANCE_MODE
            ///        != oprtr::advance::TWC_BACKWARD))
            // if (Enactor::FLAG & Size_Check)
            { SetRecord(mgpu_slice, iteration_num, peer_, stage, stream); }
            stage = 2;
            break;

          case 2:  // SubQueue Core
            // TODO: Verify this
            // if (enactor -> size_check ||
            //    (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
            //        != oprtr::advance::TWC_FORWARD &&
            //     Iteration::AdvanceKernelPolicy::ADVANCE_MODE
            //        != oprtr::advance::TWC_BACKWARD))
            // if (Enactor::FLAG & Size_Check)
            {
              if (retval = CheckRecord(mgpu_slice, iteration_num, peer_,
                                       stage - 1, stage, to_show))
                break;
              if (!to_show) break;
              /*if (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                  == oprtr::advance::TWC_FORWARD ||
                  Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                  == oprtr::advance::TWC_BACKWARD)
              {
                  frontier_attribute_->output_length[0] *= 1.1;
              }*/

              if (enactor.flag & Size_Check) iteration.Check_Queue_Size(peer_);
              if (retval) break;
            }
            if (subqueue_latency != 0)
              util::latency::Insert_Latency(
                  subqueue_latency, frontier.queue_length, stream,
                  mgpu_slice.latency_data.GetPointer(util::DEVICE));

            iteration.Core(peer_);
            if (retval) break;
#ifdef ENABLE_PERFORMANCE_PROFILING
            h_nodes_queued[peer_] = enactor_stats.nodes_queued[0];
            h_edges_queued[peer_] = enactor_stats.edges_queued[0];
            enactor_stats.nodes_queued.Move(util::DEVICE, util::HOST, 1, 0,
                                            stream);
            enactor_stats.edges_queued.Move(util::DEVICE, util::HOST, 1, 0,
                                            stream);
#endif

            if (num_gpus > 1) {
              SetRecord(mgpu_slice, iteration_num, peer__, stage, stream);
              stage = 3;
            } else {
              // SetRecord(mgpu_slice, iteration_num, peer__, 3,
              // streams[peer__]);
              stage = 4;
            }
            break;

          case 3:  // Copy
            // if (Iteration::HAS_SUBQ || peer_ != 0)
            {
              if (retval = CheckRecord(mgpu_slice, iteration_num, peer_,
                                       stage - 1, stage, to_show))
                break;
              if (!to_show) break;
            }

            // printf("size_check = %s\n", enactor -> size_check ? "true" :
            // "false");fflush(stdout);
            if ((IterationT::FLAG & Use_SubQ) == 0 && peer_ > 0) {
              frontier.queue_length = mgpu_slice.in_length_out[peer_];
            }
            if ((enactor.flag & Size_Check) == 0 &&
                (/*(enactor.flag & Debug) !=0 ||*/ num_gpus > 1)) {
              bool over_sized = false;
              if (IterationT::FLAG & Use_SubQ) {
                if (retval = CheckSize<SizeT, VertexT>(
                        false, "queue3", frontier.output_length[0] + 2,
                        frontier.Next_V_Q(), over_sized, gpu_num, iteration_num,
                        peer_, false))
                  break;
              }
              if (frontier.queue_length == 0) break;

              if (retval = CheckSize<SizeT, VertexT>(
                      false, "total_queue",
                      total_length + frontier.queue_length,
                      enactor_slices[num_gpus].frontier.V_Q(), over_sized,
                      gpu_num, iteration_num, peer_, false))
                break;

              // util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
              //    enactor_slices[num_gpus].frontier.keys[0]
              //        .GetPointer(util::DEVICE) + total_length,
              //    frontier.keys[selector].GetPointer(util::DEVICE),
              //    frontier.queue_length);
              if (retval = frontier.V_Q()->ForAll(
                      enactor_slices[num_gpus].frontier.V_Q()[0],
                      [total_length] __host__ __device__(
                          VertexT * key0, VertexT * key1, const SizeT &pos) {
                        key1[pos + total_length] = key0[pos];
                      },
                      frontier.queue_length, util::LOCATION_DEFAULT, stream))
                break;

              // if (problem -> use_double_buffer)
              if (IterationT::FLAG & Use_Double_Buffer) {
                // util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                //    data_slice->frontier_queues[num_gpus].values[0]
                //        .GetPointer(util::DEVICE) + Total_Length,
                //    frontier_queue_->values[selector].GetPointer(util::DEVICE),
                //    frontier.queue_length);
                // TODO: Use other ways to do this
                // if (retval = frontier.values[selector].ForAll(
                //    enactor_slices[num_gpus].frontier.values[0],
                //    [total_length]__host__ __device__
                //    (ValueT* val0, ValueT *val1, const SizeT &pos)
                //    {
                //        val1[pos + total_length] = val0[pos];
                //    }, frontier.queue_length,
                //    util::LOCATION_DEFAULT, stream))
                //    break;
              }
            }

            total_length += frontier.queue_length;
            // SetRecord(mgpu_slice, iteration_num, peer__, 3, streams[peer__]);
            stage = 4;
            break;

          case 4:  // End
            mgpu_slice.wait_counter++;
            to_show = false;
            stage = 5;
            break;
          default:
            // stage--;
            to_show = false;
        }

        if ((enactor.flag & Debug) && !(retval)) {
          mssg = "stage 0 @ gpu 0, peer_ 0 failed";
          mssg[6] = char(pre_stage + '0');
          mssg[14] = char(gpu_num + '0');
          mssg[23] = char(peer__ + '0');
          retval = util::GRError(mssg, __FILE__, __LINE__);
          if (retval) break;
        }
        // stages[peer__]++;
        if (retval) break;
      }
    }

    if (!iteration.Stop_Condition(gpu_num)) {
      for (int peer_ = 0; peer_ < num_gpus; peer_++)
        mgpu_slice.wait_marker[peer_] = 0;
      int wait_count = 0;
      while (wait_count < num_gpus && !iteration.Stop_Condition(gpu_num)) {
        for (int peer_ = 0; peer_ < num_gpus; peer_++) {
          if (peer_ == num_gpus || mgpu_slice.wait_marker[peer_] != 0) continue;
          cudaError_t tretval = cudaStreamQuery(enactor_slices[peer_].stream);
          if (tretval == cudaSuccess) {
            mgpu_slice.wait_marker[peer_] = 1;
            wait_count++;
            continue;
          } else if (tretval != cudaErrorNotReady) {
            enactor_slices[peer_ % num_gpus].enactor_stats.retval = tretval;
            break;
          }
        }
      }

      if (IterationT::FLAG & Unified_Receive) {
        total_length = mgpu_slice.in_length_out[0];
      } else if (num_gpus == 1)
        total_length = frontier0.queue_length;
#ifdef ENABLE_PERFORMANCE_PROFILING
      subqueue_finish_time = cpu_timer.MillisSinceStart();
      iter_sub_queue_time.push_back(subqueue_finish_time - iter_start_time);
      if (IterationT::FLAG & Use_SubQ)
        for (int peer_ = 0; peer_ < num_gpus; peer_++) {
          auto &enactor_stats = enactor_slices[peer_].enactor_stats;
          enactor_stats.iter_nodes_queued.back().push_back(
              h_nodes_queued[peer_] + enactor_stats.nodes_queued[0] -
              previous_nodes_queued[peer_]);
          previous_nodes_queued[peer_] =
              h_nodes_queued[peer_] + enactor_stats.nodes_queued[0];
          enactor_stats.nodes_queued[0] = h_nodes_queued[peer_];

          enactor_stats.iter_edges_queued.back().push_back(
              h_edges_queued[peer_] + enactor_stats.edges_queued[0] -
              previous_edges_queued[peer_]);
          previous_edges_queued[peer_] =
              h_edges_queued[peer_] + enactor_stats.edges_queued[0];
          enactor_stats.edges_queued[0] = h_edges_queued[peer_];
        }
#endif
      if (enactor.flag & Debug) {
        util::PrintMsg(std::to_string(gpu_num) + "\t " +
                       std::to_string(enactor_stats0.iteration) +
                       "\t \t Subqueue finished. Total_Length= " +
                       std::to_string(total_length));
      }

      // grid_size = Total_Length/256+1;
      // if (grid_size > 512) grid_size = 512;

      if ((enactor.flag & Size_Check) &&
          (IterationT::FLAG & Unified_Receive) == 0) {
        bool over_sized = false;
        if (enactor_stats0.retval = CheckSize<SizeT, VertexT>(
                true, "total_queue", total_length, frontier0.V_Q(), over_sized,
                gpu_num, enactor_stats0.iteration, num_gpus, true))
          break;
        // if (problem -> use_double_buffer)
        //    if (enactor_stats[0].retval =
        //        CheckSize</*true,*/ SizeT, Value> (
        //            true, "total_queue", Total_Length,
        //            &data_slice->frontier_queues[0].values[frontier_attribute[0].selector],
        //            over_sized, thread_num, enactor_stats0.iteration,
        //            num_gpus, true))
        //        break;

        SizeT offset = frontier0.queue_length;
        for (int peer_ = 1; peer_ < num_gpus; peer_++)
          if (enactor_slices[peer_].frontier.queue_length != 0) {
            auto &frontier = enactor_slices[peer_].frontier;
            // util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
            //    data_slice->frontier_queues[0    ]
            //        .keys[frontier_attribute[0    ].selector]
            //        .GetPointer(util::DEVICE) + offset,
            //    data_slice->frontier_queues[peer_]
            //        .keys[frontier_attribute[peer_].selector]
            //        .GetPointer(util::DEVICE),
            //    frontier_attribute[peer_].queue_length);
            frontier0.V_Q()->ForAll(
                frontier.V_Q()[0],
                [offset] __host__ __device__(VertexT * key0, VertexT * key1,
                                             const SizeT &pos) {
                  key0[pos + offset] = key1[pos];
                },
                frontier.queue_length, util::LOCATION_DEFAULT, stream0);

            // TODO
            // if (problem -> use_double_buffer)
            //    util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
            //        data_slice->frontier_queues[0    ]
            //            .values[frontier_attribute[0    ].selector]
            //            .GetPointer(util::DEVICE) + offset,
            //        data_slice->frontier_queues[peer_]
            //            .values[frontier_attribute[peer_].selector]
            //            .GetPointer(util::DEVICE),
            //        frontier_attribute[peer_].queue_length);
            offset += frontier.queue_length;
          }
      }
      frontier0.queue_length = total_length;
      // TODO: match here
      // if ((enactor.flag & Size_Check) == 0)
      //    frontier0.selector = 0;

      if (IterationT::FLAG & Use_FullQ) {
        int peer_ =
            ((enactor.flag & Size_Check) != 0 || num_gpus == 1) ? 0 : num_gpus;
        auto &enactor_slice = enactor_slices[peer_];
        auto &frontier = enactor_slice.frontier;
        auto &enactor_stats = enactor_slice.enactor_stats;
        // frontier_queue_     = &(data_slice->frontier_queues
        //    [(enactor -> size_check || num_gpus==1) ? 0 : num_gpus]);
        // scanned_edges_      = &(data_slice->scanned_edges
        //    [(enactor -> size_check || num_gpus==1) ? 0 : num_gpus]);
        // frontier_attribute_ = &(frontier_attribute[peer_]);
        // enactor_stats_      = &(enactor_stats[peer_]);
        // work_progress_      = &(work_progress[peer_]);
        auto &iteration_num = enactor_stats.iteration;
        auto stream = enactor_slice.stream;
        auto &retval = enactor_stats.retval;

        // frontier.queue_offset = 0;
        frontier.queue_reset = true;
        // TODO: match here
        // if ((enactor.flag & Size_Check) == 0)
        //    frontier.selector     = 0;

        iteration.Gather(peer_);
        // selector = frontier.selector;
        if (retval) break;

        if (frontier.queue_length != 0) {
          if (enactor.flag & Debug) {
            mssg = "";
            ShowDebugInfo(enactor, gpu_num, peer_, mssg, stream);
          }

          retval = iteration.Compute_OutputLength(peer_);
          if (retval) break;

          // frontier_attribute_->output_length.Move(
          //    util::DEVICE, util::HOST, 1, 0, streams[peer_]);
          if (enactor.flag & Size_Check) {
            cudaError_t tretval = cudaStreamSynchronize(stream);
            if (tretval != cudaSuccess) {
              retval = tretval;
              break;
            }

            iteration.Check_Queue_Size(peer_);
            if (retval) break;
          }

          if (fullqueue_latency != 0)
            util::latency::Insert_Latency(
                fullqueue_latency, frontier.queue_length, stream,
                mgpu_slice.latency_data.GetPointer(util::DEVICE));

          iteration.Core(peer_);
          if (retval) break;
#ifdef ENABLE_PERFORMANCE_PROFILING
          h_full_queue_nodes_queued = enactor_stats.nodes_queued[0];
          h_full_queue_edges_queued = enactor_stats.edges_queued[0];
          enactor_stats.edges_queued.Move(util::DEVICE, util::HOST, 1, 0,
                                          stream);
          enactor_stats.nodes_queued.Move(util::DEVICE, util::HOST, 1, 0,
                                          stream);
#endif
          if (retval =
                  util::GRError(cudaStreamSynchronize(stream),
                                "FullQueue_Core failed.", __FILE__, __LINE__))
            break;
            // cudaError_t tretval = cudaErrorNotReady;
            // while (tretval == cudaErrorNotReady)
            //{
            //    tretval = cudaStreamQuery(stream);
            //    if (tretval == cudaErrorNotReady)
            //    {
            //        //sleep(0);
            //        std::this_thread::sleep_for(std::chrono::microseconds(0));
            //    }
            //}
            // if (retval = util::GRError(tretval,
            //    "FullQueue_Core failed.", __FILE__, __LINE__))
            //    break;

#ifdef ENABLE_PERFORMANCE_PROFILING
          iter_full_queue_nodes_queued.push_back(
              h_full_queue_nodes_queued + enactor_stats.nodes_queued[0] -
              previous_full_queue_nodes_queued);
          previous_full_queue_nodes_queued =
              h_full_queue_nodes_queued + enactor_stats.nodes_queued[0];
          enactor_stats.nodes_queued[0] = h_full_queue_nodes_queued;

          iter_full_queue_edges_queued.push_back(
              h_full_queue_edges_queued + enactor_stats.edges_queued[0] -
              previous_full_queue_edges_queued);
          previous_full_queue_edges_queued =
              h_full_queue_edges_queued + enactor_stats.edges_queued[0];
          enactor_stats.edges_queued[0] = h_full_queue_edges_queued;
#endif
          if ((enactor.flag & Size_Check) == 0) {
            bool over_sized = false;
            if (retval = CheckSize<SizeT, VertexT>(
                    false, "queue3", frontier.output_length[0] + 2,
                    frontier.Next_V_Q(), over_sized, gpu_num, iteration_num,
                    peer_, false))
              break;
          }
          // selector = frontier_attribute[peer_].selector;
          total_length = frontier.queue_length;
        } else {
          total_length = 0;
          for (int peer__ = 0; peer__ < num_gpus; peer__++)
            mgpu_slice.out_length[peer__] = 0;
#ifdef ENABLE_PERFORMANCE_PROFILING
          iter_full_queue_nodes_queued.push_back(0);
          iter_full_queue_edges_queued.push_back(0);
#endif
        }
#ifdef ENABLE_PERFORMANCE_PROFILING
        iter_full_queue_time.push_back(cpu_timer.MillisSinceStart() -
                                       subqueue_finish_time);
#endif
        if (enactor.flag & Debug) {
          util::PrintMsg(std::to_string(gpu_num) + "\t " +
                         std::to_string(enactor_stats0.iteration) +
                         "\t \t Fullqueue finished. Total_Length= " +
                         std::to_string(total_length));
        }
        // frontier_queue_ = &(data_slice->frontier_queues[enactor ->
        // size_check?0:num_gpus]);
        if (num_gpus == 1) mgpu_slice.out_length[0] = total_length;
      }

      if (num_gpus > 1) {
        for (int peer_ = num_gpus + 1; peer_ < num_gpus * 2; peer_++)
          mgpu_slice.wait_marker[peer_] = 0;
        int wait_count = 0;
        while (wait_count < num_gpus - 1 &&
               !iteration.Stop_Condition(gpu_num)) {
          for (int peer_ = num_gpus + 1; peer_ < num_gpus * 2; peer_++) {
            if (peer_ == num_gpus || mgpu_slice.wait_marker[peer_] != 0)
              continue;
            cudaError_t tretval =
                cudaStreamQuery(enactor_slices[peer_ - num_gpus].stream2);
            if (tretval == cudaSuccess) {
              mgpu_slice.wait_marker[peer_] = 1;
              wait_count++;
              continue;
            } else if (tretval != cudaErrorNotReady) {
              enactor_slices[peer_ % num_gpus].enactor_stats.retval = tretval;
              break;
            }
          }
        }

        iteration.UpdatePreds(total_length);

        if (makeout_latency != 0)
          util::latency::Insert_Latency(
              makeout_latency, total_length, stream0,
              mgpu_slice.latency_data.GetPointer(util::DEVICE));

        iteration
            .template MakeOutput<NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>(
                total_length);
      } else {
        mgpu_slice.out_length[0] = total_length;
      }

      for (int peer_ = 0; peer_ < num_gpus; peer_++) {
        enactor_slices[peer_].frontier.queue_length =
            mgpu_slice.out_length[peer_];
#ifdef ENABLE_PERFORMANCE_PROFILING
        // if (peer_ == 0)
        enactor_slices[peer_].enactor_stats.iter_out_length.back().push_back(
            mgpu_slice.out_length[peer_]);
#endif
      }
    }

#ifdef ENABLE_PERFORMANCE_PROFILING
    iter_stop_time = cpu_timer.MillisSinceStart();
    iter_total_time.push_back(iter_stop_time - iter_start_time);
    iter_start_time = iter_stop_time;
#endif
    iteration.Change();
  }
}

/**
 * @brief Thread controls.
 * @tparam Enactor Enactor type we process on.
 * @param[in] thread_data_ Thread data.
 */
template <typename Enactor>
static CUT_THREADPROC GunrockThread(void *thread_data_) {
  // typedef typename Enactor::Problem    Problem   ;
  // typedef typename Enactor::SizeT      SizeT     ;
  // typedef typename Enactor::VertexT    VertexT   ;
  // typedef typename Enactor::ValueT     ValueT    ;
  // typedef typename Problem::GraphT     GraphT    ;
  // typedef typename GraphT ::CsrT       CsrT      ;
  // typedef typename GraphT ::GpT        GpT       ;

  ThreadSlice &thread_data = ((ThreadSlice *)thread_data_)[0];
  // Problem      *problem            =  (Problem*)     thread_data -> problem;
  Enactor &enactor = ((Enactor *)thread_data.enactor)[0];
  // int           num_gpus           =   problem     -> num_gpus;
  int thread_num = thread_data.thread_num;
  int gpu_idx = enactor.gpu_idx[thread_num];
  auto &thread_status = thread_data.status;
  auto &retval = enactor.enactor_slices[thread_num * enactor.num_gpus]
                     .enactor_stats.retval;

  if (retval = util::SetDevice(gpu_idx)) {
    thread_status = ThreadSlice::Status::Ended;
    CUT_THREADEND;
  }

  // util::PrintMsg("Thread entered.");
  thread_status = ThreadSlice::Status::Idle;
  while (thread_status != ThreadSlice::Status::ToKill) {
    while (thread_status == ThreadSlice::Status::Wait ||
           thread_status == ThreadSlice::Status::Idle) {
      // sleep(0);
      std::this_thread::sleep_for(std::chrono::microseconds(0));
      // std::this_thread::yield();
    }
    if (thread_status == ThreadSlice::Status::ToKill) break;

    // util::PrintMsg("Run started");
    enactor.Run(thread_data);
    thread_status = ThreadSlice::Status::Idle;
  }

  thread_status = ThreadSlice::Status::Ended;
  CUT_THREADEND;
}

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
