// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * louvain_enactor.cuh
 *
 * @brief Louvain Problem Enactor
 */

#pragma once

#include <gunrock/util/sort_device.cuh>
#include <gunrock/util/select_device.cuh>
#include <gunrock/util/reduce_device.cuh>
#include <gunrock/util/binary_search.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/louvain/louvain_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace louvain {

/**
 * @brief Speciflying parameters for Louvain Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<uint64_t>(
      "max-passes",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      10, "Maximum number of passes to run the louvain algorithm.", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<uint64_t>(
      "max-iters",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      10, "Maximum number of iterations to run for each pass.", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<double>(
      "pass-th",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1e-4, "Modularity threshold to continue further passes.", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<double>(
      "iter-th",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1e-6,
      "Modularity threshold to continue further iterations within a pass.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "1st-th",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      1e-4,
      "Modularity threshold to continue further iterations in the first pass.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "neighborcomm-th",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      -1.0,
      "Threshold of number of vertex-community pairs changes to quick an "
      "iteration; "
      " value less than 0 will disable this feature",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "pass-stats",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to show per-pass stats.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "iter-stats",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to show per-iteration stats.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "unify-segments",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false,
      "Whether to use cub::RadixSort instead of cub::SegmentedRadixSort.",
      __FILE__, __LINE__));

  return retval;
}

/**
 * @brief defination of Louvain iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct LouvainIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename EnactorT::Problem::EdgePairT EdgePairT;

  typedef typename EnactorT::Problem::GraphT GraphT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  LouvainIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of Louvain, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data alias the enactor works on
    auto &enactor = this->enactor[0];
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    // auto         &graph              =   data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &pass_num = enactor_stats.iteration;
    auto &w_v2 = data_slice.w_v2;
    auto &w_v2self = data_slice.w_v2self;
    auto &w_c2 = data_slice.w_c2;
    auto &current_communities = data_slice.current_communities;
    auto &next_communities = data_slice.next_communities;
    auto &community_sizes = data_slice.community_sizes;
    // auto         &edge_comms0        =   data_slice.edge_comms0;
    auto &edge_comms0 = data_slice.edge_pairs0;
    // auto         &edge_comms1        =   data_slice.edge_comms1;
    auto &edge_comms1 = data_slice.edge_pairs1;
    auto &edge_weights0 = data_slice.edge_weights0;
    auto &edge_weights1 = data_slice.edge_weights1;
    auto &seg_offsets0 = data_slice.seg_offsets0;
    auto &seg_offsets1 = data_slice.seg_offsets1;
    auto &gain_bases = data_slice.gain_bases;
    auto &max_gains = data_slice.max_gains;
    auto &cub_temp_space = data_slice.cub_temp_space;
    auto &num_neighbor_comms = data_slice.num_neighbor_comms;
    auto &edge_pairs0 = data_slice.edge_pairs0;
    auto &edge_pairs1 = data_slice.edge_pairs1;
    auto unify_segments = enactor.unify_segments;
    auto &num_new_comms = data_slice.num_new_comms;
    auto &num_new_edges = data_slice.num_new_edges;
    auto &iter_gain = data_slice.iter_gain;
    auto &pass_communities = data_slice.pass_communities;
    cudaStream_t stream = oprtr_parameters.stream;
    auto target = util::DEVICE;
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    util::CpuTimer iter_timer, pass_timer;

    auto graph_ptr = data_slice.sub_graph;
    if (enactor_stats.iteration != 0)
      graph_ptr = &(data_slice.new_graphs[enactor_stats.iteration % 2]);
    auto &graph = graph_ptr[0];
    auto &weights = graph.CsrT::edge_values;

    if (enactor.pass_stats) pass_timer.Start();
    if (enactor.iter_stats) iter_timer.Start();

    // Pass initialization
    GUARD_CU(w_v2.ForAll(
        [w_v2self, current_communities, community_sizes] __host__ __device__(
            ValueT * w_v2_, const SizeT &v) {
          w_v2_[v] = 0;
          w_v2self[v] = 0;
          current_communities[v] = v;
          community_sizes[v] = 1;
        },
        graph.nodes, target, stream));

    // Accumulate edge values
    auto accu_op = [w_v2, w_v2self, weights] __host__ __device__(
                       const VertexT &src, VertexT &dest, const SizeT &edge_id,
                       const VertexT &input_item, const SizeT &input_pos,
                       SizeT &output_pos) -> bool {
      auto old_weight = atomicAdd(w_v2 + src, weights[edge_id]);
      // printf("w_v2[%d] : %lf -> %lf\n",
      //    src, old_weight, old_weight + weights[edge_id]);
      if (src == dest) atomicAdd(w_v2self + src, weights[edge_id]);
      return false;
    };
    frontier.queue_length = graph.nodes;
    frontier.queue_reset = true;
    // oprtr_parameters.advance_mode = "";
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_frontier, null_frontier, oprtr_parameters, accu_op));

    GUARD_CU(w_c2.ForAll(
        [w_v2] __host__ __device__(ValueT * w_c, const VertexT &v) {
          w_c[v] = w_v2[v];
          // w_c = w_v;
          // printf("w_v2[%d] = %lf\n",
          //    v, w_v2[v]);
        },
        graph.nodes, target, stream));

    if (enactor.iter_stats) {
      iter_timer.Stop();
      util::PrintMsg("Pass " + std::to_string(pass_num) +
                     ", pre-iter, elapsed = " +
                     std::to_string(iter_timer.ElapsedMillis()));
    }

    int iter_num = 0;
    ValueT pass_gain = 0;
    bool to_continue = true;
    SizeT pervious_num_neighbor_comms = 0;

    // Iterations
    while (to_continue) {
      if (enactor.iter_stats) iter_timer.Start();

      if (unify_segments) {
        GUARD_CU(edge_pairs0.ForAll(
            [edge_weights0, weights, current_communities,
             graph] __host__ __device__(EdgePairT * e_pairs, const SizeT &e) {
              VertexT src, dest;
              graph.GetEdgeSrcDest(e, src, dest);
              e_pairs[e] = ProblemT::MakePair(src, current_communities[dest]);
              // edge_weights0[e] = weights[e];
            },
            graph.edges, target, stream));

        GUARD_CU(util::cubSortPairs(cub_temp_space, edge_pairs0, edge_pairs1,
                                    weights, edge_weights1, graph.edges, 0,
                                    sizeof(EdgePairT) * 8, stream));

        GUARD_CU(seg_offsets0.Set(0, graph.edges + 1, target, stream));

        GUARD_CU(seg_offsets0.ForAll(
            [edge_pairs1, graph] __host__ __device__(SizeT * offsets,
                                                     const SizeT &e) {
              bool to_keep = false;
              if (e == 0 || e == graph.edges)
                to_keep = true;
              else {
                EdgePairT pair = edge_pairs1[e];
                EdgePairT pervious_pair = edge_pairs1[e - 1];
                if (ProblemT::GetFirst(pair) !=
                        ProblemT::GetFirst(pervious_pair) ||
                    ProblemT::GetSecond(pair) !=
                        ProblemT::GetSecond(pervious_pair))
                  to_keep = true;
              }

              offsets[e] =
                  (to_keep) ? e : util::PreDefinedValues<SizeT>::InvalidValue;
            },
            graph.edges + 1, target, stream));

      } else {
        GUARD_CU(edge_comms0.ForAll(
            [edge_weights0, weights, current_communities,
             graph] __host__ __device__(EdgePairT * e_comms, const SizeT &e) {
              e_comms[e] = current_communities[graph.GetEdgeDest(e)];
              edge_weights0[e] = weights[e];
            },
            graph.edges, target, stream));

        GUARD_CU(util::cubSegmentedSortPairs(
            cub_temp_space, edge_comms0, edge_comms1, edge_weights0,
            edge_weights1, graph.edges, graph.nodes, graph.CsrT::row_offsets, 0,
            std::ceil(std::log2(graph.nodes)), stream));

        GUARD_CU(seg_offsets0.Set(0, graph.edges + 1, target, stream));

        GUARD_CU(graph.CsrT::row_offsets.ForAll(
            [seg_offsets0] __host__ __device__(SizeT * offsets,
                                               const SizeT &v) {
              seg_offsets0[offsets[v]] = 1;
            },
            graph.nodes + 1, target, stream));

        GUARD_CU(seg_offsets0.ForAll(
            [edge_comms1] __host__ __device__(SizeT * offsets, const SizeT &e) {
              bool to_keep = false;
              if (offsets[e] == 1)
                to_keep = true;
              else if (e == 0)
                to_keep = true;
              else if (edge_comms1[e] != edge_comms1[e - 1])
                to_keep = true;

              offsets[e] =
                  (to_keep) ? e : util::PreDefinedValues<SizeT>::InvalidValue;
            },
            graph.edges + 1, target, stream));
      }

      // Filter in order
      GUARD_CU(util::cubSelectIf(
          cub_temp_space, seg_offsets0, seg_offsets1, num_neighbor_comms,
          graph.edges + 1,
          [] __host__ __device__(const SizeT &e) { return util::isValid(e); },
          stream));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");

      GUARD_CU(num_neighbor_comms.Move(util::DEVICE, util::HOST, 1, 0, stream));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
      auto n_neighbor_comms = num_neighbor_comms[0] - 1;
      // util::PrintMsg("num_neigbhor_comms = " +
      // std::to_string(n_neighbor_comms));

      // GUARD_CU(oprtr::Set(neighbor_comm_offsets.GetPointer(util::DEVICE)
      //    + (num_neighbor_comms[0] + 1), graph.edges,
      //    graph.edges - num_neighbor_comms[0] - 1,
      //    target, stream));

      GUARD_CU(util::SegmentedReduce(
          cub_temp_space, edge_weights1, edge_weights0, n_neighbor_comms,
          seg_offsets1,
          [] __host__ __device__(const ValueT &a, const ValueT &b) {
            return a + b;
          },
          (ValueT)0, stream));

      GUARD_CU(seg_offsets0.ForAll(
          [seg_offsets1, n_neighbor_comms, graph] __host__ __device__(
              SizeT * offsets, const VertexT &v) {
            if (v == graph.nodes)
              offsets[v] = n_neighbor_comms;
            else
              offsets[v] = util::BinarySearch_LeftMost(
                  graph.GetNeighborListOffset(v), seg_offsets1, (SizeT)0,
                  n_neighbor_comms + 1, false);  //,
            //[] (const SizeT &a, const SizeT &b)
            //{
            //    return a < b;
            //});
            // if (offsets[v] != graph.row_offsets[v])
            // printf("offsets[%d] <- %d, row_offsets[v] = %d\n",
            //    v, offsets[v], graph.row_offsets[v]);
          },
          graph.nodes + 1, target, stream));

      auto m2 = data_slice.m2;
      GUARD_CU(gain_bases.ForAll(
          [seg_offsets0, edge_weights0, edge_comms1, seg_offsets1, w_v2,
           current_communities, w_v2self, m2, w_c2, unify_segments,
           edge_pairs1] __host__ __device__(ValueT * bases, const VertexT &v) {
            SizeT start_pos = seg_offsets0[v];
            SizeT end_pos = seg_offsets0[v + 1];
            VertexT org_comm = current_communities[v];
            ValueT w_v2c_org = 0;
            // printf("seg_range0[%d] = [%d, %d)\n",
            //    v, start_pos, end_pos);
            for (SizeT pos = start_pos; pos < end_pos; pos++) {
              SizeT seg_start = seg_offsets1[pos];
              VertexT comm =
                  (unify_segments ? ProblemT::GetSecond(edge_pairs1[seg_start])
                                  : edge_comms1[seg_start]);
              // printf("seg %d: v %d -> c %d, w_v2c = %f\n",
              //    pos, v, comm, edge_weights0[pos]);
              if (org_comm == comm) {
                w_v2c_org = edge_weights0[pos];
                break;
              }
            }

            ValueT w_v2_v = w_v2[v];
            VertexT comm = current_communities[v];
            bases[v] =
                w_v2self[v] - w_v2c_org - (w_v2_v - w_c2[comm]) * w_v2_v / m2;
            // printf("bases[%d] = %lf, w_v2[v] = %lf, comm = %d, "
            //    "w_v2c_org = %lf, w_c2[comm] = %lf, m2 = %lf\n",
            //    v, bases[v], w_v2_v, comm, w_v2c_org, w_c2[comm], m2);
          },
          graph.nodes, target, stream));

      GUARD_CU(max_gains.ForAll(
          [next_communities, current_communities] __host__ __device__(
              ValueT * gains, const VertexT &v) {
            gains[v] = 0;
            next_communities[v] = current_communities[v];
          },
          graph.nodes, target, stream));

      GUARD_CU(edge_weights0.ForAll(
          [seg_offsets0, seg_offsets1, n_neighbor_comms, gain_bases, w_c2, w_v2,
           m2, current_communities, max_gains, next_communities, edge_comms1,
           graph, unify_segments,
           edge_pairs1] __host__ __device__(ValueT * w_v2c, const SizeT &pos) {
            VertexT v = util::BinarySearch_RightMost(
                pos, seg_offsets0, (SizeT)0, graph.nodes, false);  //,
            //[] (const SizeT &a, const SizeT &b)
            //{
            //    return a < b;
            //});
            // if (pos < seg_offsets0[v] && v > 0)
            //    v--;
            // printf("seg %d: v = %d, seg_offsets0[v] = %d\n",
            //    pos, v, seg_offsets0[v]);

            VertexT comm =
                unify_segments
                    ? ProblemT::GetSecond(edge_pairs1[seg_offsets1[pos]])
                    : edge_comms1[seg_offsets1[pos]];
            ValueT gain = 0;
            if (comm != current_communities[v]) {
              gain = gain_bases[v] + w_v2c[pos] - w_c2[comm] * w_v2[v] / m2;
              ValueT old_gain = atomicMax(max_gains + v, gain);
              // printf("seg %d: v %d -> c %d, gain = %lf, gain_bases = %lf, "
              //    "w_v2c = %lf, w_c2 = %lf, w_v2 = %lf, old_gain = %lf\n",
              //    pos, v, comm, gain, gain_bases[v], w_v2c[pos],
              //    w_c2[comm], w_v2[v], old_gain);

              if (old_gain >= gain) gain = 0;
              // else
              //    next_communities[v] = comm;
            }
            w_v2c[pos] = gain;
          },
          n_neighbor_comms, target, stream));

      GUARD_CU(edge_weights0.ForAll(
          [max_gains, next_communities, seg_offsets0, graph, seg_offsets1,
           edge_comms1, unify_segments,
           edge_pairs1] __host__ __device__(ValueT * gains, const SizeT &pos) {
            auto gain = gains[pos];
            if (gain < 1e-8) return;

            VertexT v = util::BinarySearch_LeftMost(pos, seg_offsets0, (SizeT)0,
                                                    graph.nodes, false);  //,
            //[] (const SizeT &a, const SizeT &b)
            //{
            //    return a < b;
            //});
            // if (pos < seg_offsets0[v] && v > 0)
            //    v--;
            if (abs(max_gains[v] - gain) > 1e-8) return;

            next_communities[v] =
                unify_segments
                    ? ProblemT::GetSecond(edge_pairs1[seg_offsets1[pos]])
                    : edge_comms1[seg_offsets1[pos]];
            // if (next_communities[v] >= graph.nodes)
            //    printf("Invalid comm: next_comm[%d] = %d, seg_offsets1[%d] =
            //    %d\n",
            //        v, next_communities[v],
            //        pos, seg_offsets1[pos]);
          },
          n_neighbor_comms, target, stream));

      GUARD_CU(current_communities.ForAll(
          [next_communities, community_sizes, max_gains, w_v2,
           w_c2] __host__ __device__(VertexT * communities, const VertexT &v) {
            VertexT c_comm = communities[v];
            VertexT n_comm = next_communities[v];
            if (c_comm == v &&
                community_sizes[v] + community_sizes[n_comm] <= 2) {
              if (n_comm > v && next_communities[n_comm] == v) {
                max_gains[v] = 0;
                return;
              }
            }
            if (c_comm == n_comm) {
              return;
              max_gains[v] = 0;
            }
            // printf("v %d : c %d -> c %d, gain = %lf\n",
            //    v, c_comm, n_comm, max_gains[v]);

            atomicSub(community_sizes + c_comm, 1);
            atomicAdd(community_sizes + n_comm, 1);
            auto w_v2v = w_v2[v];
            atomicAdd(w_c2 + n_comm, w_v2v);
            atomicAdd(w_c2 + c_comm, -1 * w_v2v);
            communities[v] = n_comm;
          },
          graph.nodes, target, stream));

      GUARD_CU(util::cubReduce(
          cub_temp_space, max_gains, iter_gain, graph.nodes,
          [] __host__ __device__(const ValueT &a, const ValueT &b) {
            return a + b;
          },
          (ValueT)0, stream));

      // GUARD_CU(iter_gain.ForEach(
      //    [] __host__ __device__ (const ValueT &gain)
      //    {
      //        printf("iter_gain = %f\n", gain);
      //    }, 1, target, stream));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
      // GUARD_CU2(cudaDeviceSynchronize(),
      //    "cudaDeviceSynchronize failed.");
      GUARD_CU(iter_gain.Move(util::DEVICE, util::HOST, 1, 0, stream));
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
      // GUARD_CU2(cudaDeviceSynchronize(),
      //    "cudaDeviceSynchronize failed.");

      // printf("gain0 = %f\n", (double)(iter_gain[0]));
      iter_gain[0] *= 2;
      // printf("gain1 = %lf\n", iter_gain[0]);
      iter_gain[0] /= data_slice.m2;
      // printf("gain2 = %lf\n", iter_gain[0]);
      data_slice.q += iter_gain[0];
      pass_gain += iter_gain[0];
      if (enactor.iter_stats) {
        iter_timer.Stop();
        util::PrintMsg(
            "pass " + std::to_string(pass_num) + ", iter " +
            std::to_string(iter_num) + ", q = " + std::to_string(data_slice.q) +
            ", iter_gain = " + std::to_string(iter_gain[0]) +
            ", pass_gain = " + std::to_string(pass_gain) +
            ", #neighbor_comms = " + std::to_string(n_neighbor_comms) +
            ", elapsed = " + std::to_string(iter_timer.ElapsedMillis()));
      }
      iter_num++;
      if ((pass_num != 0 && iter_gain[0] < enactor.iter_gain_threshold) ||
          (pass_num == 0 && iter_gain[0] < enactor.first_threshold) ||
          iter_num >= enactor.max_iters ||
          (enactor.neighborcomm_threshold > 0 && iter_num != 1 &&
           pass_num == 0 &&
           n_neighbor_comms > (1 - enactor.neighborcomm_threshold) *
                                  pervious_num_neighbor_comms))
        to_continue = false;
      pervious_num_neighbor_comms = n_neighbor_comms;
    }
    data_slice.pass_gain = pass_gain;

    if (enactor.iter_stats) iter_timer.Start();

    // Graph contraction
    GUARD_CU(edge_comms0.ForEach(
        [] __host__ __device__(EdgePairT & comm) {
          comm = util::PreDefinedValues<EdgePairT>::InvalidValue;
        },
        graph.nodes, target, stream));

    GUARD_CU(edge_comms0.ForAll(
        [current_communities] __host__ __device__(EdgePairT * comms0,
                                                  const SizeT &v) {
          VertexT comm = current_communities[v];
          comms0[comm] = comm;
          // if (comm == 8278)
          //    printf("Comm[%d] = %d\n", v, comm);
        },
        graph.nodes, target, stream));

    GUARD_CU(util::cubSelectIf(cub_temp_space, edge_comms0, edge_comms1,
                               num_new_comms, graph.nodes,
                               [] __host__ __device__(EdgePairT & comm) {
                                 // if (comm == 8278)
                                 //    printf("Comm %d, valid = %s\n", comm,
                                 //        util::isValid(comm) ? "True" :
                                 //        "False");
                                 return (util::isValid(comm));
                               },
                               stream));

    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

    GUARD_CU(num_new_comms.Move(target, util::HOST, 1, 0, stream));

    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    auto n_new_comms = num_new_comms[0];
    // util::PrintMsg("#new_comms = " + std::to_string(n_new_comms));

    GUARD_CU(edge_comms0.ForAll(
        [edge_comms1, n_new_comms] __host__ __device__(EdgePairT * comms0,
                                                       const SizeT &new_comm) {
          comms0[edge_comms1[new_comm]] = new_comm;
          // if (edge_comms1[new_comm] == 8278)
          //    printf("Comms0[%d] = %d\n",
          //        edge_comms1[new_comm], new_comm);
        },
        n_new_comms, target, stream));

    // GUARD_CU(edge_comms0.ForAll(
    //    [] __host__ __device__ (VertexT *comms0, VertexT &v)
    //    {
    //        printf("edge_Comms0[8278] = %d\n",
    //            comms0[8278]);
    //    }, 1, target, stream));

    GUARD_CU(current_communities.ForAll(
        [edge_comms0, n_new_comms] __host__ __device__(VertexT * comms,
                                                       const VertexT &v) {
          VertexT comm = comms[v];
          comms[v] = edge_comms0[comm];
          // if (comms[v] >= n_new_comms)
          //    printf("Invalid comm: %d -> %d, edge_comms0 = %d\n",
          //        comm, comms[v], edge_comms0[comm]);
        },
        graph.nodes, target, stream));

    auto null_ptr = &current_communities;
    null_ptr = NULL;
    frontier.queue_length = graph.nodes;
    frontier.queue_reset = true;
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_ptr, null_ptr, oprtr_parameters,
        [edge_comms0, current_communities, edge_pairs0] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
          VertexT src_comm = current_communities[src];
          // src_comm = edge_comms0[src_comm];
          VertexT dest_comm = current_communities[dest];
          // dest_comm = edge_comms0[dest_comm];

          edge_pairs0[edge_id] =
              (((EdgePairT)src_comm) << (sizeof(VertexT) * 8)) +
              (EdgePairT)dest_comm;
          return false;
        }));

    GUARD_CU(util::cubSortPairs(cub_temp_space, edge_pairs0, edge_pairs1,
                                graph.CsrT::edge_values, edge_weights1,
                                graph.edges, 0, sizeof(EdgePairT) * 8, stream));

    GUARD_CU(seg_offsets0.ForEach(
        [] __host__ __device__(SizeT & offset) {
          offset = util::PreDefinedValues<SizeT>::InvalidValue;
        },
        graph.edges + 1, target, stream));

    GUARD_CU(seg_offsets0.ForAll(
        [graph, edge_pairs1] __host__ __device__(SizeT * offsets0,
                                                 const SizeT &e) {
          if (e != 0 && e != graph.edges) {
            auto edge = edge_pairs1[e];
            auto pervious_edge = edge_pairs1[e - 1];
            if (edge == pervious_edge) return;
          }
          offsets0[e] = e;
        },
        graph.edges + 1, target, stream));

    GUARD_CU(util::cubSelectIf(cub_temp_space, seg_offsets0, seg_offsets1,
                               num_new_edges, graph.edges + 1,
                               [] __host__ __device__(SizeT & offset) {
                                 return (util::isValid(offset));
                               },
                               stream));

    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

    GUARD_CU(num_new_edges.Move(target, util::HOST, 1, 0, stream));
    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    auto n_new_edges = num_new_edges[0] - 1;
    // util::PrintMsg("#new_edges = " + std::to_string(n_new_edges));

    // GraphT *new_graph = new GraphT;
    auto &new_graph = data_slice.new_graphs[(pass_num + 1) % 2];
    if (n_new_comms > new_graph.CsrT::row_offsets.GetSize() - 1 ||
        n_new_edges > new_graph.CsrT::column_indices.GetSize()) {
      GUARD_CU(new_graph.CsrT::Allocate(n_new_comms * 1.1, n_new_edges * 1.1,
                                        util::DEVICE));
    }  // else {
    new_graph.nodes = n_new_comms;
    new_graph.edges = n_new_edges;
    new_graph.CsrT::nodes = n_new_comms;
    new_graph.CsrT::edges = n_new_edges;
    //}

    GUARD_CU(util::SegmentedReduce(
        cub_temp_space, edge_weights1, new_graph.CsrT::edge_values, n_new_edges,
        seg_offsets1,
        [] __host__ __device__(const ValueT &a, const ValueT &b) {
          return a + b;
        },
        (ValueT)0, stream));

    GUARD_CU(new_graph.CsrT::column_indices.ForAll(
        [edge_pairs1, seg_offsets1, n_new_comms] __host__ __device__(
            VertexT * indices, const SizeT &e) {
          indices[e] = edge_pairs1[seg_offsets1[e]] &
                       util::PreDefinedValues<VertexT>::AllOnes;
          // if (indices[e] >= n_new_comms)
          //    printf("Invalid dest: %d, e = %d, pair = %ld\n",
          //        indices[e], e, edge_pairs1[seg_offsets1[e]]);
        },
        n_new_edges, target, stream));

    auto &new_row_offsets = new_graph.CsrT::row_offsets;
    GUARD_CU(seg_offsets1.ForAll(
        [new_row_offsets, edge_pairs1, n_new_comms,
         n_new_edges] __host__ __device__(SizeT * offsets, const SizeT &new_e) {
          VertexT src = 0, pervious_src = 0;
          if (new_e != n_new_edges)
            src = edge_pairs1[offsets[new_e]] >> (sizeof(VertexT) * 8);
          else
            src = n_new_comms;

          if (new_e != 0)
            pervious_src =
                edge_pairs1[offsets[new_e - 1]] >> (sizeof(VertexT) * 8);

          if (src == pervious_src) return;
          for (VertexT new_v = (new_e == 0 ? 0 : (pervious_src + 1));
               new_v <= src; new_v++)
            new_row_offsets[new_v] = new_e;
        },
        n_new_edges + 1, target, stream));
    GUARD_CU(new_row_offsets.ForAll(
        [] __host__ __device__(SizeT * offsets, const VertexT &v) {
          offsets[0] = 0;
        },
        1, target, stream));

    if (enactor.iter_stats) {
      iter_timer.Stop();
      util::PrintMsg("pass " + std::to_string(pass_num) +
                     ", graph compaction, elapsed = " +
                     std::to_string(iter_timer.ElapsedMillis()));
    }

    if (enactor.pass_stats) {
      pass_timer.Stop();
      util::PrintMsg(
          "pass " + std::to_string(pass_num) + ", #v = " +
          std::to_string(graph.nodes) + " -> " + std::to_string(n_new_comms) +
          ", #e = " + std::to_string(graph.edges) + " -> " +
          std::to_string(n_new_edges) + ", #iter = " +
          std::to_string(iter_num) + ", q = " + std::to_string(data_slice.q) +
          ", pass_gain = " + std::to_string(pass_gain) +
          ", elapsed = " + std::to_string(pass_timer.ElapsedMillis()));
    }
    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

    // util::Array1D<SizeT, VertexT> &pass_comm = new util::Array1D<SizeT,
    // VertexT>;
    auto &pass_comm = data_slice.pass_communities[pass_num];
    GUARD_CU(pass_comm.EnsureSize_(graph.nodes, util::HOST));
    GUARD_CU2(cudaMemcpyAsync(pass_comm.GetPointer(util::HOST),
                              current_communities.GetPointer(util::DEVICE),
                              sizeof(VertexT) * graph.nodes,
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync failed.");
    data_slice.num_pass = pass_num;

    // pass_communities.push_back(pass_comm);

    GUARD_CU(new_graph.FromCsr(new_graph.csr(), target, stream, true, true));
    // GUARD_CU(new_graph.csr().Move(target, util::HOST, stream));
    if (enactor_stats.iteration != 0) {
      // util::PrintMsg("Release graph");
      // GUARD_CU(data_slice.new_graph -> Release(target));
      // delete data_slice.new_graph;
    }
    // data_slice.new_graph = new_graph;

    // GUARD_CU2(cudaStreamSynchronize(stream),
    //    "cudaStreamSynchronize failed");
    // GUARD_CU(new_graph -> csr().Display());
    // util::PrintMsg("Pass finished");

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    int num_gpus = this->enactor->num_gpus;
    auto &enactor_slices = this->enactor->enactor_slices;

    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
      auto &retval = enactor_slices[gpu].enactor_stats.retval;
      if (retval == cudaSuccess) continue;
      printf("(CUDA error %d @ GPU %d: %s\n", retval, gpu % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor->enactor_slices[this->gpu_num * this->enactor->num_gpus];
    ValueT pass_gain_threshold =
        this->enactor->problem->parameters.template Get<ValueT>("pass-th");

    // printf("iter = %lld, pass_gain = %lf\n",
    //    enactor_slice.enactor_stats.iteration, data_slice.pass_gain);

    if (enactor_slice.enactor_stats.iteration >= data_slice.max_iters)
      return true;

    if (enactor_slice.enactor_stats.iteration <= 1 ||
        data_slice.pass_gain >= pass_gain_threshold)
      return false;

    return true;
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

    cudaError_t retval = LouvainIterationLoop::template ExpandIncomingBase<
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>(received_length, peer_,
                                                      expand_op);
    return retval;
  }
};  // end of LouvainIteration

/**
 * @brief Louvain enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT,
          typename _Problem::GraphT::VertexT,  // TODO: change to other label
                                               // types used for the operators,
                                               // e.g.: typename
                                               // _Problem::LabelT,
          typename _Problem::GraphT::ValueT,   // TODO: change to other value
                                               // types used for inter GPU
                                               // communication, e.g.: typename
                                               // _Problem::ValueT,
          ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT
      LabelT;  // e.g. typedef typename Problem::LabelT LabelT;
  typedef typename GraphT::ValueT
      ValueT;  // e.g. typedef typename Problem::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef LouvainIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  VertexT max_passes;
  VertexT max_iters;
  bool pass_stats;
  bool iter_stats;
  bool unify_segments;
  ValueT pass_gain_threshold;
  ValueT iter_gain_threshold;
  ValueT first_threshold;
  ValueT neighborcomm_threshold;

  /**
   * @brief LouvainEnactor constructor
   */
  Enactor() : BaseEnactor("Louvain"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief LouvainEnactor destructor
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
   * @param[in] parameters Running parameters.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(
      // util::Parameters &parameters,
      Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    max_passes = problem.parameters.template Get<VertexT>("max-passes");
    max_iters = problem.parameters.template Get<VertexT>("max-iters");
    pass_stats = problem.parameters.template Get<bool>("pass-stats");
    iter_stats = problem.parameters.template Get<bool>("iter-stats");
    pass_gain_threshold = problem.parameters.template Get<ValueT>("pass-th");
    iter_gain_threshold = problem.parameters.template Get<ValueT>("iter-th");
    first_threshold = problem.parameters.template Get<ValueT>("1st-th");
    unify_segments = problem.parameters.template Get<bool>("unify-segments");
    neighborcomm_threshold =
        problem.parameters.template Get<ValueT>("neighborcomm-th");

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 0, NULL, target, false));
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

  cudaError_t Check_Queue_Size(int peer_) {
    // no need to check queue size for PR
    return cudaSuccess;
  }

  /**
   * @brief one run of Louvain, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // TODO: change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 0, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(
      // TODO: add problem specific info, e.g.:
      // VertexT src,
      util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    // TODO: Initialize frontiers according to the algorithm, e.g.:
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      //    if ((this->num_gpus == 1) ||
      //         (gpu == this->problem->org_graph->GpT::partition_table[src]))
      //    {
      //        this -> thread_slices[gpu].init_size = 1;
      //        for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
      //        {
      //            auto &frontier = this ->
      //                enactor_slices[gpu * this -> num_gpus + peer_].frontier;
      //            frontier.queue_length = (peer_ == 0) ? 1 : 0;
      //            if (peer_ == 0)
      //            {
      //                GUARD_CU(frontier.V_Q() -> ForEach(
      //                    [src]__host__ __device__ (VertexT &v)
      //                {
      //                    v = src;
      //                }, 1, target, 0));
      //            }
      //        }
      //    }
      //
      //    else {
      this->thread_slices[gpu].init_size = 1;
      for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
        this->enactor_slices[gpu * this->num_gpus + peer_]
            .frontier.queue_length = 1;
      }
      //    }
    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a Louvain computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Louvain Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace louvain
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
