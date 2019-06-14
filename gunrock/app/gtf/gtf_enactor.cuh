// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtf_enactor.cuh
 *
 * @brief Max Flow Problem Enactor
 */

#pragma once
#include <gunrock/util/sort_device.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/gtf/gtf_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#define debug_aml(a...)
#include <gunrock/app/mf/mf_enactor.cuh>
#include <gunrock/app/gtf/gtf_test.cuh>

//#define debug_aml(a...) \
  {printf("%s:%d ", __FILE__, __LINE__); printf(a); printf("\n");}

namespace gunrock {
namespace app {
namespace gtf {

/**
 * @brief Speciflying parameters for gtf Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 *		      info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of gtf iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct GTFIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT GraphT;
  typedef typename GraphT::CsrT CsrT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  GTFIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of gtf, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    auto enactor = this->enactor;
    auto gpu_num = this->gpu_num;
    auto num_gpus = enactor->num_gpus;
    auto gpu_offset = num_gpus * gpu_num;
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];

    // MF specific
    auto &mf_data_slice = enactor->problem->mf_problem.data_slices[gpu_num][0];
    auto &mf_problem = enactor->problem->mf_problem;
    auto &mf_enactor = enactor->mf_enactor;
    auto &mf_flow = mf_data_slice.flow;
    auto mf_target = util::DEVICE;
    auto &h_reverse = data_slice.reverse;

    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    //!!! allowed?
    auto num_nodes = graph.nodes;                 // n + 2 = V
    auto num_org_nodes = num_nodes - 2;           // n
    auto num_edges = graph.edges;                 // m + n*4
    auto offset = num_edges - (num_org_nodes)*2;  //!!!
    auto source = data_slice.source;
    auto sink = data_slice.sink;

    auto &next_communities = data_slice.next_communities;
    auto &curr_communities = data_slice.curr_communities;
    auto &community_sizes = data_slice.community_sizes;
    auto &community_weights = data_slice.community_weights;
    auto &community_active = data_slice.community_active;
    auto &community_accus = data_slice.community_accus;
    auto &vertex_active = data_slice.vertex_active;
    auto &vertex_reachabilities = data_slice.vertex_reachabilities;

    auto &edge_residuals = data_slice.edge_residuals;
    auto &edge_flows = data_slice.edge_flows;
    auto &active = data_slice.active;
    auto &num_comms = data_slice.num_comms;
    auto &previous_num_comms = data_slice.previous_num_comms;
    auto &num_updated_vertices = data_slice.num_updated_vertices;
    auto &Y = data_slice.Y;
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    /*
    printf("iteration %d \n", iteration);
    GUARD_CU(edge_residuals.ForAll(
       [mf_flow, graph, source] __host__ __device__ (ValueT *edge_residuals,
    const SizeT &e){ if(e < 10) printf("GPU: e_idx %d, e_val %f\n", e,
    graph.edge_values[e]);
           //edge_residuals[e] = graph.edge_values[e]; // just for debugging
    purposes #!!!
       }, graph.edges, util::DEVICE, oprtr_parameters.stream));
    */

    cpu_timer.Start();
    GUARD_CU(graph.edge_values.Move(util::DEVICE, util::HOST, graph.edges, 0,
                                    oprtr_parameters.stream));
    GUARD_CU(cudaDeviceSynchronize());
    cpu_timer.Stop();
    // printf("move: %f \n", cpu_timer.ElapsedMillis());

    mf_problem.parameters.Set("source", source);
    mf_problem.parameters.Set("sink", sink);
    cpu_timer.Start();
    GUARD_CU(mf_problem.Reset(graph, h_reverse + 0, mf_target));
    GUARD_CU(cudaDeviceSynchronize());
    cpu_timer.Stop();
    // printf("problem reset: %f \n", cpu_timer.ElapsedMillis());

    cpu_timer.Start();
    GUARD_CU(mf_enactor.Reset(source, mf_target));
    GUARD_CU(cudaDeviceSynchronize());
    cpu_timer.Stop();
    // printf("enact reset: %f \n", cpu_timer.ElapsedMillis());

    cpu_timer.Start();
    GUARD_CU(mf_enactor.Enact());
    GUARD_CU(cudaDeviceSynchronize());
    cpu_timer.Stop();
    // printf("mf: %f \n", cpu_timer.ElapsedMillis());

    cpu_timer.Start();
    // min cut
    GUARD_CU(edge_residuals.ForAll(
        [mf_flow, graph, source] __host__ __device__(ValueT * edge_residuals,
                                                     const SizeT &e) {
          // if(e == 0) printf("in residual assignment beginning of gtf\n");
          edge_residuals[e] = graph.edge_values[e] - mf_flow[e];
          mf_flow[e] = 0.;
          // if(e < 10)printf("GPU: er_idx %d, e_res %f \n", e,
          // edge_residuals[e]);
        },
        graph.edges, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(vertex_reachabilities.ForAll(
        [] __host__ __device__(bool *vertex_reachabilities, const SizeT &v) {
          vertex_reachabilities[v] = false;
          // if(v == 0) printf("in reach\n");
        },
        graph.nodes, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(vertex_reachabilities.ForAll(
        [edge_residuals, graph, community_sizes, source] __host__ __device__(
            bool *vertex_reachabilities, const SizeT &idx) {
          VertexT head = 0;
          VertexT tail = 0;
          VertexT *queue = community_sizes + 0;
          queue[head] = source;
          while (tail <= head) {
            VertexT v = queue[tail];
            auto e_start = graph.GetNeighborListOffset(v);
            auto num_neighbors = graph.GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; e++) {
              VertexT u = graph.GetEdgeDest(e);
              if (vertex_reachabilities[u] == false &&
                  abs(edge_residuals[e]) > 1e-6) {
                head++;
                queue[head] = u;
                vertex_reachabilities[u] = true;
              }
            }
            tail++;
          }
          // if(idx == 0) printf("in min-cut\n");
        },
        1, util::DEVICE, oprtr_parameters.stream));

    //////////////////////////////////////////////////

    GUARD_CU(community_weights.ForAll(
        [vertex_active,  // vertex specific
         vertex_reachabilities,
         next_communities,  // community specific
         curr_communities, community_active, community_sizes, community_accus,
         edge_residuals,  // intermediate output
         // others
         num_comms, num_edges, num_org_nodes, graph,
         active] __host__ __device__(ValueT * community_weights,
                                     const VertexT &idx) {
          {
            auto &edge_capacities = graph.edge_values;
            unsigned int comm;
            for (comm = 0; comm < num_comms[0]; comm++) {
              community_weights[comm] = 0;
              community_sizes[comm] = 0;
              next_communities[comm] = 0;
            }
            auto pervious_num_comms = num_comms[0];

            for (VertexT v = 0; v < num_org_nodes; v++) {
              if (!vertex_active[v]) continue;
              if (vertex_reachabilities[v] == 1) {  // reachable by source
                comm = next_communities[curr_communities[v]];
                if (comm == 0) {  // not assigned yet
                  comm = num_comms[0];
                  next_communities[curr_communities[v]] = num_comms[0];
                  community_active[comm] = true;
                  num_comms[0]++;
                  community_weights[comm] = 0;
                  community_sizes[comm] = 0;
                  next_communities[comm] = 0;
                  community_accus[comm] = community_accus[curr_communities[v]];
                }
                curr_communities[v] = comm;
                community_weights[comm] +=
                    edge_residuals[num_edges - num_org_nodes * 2 + v];
                community_sizes[comm]++;
                // printf("++ %d %f %f\n", comm, community_weights[comm],
                // community_accus[comm]);
              } else {  // otherwise
                comm = curr_communities[v];
                SizeT e_start = graph.GetNeighborListOffset(v);
                SizeT num_neighbors = graph.GetNeighborListLength(v);
                community_weights[comm] -=
                    edge_residuals[e_start + num_neighbors - 1];
                community_sizes[comm]++;

                auto e_end = e_start + num_neighbors - 2;
                for (auto e = e_start; e < e_end; e++) {
                  VertexT u = graph.GetEdgeDest(e);
                  if (vertex_reachabilities[u] == 1) {
                    edge_residuals[e] = 0;
                  }
                }
                // printf("-- %d %f %f\n", comm, community_weights[comm],
                // community_accus[comm]);
              }
            }  // end of for v
            // printf("%d %f %f\n", comm, community_weights[comm],
            // community_accus[comm]);

            for (comm = 0; comm < pervious_num_comms; comm++) {
              if (community_active[comm]) {
                if (next_communities[comm] == 0) {
                  community_weights[comm] = 0;
                  community_active[comm] = false;
                } else if (community_sizes[comm] == 0) {
                  community_active[comm] = false;
                  community_active[next_communities[comm]] = false;
                  community_weights[next_communities[comm]] = 0;
                } else {
                  // printf("values: comm: %d, sizes: %d, weights: %f, accus:
                  // %f.\n",
                  //    comm, community_sizes[comm], community_weights[comm],
                  //    community_accus[comm]);
                  community_weights[comm] /= community_sizes[comm];
                  community_accus[comm] += community_weights[comm];
                }
              } else {
                community_weights[comm] = 0;
              }
            }

            for (; comm < num_comms[0]; comm++) {
              community_weights[comm] /= community_sizes[comm];
              community_accus[comm] += community_weights[comm];
              // printf("comm %d, accus %f, sizes %d \n",
              //    comm, community_accus  [comm], community_sizes  [comm]);
              // printf("values: comm: %d, sizes: %d, weights: %f, accus:
              // %f.\n",
              //    comm, community_sizes[comm], community_weights[comm],
              //    community_accus[comm]);
            }

            active[0] = false;
            for (VertexT v = 0; v < num_org_nodes; v++) {
              if (!vertex_active[v]) continue;

              auto comm = curr_communities[v];
              if (!community_active[comm] ||
                  abs(community_weights[comm]) <= 1e-6) {
                if (vertex_reachabilities[v] == 1)
                  edge_residuals[num_edges - num_org_nodes * 2 + v] = 0;

                if (vertex_reachabilities[v] != 1) {
                  SizeT e = graph.GetNeighborListOffset(v) +
                            graph.GetNeighborListLength(v) - 1;
                  edge_residuals[e] = 0;
                }
                vertex_active[v] = false;
                community_active[comm] = false;
              } else {
                active[0] = true;
                SizeT e_from_src = num_edges - num_org_nodes * 2 + v;
                SizeT e_to_dest = graph.GetNeighborListOffset(v) +
                                  graph.GetNeighborListLength(v) - 1;
                if (vertex_reachabilities[v] == 1) {
                  edge_residuals[e_from_src] -= community_weights[comm];
                  if (edge_residuals[e_from_src] < 0) {
                    double temp = -1 * edge_residuals[e_from_src];
                    edge_residuals[e_from_src] = edge_residuals[e_to_dest];
                    edge_residuals[e_to_dest] = temp;
                  }
                } else {
                  edge_residuals[e_to_dest] += community_weights[comm];
                  if (edge_residuals[e_to_dest] < 0) {
                    double temp = -1 * edge_residuals[e_to_dest];
                    edge_residuals[e_to_dest] = edge_residuals[e_from_src];
                    edge_residuals[e_from_src] = temp;
                  }
                }
              }
            }  // end of for v

            // for (SizeT e = 0; e < graph.edges; e ++){
            //  edge_capacities[e] = edge_residuals[e];
            // printf("CPU: eidx %d, edge_v %f \n", e, edge_capacities[e]);
            //}
          }
        },
        1, util::DEVICE, oprtr_parameters.stream));

    /*
    ////////////////////////////////
    GUARD_CU(community_weights.ForAll(
    [community_sizes, next_communities, num_comms, vertex_reachabilities,
    community_accus]
    __host__ __device__ (ValueT *community_weight, const SizeT &pos){
          if(pos < num_comms[0]){
              community_weight [pos] = 0;
              community_sizes  [pos] = 0;
              next_communities [pos] = 0;
              //printf("vertext value %f \n", community_accus[0]);
          }
          //printf("%d, ", vertex_reachabilities[pos]);
        }, num_nodes, util::DEVICE, oprtr_parameters.stream));
    //printf("core runs permantly1 \n");

    GUARD_CU(previous_num_comms.ForAll(
        [num_comms] __host__ __device__ (VertexT *previous_num_comm, const SizeT
    &pos){ previous_num_comm[pos] = num_comms[pos];
        }, 1, util::DEVICE, oprtr_parameters.stream));


    GUARD_CU(community_weights.ForAll(
        [vertex_active, // vertex specific
        vertex_reachabilities,
        next_communities, //community specific
        curr_communities,
        community_active,
        community_sizes,
        community_accus,
        edge_residuals, // intermediate output
        // others
        num_comms, num_edges,
        num_org_nodes, graph]
        __host__ __device__ (ValueT *community_weights, const VertexT &idx){
        {
            VertexT comm;
            //if(idx == 0) printf("in 1st for loop begin\n");
            for (VertexT v = 0; v < num_org_nodes; v++)
            {
                if (!vertex_active[v])
                    continue;

                if (vertex_reachabilities[v])
                { // reachable by source
                    comm = next_communities[curr_communities[v]];
                    if (comm == 0)
                    { // not assigned yet
                        comm = num_comms[0];
                        next_communities[curr_communities[v]] = num_comms[0];
                        community_active [comm] = true;
                        num_comms[0] ++;
                        community_weights[comm] = 0;
                        community_sizes  [comm] = 0;
                        next_communities [comm] = 0;
                        community_accus  [comm] =
    community_accus[curr_communities[v]];
                    }
                    curr_communities[v] = comm;
                    community_weights[comm] +=
                        edge_residuals[num_edges - num_org_nodes * 2 + v];
                    community_sizes  [comm] ++;
                    //printf("++ %d %f %f\n", comm, community_weights[comm],
    community_accus[comm]);
                }

                else { // otherwise
                    comm = curr_communities[v];
                    SizeT e_start = graph.GetNeighborListOffset(v);
                    SizeT num_neighbors = graph.GetNeighborListLength(v);
                    community_weights[comm] -= edge_residuals[e_start +
    num_neighbors - 1]; community_sizes  [comm] ++;

                    auto e_end = e_start + num_neighbors - 2;
                    for (auto e = e_start; e < e_end; e++)
                    {
                        VertexT u = graph.GetEdgeDest(e);
                        if (vertex_reachabilities[u] == 1)
                        {
                            edge_residuals[e] = 0;
                        }
                    }
                    //printf("-- %d %f %f\n", comm, community_weights[comm],
    community_accus[comm]);
                }
            }
            //if(idx == 0) printf("in 1st for loop end\n");
          }
        }, 1, util::DEVICE, oprtr_parameters.stream)); //loop only once

        GUARD_CU(community_weights.ForAll(
             [next_communities, //community specific
             curr_communities,
             community_active,
             community_sizes,
             community_accus,
             // others
             previous_num_comms]
             __host__ __device__ (ValueT *community_weights, unsigned int &idx){
               {
                 for (auto comm = 0; comm < previous_num_comms[0]; comm ++)
                 {
                     if (community_active[comm])
                     {
                         if (next_communities[comm] == 0)
                         {
                             community_weights[comm] = 0;
                             community_active [comm] = false;
                         } else if (community_sizes[comm] == 0) {
                             community_active [comm] = false;
                             community_active [next_communities[comm]] = false;
                             community_weights[next_communities[comm]] = 0;
                         } else {
                             //printf("values: comm: %d, sizes: %d, weights: %f,
    accus: %f.\n", comm, community_sizes[comm], community_weights[comm],
    community_accus[comm]); community_weights[comm] /= community_sizes  [comm];
                             community_accus  [comm] += community_weights[comm];
                         }
                     } else {
                         community_weights[comm] = 0;
                     }
                 }
               }
               //if(comm == 0) printf("in 2st for loop end\n");
             }, 1, util::DEVICE, oprtr_parameters.stream));


     GUARD_CU(community_weights.ForAll(
        [next_communities, //community specific
        community_sizes,
        community_accus,
        // others
        previous_num_comms, num_comms, active] //!!!
          __host__ __device__ (ValueT *community_weights, unsigned int &comm){
        {
          if(comm < num_comms[0] && comm >= previous_num_comms[0]){
            community_weights[comm] /= community_sizes  [comm];
            community_accus  [comm] += community_weights[comm];
            printf("comm %d, accus %f, sizes %d \n", comm, community_accus
    [comm], community_sizes  [comm]);
          }
          active[0] = 0;
        }
      }, num_org_nodes, util::DEVICE, oprtr_parameters.stream));

      GUARD_CU(community_weights.ForAll(
           [vertex_active, // vertex specific
           vertex_reachabilities,
           next_communities, //community specific
           curr_communities,
           community_active,
           community_sizes,
           community_accus,
           edge_residuals, // intermediate output
           // others
           num_comms, num_edges,
           num_org_nodes, graph,
           active]
           __host__ __device__ (ValueT *community_weights, const VertexT &idx){
             {
               for (VertexT v = 0; v < num_org_nodes; v++)
               {
                   if (!vertex_active[v])
                       continue;

                   auto comm = curr_communities[v];
                   if (!community_active[comm] ||
                       abs(community_weights[comm]) < 1e-6)
                   {
                       if (vertex_reachabilities[v] == 1)
                           edge_residuals[num_edges - num_org_nodes * 2 + v] =
    0; if (vertex_reachabilities[v] != 1)
                       {
                           SizeT  e = graph.GetNeighborListOffset(v)
                               + graph.GetNeighborListLength(v) - 1;
                           edge_residuals[e] = 0;
                       }
                       vertex_active[v] = false;
                       community_active[comm] = false;
                   }

                   else {
                       active[0] = 1;
                       SizeT e_from_src = num_edges - num_org_nodes * 2 + v;
                       SizeT e_to_dest  = graph.GetNeighborListOffset(v)
                           + graph.GetNeighborListLength(v) - 1;
                       if (vertex_reachabilities[v] == 1)
                       {
                           edge_residuals[e_from_src] -=
    community_weights[comm]; if (edge_residuals[e_from_src] < 0)
                           {
                               double temp = -1 * edge_residuals[e_from_src];
                               edge_residuals[e_from_src] =
    edge_residuals[e_to_dest]; edge_residuals[e_to_dest] = temp;
                           }
                       } else {
                           edge_residuals[e_to_dest] += community_weights[comm];
                           if (edge_residuals[e_to_dest] < 0)
                           {
                               double temp = -1 * edge_residuals[e_to_dest];
                               edge_residuals[e_to_dest] =
    edge_residuals[e_from_src]; edge_residuals[e_from_src] = temp;
                           }
                       }
                   }
               } // end of for v
               */

    // below is parallel
    /*
    if (!vertex_active[v]) return;

    auto comm = curr_communities[v];
    if (!community_active[comm] ||
        abs(community_weights[comm]) < 1e-6)
    {
        if (vertex_reachabilities[v] == 1)
            edge_residuals[num_edges - num_org_nodes * 2 + v] = 0;
        if (vertex_reachabilities[v] != 1)
        {
            SizeT  e = graph.GetNeighborListOffset(v)
                + graph.GetNeighborListLength(v) - 1;
            edge_residuals[e] = 0;
        }
        vertex_active[v] = false;
        community_active[comm] = false;
    }

    else {
        active[0] = 1;
        SizeT e_from_src = num_edges - num_org_nodes * 2 + v;
        SizeT e_to_dest  = graph.GetNeighborListOffset(v)
            + graph.GetNeighborListLength(v) - 1;
        if (vertex_reachabilities[v] == 1)
        {
            edge_residuals[e_from_src] -= community_weights[comm];
            if (edge_residuals[e_from_src] < 0)
            {
                auto temp = -1 * edge_residuals[e_from_src];
                edge_residuals[e_from_src] = edge_residuals[e_to_dest];
                edge_residuals[e_to_dest] = temp;
            }
        } else {
            edge_residuals[e_to_dest] += community_weights[comm];
            if (edge_residuals[e_to_dest] < 0)
            {
                auto temp = -1 * edge_residuals[e_to_dest];
                edge_residuals[e_to_dest] = edge_residuals[e_from_src];
                edge_residuals[e_from_src] = temp;
            }
        }
    }
    //if(v == 0) printf("in 3st for loop end\n");
    */

    //!!!}
    //!!!}, 1, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(edge_residuals.ForAll(
        [graph, iteration, active] __host__ __device__(ValueT * edge_residuals,
                                                       SizeT & e) {
          {
            if (false) {
              // if(iteration == 0){
              active[0] = 0;
              edge_residuals[e] =
                  graph.edge_values[e];  // just for debugging purposes #!!!
            } else {
              graph.edge_values[e] = edge_residuals[e];
            }
          }
        },
        graph.edges, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(community_accus.ForAll(
        [active, community_accus, curr_communities] __host__ __device__(
            ValueT * community_accus, SizeT & v) {
          {
            if (active[0] == 0) {
              ValueT tmp = max(community_accus[v] - 3., 0.0);
              community_accus[v] = tmp + min(community_accus[v] + 3., 0.0);
              // printf("%d %f \n", v, community_accus[curr_communities[v]]);
            }
          }
        },
        num_org_nodes, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(community_accus.ForAll(
        [active, community_accus, curr_communities, Y] __host__ __device__(
            ValueT * community_accus, SizeT & v) {
          {
            if (active[0] == 0) {
              Y[v] = community_accus[curr_communities[v]];
            }
            // if(v == 0) printf("in last for loop end\n");
          }
        },
        num_org_nodes, util::DEVICE, oprtr_parameters.stream));
    GUARD_CU(cudaDeviceSynchronize());
    cpu_timer.Stop();
    // printf("gtf: %f \n", cpu_timer.ElapsedMillis());

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed");

    // printf("new updated vertices %d\n", frontier.queue_length);
    cpu_timer.Start();
    frontier.queue_reset = true;
    oprtr_parameters.filter_mode = "BY_PASS";
    GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        [active] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool { return active[0] > 0; }));
    cpu_timer.Stop();

    frontier.queue_index++;
    // Get back the resulted frontier length
    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed");

    //	printf("new updated vertices %d (version after filter)\n", \
            frontier.queue_length);\
        fflush(stdout);

    data_slice.num_updated_vertices = frontier.queue_length;

    return retval;
  }

  /* cudaError_t Compute_OutputLength(int peer_)
  {
      // No need to load balance or get output size
      return cudaSuccess;
  }*/

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES  Number of data associated with each
   *				      transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES  Number of data associated with each
                                    transmition item, typed ValueT
   * @param[in] received_length     The number of transmition items received
   * @param[in] peer_		      which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    auto &enactor = this->enactor;
    auto &problem = enactor->problem;
    auto gpu_num = this->gpu_num;
    auto gpu_offset = gpu_num * enactor->num_gpus;
    auto &data_slice = problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;

    debug_aml("ExpandIncomming do nothing");
    /*	for key " +
                std::to_string(key) + " and for in_pos " +
                std::to_string(in_pos) + " and for vertex ass ins " +
                std::to_string(vertex_associate_ins[in_pos]) +
                " and for value ass ins " +
                std::to_string(value__associate_ins[in_pos]));*/

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    debug_aml("expand incoming\n");
    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto enactor = this->enactor;
    int num_gpus = enactor->num_gpus;
    auto &enactor_slice = enactor->enactor_slices[0];
    auto iteration = enactor_slice.enactor_stats.iteration;

    auto &retval = enactor_slice.enactor_stats.retval;
    if (retval != cudaSuccess) {
      printf("(CUDA error %d @ GPU %d: %s\n", retval, 0 % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    auto &data_slice = enactor->problem->data_slices[gpu_num][0];

    if (data_slice.num_updated_vertices == 0) return true;

    return false;
  }

};  // end of gtfIteration

/**
 * @brief gtf enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::VertexT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::MfProblemT MfProblemT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef GTFIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;
  // typedef mf::Problem<GraphT> MfProblemT;
  typedef mf::Enactor<MfProblemT> MfEnactorT;
  MfEnactorT mf_enactor;

  /**
   * @brief gtfEnactor constructor
   */
  Enactor() : BaseEnactor("gtf"), mf_enactor(), problem(NULL) {
    // TODO: change according to algorithmic needs
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief MFEnactor destructor
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
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));
    GUARD_CU(mf_enactor.Init(problem.mf_problem, target));

    auto num_gpus = this->num_gpus;

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto gpu_offset = gpu * num_gpus;
      auto &enactor_slice = this->enactor_slices[gpu_offset + 0];
      auto &graph = problem.sub_graphs[gpu];
      auto nodes = graph.nodes;
      auto edges = graph.edges;
      GUARD_CU(
          enactor_slice.frontier.Allocate(nodes, edges, this->queue_factors));
    }
    iterations = new IterationT[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of gtf, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    debug_aml("Run enact");
    gunrock::app::Iteration_Loop<0,  // NUM_VERTEX_ASSOCIATES
                                 1,  // NUM_VALUE__ASSOCIATES
                                 IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(const VertexT &src, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    debug_aml("Enactor Reset, src %d", src);

    typedef typename EnactorT::Problem::GraphT::GpT GpT;
    auto num_gpus = this->num_gpus;

    GUARD_CU(BaseEnactor::Reset(target));

    // Initialize frontiers according to the algorithm gtf
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      auto gpu_offset = gpu * num_gpus;
      if (num_gpus == 1 ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < num_gpus; ++peer_) {
          auto &frontier = this->enactor_slices[gpu_offset + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [src] __host__ __device__(VertexT & v) { v = src; }, 1, target,
                0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < num_gpus; peer_++) {
          auto &frontier = this->enactor_slices[gpu_offset + peer_].frontier;
          frontier.queue_length = 0;
        }
      }
    }
    GUARD_CU(BaseEnactor::Sync());
    debug_aml("Enactor Reset end");
    return retval;
  }

  /**
   * @brief Enacts a gtf computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    debug_aml("enact");
    printf("enact calling successfully!!!!!!!!!!!\n");
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU gtf Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace gtf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
