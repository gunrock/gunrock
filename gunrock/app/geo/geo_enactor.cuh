// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_enactor.cuh
 *
 * @brief Geo Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/geo/geo_problem.cuh>
#include <gunrock/app/geo/geo_spatial.cuh>

namespace gunrock {
namespace app {
namespace geo {

/**
 * @brief Speciflying parameters for Geo Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of Geo iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct GEOIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  GEOIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of Geo, one iteration
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

    auto &latitude = data_slice.latitude;
    auto &longitude = data_slice.longitude;
    auto &active = data_slice.active;
    auto &spatial_iter = data_slice.spatial_iter;
    auto &geo_complete = data_slice.geo_complete;
    auto &Dinv = data_slice.Dinv;

    util::Location target = util::DEVICE;

    // --
    // Define operations

    /**
     * @brief Compute "center" of a set of points.
     *
     *      For set X ->
     *        if points == 1; center = point;
     *        if points == 2; center = midpoint;
     *        if points > 2; center = spatial median;
     */
    auto spatial_center_op =
        [graph, latitude, longitude, Dinv, target, spatial_iter] __host__
        __device__(VertexT * v_q, const SizeT &pos) {
          VertexT v = v_q[pos];

          // if no predicted location, and neighbor locations exists
          // Custom spatial center kernel for geolocation
          if (!util::isValid(latitude[v]) && !util::isValid(longitude[v])) {
            SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

            SizeT i = 0;
            ValueT neighbor_lat[2],
                neighbor_lon[2];  // for length <=2 use registers

            for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
              VertexT u = graph.CsrT::GetEdgeDest(e);
              if (util::isValid(latitude[u]) && util::isValid(longitude[u])) {
                neighbor_lat[i % 2] = latitude[u];   // last valid latitude
                neighbor_lon[i % 2] = longitude[u];  // last valid longitude
                i++;
              }
            }

            SizeT valid_neighbors = i;

            // If one location found, point at that location
            if (valid_neighbors == 1) {
              latitude[v] = neighbor_lat[0];
              longitude[v] = neighbor_lon[0];
              return;
            }

            // If two locations found, compute a midpoint
            else if (valid_neighbors == 2) {
              midpoint(neighbor_lat[0], neighbor_lon[0], neighbor_lat[1],
                       neighbor_lon[1], latitude.GetPointer(target),
                       longitude.GetPointer(target), v);
              return;
            }

            // if locations more than 2, compute spatial
            // median.
            else if (valid_neighbors > 2) {
              spatial_median(
                  graph, valid_neighbors, latitude.GetPointer(target),
                  longitude.GetPointer(target), v, Dinv.GetPointer(target),
                  false, target, spatial_iter);
            }

            // if no valid locations are found
            else {
              latitude[v] = util::PreDefinedValues<ValueT>::InvalidValue;
              longitude[v] = util::PreDefinedValues<ValueT>::InvalidValue;
            }

          }  // -- median calculation.
        };

    auto status_op = [latitude, longitude, active] __host__ __device__(
                         VertexT * v_q, const SizeT &pos) {
      VertexT v = v_q[pos];
      if (util::isValid(latitude[v]) && util::isValid(longitude[v])) {
        atomicAdd(&active[0], 1);
      }
    };

    // Run --

    GUARD_CU(frontier.V_Q()->ForAll(spatial_center_op, frontier.queue_length,
                                    util::DEVICE, oprtr_parameters.stream));

    if (geo_complete) {
      GUARD_CU(frontier.V_Q()->ForAll(status_op, frontier.queue_length,
                                      util::DEVICE, oprtr_parameters.stream));

      GUARD_CU(data_slice.active.SetPointer(&data_slice.active_, sizeof(SizeT),
                                            util::HOST));
      GUARD_CU(data_slice.active.Move(util::DEVICE, util::HOST));
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
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

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

  bool Stop_Condition(int gpu_num = 0) {
    auto &enactor_slice = this->enactor->enactor_slices[0];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &graph = data_slice.sub_graph[0];

    auto iter = enactor_stats.iteration;

    // Anymore work to do?
    // printf("Predictions active in Stop: %u vs. needed %u.\n",
    // data_slice.active_, graph.nodes);
    if (data_slice.geo_complete) {
      if (data_slice.active_ >= graph.nodes) return true;
    } else {
      if (iter >= data_slice.geo_iter) return true;
    }

    // else, keep running
    return false;
  }

};  // end of GEOIteration

/**
 * @brief Geolocation enactor class.
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
  typedef GEOIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief geo constructor
   */
  Enactor() : BaseEnactor("Geolocation"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief geo destructor
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
   * @brief one run of geo, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
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
        // MULTIGPU INCOMPLETE
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a geo computing on the specified graph.
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

}  // namespace geo
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
