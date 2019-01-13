// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * color_enactor.cuh
 *
 * @brief color Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/sort_device.cuh>
#include <gunrock/util/track_utils.cuh>

#include <gunrock/app/color/color_problem.cuh>

#include <curand.h>
#include <curand_kernel.h>

namespace gunrock {
namespace app {
namespace color {

/**
 * @brief Speciflying parameters for hello Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of hello iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct ColorIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  ColorIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of hello, one iteration
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

    auto &colors = data_slice.colors;
    auto &rand = data_slice.rand;
    auto &color_predicate = data_slice.color_predicate;
    auto &color_temp = data_slice.color_temp;
    auto &color_temp2 = data_slice.color_temp2;
    auto &prohibit = data_slice.prohibit;
    auto &gen = data_slice.gen;
    auto &color_balance = data_slice.color_balance;
    auto &colored = data_slice.colored;
    auto &use_jpl = data_slice.use_jpl;
    auto &no_conflict = data_slice.no_conflict;
    auto &hash_size = data_slice.hash_size;
    auto &test_run = data_slice.test_run;
    auto &min_color = data_slice.min_color;
    auto &loop_color = data_slice.loop_color;
    auto stream = oprtr_parameters.stream;
    util::Array1D<SizeT, VertexT>* null_frontier = NULL;
    auto null_ptr = null_frontier;
    // curandGenerateUniform(gen, rand.GetPointer(util::DEVICE), graph.nodes);
    // --
    // Define operations

      // =======================================================================
      /* color_op
      @Description: non-jpl vertex coloring operation. Based on parameter
      choices, different type of coloring is utilized. Max and min independent
      sets are generated prior to coloring. Coloring can either be from hash
      funtion or iteration.
      */
      // =======================================================================
      auto color_op =
          [graph, colors, rand, iteration, hash_size, prohibit] __host__
          __device__(VertexT * v_q, const SizeT &pos) {
            VertexT v = v_q[pos];
            SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto temp = rand[v];

            VertexT max = v; // active max vertex
            VertexT min = v; // active min vertex

            for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
              VertexT u = graph.CsrT::GetEdgeDest(e);
              if (rand[u] > temp)
                max = u;

              if (rand[u] < temp)
                min = u;

              // printf("Let's see what rand[u] = %f\n", rand[u]);
              temp = rand[u]; // compare against e-1
            }

            // max hash coloring
            SizeT prohibit_offset = max * hash_size;
            for (int c = 1, n = 0; (c < iteration * 2 + 1) || (n < hash_size);
                 c++, n++) {
              if (prohibit[prohibit_offset + n] != c) {
                colors[max] = c;
                break;
              }
            }

            // min hash coloring
            prohibit_offset = min * hash_size;
            for (int c = 1, n = 0; (c < iteration * 2 + 1) || (n < hash_size);
                 c++, n++) {
              if (prohibit[prohibit_offset + n] != c) {
                colors[min] = c;
                break;
              }
            }

            // if hash coloring fail because not enough space, fall back to
            // color by iteration
            if (!util::isValid(colors[max]))
              colors[max] = iteration * 2 + 1;

            if (!util::isValid(colors[min]))
              colors[min] = iteration * 2 + 2;
          };

      // =======================================================================
      /* gen_op
      @Description: populate @prohibit list with first @hash_size^th neighbor
      colors
      @hash_size. Each thread handle one element inside @prohibit, no thread
      divergence.
      */
      // =======================================================================
      auto gen_op = [graph, colors, hash_size] __host__ __device__(
                        VertexT * prohibit_, const SizeT &pos) {
        VertexT v = pos / hash_size;
        SizeT a_idx = pos % hash_size;
        SizeT e = graph.CsrT::GetNeighborListOffset(v) + a_idx;

        VertexT u = graph.CsrT::GetEdgeDest(e);
        prohibit_[pos] = colors[u];
      };

      // =======================================================================
      /* resolve_op
      @Description: resolve conflicts after non-jpl coloring. This operation is
      called only when @no_conflict != 0. @no_conflict == 1 means resolve by
      comparing @rand. @no_conflict == 2 means resolve by comparing node degree
      */
      // =======================================================================
      auto resolve_op = [graph, colors, rand, no_conflict] __host__ __device__(
                            VertexT * v_q, const SizeT &pos) {
        VertexT v = v_q[pos];
        SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

        if (util::isValid(colors[v])) {
          for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
            VertexT u = graph.CsrT::GetEdgeDest(e);
            if (colors[u] == colors[v]) {

              // decide by random number
              if (rand[u] >= rand[v] && no_conflict == 1) {
                colors[v] = util::PreDefinedValues<VertexT>::InvalidValue;
                break;
              }

              // decide by degree heuristic
              else if (graph.CsrT::GetNeighborListLength(u) >= num_neighbors &&
                       no_conflict == 2) {
                colors[v] = util::PreDefinedValues<VertexT>::InvalidValue;
                break;
              }
            }
          }
        }
      };

      // =======================================================================
      /* jpl_color_op
      @Description: jpl vertex coloring operation. No conflict resolution is
      needed
      */
      // =======================================================================
      auto jpl_color_op = [graph, colors, rand, iteration, min_color] __host__ __device__(
                              VertexT * v_q, const SizeT &pos) {
        VertexT v = v_q[pos];
        if (util::isValid(colors[v]))
          return;

        SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

        bool colormax = true;
        bool colormin = true;
        int  color = iteration * 2;
        
        for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
          VertexT u = graph.CsrT::GetEdgeDest(e);
          
          if ((util::isValid(colors[u])) && (colors[u] != color + 1) && (colors[u] != color + 2) || 
	      (v == u))
            continue;
          if (rand[v] <= rand[u])
            colormax = false;
	  if (min_color) {
            if (rand[v] >= rand[u])
              colormin = false;
  	  }
        }

        if (colormax)
          colors[v] = color + 1;
	if (min_color) {
          if (colormin)
            //printf("DEBUG: coloring with min\n");
            colors[v] = color + 2;
         }
      };

      // =======================================================================
      /* advance_op
      @Description: advance to neighbor random number for coloring comparison
      */
      //========================================================================
      auto advance_op = [graph, iteration, colors] __host__ __device__ (
		const VertexT &src, VertexT &dest, const SizeT &edge_id,
                const VertexT &input_item, const SizeT &input_pos,
                SizeT &output_pos) -> ValueT 
      {
	printf("ADVANCE: At iteration = %d\n",iteration);
	printf("ADVANCE: src = %d\n", src);
	printf("ADVANCE: input_item = %d\n", input_item);
	printf("ADVANCE: input_pos = %d\n", input_pos);
	printf("ADVANCE: dest = %d and %f\n", dest, (ValueT)dest); 
	return (ValueT) dest;
      };

      // =======================================================================
      /* reduce_op
      @Description: coloring comparison for max rand
      */
      //========================================================================
      auto max_reduce_op = [rand, colors, iteration] __host__ __device__ (
	const ValueT &a, const ValueT &b) -> ValueT
      {
	printf("REDUCE: (a, b) = (%f, %f)\n", a, b);
        printf("REDUCE: rand(a, b) = (%f, %f)\n", rand[a], rand[b]);

	VertexT v = (VertexT) a;
	VertexT u = (VertexT) b;

	ValueT randa = rand[v];
	ValueT randb = rand[u];

	if ((util::isValid(colors[u])) && (colors[u] != iteration) || (v == u)) {
		randb = (ValueT) -1;
		printf("REDUCE: vertex was colored. \n");
	}

	return (randa < randb) ? u : v;
      };     

      // =======================================================================
      /* filterAndColor_op
      @Description: color selected node then remove it from frontier.
      */
      //========================================================================

	auto filterAndColor_op = [iteration, colors, color_predicate] __host__ __device__ (
		const VertexT &src, VertexT &dest, const SizeT &edge_id,
            	const VertexT &input_item, const SizeT &input_pos,
            	SizeT &output_pos) -> bool
	{	
		printf("FILTER: src = %d \n",src);
		printf("FILTER: dest = %d \n",dest);
		printf("FILTER: edge_id = %d \n",edge_id);
		printf("FILTER: input_item = %d \n", input_item);
		printf("FILTER: input_pos = %d \n", input_pos);
		printf("FILTER: output_pos = %d \n", output_pos);	
		printf("FILTER: color_predicate[] = %f and (vertext) %u \n", 
				color_predicate[dest], (VertexT) color_predicate[dest]);		

		//if the node is not selected to be colored, keep it in frontier
		if (!util::isValid(color_predicate[dest])) return true;
		VertexT id = (VertexT) color_predicate[dest];

		//after color the node, drop it from frontier
		printf("FILTER: coloring node %d, to color = %d\n", id, iteration);
		colors[id] = iteration;
		return false;

	        if (!util::isValid(colors[input_item])) return true;
	};

      // =======================================================================
      /* status_op
      @Description: check coloring status.
      */
      // =======================================================================
      auto status_op = [colors, colored] __host__ __device__(VertexT * v_q,
                                                             const SizeT &pos) {
        VertexT v = v_q[pos];
        if (util::isValid(colors[v])) {
          atomicAdd(&colored[0], 1);
        }
      };

      //======================================================================//
      // Run --                                                               //
      //======================================================================//

      // JPL exact method
      //printf("DEBUG: =====Start Iteration====\n");
      if (use_jpl) {
	if (!color_balance) {
		//printf("DEBUG: using for loop \n");
        	GUARD_CU(frontier.V_Q()->ForAll(jpl_color_op, frontier.queue_length,
                                        util::DEVICE, stream));
	}

      	else {

		oprtr_parameters.reduce_values_out   = &color_predicate;
                oprtr_parameters.reduce_values_temp  = &color_temp;
		oprtr_parameters.reduce_values_temp2 = &color_temp2;
            	oprtr_parameters.reduce_reset        = true;
            	oprtr_parameters.advance_mode        = "ALL_EDGES";

		frontier.queue_length = graph.nodes;
		frontier.queue_reset = true;
		static ValueT Identity = util::PreDefinedValues<ValueT>::MinValue;

		GUARD_CU(oprtr::NeighborReduce<oprtr::OprtrType_V2V |
			 oprtr::OprtrMode_REDUCE_TO_SRC | oprtr::ReduceOp_None>(
			 graph.csr(), 
			 null_ptr, /*frontier.V_Q(),*/ 
			 null_ptr, /*frontier.Next_V_Q(),*/
			 oprtr_parameters, 
			 advance_op,
			 max_reduce_op,
			 (ValueT) -1));

	      auto reduce_color_op = [colors, color_predicate, iteration] __host__ __device__(
				      VertexT * v_q, const SizeT &pos) {

		// VertexT v = v_q[pos];

		if (!util::isValid(color_predicate[pos])) return;
		VertexT id = color_predicate[pos];
		colors[id] = iteration;
		return;	
	      };

        	GUARD_CU(frontier.V_Q()->ForAll(reduce_color_op, frontier.queue_length,
                                        util::DEVICE, stream));

#if 0
		printf("AFTER RED: queue length = %d \n queue index = %d\n",
                                frontier.queue_length, frontier.queue_index);

		frontier.queue_index++;	
	        GUARD_CU(frontier.work_progress.GetQueueLength(
        	    frontier.queue_index, frontier.queue_length,
            	    false, oprtr_parameters.stream, false));

		printf("AFTER ADV: queue length = %d \n queue index = %d\n", 
				frontier.queue_length, frontier.queue_index);
#endif

	}
      }

      // Current method in development
      else {
        // color by max and min independent set, non-exactsolution
        GUARD_CU(frontier.V_Q()->ForAll(color_op, frontier.queue_length,
                                        util::DEVICE, stream));
      }

      // optional resolution to make method exact solution
      if (no_conflict == 1 || no_conflict == 2) {

        // optinal coloring by hash function n * hash_size (non-exact)
        if (hash_size != 0)
          GUARD_CU(prohibit.ForAll(gen_op, graph.nodes * hash_size,
                                   util::DEVICE, stream));

        GUARD_CU(frontier.V_Q()->ForAll(resolve_op, frontier.queue_length,
                                        util::DEVICE, stream));
      }

      if (test_run && !color_balance) {
		
	//reset atomic count
        GUARD_CU(data_slice.colored.ForAll(
            [] __host__ __device__(SizeT * x, const VertexT &pos) {
              x[pos] = 0;
            },
            1, util::DEVICE, stream));
	
	printf("DEBUG: after reseting colored\n");	

        GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");

        GUARD_CU(frontier.V_Q()->ForAll(status_op, frontier.queue_length,
                                        util::DEVICE, stream));

	printf("DEBUG: after update colored\n");

        GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");
        
	GUARD_CU(data_slice.colored.Move(util::DEVICE, util::HOST));

	printf("DEBUG: after move colored to HOST \n");
        
	GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");
      }

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;
    auto user_iter = data_slice.user_iter;
    auto &graph = data_slice.sub_graph[0];
    auto test_run = data_slice.test_run;
    auto frontier = enactor_slices[0].frontier;
    auto color_balance = data_slice.color_balance;
    // printf("DEBUG: iteration number %d, colored: %d\n", iter,
    //       data_slice.colored[0]);

	// atomic based stop condition
           if (test_run && (data_slice.colored[0] >= graph.nodes)) {
             printf("Max iteration: %d\n", iter);
             return true;
           }

           // user defined stop condition
           if (!test_run && (iter == user_iter)) return true;

           return false;
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
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // auto         &distances          =   data_slice.distances;

    auto expand_op = []  __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
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
}; // end of colorIteration

/**
 * @brief Color enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT,
          typename _Problem::GraphT::VertexT, 
                                              // types used for the operators,
                                              // e.g.: typename
                                              // _Problem::LabelT,
          typename _Problem::GraphT::ValueT,  
                                              // types used for inter GPU
                                              // communication, e.g.: typename
                                              // _Problem::ValueT,
          ARRAY_FLAG, cudaHostRegisterFlag> {
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
  typedef ColorIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief color constructor
   */
  Enactor() : BaseEnactor("Color"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief hello destructor
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

    GUARD_CU(this->Init_Threads(this, (CUT_THREADROUTINE) &
                                          (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of hello, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        //       per element in the inter-GPU sub-frontiers
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

    SizeT num_nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    // In this case, we add a single `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = num_nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_nodes : 0;
          if (peer_ == 0) {

            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(num_nodes, target | util::HOST);
            for (SizeT i = 0; i < num_nodes; ++i) {
              tmp[i] = (VertexT)i % num_nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                num_nodes, target, 0));

            tmp.Release();
          }
        }
      } else {
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a hello computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Color Done.", this->flag & Debug);
    return retval;
  }
};

} // namespace color
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
