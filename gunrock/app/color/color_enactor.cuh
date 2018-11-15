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


#include <gunrock/util/track_utils.cuh>
#include <gunrock/util/sort_device.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

// <DONE> change includes
#include <gunrock/app/color/color_problem.cuh>

#include <curand.h>
#include <curand_kernel.h>
// </DONE>


namespace gunrock {
namespace app {
// <DONE> change namespace
namespace color {
// </DONE>

/**
 * @brief Speciflying parameters for hello Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));

    return retval;
}

/**
 * @brief defination of hello iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct ColorIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push>
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;

    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push> BaseIterationLoop;

    ColorIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of hello, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // --
        // Alias variables

        auto &data_slice = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];

        auto &enactor_slice = this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];

        auto &enactor_stats    = enactor_slice.enactor_stats;
        auto &graph            = data_slice.sub_graph[0];
        auto &frontier         = enactor_slice.frontier;
        auto &oprtr_parameters = enactor_slice.oprtr_parameters;
        auto &retval           = enactor_stats.retval;
        auto &iteration        = enactor_stats.iteration;

        // <DONE> add problem specific data alias here:
        auto &colors 	 = data_slice.colors;
        auto &rand 	   = data_slice.rand;
	auto &gen	           = data_slice.gen;
	auto &color_balance  = data_slice.color_balance;
	auto &colored	       = data_slice.colored;
  auto &use_jpl        = data_slice.use_jpl;
        // </DONE>

	curandGenerateUniform(gen, rand.GetPointer(util::DEVICE), graph.nodes);

        // --
        // Define operations

	if(color_balance) {

#if 0
        // advance operation
        auto advance_op = [
            colors,
            rand
        ] __host__ __device__ (
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            // <TODO> Implement advance operation

            // Mark src and dest as visited
            atomicMax(visited + src, 1);
            auto dest_visited = atomicMax(visited + dest, 1);

            // Increment degree of src
            atomicAdd(degrees + src, 1);

            // Add dest to queue if previously unsen
            return dest_visited == 0;

            // </TODO>
        };


	oprtr_parameters.reduce_values_out   = &rand_max;
	oprtr_parameters.reduce_reset        = true;
        oprtr_parameters.reduce_values_temp  = &color_temp;
 	oprtr_parameters.reduce_values_temp2 = &color_temp2;

        frontier.queue_length = graph.nodes;
        frontier.queue_reset  = true;

        GUARD_CU(oprtr::NeighborReduce<oprtr::OprtrType_V2V |
                oprtr::OprtrMode_REDUCE_TO_SRC | oprtr::ReduceOp_Maximum>(
                graph, null_ptr, null_ptr,
                oprtr_parameters, advance_op,
                []__host__ __device__ (const ValueT &a, const ValueT &b)
                {
                    return (a < b) ? b : a;
         }, (ValueT)0));

        // --
        // Run

        // <TODO> some of this may need to be edited depending on algorithmic needs
        // !! How much variation between apps is there in these calls?

        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
            oprtr_parameters, advance_op, filter_op));
#endif

        // Get back the resulted frontier length
        // GUARD_CU(frontier.work_progress.GetQueueLength(
        //    frontier.queue_index, frontier.queue_length,
        //    false, oprtr_parameters.stream, true));

	}

  else {

	    auto color_op = [
        use_jpl,
		    graph,
	    	colors,
	    	rand,
	    	iteration
	    ] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

        VertexT v 		= v_q[pos];
        SizeT start_edge 	= graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors 	= graph.CsrT::GetNeighborListLength(v);
        ValueT temp = rand[v];

        if(use_jpl) {

          if(!util::isValid(colors[v])) {
              bool colormax = true;
              bool colormin = true;
              for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
                VertexT u = graph.CsrT::GetEdgeDest(e);
                if(!util::isValid(colors[u]) && rand[u] >= temp) {
                  printf("Max: Node %d with %f defeated by node %d with %f\n",
                  v,rand[v],u,rand[u]);
                  colormax = false;
                }
                if(!util::isValid(colors[u]) && rand[u] <= temp) {
                  printf("Min: Node %d with %f defeated by node %d with %f\n",
                  v,rand[v],u,rand[u]);
                  colormin = false;
                }
              }

              if(colormax)
                colors[v] = iteration*2+1;
              if(colormin)
                colors[v] = iteration*2+2;
            }
        }
        else {
          VertexT max = v;    // active max vertex
  		    VertexT min = v;    // active min vertex

  	    	for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
    		    VertexT u = graph.CsrT::GetEdgeDest(e);
    		      if (rand[u] > temp)
    		    	max = u;

    		    if (rand[u] < temp)
    			    min = u;

    		    printf("Let's see what rand[u] = %f\n", rand[u]);
    		    temp = rand[u]; // compare against e-1
  	    	}

          // Assign two colors per iteration
      		if (!util::isValid(colors[max]))
      		    colors[max] = iteration*2+1;

      		if (!util::isValid(colors[min]))
      		    colors[min] = iteration*2+2;

      		printf("iteration number = %u\n", iteration);
      		printf("colors[%u, %u] = [%u, %u]\n", min, max, colors[min], colors[max]);
        }
	  };


	  auto status_op = [
		  colors,
		  colored
	    ] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

		   VertexT v	= v_q[pos];

		     if(util::isValid(colors[v]))
    		    atomicAdd(&colored[0], 1);
    	 };

	    // Run --
      GUARD_CU(frontier.V_Q()->ForAll(
           color_op, frontier.queue_length,
           util::DEVICE, oprtr_parameters.stream));

	    GUARD_CU(frontier.V_Q()->ForAll(
                 status_op, frontier.queue_length,
                 util::DEVICE, oprtr_parameters.stream));

	    GUARD_CU(data_slice.colored .SetPointer(&data_slice.colored_, sizeof(SizeT), util::HOST));
            GUARD_CU(data_slice.colored .Move(util::DEVICE, util::HOST));

	}

        return retval;
}


    bool Stop_Condition(int gpu_num = 0)
    {
        auto &data_slice = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto &enactor_slices = this -> enactor -> enactor_slices;
        auto iter = enactor_slices[0].enactor_stats.iteration;
        auto usr_iter = data_slice.usr_iter;
	      auto &graph = data_slice.sub_graph[0];
        printf("Max Iteration: %d\n",usr_iter);
        printf("Iteration: %d\n",iter);
        printf("colored_: %d\n", data_slice.colored_);
        printf("Num Nodes: %d\n", graph.nodes);

        //old stop condition
        //if(data_slice.colored_ >= graph.nodes)
	      //   return true;

        //user defined stop condition
        if(iter == usr_iter)
             return true;

        return false;
    }

    /**
     * @brief Routine to combine received data and local data
     * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each transmition item, typed VertexT
     * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each transmition item, typed ValueT
     * @param  received_length The numver of transmition items received
     * @param[in] peer_ which peer GPU the data came from
     * \return cudaError_t error message(s), if any
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {

        // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

        auto &data_slice    = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto &enactor_slice = this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        //auto iteration = enactor_slice.enactor_stats.iteration;
        // TODO: add problem specific data alias here, e.g.:
        // auto         &distances          =   data_slice.distances;

        auto expand_op = [
        // TODO: pass data used by the lambda, e.g.:
        // distances
        ] __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            // TODO: fill in the lambda to combine received and local data, e.g.:
            // ValueT in_val  = value__associate_ins[in_pos];
            // ValueT old_val = atomicMin(distances + key, in_val);
            // if (old_val <= in_val)
            //     return false;
            return true;
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }
}; // end of colorIteration

/**
 * @brief Color enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT,
        typename _Problem::GraphT::VertexT, // TODO: change to other label types used for the operators, e.g.: typename _Problem::LabelT,
        typename _Problem::GraphT::ValueT, // TODO: change to other value types used for inter GPU communication, e.g.: typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename GraphT::VertexT   LabelT  ;
    typedef typename GraphT::ValueT    ValueT  ;
    typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef ColorIterationLoop<EnactorT>
        IterationT;

    Problem *problem;
    IterationT *iterations;

    /**
     * @brief color constructor
     */
    Enactor() :
        BaseEnactor("Color"),
        problem    (NULL  )
    {
        // <TODO> change according to algorithmic needs
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
        // </TODO>
    }

    /**
     * @brief hello destructor
     */
    virtual ~Enactor() { /*Release();*/ }

    /*
     * @brief Releasing allocated memory space
     * @param target The location to release memory from
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Release(target));
        delete []iterations; iterations = NULL;
        problem = NULL;
        return retval;
    }

    /**
     * @brief Initialize the problem.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        Problem          &problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;

        // Lazy initialization
        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None, 2, NULL, target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++) {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++) {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
      * @brief one run of hello, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            // <DONE> change to how many {VertexT, ValueT} data need to communicate
            //       per element in the inter-GPU sub-frontiers
            0, 1,
            // </DONE>
            IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Reset enactor
...
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(
        util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));

	SizeT num_nodes = this -> problem -> data_slices[0][0].sub_graph[0].nodes;

        // <DONE> Initialize frontiers according to the algorithm:
        // In this case, we add a single `src` to the frontier
        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
           if (this->num_gpus == 1) {
               this -> thread_slices[gpu].init_size = num_nodes;
               for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                   auto &frontier = this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                   frontier.queue_length = (peer_ == 0) ? num_nodes : 0;
                   if (peer_ == 0) {

                      util::Array1D<SizeT, VertexT> tmp;
                      tmp.Allocate(num_nodes, target | util::HOST);
                      for(SizeT i = 0; i < num_nodes; ++i) {
                          tmp[i] = (VertexT)i % num_nodes;
                      }
                      GUARD_CU(tmp.Move(util::HOST, target));

                      GUARD_CU(frontier.V_Q() -> ForEach(tmp,
                          []__host__ __device__ (VertexT &v, VertexT &i) {
                          v = i;
                      }, num_nodes, target, 0));

                      tmp.Release();
		   }
               }
           } else {
           }
        }
        // </DONE>

        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
     * @brief Enacts a hello computing on the specified graph.
...
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact()
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU Color Done.", this -> flag & Debug);
        return retval;
    }
};

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
