// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * Template_enactor.cuh
 *
 * @brief hello Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>
 
// <DONE> change includes
#include <gunrock/app/geo/geo_problem.cuh>
// </DONE>


namespace gunrock {
namespace app {
// <DONE> change namespace
namespace geo {
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

    // <TODO> if needed, add command line parameters used by the enactor here
    // </TODO>
    
    return retval;
}

/**
 * @brief defination of hello iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct GEOIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push
    // <TODO>if needed, stack more option, e.g.:
    // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
    // Update_Predecessors : 0x0)
    // </TODO>
    >
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push
        // <TODO> add the same options as in template parameters here, e.g.:
        // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
        // Update_Predecessors : 0x0)
        // </TODO>
        > BaseIterationLoop;

    GEOIterationLoop() : BaseIterationLoop() {}

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
        auto &locations	       = data_slice.locations;
        auto &predicted 	       = data_slice.predicted;
        // </DONE>

	util::Location target = util::DEVICE;
        
        // --
        // Define operations

	// compute operation, substitute for neighbor reduce,
	// ForAll() with a for loop inside (nested forloop),
	// visiting all the neighbors and calculating the spatial center
	// after a gather.
	auto gather_op = [
	    graph,
	    locations,
	    predicted,
	    iteration
	] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

	    VertexT v 		= v_q[pos];
	    SizeT start_edge 	= graph.CsrT::GetNeighborListOffset(v);
	    SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
	
	    locations[v] 	= new ValueT[num_neighbors];
	    SizeT i 		= 0;

	    for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
		VertexT u = graph.CsrT::GetEdgeDest(e);
		if (util::isValid(predicted[u])) {
		    // gather locations from neighbors	
		    locations[v][i] = predicted[u];   
		} else {
		    // if the neighbor had no location, use an invalid.
		    locations[v][i] = util::PreDefinedValues<ValueT>::InvalidValue;
		}

		i++;
	    }

	};

	auto compute_op =  [
	    graph,
	    locations,
	    predicted,
            iteration
	] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

	    VertexT v 		= v_q[pos];
	    SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

	    // if no predicted location, and neighbor locations exists
	    if (!util::isValid(predicted[v]) && locations[v]) {
		predicted[v] = spatial_center(locations[v], num_neighbors);
	    }
	};


	// Run --
        GUARD_CU(frontier.V_Q()->ForAll(
            gather_op, frontier.queue_length,
            util::DEVICE, oprtr_parameters.stream));

	GUARD_CU(frontier.V_Q()->ForAll(
	    compute_op, frontier.queue_length,
	    util::DEVICE, oprtr_parameters.stream));
        
        return retval;
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
}; // end of GEOIteration

/**
 * @brief Geolocation enactor class.
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
        typename _Problem::GraphT::ValueT,  // TODO: change to other value types used for inter GPU communication, e.g.: typename _Problem::ValueT,
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
    typedef GEOIterationLoop<EnactorT> 
        IterationT;

    Problem *problem;
    IterationT *iterations;

    /**
     * @brief hello constructor
     */
    Enactor() :
        BaseEnactor("Geolocation"),
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
            problem, Enactor_None,
            // <TODO> change to how many frontier queues, and their types
            2, NULL,
            // </TODO>
            target, false));
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
            // <TODO> change to how many {VertexT, ValueT} data need to communicate
            //       per element in the inter-GPU sub-frontiers
            0, 1,
            // </TODO>
            IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Reset enactor
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(
        // <TODO> problem specific data if necessary, eg
        VertexT src = 0,
        // </TODO>
        util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));

        // <TODO> Initialize frontiers according to the algorithm:
        // In this case, we add a single `src` to the frontier
	// For Geolocation, can I queue in more work as an initial frontier?
        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
           if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src])) {
               this -> thread_slices[gpu].init_size = 1;
               for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                   auto &frontier = this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                   frontier.queue_length = (peer_ == 0) ? 1 : 0;
                   if (peer_ == 0) {
                       GUARD_CU(frontier.V_Q() -> ForEach(
                           [src]__host__ __device__ (VertexT &v) {
                           v = src;
                       }, 1, target, 0));
                   }
               }
           } else {
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier.queue_length = 0;
                }
           }
        }
        // </TODO>
        
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
     * @brief Enacts a hello computing on the specified graph.
...
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact(
        // <TODO> problem specific data if necessary, eg
        VertexT src = 0
        // </TODO>
    )
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU Template Done.", this -> flag & Debug);
        return retval;
    }
};

} // namespace geo
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
