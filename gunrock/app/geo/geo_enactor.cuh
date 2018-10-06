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
       
	//auto &weights	       = graph.CsrT::edge_values; 

        // <DONE> add problem specific data alias here:
        auto &locations_lat    = data_slice.locations_lat;
        auto &locations_lon    = data_slice.locations_lon;

        auto &latitude	       = data_slice.latitude;
	auto &longitude	       = data_slice.longitude;

	auto &valid_locations  = data_slice.valid_locations;
	auto &active	       = data_slice.active;
        // </DONE>

	util::Location target = util::DEVICE;
 
        // --
        // Define operations

	// Custom spatial center kernel for geolocation
	// <TODO> Needs proper implementation for median calculation
	// </TODO> -- median calculation.

	printf("Gather operator ... \n");
	// compute operation, substitute for neighbor reduce,
	// ForAll() with a for loop inside (nested forloop),
	// visiting all the neighbors and calculating the spatial center
	// after a gather.
	auto gather_op = [
	    graph,
	    locations_lat,
	    locations_lon,
	    latitude,
	    longitude,
	    valid_locations,
	    iteration
	] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

	    VertexT v 		= v_q[pos];
	    SizeT start_edge 	= graph.CsrT::GetNeighborListOffset(v);
	    SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

	    // printf("Initialize per vertex location array ...\n");

	    SizeT i 		= 0;

	    for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
		VertexT u = graph.CsrT::GetEdgeDest(e);
		// printf("For each vertex, perform a gather and obtain predicted locations...\n");
		if (util::isValid(latitude[u]) && util::isValid(longitude[u])) {
		    // gather locations from neighbors	
		    locations_lat[(v * num_neighbors) + i] = latitude[u];
                    locations_lon[(v * num_neighbors) + i] = longitude[u];
		    i++;
		}
	    }
	    // printf("Number of neighbors for vertex %u is %u.\n", v, num_neighbors);
	    // printf("Valid locations for vertex %u at %u is %u.\n", v, pos, i+1);
	    valid_locations[v] = i;

	};

	printf("Spatial Median operator ... \n");

	auto compute_op =  [
	    graph,
	    locations_lat,
	    locations_lon,
	    latitude,
	    longitude,
	    valid_locations,
            iteration,
	    active
	] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

	    VertexT v 		= v_q[pos];
	    SizeT n		= graph.CsrT::GetNeighborListLength(v);

	    // if no predicted location, and neighbor locations exists
	    // Custom spatial center kernel for geolocation
            // <TODO> Needs proper implementation for median calculation
	    if (!util::isValid(latitude[v]) && !util::isValid(longitude[v])) {
		
		if (valid_locations[v] == 0) {
                    latitude[v]  = util::PreDefinedValues<ValueT>::InvalidValue;
                    longitude[v] = util::PreDefinedValues<ValueT>::InvalidValue;
		} else if (valid_locations[v] == 1) {
                    // the only location that is valid
		    latitude[v] = locations_lat[(v * n) + 0];
                    longitude[v] = locations_lon[(v * n) + 0];

		} else if (valid_locations[v] == 2) {
                    // mid-point of the two locations
		    latitude[v] = (ValueT)((locations_lat[(v * n) + 0] + locations_lat[(v * n) + 1]) / 2);
		    longitude[v] = (ValueT)((locations_lon[(v * n) + 0] + locations_lon[(v * n) + 1]) / 2);
                } else {
		    ValueT temp_lat = 0;
		    ValueT temp_lon = 0;
                    // calculate spatial median
		    for (SizeT i = 0; i < valid_locations[v]; i++) {
			temp_lat += locations_lat[(v * n) + i];
			temp_lon += locations_lon[(v * n) + i];
		    }
		    latitude[v] = (ValueT) (temp_lat/valid_locations[v]);
		    longitude[v] = (ValueT) (temp_lon/valid_locations[v]);
                }
	    } // </TODO> -- median calculation.
	};

	auto status_op = [
	    latitude,
	    longitude,
	    active
	] __host__ __device__ (VertexT *v_q, const SizeT &pos) {

	    VertexT v           = v_q[pos];
	    if (util::isValid(latitude[v]) && util::isValid(longitude[v])) {
                // TODO: Confirm atomic usage:
                // I don't think I need an atomic
                // here, it doesn't matter who reaches the int
                // first, as long as it gets to add 1 to it.
		atomicAdd(&active[0], 1);
	    }
	};

	// Run --
        GUARD_CU(frontier.V_Q()->ForAll(
            gather_op, frontier.queue_length,
            util::DEVICE, oprtr_parameters.stream));

	GUARD_CU(frontier.V_Q()->ForAll(
	    compute_op, frontier.queue_length,
	    util::DEVICE, oprtr_parameters.stream));

	// GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
	//    "cudaStreamSynchronize failed");
 
	GUARD_CU(frontier.V_Q()->ForAll(
            status_op, frontier.queue_length,
            util::DEVICE, oprtr_parameters.stream));

        GUARD_CU(data_slice.active .SetPointer(&data_slice.active_, sizeof(SizeT), util::HOST));
        GUARD_CU(data_slice.active .Move(util::DEVICE, util::HOST));

	// printf("Current CPU active predictions = %u.\n", data_slice.active_);
 
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

    bool Stop_Condition(int gpu_num = 0) 
    {
        auto &enactor_slice = this -> enactor -> enactor_slices[0];
        auto &enactor_stats = enactor_slice.enactor_stats;
        auto &data_slice    = this -> enactor -> problem -> data_slices[this -> gpu_num][0];
	auto &graph 	    = data_slice.sub_graph[0];

        auto iter 	    = enactor_stats.iteration;

	// Anymore work to do?
	// printf("Predictions active in Stop: %u vs. needed %u.\n", data_slice.active_, graph.nodes);	
	if(data_slice.active_ >= graph.nodes)
	    return true;

	// else, keep running
        return false;
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
        // </TODO>
        util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
       
        GUARD_CU(BaseEnactor::Reset(target));

        SizeT nodes = this -> problem -> data_slices[0][0].sub_graph[0].nodes;
        printf("nodes=%d\n", nodes);

        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
           if (this->num_gpus == 1) {
               this -> thread_slices[gpu].init_size = nodes;
               for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                   auto &frontier = this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                   frontier.queue_length = (peer_ == 0) ? nodes : 0;
                   if (peer_ == 0) {

                      util::Array1D<SizeT, VertexT> tmp;
                      tmp.Allocate(nodes, target | util::HOST);
                      for(SizeT i = 0; i < nodes; ++i) {
                          tmp[i] = (VertexT)i % nodes;
                      }
                      GUARD_CU(tmp.Move(util::HOST, target));

                      GUARD_CU(frontier.V_Q() -> ForEach(tmp,
                          []__host__ __device__ (VertexT &v, VertexT &i) {
                          v = i;
                      }, nodes, target, 0));

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
     * @brief Enacts a hello computing on the specified graph.
...
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact(
        // <TODO> problem specific data if necessary, eg
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
