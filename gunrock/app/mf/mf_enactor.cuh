// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_enactor.cuh
 *
 * @brief Max Flow Problem Enactor
 */

#pragma once
#include <gunrock/util/sort_device.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/mf/mf_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#define debug_aml(a...) 
//#define debug_aml(a...) printf(a);

//#define debug_aml2(a...) printf(a);
#define debug_aml2(a...)

namespace gunrock {
namespace app {
namespace mf {

/**
 * @brief Speciflying parameters for MF Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter 
 *		      info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));
    return retval;
}
/**
 * @brief defination of MF iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct MFIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push>
{
    typedef typename EnactorT::VertexT	VertexT;
    typedef typename EnactorT::ValueT	ValueT;
    typedef typename EnactorT::SizeT	SizeT;
    typedef typename EnactorT::Problem	ProblemT;
    typedef typename ProblemT::GraphT	GraphT;
    typedef typename GraphT::CsrT	CsrT;

    typedef IterationLoopBase <EnactorT, Use_FullQ | Push> BaseIterationLoop;

    MFIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of mf, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        auto enactor	    	= this -> enactor;
        auto gpu_num	    	= this -> gpu_num;
        auto num_gpus	    	= enactor -> num_gpus;
        auto gpu_offset	    	= num_gpus * gpu_num;
        auto &data_slice	= enactor -> problem -> data_slices[gpu_num][0];
        auto &enactor_slice	= enactor -> enactor_slices[gpu_offset + peer_];
        auto &enactor_stats	= enactor_slice.enactor_stats;
        auto &graph		= data_slice.sub_graph[0];
        auto &frontier        	= enactor_slice.frontier;
        auto &oprtr_parameters	= enactor_slice.oprtr_parameters;
        auto &retval          	= enactor_stats.retval;
        auto &iteration       	= enactor_stats.iteration;

        auto source	           	= data_slice.source;
        auto sink	        	= data_slice.sink;
        bool &was_changed       = data_slice.was_changed;
        auto &capacity        	= graph.edge_values;
        auto &reverse		    = data_slice.reverse;
        auto &flow            	= data_slice.flow;
        auto &excess          	= data_slice.excess;
        auto &height	      	= data_slice.height;
        auto &lowest_neighbor	= data_slice.lowest_neighbor;
        auto &local_vertices	= data_slice.local_vertices;
        auto &active	    	= data_slice.active;
        auto null_ptr	    	= &local_vertices;
        null_ptr = NULL;

        auto advance_preflow_op = [capacity, flow, excess, height, reverse, 
             source]
             __host__ __device__
             (const VertexT &src, VertexT &dest, const SizeT &edge_id, 
              const VertexT &input_item, const SizeT &input_pos,
              const SizeT &output_pos) -> bool
             {
                 if (!util::isValid(dest) or !util::isValid(src) or 
                         src != source)
                     return false;
                 flow[edge_id] = capacity[edge_id];
                 flow[reverse[edge_id]] = ((ValueT)0) - capacity[edge_id];
                 atomicAdd(&excess[dest], capacity[edge_id]);
                 //debug_aml("flow[%d->%d] = %lf\n", src, dest, capacity[edge_id]);
                 return true;
             };

        auto advance_lockfree_op = 
	        [graph, excess, capacity, flow, reverse, height, iteration, source, 
            sink, active] 
            __host__ __device__
            (const VertexT &src, VertexT &dest, const SizeT &edge_id,
             const VertexT &input_item, const SizeT &input_pos,
             SizeT &output_pos) -> bool
            {
                if (util::isValid(src) && src != source && src != sink)
                {
                    SizeT e_id = graph.GetNeighborListOffset(src);
                    SizeT neighbor_num = graph.GetNeighborListLength(src);
                    SizeT e_end = e_id + neighbor_num;
                    int iter = 0;
                    while (e_id == edge_id && neighbor_num > 0 && 
                            excess[src] > 0)
                    {
                        debug_aml2("src %d\n", src);
                        debug_aml2("excess[%d] = %lf\n", src, excess[src]);
                        VertexT lowest_id = 
                            util::PreDefinedValues<VertexT>::InvalidValue;
                        VertexT lowest_h;
                        while (e_id < e_end){
                            VertexT v = graph.GetEdgeDest(e_id);
                            if ((capacity[e_id] - flow[e_id] > 0) &&
                                    (!util::isValid(lowest_id) ||
                                     height[v] < lowest_h)){
                                lowest_id = e_id;
                                lowest_h = height[v];
                            }
                            ++e_id;
                        }
                        debug_aml2("lowest for %d is %d\n",
                                src, graph.GetEdgeDest(lowest_id));
                        if (lowest_h < height[src])
                        {
                            //push
                            ValueT f = 
                                fminf(capacity[lowest_id] - flow[lowest_id], 
                                        excess[src]);
                            ValueT old = atomicAdd(&excess[src], -f);
                            if (f > 0 && (old > f || almost_eql(old, f)))
                            {
                                VertexT l_dest = graph.GetEdgeDest(lowest_id);
                                atomicAdd(&excess[l_dest], f);
                                atomicAdd(&flow[lowest_id], f);
                                atomicAdd(&flow[reverse[lowest_id]], -f);
                                debug_aml2("push, %lf, %d->%d, e[%d] = %lf\n", 
                                        f, src, l_dest, l_dest, excess[l_dest]);
                                active[0] = 1; 
                            }else{
                                atomicAdd(&excess[src], f);
                                debug_aml2("push back, %lf, %lf\n", 
                                        f, excess[src]);
                            } 
                        }else{
                            //relabel
                            height[src] = lowest_h + 1;
                            debug_aml2("relabel, %d new height %d\n", 
                                    src, lowest_h + 1);
                            active[0] = 1;
                        }
                    }
                }
            };

            
        if (iteration == 0){
            debug_aml("iteration 0, preflow operator is comming\n");
            was_changed = true;
            // ADVANCE_PREFLOW_OP
            oprtr_parameters.advance_mode = "ALL_EDGES";
            GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
                    graph.csr(), &local_vertices, null_ptr,
                    oprtr_parameters, advance_preflow_op));

            debug_aml("iteration 0, preflow ends, results:\n");
        }

        //Global relabeling
        if (was_changed == true and (iteration % 50 == 0)){
            debug_aml("iteration %d, relabeling\n", iteration);
            GUARD_CU(height.Move(util::DEVICE, util::HOST, graph.nodes, 0,
                        oprtr_parameters.stream));
            GUARD_CU(flow.Move(util::DEVICE, util::HOST, graph.edges, 0, 
                        oprtr_parameters.stream));
            GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), 
                    "cudaStreamSynchronize failed.");
            int num_changes = relabeling(graph, source, sink, 
                    height.GetPointer(util::HOST), 
                    reverse.GetPointer(util::HOST), 
                    flow.GetPointer(util::HOST));
            GUARD_CU(height.Move(util::HOST, util::DEVICE, graph.nodes, 0, 
                      oprtr_parameters.stream));
            if (num_changes > 0)
                was_changed = true;
            debug_aml("iteration %d, relabeling finished\n", iteration);
        }

        GUARD_CU(active.ForAll(
            [] __host__ __device__ (SizeT *a, const SizeT &v)
            {
                a[v] = 0;
            }, 1, util::DEVICE, oprtr_parameters.stream));

        // ADVANCE_LOCKFREE_OP
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
                    graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
                    //graph.csr(), &local_vertices, null_ptr,
                    oprtr_parameters, advance_lockfree_op));

        frontier.queue_reset = true;
        oprtr_parameters.filter_mode = "BY_PASS";
        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
                    graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
                    oprtr_parameters, 
                    [active] __host__ __device__
                    (const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool
                    {
                        return active[0] > 0;
                    }));

        frontier.queue_index++;

        // Get back the resulted frontier length
        GUARD_CU(frontier.work_progress.GetQueueLength(
                    frontier.queue_index, frontier.queue_length,
                    false, oprtr_parameters.stream, true));

        GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                "cudaStreamSynchronize failed");

        debug_aml2("new updated vertices %d (version after filter)\n", \
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
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {
        auto &enactor	    = this -> enactor;
        auto &problem	    = enactor -> problem;
        auto gpu_num	    = this -> gpu_num;
        auto gpu_offset	    = gpu_num * enactor -> num_gpus;
        auto &data_slice    = problem -> data_slices[gpu_num][0];
        auto &enactor_slice = enactor -> enactor_slices[gpu_offset + peer_];
        auto iteration	    = enactor_slice.enactor_stats.iteration;

        auto &capacity	    = data_slice.sub_graph[0].edge_values; 
        auto &flow  	    = data_slice.flow;
        auto &excess	    = data_slice.excess;
        auto &height	    = data_slice.height;

/*	for key " + 
		    std::to_string(key) + " and for in_pos " +
		    std::to_string(in_pos) + " and for vertex ass ins " +
		    std::to_string(vertex_associate_ins[in_pos]) +
		    " and for value ass ins " +
		    std::to_string(value__associate_ins[in_pos]));*/
    
        auto expand_op = [capacity, flow, excess, height] 
        __host__ __device__(VertexT &key, const SizeT &in_pos,
        VertexT *vertex_associate_ins, ValueT  *value__associate_ins) -> bool
        {

            // TODO: fill in the lambda to combine received and local data, e.g.:
            // ValueT in_val  = value__associate_ins[in_pos];
            // ValueT old_val = atomicMin(distances + key, in_val);
            // if (old_val <= in_val)
            //     return false;
            return true;
        };

        cudaError_t retval = BaseIterationLoop::template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval; 
    }

    bool Stop_Condition(int gpu_num = 0)
    {
        auto enactor        = this -> enactor;
        int num_gpus        = enactor -> num_gpus;
        auto &data_slice    = enactor -> problem -> data_slices[gpu_num][0];
        auto &enactor_slice = enactor -> enactor_slices[0];
        auto &retval        = enactor_slice.enactor_stats.retval;
        auto &oprtr_parameters	= enactor_slice.oprtr_parameters;
        if (retval != cudaSuccess){
            printf("(CUDA error %d @ GPU %d: %s\n", retval, 0 % num_gpus,
                    cudaGetErrorString(retval));
            fflush(stdout);
            return true;
        }
        if (data_slice.num_updated_vertices == 0) return true;
        return false;
    }

}; // end of MFIteration

/* MF enactor class.
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
        typename _Problem::VertexT,
        typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
    public:
        typedef _Problem                  Problem;
        typedef typename Problem::VertexT VertexT;
        typedef typename Problem::ValueT  ValueT;
        typedef typename Problem::SizeT   SizeT;
        typedef typename Problem::GraphT  GraphT;
        typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, 
                cudaHostRegisterFlag> BaseEnactor;
        typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;

        typedef MFIterationLoop<EnactorT> IterationT;

        Problem     *problem   ;
        IterationT  *iterations;

        /**
         * @brief MFEnactor constructor
         */
        Enactor(): BaseEnactor("mf"), problem(NULL)
        {
            // TODO: change according to algorithmic needs
            this -> max_num_vertex_associates = 0;
            this -> max_num_value__associates = 1;
        }

        /**
         * @brief MFEnactor destructor
         */
        virtual ~Enactor()
        {
            //Release();
        }

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
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Initialize the problem.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        Problem		&problem,
        util::Location	target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;
        
	// Lazy initialization
        GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, 
		    false));

	auto num_gpus = this->num_gpus;

	for (int gpu = 0; gpu < num_gpus; ++gpu)
	{
	    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
	    auto gpu_offset = gpu * num_gpus;
	    auto &enactor_slice = this->enactor_slices[gpu_offset + 0];
	    auto &graph = problem.sub_graphs[gpu];
	    auto nodes = graph.nodes;
	    auto edges = graph.edges;
	    GUARD_CU(enactor_slice.frontier.Allocate(nodes, edges, 
			this->queue_factors));
	}

	iterations = new IterationT[num_gpus];
        for (int gpu = 0; gpu < num_gpus; gpu ++)
        {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

	GUARD_CU(this -> Init_Threads(this, 
		    (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
	return retval;
    }

    /**
      * @brief one run of mf, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
	    debug_aml("Run enact\n");
        gunrock::app::Iteration_Loop<0,1, IterationT>(
		thread_data, iterations[thread_data.thread_num]);
        
        return cudaSuccess;
    }

    /**
     * @brief Reset enactor
     * @param[in] src Source node to start primitive.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(const VertexT& src, util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
	debug_aml("Enactor Reset, src %d\n", src);
       
	typedef typename EnactorT::Problem::GraphT::GpT GpT;
	auto num_gpus = this->num_gpus;

	GUARD_CU(BaseEnactor::Reset(target));

        // Initialize frontiers according to the algorithm MF
	for (int gpu = 0; gpu < num_gpus; gpu++)
	{
	    auto gpu_offset = gpu * num_gpus;
	    if (num_gpus == 1 ||
		(gpu == this->problem->org_graph->GpT::partition_table[src]))
	    {
		this -> thread_slices[gpu].init_size = 1;
		for (int peer_ = 0; peer_ < num_gpus; ++peer_)
		{
		    auto &frontier = 
			this -> enactor_slices[gpu_offset + peer_].frontier;
		    frontier.queue_length = (peer_ == 0) ? 1 : 0;
		    if (peer_ == 0)
		    {
			GUARD_CU(frontier.V_Q() -> ForEach(
			    [src]__host__ __device__ (VertexT &v){v = src;}, 
			    1, target, 0));
		    }
		}
	    }
            else 
	    {
		this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < num_gpus; peer_++)
                {
		    auto &frontier = 
			this -> enactor_slices[gpu_offset + peer_].frontier;
		    frontier.queue_length = 0;
                }
	    }
        }
        GUARD_CU(BaseEnactor::Sync());
	debug_aml("Enactor Reset end\n");
        return retval;
    }

    /**
     * @brief Enacts a MF computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact()
    {
        cudaError_t  retval     = cudaSuccess;
	debug_aml("enact\n");
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU MF Done.", this -> flag & Debug);
        return retval;
    }

    /** @} */
};

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
