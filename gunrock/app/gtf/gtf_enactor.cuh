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
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));
    return retval;
}

/**
 * @brief defination of gtf iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct GTFIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push>
{
    typedef typename EnactorT::VertexT	VertexT;
    typedef typename EnactorT::ValueT	ValueT;
    typedef typename EnactorT::SizeT	SizeT;
    typedef typename EnactorT::Problem	ProblemT;
    typedef typename ProblemT::GraphT	GraphT;
    typedef typename GraphT::CsrT	CsrT;
    typedef IterationLoopBase <EnactorT, Use_FullQ | Push> BaseIterationLoop;

    GTFIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of gtf, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
	auto enactor		= this -> enactor;
	auto gpu_num		= this -> gpu_num;
	auto num_gpus		= enactor -> num_gpus;
	auto gpu_offset		= num_gpus * gpu_num;
        auto &data_slice	= enactor -> problem -> data_slices[gpu_num][0];
        auto &enactor_slice	= enactor ->
				  enactor_slices[gpu_offset + peer_];
        auto &enactor_stats	= enactor_slice.enactor_stats;
        auto &graph		= data_slice.sub_graph[0];
        auto &frontier        	= enactor_slice.frontier;
        auto &oprtr_parameters	= enactor_slice.oprtr_parameters;
        auto &retval          	= enactor_stats.retval;
        auto &iteration       	= enactor_stats.iteration;

	auto source		= data_slice.source;
	auto sink		= data_slice.sink;
	auto &capacity        	= graph.edge_values;
	auto &reverse		= data_slice.reverse;
        auto &flow            	= data_slice.flow;
        auto &excess          	= data_slice.excess;
        auto &height	      	= data_slice.height;
	auto &lowest_neighbor	= data_slice.lowest_neighbor;
	auto &local_vertices	= data_slice.local_vertices;
	auto &active		= data_slice.active;
	auto null_ptr		= &local_vertices;
	null_ptr = NULL;

	auto advance_push_op = [capacity, flow, excess, height, reverse,
	     source, sink, active]
	    __host__ __device__
	    (const VertexT &src, VertexT &dest, const SizeT &edge_id,
	    const VertexT &input_item, const SizeT &input_pos,
	    const SizeT &output_pos) -> bool
	{
	    if (!util::isValid(dest) or !util::isValid(src) or
		    src == source or src == sink)
		return false;
	    auto e = excess[src];
	    auto cf = capacity[edge_id] - flow[edge_id];
	    auto f = min(cf, e);
	    auto rev_id = reverse[edge_id];
	    if (f > 0 && height[src] == height[dest] + 1)
	    {
		if (atomicAdd(&excess[src], -f) >= f)
		{
		    atomicAdd(&excess[dest], f);
		    atomicAdd(&flow[edge_id], f);
		    atomicAdd(&flow[rev_id], -f);
//		    printf("push %d->%d, flow %lf, e[%d] %lf, e[%d] %lf\n", \
			    src, dest, f, src, excess[src], dest, excess[dest]);
		}else{
		    atomicAdd(&excess[src], f);
//		    printf("rollback push %d->%d, excess[%d] = %lf\n", \
			    src, dest, src, excess[src]);
		}
		active[0] = 1;
		return true;
	    }
	    return false;
	};

	auto advance_find_lowest_op =
	    [excess, capacity, flow, lowest_neighbor, height, iteration,
	    source, sink, active]
	    __host__ __device__
	    (const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
	    if (!util::isValid(dest) or !util::isValid(src) or
		    src == source or src == sink)
		return false;
	    if (excess[src] > (ValueT)0 and
		    capacity[edge_id] - flow[edge_id] > (ValueT)0)
	    {
		auto l = lowest_neighbor[src];
		auto height_dest = height[dest];
		while (!util::isValid(l) or height_dest < height[l]){
		    l = atomicCAS(&lowest_neighbor[src], l, dest);
		}
		if (lowest_neighbor[src] == dest){
		    return true;
		}
	    }
	    return false;
	};

	auto advance_relabel_op =
	    [excess, capacity, flow, lowest_neighbor, height, iteration,
	    source, sink, active]
	    __host__ __device__
	    (const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
	    if (!util::isValid(dest) or !util::isValid(src) or
		    src == source or src == sink)
		return false;
	    if (excess[src] > (ValueT)0 and
		    capacity[edge_id] - flow[edge_id] > (ValueT)0 and
		    lowest_neighbor[src] == dest){
		if (height[src] <= height[dest])
		{
	//	    printf("relabel src %d, dest %d, H[%d]=%d, -> %d\n",\
			src, dest, src, height[src], height[dest]+1);
		    height[src] = height[dest] + 1;
		    active[0] = 1;
		    return true;
		}
	    }
	    return false;
	};

//	GUARD_CU(excess.ForAll(
//	    [] __host__ __device__ (ValueT *excess_, const SizeT &v){
//	      printf("excess_[%d] = %lf\n", v, excess_[v]);
//	    }, graph.nodes, util::DEVICE, oprtr_parameters.stream));
//
//	GUARD_CU(height.ForAll(
//	    [] __host__ __device__ (VertexT *h, const SizeT &v){
//	      printf("height[%d] = %d\n", v, h[v]);
//	    }, graph.nodes, util::DEVICE, oprtr_parameters.stream));
//
//	GUARD_CU(flow.ForAll(
//	    [] __host__ __device__ (ValueT *f, const SizeT &v){
//	      printf("flow[%d] = %lf\n", v, f[v]);
//	    }, graph.edges, util::DEVICE, oprtr_parameters.stream));


	oprtr_parameters.advance_mode = "ALL_EDGES";
	GUARD_CU(active.ForAll(
	    [] __host__ __device__ (SizeT *a, const SizeT &v){
	      a[v] = 0;
	    }, 1, util::DEVICE, oprtr_parameters.stream));

	// ADVANCE_PUSH_OP
	GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
		    graph.csr(), &local_vertices, null_ptr,
		    oprtr_parameters, advance_push_op));
	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
	    "cudaStreamSynchronize failed");

	GUARD_CU(lowest_neighbor.ForAll(
          [] __host__ __device__ (VertexT *el, const SizeT &v){
	    el[v] = util::PreDefinedValues<VertexT>::InvalidValue;
          }, graph.nodes, util::DEVICE, oprtr_parameters.stream));
	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
          "cudaStreamSynchronize failed");


	// ADVANCE_FIND_LOWEST_OP
	GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
		    graph.csr(), &local_vertices, null_ptr,
		    oprtr_parameters, advance_find_lowest_op));
	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
	    "cudaStreamSynchronize failed");

//	GUARD_CU(lowest_neighbor.ForAll(
//          [] __host__ __device__ (VertexT *el, const SizeT &v){
//            printf("lowest_neighbor[%d] = %d\n", v, el[v]);
//          }, graph.nodes, util::DEVICE, oprtr_parameters.stream));
	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
          "cudaStreamSynchronize failed");

	// ADVANCE RELABEL OP
	GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
		    graph.csr(), &local_vertices, null_ptr,
		    oprtr_parameters, advance_relabel_op));
	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
	    "cudaStreamSynchronize failed");

//	GUARD_CU(active.ForAll(
//	    [] __host__ __device__ (SizeT *a, const SizeT &v){
//	      if (a[v]){
//		printf("there are");
//	      }else{
//		printf("there are not");
//	      }
//	      printf(" active nodes\n");
//	    }, 1, util::DEVICE, oprtr_parameters.stream));

	GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
	    "cudaStreamSynchronize failed");

	//printf("new updated vertices %d\n", frontier.queue_length);

	frontier.queue_reset = true;
	oprtr_parameters.filter_mode = "BY_PASS";
	GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
		    graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
		    oprtr_parameters,
	[active]
	__host__ __device__
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

	debug_aml("ExpandIncomming do nothing");
/*	for key " +
		    std::to_string(key) + " and for in_pos " +
		    std::to_string(in_pos) + " and for vertex ass ins " +
		    std::to_string(vertex_associate_ins[in_pos]) +
		    " and for value ass ins " +
		    std::to_string(value__associate_ins[in_pos]));*/
	auto expand_op = [capacity, flow, excess, height]
	__host__ __device__(VertexT &key, const SizeT &in_pos,
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

	debug_aml("expand incoming\n");
        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
	return retval;
    }

    bool Stop_Condition(int gpu_num = 0)
    {
	auto enactor = this -> enactor;
	int num_gpus = enactor -> num_gpus;
	auto &enactor_slice = enactor -> enactor_slices[0];
	auto iteration	= enactor_slice.enactor_stats.iteration;

	auto &retval = enactor_slice.enactor_stats.retval;
	if (retval != cudaSuccess){
	    printf("(CUDA error %d @ GPU %d: %s\n", retval, 0 % num_gpus,
		cudaGetErrorString(retval));
	    fflush(stdout);
	    return true;
	}

	auto &data_slice = enactor -> problem -> data_slices[gpu_num][0];

	if (data_slice.num_updated_vertices == 0)
	    return true;

	return false;
    }

}; // end of gtfIteration

/**
 * @brief gtf enactor class.
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
	    cudaHostRegisterFlag>				BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
								EnactorT;
    typedef GTFIterationLoop<EnactorT>				IterationT;

    Problem     *problem   ;
    IterationT  *iterations;

    /**
     * @brief gtfEnactor constructor
     */
    Enactor() :
        BaseEnactor("gtf"),
        problem    (NULL)
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
      * @brief one run of gtf, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
	debug_aml("Run enact");
        gunrock::app::Iteration_Loop<
            0, // NUM_VERTEX_ASSOCIATES
	    1, // NUM_VALUE__ASSOCIATES
            IterationT>(thread_data, iterations[thread_data.thread_num]);
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
	debug_aml("Enactor Reset, src %d", src);

	typedef typename EnactorT::Problem::GraphT::GpT GpT;
	auto num_gpus = this->num_gpus;

	GUARD_CU(BaseEnactor::Reset(target));

        // Initialize frontiers according to the algorithm gtf
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
	debug_aml("Enactor Reset end");
        return retval;
    }

    /**
     * @brief Enacts a gtf computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact()
    {
        cudaError_t  retval     = cudaSuccess;
	debug_aml("enact");
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU gtf Done.", this -> flag & Debug);
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
