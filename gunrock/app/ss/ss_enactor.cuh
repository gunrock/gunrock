// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * ss_enactor.cuh
 *
 * @brief SS Problem Enactor
 */

#pragma once

#include <gunrock/util/sort_device.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/ss/ss_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace ss {

/**
 * @brief Speciflying parameters for Scan Statistics Enactor
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
 * @brief defination of SS iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SSIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push>
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;

    typedef typename EnactorT::Problem::GraphT       GraphT;
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push> BaseIterationLoop;

    SSIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of ss, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // Data ss works on
        auto         &enactor            =   this -> enactor[0];
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &scan_stats         =   data_slice.scan_stats;
        auto         &row_offsets        =   graph.CsrT::row_offsets;
        auto         &frontier           =   enactor_slice.frontier;
        auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
        auto         &retval             =   enactor_stats.retval;
        auto         &stream             =   oprtr_parameters.stream;
        auto         &iteration          =   enactor_stats.iteration;
        auto          target             =   util::DEVICE;

        // First add degrees to scan statistics
        GUARD_CU(scan_stats.ForAll([scan_stats, row_offsets]
            __host__ __device__ (ValueT *scan_stats_, const SizeT & v)
            {
                scan_stats_[v] = row_offsets[v + 1] - row_offsets[v];
            }, graph.nodes, target, stream));

        // Compute number of triangles for each edge of each node divided by 2

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
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];

        auto expand_op = []
        __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            return true;
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }
}; // end of SSIteration

/**
 * @brief SS enactor class.
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
        typename _Problem::LabelT,
        typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    // Definations
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::ValueT   ValueT  ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename Problem::LabelT   LabelT  ;
    typedef EnactorBase<GraphT , LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef SSIterationLoop<EnactorT> IterationT;

    // Members
    Problem     *problem   ;
    IterationT  *iterations;

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSEnactor constructor
     */
    Enactor() :
        BaseEnactor("ss"),
        problem    (NULL  )
    {
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
    }

    /**
     * @brief SSEnactor destructor
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
     * @brief Initialize the enactor.
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

        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None, 2, NULL, target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice
                = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
      * @brief one run of ss, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            // TODO: change to how many {VertexT, ValueT} data need to communicate
            //       per element in the inter-GPU sub-frontiers
            0, 1,
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
    cudaError_t Reset(VertexT src, util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));
        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
	    this -> thread_slices[gpu].init_size = 1;
	    for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
	    {
		this -> enactor_slices[gpu * this -> num_gpus + peer_]
		    .frontier.queue_length = 1;
	    }
        }
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
     * @brief Enacts a SS computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact()
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU SS Done.", this -> flag & Debug);
        return retval;
    }

    /** @} */
};

} // namespace ss
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

