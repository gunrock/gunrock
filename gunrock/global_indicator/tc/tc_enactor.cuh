// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * tc_enactor.cuh
 *
 * @brief Problem enactor for Triangle Counting
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/global_indicator/tc/tc_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <moderngpu.cuh>
#include <cub/cub.cuh>

#include <fstream>


using namespace gunrock::app;

namespace gunrock {
namespace global_indicator {
namespace tc {

/**
 * @brief Specify parameters for TC Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));
    return retval;
}

/// Selection functor type
struct GreaterThan
{
    int compare;

    __host__ __device__ __forceinline__
        GreaterThan(int compare) : compare(compare) {}

    __host__ __device__ __forceinline__
        bool operator()(const int &a) const {
            return (a > compare);
        }
};

using namespace gunrock::app;
using namespace mgpu;
using namespace cub;

/**
 * @brief definition of TC iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct TCIterationLoop : puplic IterationLoopBase
    <EanctorT, Use_FullQ | Push |
    (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
    Update_Predecessors : 0x0)>
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push |
        (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
         Update_Predecessors : 0x0)> BaseIterationLoop;

    TCIterationLoop() : BaseIterationLoop() {}    
}

/**
 * @brief TC enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class TCEnactor :
    public EnactorBase<typename _Problem::SizeT,
		       ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                    Problem;
    typedef typename Problem::SizeT     SizeT;
    typedef typename Problem::VertexId  VertexId;
    typedef typename Problem::Value     Value;
    typedef EnactorBase<SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef TCIterationLoop<EnactorT> IterationT;

    Problem                            *problem;
    ContextPtr                         *context;

    /**
     * @brief TCEnactor constructor.
     */
    TCEnactor() :
        BaseEnactor("tc"),
        problem    (NULL)
    {
        this -> max_num_vertex_associates
            = (Problem::FLAG & Mark_Predecessors) != 0 ? 1 : 0;
        this -> max_num_value__associates = 1;
    }

    /**
     * @brief TCEnactor destructor
     */
    virtual ~TCEnactor()
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
    cudaError_t InitTC(
        ContextPtr  &context,
        Problem     &problem,
	util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this -> problem = problem;
        this -> context = context;

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

            for (int peer = 0; peer < this -> num_gpus; peer ++)
            {
                this -> enactor_slices[gpu * this -> num_gpus + peer]
                    .oprtr_parameters.labels
                    = &(problem.data_slices[gpu] -> labels);
            }
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
            if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src]))
            {
                this -> thread_slices[gpu].init_size = 1;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    auto &frontier = this ->
                        enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                    frontier.queue_length = (peer_ == 0) ? 1 : 0;
                    if (peer_ == 0)
                    {
                        GUARD_CU(frontier.V_Q() -> ForEach(
                            [src]__host__ __device__ (VertexT &v)
                        {
                            v = src;
                        }, 1, target, 0));
                    }
                }
            }

            else {
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_]
                        .frontier.queue_length = 0;
                }
            }
        }
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
      * @brief one run of tc, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0,
            1, IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Enacts a tc computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact(VertexT src)
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU SSSP Done.", this -> flag & Debug);
        return retval;
    }

  /** @} */

};

} // namespace tc
} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
