// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>

namespace gunrock {
namespace app {
namespace sssp {

cudaError_t UseParameters2(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    retval = app::UseParameters2(parameters);
    if (retval) return retval;
    return retval;
}

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template <
    //typename AdvanceKernelPolicy,
    //typename FilterKernelPolicy,
    typename Enactor>
static CUT_THREADPROC SSSPThread(
    void * thread_data_)
{
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::VertexT    VertexT   ;
    typedef typename Enactor::ValueT     ValueT    ;
    //typedef typename Problem::DataSlice  DataSlice ;
    //typedef GraphSlice <VertexId, SizeT, Value>          GraphSliceT;
    //typedef SSSPFunctor<VertexId, SizeT, Value, Problem> Functor;

    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data -> problem;
    Enactor      *enactor            =  (Enactor*)     thread_data -> enactor;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx[thread_num] ;
    auto         &thread_status      =   thread_data -> status;
    //DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    //FrontierAttribute<SizeT>
    //             *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    auto         &enactor_slice      =   enactor     -> enactor_slices[thread_num * num_gpus];
    auto         &enactor_stats      =   enactor_slice.enactor_stats;
    auto         &thread_retval      =   enactor_stats.retval;

    if (thread_retval = util::SetDevice(gpu_idx))
    {
        thread_status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

    thread_status = ThreadSlice::Status::Idle;
    while (thread_status != ThreadSlice::Status::ToKill)
    {
        while (thread_status == ThreadSlice::Status::Wait ||
               thread_status == ThreadSlice::Status::Idle)
        {
            sleep(0);
            //std::this_thread::yield();
        }
        if (thread_status == ThreadSlice::Status::ToKill)
            break;

        for (int peer_=0;peer_<num_gpus;peer_++)
        {
            //frontier_attribute[peer_].queue_index  = 0;        // Work queue index
            //frontier_attribute[peer_].queue_length = peer_==0?thread_data->init_size:0;
            //frontier_attribute[peer_].selector     = 0;//frontier_attrbute[peer_].queue_length == 0 ? 0 : 1;
            //frontier_attribute[peer_].queue_reset  = true;
            //enactor_stats     [peer_].iteration    = 0;
        }

        //gunrock::app::Iteration_Loop
        //    <Enactor, Functor,
        //    SSSPIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
        //    Problem::MARK_PATHS ? 1:0, 1>
        //    (thread_data);
        //printf("SSSP_Thread finished\n");fflush(stdout);
        thread_status = ThreadSlice::Status::Idle;
    }

    thread_status = ThreadSlice::Status::Ended;
    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on
 */
template <typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<typename _Problem::GraphT, ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::ValueT   ValueT  ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef EnactorBase<GraphT , ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;

    Problem     *problem      ;

    /**
     * @brief BFSEnactor constructor
     */
    Enactor() :
        BaseEnactor("sssp"),
        problem    (NULL  )
    {}

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~Enactor()
    {
        Release();
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Release(target)) return retval;
        problem = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

    /**
     * @brief Initialize the problem.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    //template<
    //    typename AdvanceKernelPolicy,
    //    typename FilterKernelPolicy>
    cudaError_t InitSSSP(
        util::Parameters &parameters,
        Problem          *problem,
        //Enactor_Flag      flag   = Enactor_None,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = BaseEnactor::Init(parameters, Enactor_None, 2, NULL, 1024, target))
            return retval;

        this->problem = problem;

        retval = this -> Init_Threads(this, (CUT_THREADROUTINE)&(SSSPThread<EnactorT>));
        if (retval) return retval;
        return retval;
    }

    /**
     * @brief Reset enactor
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(VertexT src, util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Reset(target))
            return retval;
        return retval;
    }

    /** @} */

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] src Source node to start primitive.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    //template<
    //    typename AdvanceKernelPolicy,
    //    typename FilterKernelPolicy>
    cudaError_t EnactSSSP(
        VertexT src)
    {
        cudaError_t  retval     = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src]))
                 this -> thread_slices[gpu].init_size = 1;
            else this -> thread_slices[gpu].init_size = 0;
            // TODO: move to somewhere else
            //this->frontier_attribute[gpu*this->num_gpus].queue_length
            //    = thread_slices[gpu].init_size;
        }

        retval = this -> Run_Threads();
        if (retval) return retval;

        if (this -> flag & Debug) util::PrintMsg("\nGPU SSSP Done.\n");
        return retval;
    }

    /*typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        8,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        10,                                 // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        1,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    TWC_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LB_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_CULL>
    LB_CULL_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    LB_LIGHT_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT_CULL>
    LB_LIGHT_CULL_AdvanceKernelPolicy;

    template <typename Dummy, gunrock::oprtr::advance::MODE A_MODE>
    struct MODE_SWITCH{};

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::TWC_FORWARD>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<TWC_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <TWC_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };*/

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @param[in] src Source node to start primitive.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    //template <typename SSSPProblem>
    cudaError_t Enact(
        VertexT     src,
        std::string traversal_mode = "LB")
    {
        /*if (this -> min_sm_version >= 300)
        {
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Enact(*this, src);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Enact(*this, src);
        }*/

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr   *context,
        Problem      *problem,
        int          max_grid_size = 0,
        std::string  traversal_mode = "LB")
    {
        /*if (this -> min_sm_version >= 300)
        {
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else printf("Traversal_mode %s is undefined for SSSP\n", traversal_mode.c_str());
        }*/

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */
};

} // namespace sssp
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
