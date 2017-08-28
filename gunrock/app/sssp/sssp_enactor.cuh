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
#include <gunrock/oprtr/oprtr.cuh>

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
 * @tparam Enactor Enactor type we process on.
 * @param[in] thread_data_ Thread data.
 */
template <typename EnactorT>
static CUT_THREADPROC SSSPThread(
    void * thread_data_)
{
    typedef typename EnactorT::Problem    Problem   ;
    typedef typename EnactorT::SizeT      SizeT     ;
    typedef typename EnactorT::VertexT    VertexT   ;
    typedef typename EnactorT::ValueT     ValueT    ;
    typedef typename Problem ::GraphT     GraphT    ;
    typedef typename GraphT  ::CsrT       CsrT      ;
    typedef typename GraphT  ::GpT        GpT       ;
    //typedef typename Problem::DataSlice  DataSlice ;
    //typedef GraphSlice <VertexId, SizeT, Value>          GraphSliceT;
    //typedef SSSPFunctor<VertexId, SizeT, Value, Problem> Functor;

    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data -> problem;
    EnactorT     *enactor            =  (EnactorT*)    thread_data -> enactor;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx[thread_num] ;
    auto         &thread_status      =   thread_data -> status;
    auto         &data_slice         =   problem     -> data_slices[thread_num];
    //FrontierAttribute<SizeT>
    //             *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    auto         &enactor_slice      =   enactor     -> enactor_slices[thread_num * num_gpus];
    auto         &enactor_stats      =   enactor_slice.enactor_stats;
    auto         &graph              =   data_slice  -> sub_graph[0];
    auto         &distances          =   data_slice  -> distances;
    auto         &labels             =   data_slice  -> labels;
    auto         &preds              =   data_slice  -> preds;
    auto         &weights            =   graph.CsrT::edge_values;
    auto         &original_vertex    =   graph.GpT::original_vertex;
    auto         &frontier           =   enactor_slice.frontier;
    auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
    auto         &retval             =   enactor_stats.retval;
    SizeT         iteration          =   0;

    if (retval = util::SetDevice(gpu_idx))
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

        oprtr::Advance<oprtr::OprtrType_V2V>(
            (CsrT)graph, frontier.V_Q(frontier.queue_index),
            frontier.V_Q(frontier.queue_index + 1), oprtr_parameters,
            [distances, weights, original_vertex, preds]__host__ __device__ (
                const VertexT &src, VertexT &dest, const SizeT &edge_id,
                const VertexT &input_item, const SizeT &input_pos,
                SizeT &output_pos) -> bool {
                ValueT src_distance, edge_weight;

                util::io::ModifiedLoad<oprtr::COLUMN_READ_MODIFIER>::Ld(
                    src_distance, distances + src);
                util::io::ModifiedLoad<oprtr::COLUMN_READ_MODIFIER>::Ld(
                    edge_weight, weights + edge_id);
                ValueT new_distance = src_distance + edge_weight;

                // Check if the destination node has been claimed as someone's child
                ValueT old_distance = atomicMin(distances + dest, new_distance);
                if (new_distance < old_distance)
                {
                    VertexT pred = src;
                    if (original_vertex + 0 != NULL)
                        pred = original_vertex[src];
                    util::io::ModifiedStore<oprtr::QUEUE_WRITE_MODIFIER>::St(
                        pred, preds + dest);
                    return true;
                }
                return false;
            });

        oprtr::Filter<oprtr::OprtrType_V2V>(
            graph, frontier.V_Q(frontier.queue_index),
            frontier.V_Q(frontier.queue_index + 1), oprtr_parameters,
            [labels, iteration] __host__ __device__(
                const VertexT &src, VertexT &dest, const SizeT &edge_id,
                const VertexT &input_item, const SizeT &input_pos,
                SizeT &output_pos) -> bool {

                if (!util::isValid(dest)) return false;
                if (labels[dest] == iteration) return false;
                (labels + dest)[0] = iteration;
                return true;
            });
        thread_status = ThreadSlice::Status::Idle;
    }

    thread_status = ThreadSlice::Status::Ended;
    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 * @tparam _Problem Problem type we process on
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT, typename _Problem::LabelT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::ValueT   ValueT  ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename Problem::LabelT   LabelT  ;
    typedef EnactorBase<GraphT , LabelT, ARRAY_FLAG, cudaHostRegisterFlag>
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
        GUARD_CU(BaseEnactor::Release(target));
        problem = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Initialize the problem.
     * @param[in] parameters Running parameters.
     * @param[in] problem Pointer to Problem object.
     * @param[in] target Target location of data
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        util::Parameters &parameters,
        Problem          *problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = problem;

        // Lazy initialization
        GUARD_CU(BaseEnactor::Init(parameters, Enactor_None, 2, NULL, /*1024,*/ target));
        GUARD_CU(this -> Init_Threads(this, (CUT_THREADROUTINE)&(SSSPThread<EnactorT>)));
        return retval;
    }

    /**
     * @brief Reset enactor
     * @param[in] src Source node to start primitive.
     * @param[in] target Target location of data
     * \return cudaError_t object Indicates the success of all CUDA calls.
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
                 this -> thread_slices[gpu].init_size = 1;
            else this -> thread_slices[gpu].init_size = 0;
            // TODO: move to somewhere else
            //this->frontier_attribute[gpu*this->num_gpus].queue_length
            //    = thread_slices[gpu].init_size;
        }
        return retval;
    }

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Enact(VertexT src)
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads());
        if (this -> flag & Debug)
            util::PrintMsg("\nGPU SSSP Done.\n");
        return retval;
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
