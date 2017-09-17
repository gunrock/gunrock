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
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sssp {

cudaError_t UseParameters2(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters2(parameters));
    return retval;
}

template <typename EnactorT>
struct SSSPIteration : public IterationBase
    <EnactorT, Use_FullQ | Push |
    (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
    Update_Predecessors : 0x0)>
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;

    typedef IterationBase
        <EnactorT, Use_FullQ | Push |
        (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
         Update_Predecessors : 0x0)> BaseIteration;

    SSSPIteration(
        EnactorT *enactor,
        int      gpu_num) :
        BaseIteration(enactor, gpu_num)
    {}

    cudaError_t Core(int peer_ = 0)
    {
        auto         &data_slice         =   this -> enactor -> problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor -> enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &distances          =   data_slice.distances;
        auto         &labels             =   data_slice.labels;
        auto         &preds              =   data_slice.preds;
        auto         &weights            =   graph.CsrT::edge_values;
        auto         &original_vertex    =   graph.GpT::original_vertex;
        auto         &frontier           =   enactor_slice.frontier;
        auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
        auto         &retval             =   enactor_stats.retval;
        //auto         &stream             =   enactor_slice.stream;
        auto         &iteration          =   enactor_stats.iteration;

        auto advance_op = [distances, weights, original_vertex, preds]
        __host__ __device__ (
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            ValueT src_distance = Load<cub::LOAD_CG>(distances + src);
            ValueT edge_weight  = Load<cub::LOAD_CS>(weights + edge_id);
            ValueT new_distance = src_distance + edge_weight;

            // Check if the destination node has been claimed as someone's child
            ValueT old_distance = atomicMin(distances + dest, new_distance);
            if (new_distance < old_distance)
            {
                if (!preds.isEmpty())
                {
                    VertexT pred = src;
                    if (!original_vertex.isEmpty())
                        pred = original_vertex[src];
                    Store(preds + dest, pred);
                }
                return true;
            }
            return false;
        };
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
            oprtr_parameters, advance_op));

        oprtr_parameters.label = iteration + 1;
        oprtr_parameters.frontier -> queue_reset = false;

        auto filter_op = [labels, iteration] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            if (!util::isValid(dest)) return false;
            if (labels[dest] == iteration) return false;
            labels[dest] = iteration;
            return true;
        };
        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
            oprtr_parameters, filter_op));

        return retval;
    }

}; // end of struct SSSPIteration

/**
 * @brief Thread controls.
 * @tparam Enactor Enactor type we process on.
 * @param[in] thread_data_ Thread data.
 */
template <typename Enactor>
static CUT_THREADPROC SSSPThread(
    void * thread_data_)
{
    typedef SSSPIteration<Enactor>       IterationT;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::VertexT    VertexT   ;
    typedef typename Enactor::ValueT     ValueT    ;
    typedef typename Problem::GraphT     GraphT    ;
    typedef typename GraphT ::CsrT       CsrT      ;
    typedef typename GraphT ::GpT        GpT       ;

    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    //Problem      *problem            =  (Problem*)     thread_data -> problem;
    Enactor      *enactor            =  (Enactor*)     thread_data -> enactor;
    //int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   enactor -> gpu_idx[thread_num] ;
    auto         &thread_status      =   thread_data -> status;
    auto         &retval             =   enactor -> enactor_slices[thread_num * enactor -> num_gpus].enactor_stats.retval;

    if (retval = util::SetDevice(gpu_idx))
    {
        thread_status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

    IterationT iteration(enactor, thread_num);
    util::PrintMsg("Thread entered.");
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

        util::PrintMsg("Run started");
        gunrock::app::Iteration_Loop<
            ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0, 1, IterationT>(thread_data[0], iteration);
        /*while (frontier.queue_length != 0 && !(retval))
        {
            // Iteration core here
            frontier.GetQueueLength(stream);
            retval = util::GRError(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize failed.", __FILE__, __LINE__);
            iteration ++;
            if (retval) break;
        }*/
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
        GUARD_CU(BaseEnactor::Init(parameters, Enactor_None, 2, NULL, target));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem -> sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));

            for (int peer = 0; peer < this -> num_gpus; peer ++)
            {
                this -> enactor_slices[gpu * this -> num_gpus + peer].oprtr_parameters.labels
                    = &(problem -> data_slices[gpu] -> labels);
            }
        }
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
            {
                this -> thread_slices[gpu].init_size = 1;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    auto &frontier = this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                    frontier.queue_length = (peer_ == 0) ? 1 : 0;
                    if (peer_ == 0)
                    {
                        GUARD_CU(frontier.V_Q() -> ForEach([src]__host__ __device__ (VertexT &v){
                            v = src;
                        }, 1, target, 0));
                    }
                }
            }

            else {
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier.queue_length
                        = 0;
                }
            }
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
