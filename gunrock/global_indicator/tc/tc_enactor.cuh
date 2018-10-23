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

using namespace gunrock::app;

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
    /**
     * @brief Core computation of tc, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // Data tc that works on
        auto         &enactor            =   this -> enactor[0];
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &labels             =   data_slice.labels;
        auto         &frontier           =   enactor_slice.frontier;
        auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
	auto         &cub_temp_space     =   data_slice.cub_temp_space;
        auto         &retval             =   enactor_stats.retval;
        //auto         &stream             =   enactor_slice.stream;
        auto         &iteration          =   enactor_stats.iteration;
        auto          graph_ptr          =   data_slice.sub_graph;
        if (enactor_stats.iteration != 0)
            graph_ptr = &(data_slice.new_graphs[enactor_stats.iteration % 2]);
        auto         &graph              =   graph_ptr[0];

        // The advance operation
        auto advance_op = [src_node_ids]
        __host__ __device__ (
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            bool res = src < dest;
            VertexT id = (res) ? 1 : 0;
            Store(src_node_ids + edge_id, id);
            return res;
        };

        // The filter operation
        auto filter_op = [] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            if (!util::isValid(dest)) return false;
            return true;
        };

        oprtr_parameters.label = iteration + 1;
        // Call the advance operator, using the advance operation
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
            oprtr_parameters, advance_op, filter_op));

        if (oprtr_parameters.advance_mode != "LB_CULL" &&
            oprtr_parameters.advance_mode != "LB_LIGHT_CULL")
        {
            frontier.queue_reset = false;
            // Call the filter operator, using the filter operation
            GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
                graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
                oprtr_parameters, filter_op));
        }

        // Get back the resulted frontier length
        GUARD_CU(frontier.work_progress.GetQueueLength(
            frontier.queue_index, frontier.queue_length,
            false, oprtr_parameters.stream, true));

        GUARD_CU(util::CUBSelect_if(queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
                                    graph_slice->column_indices.GetPointer(util::DEVICE),
                                    graph_slice->column_indices.GetPointer(util::DEVICE) + graph.edges / 2,
                                    graph.edges));

        GUARD_CU(util::CUBSegReduce_sum(data_slice->d_src_node_ids.GetPointer(util::DEVICE),
                                        data_slice->d_edge_list.GetPointer(util::DEVICE),
                                        graph_slice->row_offsets.GetPointer(util::DEVICE),
                                        graph.nodes));

        mgpu::Scan<mgpu::MgpuScanTypeExc>(
            data_slice->d_edge_list.GetPointer(util::DEVICE),
            graph_slice->nodes+1,
            (int)0,
            mgpu::plus<int>(),
            (int*)0,
            (int*)0,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            context[0]);

        mgpu::IntervalExpand(
            graph_slice->edges/2,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            graph_slice->nodes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            context[0]);

        // 2) Do intersection using generated edge lists from the previous step.
        //gunrock::oprtr::intersection::LaunchKernel
        //<IntersectionKernelPolicy, TCProblem, TCFunctor>(
        //);

        // Reuse d_scanned_edges
        SizeT *d_output_triplets = d_scanned_edges;
        util::MemsetKernel<<<256, 1024>>>(d_output_triplets, (SizeT)0, graph_slice->edges);

        // Should make tc_count a member var to TCProblem
        float cc = gunrock::oprtr::intersection::LaunchKernel
            <IntersectionKernelPolicy, Problem, TCFunctor>(
            statistics[0],
            attributes[0],
            d_data_slice,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            data_slice->d_degrees.GetPointer(util::DEVICE),
            data_slice->d_edge_tc.GetPointer(util::DEVICE),
            d_output_triplets,
            graph_slice->edges/2,
            graph_slice->nodes,
            graph_slice->edges/2,
            work_progress[0],
            context[0],
            stream);

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
        auto iteration = enactor_slice.enactor_stats.iteration;
        auto         &distances          =   data_slice.distances;
        auto         &labels             =   data_slice.labels;
        auto         &preds              =   data_slice.preds;
        auto          label              =   this -> enactor ->
            mgpu_slices[this -> gpu_num].in_iteration[iteration % 2][peer_];

        auto expand_op = [distances, labels, label, preds]
        __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            ValueT in_val  = value__associate_ins[in_pos];
            ValueT old_val = atomicMin(distances + key, in_val);
            if (old_val <= in_val)
                return false;
            if (labels[key] == label)
                return false;
            labels[key] = label;
            if (!preds.isEmpty())
                preds[key] = vertex_associate_ins[in_pos];
            return true;
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }
}; // end of TCIteration


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
    typedef TCEnactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
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
