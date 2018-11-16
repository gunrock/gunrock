// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * ss_problem.cuh
 *
 * @brief GPU Storage management Structure for SS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/ss/ss_test.cuh>

namespace gunrock {
namespace app {
namespace ss {

/**
 * @brief Speciflying parameters for SS Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    return retval;
}

/**
 * @brief Scan statistics Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in ss
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    typename _LabelT = typename _GraphT::VertexT,
    typename _ValueT = typename _GraphT::ValueT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;
    typedef          _LabelT         LabelT;
    typedef          _ValueT         ValueT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;

    //Helper structures

    /**
     * @brief Data structure containing SS-specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {
        // ss-specific storage arrays
        util::Array1D<SizeT, ValueT >    scan_stats  ; // scan statistis values
        util::Array1D<SizeT, VertexT>    nodes      ; // node ids corresponding to sorted values

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            scan_stats       .SetName("scan_stats");
            nodes            .SetName("nodes"     );
        }

        /*
         * @brief Default destructor
         */
        virtual ~DataSlice()
        {
            Release();
        }

        /*
         * @brief Releasing allocated memory space
         * @param[in] target      The location to release memory from
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Release(util::Location target = util::LOCATION_ALL)
        {
            cudaError_t retval = cudaSuccess;
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx));

            GUARD_CU(scan_stats      .Release(target));
            GUARD_CU(nodes           .Release(target));
            GUARD_CU(BaseDataSlice ::Release(target));
            return retval;
        }

        /**
         * @brief initializing ss-specific data on each gpu
         * @param     sub_graph   Sub graph on the GPU.
         * @param[in] num_gpus    Number of GPUs
         * @param[in] gpu_idx     GPU device index
         * @param[in] target      Targeting device location
         * @param[in] flag        Problem flag containling options
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Init(
            GraphT        &sub_graph,
            int            num_gpus = 1,
            int            gpu_idx = 0,
            util::Location target  = util::DEVICE,
            ProblemFlag    flag    = Problem_None)
        {
            cudaError_t retval  = cudaSuccess;

            GUARD_CU(BaseDataSlice::Init(
                sub_graph, num_gpus, gpu_idx, target, flag));
            GUARD_CU(scan_stats.Allocate(sub_graph.nodes, target));
            GUARD_CU(nodes     .Allocate(sub_graph.nodes, target));

            /*if (target & util::DEVICE)
            {
                GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
            }*/
            if (target & util::DEVICE)
            {
                GUARD_CU(sub_graph.Move(util::HOST, target, this -> stream));
            }
            return retval;
        } // Init

        /**
         * @brief Reset problem function. Must be called prior to each run.
         * @param[in] target      Targeting device location
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Reset(util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT num_nodes = this -> sub_graph -> nodes;

            // Ensure data are allocated
            GUARD_CU(scan_stats.EnsureSize_(num_nodes, target));
            GUARD_CU(nodes     .EnsureSize_(num_nodes, target));

            // Reset data
            GUARD_CU(scan_stats.ForEach([]__host__ __device__
            (ValueT &scan_stat){
                scan_stat = util::PreDefinedValues<ValueT>::MinValue;
            }, num_nodes, target, this -> stream));

            GUARD_CU(nodes   .ForEach([]__host__ __device__
            (VertexT & node){
                node = util::PreDefinedValues<VertexT>::InvalidValue;
            }, num_nodes, target, this -> stream));

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Methods

    /**
     * @brief SSProblem default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief SSProblem default destructor
     */
    virtual ~Problem()
    {
        Release();
    }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices == NULL) return retval;
        for (int i = 0; i < this->num_gpus; i++)
            GUARD_CU(data_slices[i].Release(target));

        if ((target & util::HOST) != 0 &&
            data_slices[0].GetPointer(util::DEVICE) == NULL)
        {
            delete[] data_slices; data_slices=NULL;
        }
        GUARD_CU(BaseProblem::Release(target));
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result distancess computed on GPUs back to host-side arrays.
     * @param[out] h_distances Host array to store computed vertex distances from the source.
     * @param[out] h_preds     Host array to store computed vertex predecessors.
     * @param[in]  target where the results are stored
     * \return     cudaError_t Error message(s), if any
     */
    cudaError_t Extract(
        ValueT         *h_scan_stat,
        VertexT        *h_node     = NULL,
        util::Location  target      = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;

        if (this-> num_gpus == 1)
        {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE)
            {
                GUARD_CU(util::SetDevice(this->gpu_idx[0]));

                GUARD_CU(data_slice.scan_stats.SetPointer(
                    h_scan_stat, nodes, util::HOST));
                GUARD_CU(data_slice.scan_stats.Move(util::DEVICE, util::HOST));

                GUARD_CU(data_slice.nodes.SetPointer(h_node, 1, util::HOST));
                GUARD_CU(data_slice.nodes.Move(util::DEVICE, util::HOST));
            }
            else if (target == util::HOST) {
                GUARD_CU(data_slice.scan_stats.ForEach(h_scan_stat,
                    []__host__ __device__
                    (const ValueT &scan_stat, ValueT &h_scan_stat_){
                        h_scan_stat_ = scan_stat;
                    }, nodes, util::HOST));

                GUARD_CU(data_slice.nodes.ForEach(h_node,
                    []__host__ __device__
                    (const VertexT &node, VertexT &h_node_){
                        h_node_ = node;
                    }, 1, util::HOST));
            }
        }
        else { // num_gpus != 1
            util::Array1D<SizeT, ValueT *> th_scan_stats;
            util::Array1D<SizeT, VertexT*> th_nodes;
            th_scan_stats.SetName("bfs::Problem::Extract::th_scan_stats");
            th_nodes     .SetName("bfs::Problem::Extract::th_nodes");
            GUARD_CU(th_scan_stats.Allocate(this->num_gpus, util::HOST));
            GUARD_CU(th_nodes    .Allocate(this->num_gpus, util::HOST));

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE)
                {
                    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                    GUARD_CU(data_slice.scan_stats.Move(util::DEVICE, util::HOST));
                    GUARD_CU(data_slice.nodes.Move(util::DEVICE, util::HOST));
                }
                th_scan_stats[gpu] = data_slice.scan_stats.GetPointer(util::HOST);
                th_nodes    [gpu] = data_slice.nodes    .GetPointer(util::HOST);
            } //end for(gpu)

            for (VertexT v = 0; v < nodes; v++)
            {
                int gpu = this -> org_graph -> GpT::partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph -> GpT::convertion_table[v];

                h_scan_stat[v] = th_scan_stats[gpu][v_];
                h_node      [v] = th_nodes     [gpu][v_];
            }

            GUARD_CU(th_scan_stats.Release());
            GUARD_CU(th_nodes     .Release());
        } //end if

        return retval;
    }

    /**
     * @brief initialization function.
     * @param     graph       The graph that SS processes on
     * @param[in] Location    Memory location to work on
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(
        GraphT           &graph,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseProblem::Init(graph, target));
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        if (this -> parameters.template Get<bool>("mark-pred"))
            this -> flag = this -> flag | Mark_Predecessors;

        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

            auto &data_slice = data_slices[gpu][0];
            GUARD_CU(data_slice.Init(this -> sub_graphs[gpu],
                this -> num_gpus, this -> gpu_idx[gpu], target, this -> flag));
        } // end for (gpu)

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] src      Source vertex to start.
     * @param[in] location Memory location to work on
     * \return cudaError_t Error message(s), if any
     */
    cudaError_t Reset(
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
            // Set device
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            GUARD_CU(data_slices[gpu] -> Reset(target));
            GUARD_CU(data_slices[gpu].Move(util::HOST, target));
        }

        GUARD_CU2(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed");
        return retval;
    }

    /** @} */
};

} //namespace ss
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
