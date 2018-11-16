// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file tc_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/tc/tc_test.cuh>

namespace gunrock {
namespace app {
namespace tc {

/**
 * @brief Specifying parameters for TC Problem
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
 * @brief Triangle Counting Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    typename _LabelT = typename _GraphT::VertexT,
    typename _ValueT = typename _GraphT::ValueT,
    ProblemFlag _FLAG = Problem_None>
struct TCProblem : ProblemBase <_GraphT, _FLAG>
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
     * @brief Data slice structure which contains problem specific data.
     *
     */
     
    struct DataSlice : BaseDataSlice 
    {
        // device storage arrays
	util::Array1D<SizeT, VertexT> 	tc_counts;

	/*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            tc_counts.SetName("tc_counts");
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

            GUARD_CU(tc_counts      .Release(target));
            GUARD_CU(BaseDataSlice ::Release(target));
            return retval;
        }

        /**
         * @brief initializing tc-specific data on each gpu
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
            GUARD_CU(tc_counts.Allocate(sub_graph.edges, target));

            if (target & util::DEVICE) 
            {
                GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
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
            SizeT nodes = this -> sub_graph -> nodes;
            SizeT edges = this -> sub_graph -> edges;

            // Ensure data are allocated
            GUARD_CU(tc_counts.EnsureSize_(edges, target));

            // Reset data
            GUARD_CU(tc_counts.ForEach([]__host__ __device__
            (VertexT &x){
                x = (VertexT)0;
            }, edges, target, this -> stream));

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    /**
     * @brief Default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }
    /**
     * @brief Default destructor
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
     * @brief Copy results computed on the GPU back to host-side vectors.
     * @param[out] h_labels
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(
	VertexT 	*h_tc_counts, 
        util::Location  target      = util::DEVICE) 
    {
        cudaError_t retval = cudaSuccess;
        SizeT edges = this -> org_graph -> edges;

	if (this -> num_gpus == 1) {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE) 
            {
		GUARD_CU(util::SetDevice(this->gpu_idx[0]));
                GUARD_CU(data_slice.tc_counts.SetPointer(h_tc_counts, edges, util::HOST));
                GUARD_CU(data_slice.tc_counts.Move(util::DEVICE, util::HOST));
            } else if (target == util::HOST) {
                GUARD_CU(data_slice.scan_stats.ForEach(h_tc_counts,
                    []__host__ __device__
                    (const VertexT &d_x, VertexT &h_x){
                        h_x = d_x;
                    }, edges, util::HOST));
            }

        } else {
/*            util::Array1D<SizeT, VertexT *> th_src_ids;
            util::Array1D<SizeT, SizeT *> th_edge_tc;
            th_src_ids.SetName("tc::TCProblem::Extract::th_src_ids");
            th_edge_tc.SetName("tc::TCProblem::Extract::th_edge_tc");
            GUARD_CU(th_src_ids.Allocate(this->num_gpus, util::HOST));
            GUARD_CU(th_edge_tc.Allocate(this->num_gpus, util::HOST));

            for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE) {
                    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                    GUARD_CU(data_slice.d_src_node_ids.Move(util::DEVICE, util::HOST));
                    GUARD_CU(data_slice.d_edge_tc.Move(util::DEVICE, util::HOST));
                }
                th_src_ids[gpu] = data_slice.d_src_ids.GetPointer(util::HOST);
                th_edge_tc[gpu] = data_slice.d_edge_tc.GetPointer(util::HOST);
            } // end for (gpu)
            for (VertexT v = 0; v < nodes; ++v) {
                int gpu = this -> org_graph -> GpT:: partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph -> GpT::convertion_table[v];

                source_ids[v] = th_src_ids[gpu][v_];
		edge_tc   [v] = th_edge_tc[gpu][v_];
            }

            GUARD_CU(th_src_ids.Release());
            GUARD_CU(th_edge_tc.Release());
*/
        } // end if

        return retval;
    }

    /**
     * @brief initialization function.
     * @param     graph       The graph that SSSP processes on
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

}  // namespace tc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
