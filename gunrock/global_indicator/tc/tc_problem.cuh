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
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

using namespace gunrock::app;

namespace gunrock {
namespace global_indicator {
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
    GUARD_CU(parameters.Use<bool>(
        "mark-pred",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        false,
        "Whether to mark predecessor info.",
        __FILE__, __LINE__));

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
	util::Array1D<SizeT, LabelT> 	d_src_node_ids;  // source node ids
	util::Array1D<SizeT, SizeT> 	d_edge_tc     ;  // edge ids
        util::Array1D<SizeT, LabelT> 	labels        ;  // labels to mark latest iteration on the vertex visited
        util::Array1D<SizeT, LabelT> 	d_edge_list   ;  // edge list
        util::Array1D<SizeT, SizeT> 	d_degrees     ;  // node degrees

	/*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
	    labels		.SetName("labels"		);
	    d_src_node_ids	.SetName("d_src_node_ids"	);
	    d_edge_tc	    	.SetName("d_edge_tc"		);
	    d_edge_list	    	.SetName("d_edge_list"		);
            d_degrees		.SetName("d_degrees"		);
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
            GUARD_CU(BaseDataSlice ::Release(target));
            GUARD_CU(d_src_node_ids .Release(target));
            GUARD_CU(d_edge_tc      .Release(target));
            GUARD_CU(d_edge_list    .Release(target));
            GUARD_CU(d_degrees      .Release(target));
            GUARD_CU(labels         .Release(target));
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
            GUARD_CU(d_degrees       .Allocate(sub_graph.nodes, target));
            GUARD_CU(labels          .Allocate(sub_graph.nodes, target));
            GUARD_CU(d_src_node_ids  .Allocate(sub_graph.edges, target));
            GUARD_CU(d_edge_tc       .Allocate(sub_graph.edges, target));
            GUARD_CU(d_edge_list     .Allocate(sub_graph.edges, target));
            if (flag & Mark_Predecessors)
            {
                GUARD_CU(preds      .Allocate(sub_graph.nodes, target));
                GUARD_CU(temp_preds .Allocate(sub_graph.nodes, target));
            }

            GUARD_CU(sub_graph.Move(util::HOST, target, this -> stream));
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
            GUARD_CU(d_src_node_ids.EnsureSize_(edges, target));
            GUARD_CU(d_edge_tc     .EnsureSize_(edges, target));
            GUARD_CU(d_edge_list   .EnsureSize_(edges, target));
            GUARD_CU(labels        .EnsureSize_(nodes, target));
            GUARD_CU(d_degrees     .EnsureSize_(nodes, target));
            if (this -> flag & Mark_Predecessors)
            {
                GUARD_CU(preds.EnsureSize_(nodes, target));
                GUARD_CU(temp_preds.EnsureSize_(nodes, target));
            }

            // Reset data
            GUARD_CU(d_edge_tc.ForEach([]__host__ __device__
            (SizeT &d_edge_tc){
                d_edge_tc = (SizeT) 0;
            }, edges, target, this -> stream));

            GUARD_CU(labels   .ForEach([]__host__ __device__
            (LabelT &label){
                label = util::PreDefinedValues<LabelT>::InvalidValue;
            }, nodes, target, this -> stream));

            if (this -> flag & Mark_Predecessors)
            {
                GUARD_CU(preds.ForAll([]__host__ __device__
                (VertexT *preds_, const SizeT &pos){
                    preds_[pos] = pos;
                }, nodes, target, this -> stream));

                GUARD_CU(temp_preds.ForAll([]__host__ __device__
                (VertexT *preds_, const SizeT &pos){
                    preds_[pos] = pos;
                }, nodes, target, this -> stream));
            }

            return retval;
        }
    }; // DataSlice

    // Members

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    /**
     * @brief Default constructor
     */
    TCProblem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief Default destructor
     */
    virtual ~TCProblem() 
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
    cudaError_t Extract(VertexT *source_ids, VertexT *dest_ids, SizeT *edge_tc,
                        util::Location  target      = util::DEVICE) 
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;
        SizeT edges = this -> org_graph -> edges;

	if (this -> num_gpus == 1) {
            auto &data_slice = data_slices[0][0];
            if (target == util::DEVICE) {
                // Set device
		GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                GUARD_CU(data_slice.d_src_node_ids.SetPointer(source_ids, edges, util::HOST));
                GUARD_CU(data_slice.d_src_node_ids.Move(util::DEVICE, util::HOST));
                GUARD_CU(data_slice.d_edge_tc.SetPointer(edge_tc, edges, util::HOST));
                GUARD_CU(data_slice.d_edge_tc.Move(util::DEVICE, util::HOST));
            } else if (target == util::HOST) {
                GUARD_CU(data_slice.d_src_node_ids.ForEach(source_ids,
                    []__host__ __device__
                    (const VertexT &d_src_node_ids, VertexT &source_ids){
                        source_ids = d_src_node_ids;
                    }, edges, util::HOST));
                GUARD_CU(data_slice.d_edge_tc.ForEach(edge_tc,
                    []__host__ __device__
                    (const VertexT &d_edge_tc, VertexT &edge_tc){
                        edge_tc, util::HOST}));

            }

        } else {
            util::Array1D<SizeT, VertexT *> th_src_ids;
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
        VertexT    src,
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

        // Fillin the initial input_queue for TC problem
        int gpu;
        VertexT src_;
        if (this->num_gpus <= 1)
        {
            gpu = 0; src_=src;
        } else {
            gpu = this -> org_graph -> partition_table[src];
            if (this -> flag & partitioner::Keep_Node_Num)
                src_ = src;
            else
                src_ = this -> org_graph -> GpT::convertion_table[src];
        }

        if (target & util::DEVICE)
        {
            GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            GUARD_CU2(cudaDeviceSynchronize(),
                "cudaDeviceSynchronize failed");
        }

        if (target & util::HOST)
        {
            data_slices[gpu] -> d_edge_tc[src_] = 0;
        }
        if (target & util::DEVICE)
        {
            GUARD_CU(cudaMemcpy(
                data_slices[gpu]->d_src_ids.GetPointer(util::DEVICE) + src_,
                &src_ids, sizeof(VertexT),
                cudaMemcpyHostToDevice),
                "TCProblem cudaMemcpy d_src_node_ids failed");
            GUARD_CU(cudaMemcpy(
                data_slices[gpu]->d_edge_tc.GetPointer(util::DEVICE) + src_,
                &edge_tc, sizeof(VertexT),
                cudaMemcpyHostToDevice),
                "TCProblem cudaMemcpy d_edge_tc failed");
	    GUARD_CU2(cudaDeviceSynchronize(),
		"cudaDeviceSynchronize failed");
        }

	return retval;
    }

    /**
     *  @brief Performs any initialization work needed for primitive
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation
     *  \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Reset(
        FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
        double queue_sizing,
        double queue_sizing1 = -1.0) 
    {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.


        cudaError_t retval = cudaSuccess;

        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) 
        {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(this->gpu_idx[gpu]),
                    "TCProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

	        data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                this->use_double_buffer,
                queue_sizing, queue_sizing1);

            if (retval = data_slices[gpu]->frontier_queues[0].keys[0].EnsureSize(
                this->nodes, util::DEVICE));
            if (retval = data_slices[gpu]->frontier_queues[0].keys[1].EnsureSize(
                this->edges, util::DEVICE));

                // allocate output labels if necessary
            if (data_slices[gpu]->d_src_node_ids.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_src_node_ids.Allocate(this->edges, util::DEVICE)) 
                    return retval;
            if (data_slices[gpu]->d_edge_tc.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_edge_tc.Allocate(this->edges, util::DEVICE)) 
                    return retval;
            if (data_slices[gpu]->d_edge_list.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_edge_list.Allocate(this->edges, util::DEVICE)) 
                    return retval;
            if (data_slices[gpu]->d_degrees.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_degrees.Allocate(this->nodes, util::DEVICE)) 
                    return retval;
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->labels.Allocate(this->nodes, util::DEVICE)) 
                    return retval;
            // TODO: code to for other allocations here  

            // TODO: fill in the initial input_queue for problem
            util::MemsetIdxKernel<<<256, 1024>>>(
                data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), this->nodes);

            util::MemsetKernel<<<256, 1024>>>(data_slices[gpu]->d_edge_tc.GetPointer(util::DEVICE), (SizeT)0, this->edges);

            util::MemsetMadVectorKernel<<<128, 128>>>(
                data_slices[gpu]->d_degrees.GetPointer(util::DEVICE),
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                this->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE)+1, -1, this->nodes);

            if (retval = util::GRError(cudaMemcpy(
                d_data_slices[gpu], data_slices[gpu], sizeof(DataSlice), cudaMemcpyHostToDevice),
                "Problem cudaMemcpy data_slices to d_data_slices failed",
                __FILE__, __LINE__)) return retval;
        }

        return retval;
    }

    /** @} */
};

}  // namespace tc
}  // namespace global_indicator
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
