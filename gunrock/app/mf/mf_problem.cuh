// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_problem.cuh
 *
 * @brief GPU Storage management Structure for Max Flow Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>

namespace gunrock {
namespace app {
namespace mf {

/**
 * @brief Speciflying parameters for MF Problem
 * @param  parameters  The util::Parameter<...> structure holding all 
 *			parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    // TODO: Add problem specific command-line parameter usages here, e.g.:
    GUARD_CU(parameters.Use<bool>(
        "mark-pred",
        util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | 
	util::OPTIONAL_PARAMETER,
        false,
        "Whether to mark predecessor info.",
        __FILE__, __LINE__));

    return retval;
}

/**
 * @brief Max Flow Problem structure stores device-side arrays
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in mf
 * @tparam _ValueT  Type of signed integer to use as capacity and flow 
		    of edges and as excess and height values of vertices.
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    typename _ValueT  = typename _GraphT::ValueT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef	    _GraphT	      GraphT;
    typedef typename GraphT::VertexT  VertexT;
    typedef typename GraphT::SizeT    SizeT;
    typedef typename GraphT::GpT      GpT;
    typedef	    _ValueT	      ValueT;

    static const ProblemFlag FLAG =	_FLAG;
    typedef ProblemBase	 <GraphT, FLAG>	BaseProblem;
    typedef DataSliceBase<GraphT, FLAG>	BaseDataSlice;

    //Helper structures

    /**
     * @brief Data structure containing MF-specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {
        // MF-specific storage arrays:
        util::Array1D<SizeT, ValueT>  capacity;  // edge capacity
        util::Array1D<SizeT, ValueT>  flow;      // edge flow 
        util::Array1D<SizeT, ValueT>  excess;    // vertex excess
        util::Array1D<SizeT, VertexT> height;    // vertex height

	VertexT	source;	// source vertex
	VertexT sink;	// sink vertex

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
	    source = util::PreDefinedValues<VertexT>::InvalidValue;
	    sink = util::PreDefinedValues<VertexT>::InvalidValue;
            capacity.SetName("capacity"	);
            excess  .SetName("excess"	);
            flow    .SetName("flow"	);
            height  .SetName("height"	);
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

            // TODO: Release problem specific data, e.g.:
            GUARD_CU(capacity.Release(target));
            GUARD_CU(excess.Release(target));
            GUARD_CU(flow.Release(target));
            GUARD_CU(height.Release(target));
            GUARD_CU(BaseDataSlice ::Release(target));
            return retval;
        }

        /**
         * @brief initializing mf-specific data on each gpu
         * @param     sub_graph   Sub graph on the GPU.
         * @param[in] gpu_idx     GPU device index
         * @param[in] target      Targeting device location
         * @param[in] flag        Problem flag containling options
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Init(
            GraphT        &sub_graph,
	    int		   num_gpus = 1,
            int            gpu_idx = 0,
            util::Location target  = util::DEVICE,
            ProblemFlag    flag    = Problem_None)
        {
            cudaError_t retval  = cudaSuccess;
	    SizeT nodes_size = sub_graph.nodes;
	    SizeT edges_size = sub_graph.edges;

            GUARD_CU(BaseDataSlice::Init(
				sub_graph, num_gpus, gpu_idx, target, flag));

            // 
	    // Allocate data on Gpu
	    //
            GUARD_CU(capacity .Allocate(edges_size, target));
            GUARD_CU(flow     .Allocate(edges_size, target));
            GUARD_CU(excess   .Allocate(nodes_size, target));
            GUARD_CU(height   .Allocate(nodes_size, target));

            //GUARD_CU(sub_graph.Move(util::HOST, target, this -> stream));
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
            SizeT nodes_size = this -> sub_graph -> nodes;
            SizeT edges_size = this -> sub_graph -> edges;

            // Ensure data are allocated
            // TODO: ensure size of problem specific data, e.g.:
            // GUARD_CU(distances.EnsureSize_(nodes, target));
            GUARD_CU(capacity.EnsureSize_(edges_size, target));
            GUARD_CU(excess.EnsureSize_(nodes_size, target));
            GUARD_CU(flow.EnsureSize_(edges_size, target));
            GUARD_CU(height.EnsureSize_(nodes_size, target));

            // Reset data
            // TODO: reset problem specific data, e.g.:
            GUARD_CU(capacity.ForEach([]__host__ __device__
            (ValueT &capacity){
              capacity = util::PreDefinedValues<ValueT>::InvalidValue;
            }, edges_size, target, this -> stream));

            GUARD_CU(excess.ForEach([]__host__ __device__
            (ValueT &excess){
              excess = util::PreDefinedValues<ValueT>::InvalidValue;
            }, nodes_size, target, this -> stream));

            GUARD_CU(flow.ForEach([]__host__ __device__
            (ValueT &flow){
              flow = util::PreDefinedValues<ValueT>::InvalidValue;
            }, edges_size, target, this -> stream));

            GUARD_CU(height.ForEach([]__host__ __device__
            (ValueT &height){
              height = util::PreDefinedValues<ValueT>::InvalidValue;
            }, nodes_size, target, this -> stream));

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Methods

    /**
     * @brief MFProblem default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief MFProblem default destructor
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
     * @param[out] h_distances Host array to store computed vertex distances 
     *		   from the source.
     * \return     cudaError_t Error message(s), if any
     */
    cudaError_t Extract(
        // TODO: add list of results to extract, e.g.:
        ValueT	       **h_flow,
        util::Location  target      = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
	/* aga
        SizeT nodes = this -> org_graph -> nodes;

        if (this-> num_gpus == 1)
        {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE)
            {
                GUARD_CU(util::SetDevice(this->gpu_idx[0]));

                // TODO: extract the results from single GPU, e.g.:
                GUARD_CU(data_slice.excess.SetPointer(
					h_excess, nodes, util::HOST));
                GUARD_CU(data_slice.excess.Move(util::DEVICE, util::HOST));

            }
            else if (target == util::HOST)
            {
                // TODO: extract the results from single CPU, e.g.:
		GUARD_CU(data_slice.excess.ForEach(h_excess,
		  []__host__ __device__
                  (const ValueT &excess, ValueT &h_excess){
                    h_excess = excess;
                  }, nodes, util::HOST));
            }
        }
        else
        { // num_gpus != 1
            // TODO: extract the results from multiple GPUs, e.g.:
            util::Array1D<SizeT, ValueT *> th_excess;
            th_excess.SetName("mf::Problem::Extract::th_excess");
            GUARD_CU(th_excess.Allocate(this->num_gpus, util::HOST));

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE)
                {
                    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                    GUARD_CU(data_slice.excess.Move(util::DEVICE, util::HOST));
                }
                th_excess[gpu] = data_slice.excess.GetPointer(util::HOST);
            } //end for(gpu)

            for (VertexT v = 0; v < nodes; v++)
            {
                int gpu = this -> org_graph -> GpT::partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph -> GpT::convertion_table[v];

                h_excess[v] = th_excess[gpu][v_];
            }

            GUARD_CU(th_excess.Release());
        } //end if
	*/
        return retval;
    }

    /**
     * @brief initialization function.
     * @param     graph       The graph that MF processes on
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

        // TODO get problem specific flags from parameters, e.g.:
        // if (this -> parameters.template Get<bool>("mark-pred"))
        //    this -> flag = this -> flag | Mark_Predecessors;

        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            data_slices[gpu].SetName(
			    "data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

            auto &data_slice = data_slices[gpu][0];
            GUARD_CU(data_slice.Init(this -> sub_graphs[gpu],
				this -> gpu_idx[gpu], target, this -> flag));
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
        // TODO: add problem specific info, e.g.:
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

        // TODO: Initial problem specific starting point, e.g.:
        int gpu;
        VertexT src_;
        if (this->num_gpus <= 1)
	{
	    gpu = 0; 
	    src_=src;
        } else 
	{
            gpu = this -> org_graph -> partition_table[src];
            if (this -> flag & partitioner::Keep_Node_Num)
                src_ = src;
            else
                src_ = this -> org_graph -> GpT::convertion_table[src];
         }
         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
         GUARD_CU2(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed");

         ValueT src_excess = std::numeric_limits<ValueT>::max();
         if (target & util::HOST)
         {
             data_slices[gpu] -> excess[src_] = src_excess;
         }
         if (target & util::DEVICE)
         {
            GUARD_CU2(cudaMemcpy(
                data_slices[gpu]->excess.GetPointer(util::DEVICE) + src_,
                &src_excess, sizeof(ValueT),
                cudaMemcpyHostToDevice),
                "MFProblem cudaMemcpy excess failed");
         }

        GUARD_CU2(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed");
        return retval;
    }

    /** @} */
};

} //namespace mf
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

