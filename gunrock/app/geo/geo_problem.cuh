// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_problem.cuh
 *
 * @brief GPU Storage management Structure for hello Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace geo {
// </DONE>


/**
 * @brief Speciflying parameters for hello Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    // <TODO> Add problem specific command-line parameter usages here, e.g.:
    // GUARD_CU(parameters.Use<bool>(
    //    "mark-pred",
    //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    false,
    //    "Whether to mark predecessor info.",
    //    __FILE__, __LINE__));
    // </TODO>

    return retval;
}

/**
 * @brief Template Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;

    // ----------------------------------------------------------------
    // Dataslice structure

    /**
     * @brief Data structure containing problem specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {
        // <DONE> add problem specific storage arrays:
        util::Array1D<SizeT, ValueT>  locations_lat;
	util::Array1D<SizeT, ValueT>  locations_lon;

        util::Array1D<SizeT, ValueT>  latitude;
	util::Array1D<SizeT, ValueT>  longitude;

	util::Array1D<SizeT, SizeT>   valid_locations;
	util::Array1D<SizeT, SizeT>   active;
   	SizeT			      active_; 
        // </DONE>

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            // <DONE> name of the problem specific arrays:
            locations_lat	.SetName("locations_lat");
	    locations_lon	.SetName("locations_lon");
            latitude		.SetName("latitude");
	    longitude		.SetName("longitude");
	    valid_locations	.SetName("valid_locations");
	    active		.SetName("active");
            // </DONE>
        }

        /*
         * @brief Default destructor
         */
        virtual ~DataSlice() { Release(); }

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

            // <DONE> Release problem specific data, e.g.:
            GUARD_CU(locations_lat	.Release(target));
	    GUARD_CU(locations_lon	.Release(target));

            GUARD_CU(latitude		.Release(target));
            GUARD_CU(longitude          .Release(target));

	    GUARD_CU(valid_locations	.Release(target));
	    GUARD_CU(active		.Release(target));
            // </DONE>

            GUARD_CU(BaseDataSlice ::Release(target));
            return retval;
        }

        /**
         * @brief initializing sssp-specific data on each gpu
         * @param     sub_graph   Sub graph on the GPU.
         * @param[in] gpu_idx     GPU device index
         * @param[in] target      Targeting device location
         * @param[in] flag        Problem flag containling options
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Init(
            GraphT        &sub_graph,
            int            num_gpus,
            int            gpu_idx,
            util::Location target,
            ProblemFlag    flag)
        {
            cudaError_t retval  = cudaSuccess;

            GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
	    SizeT nodes = this -> sub_graph -> nodes;

	    printf("Number of nodes for allocation: %u\n", nodes);

            // <DONE> allocate problem specific data here, e.g.:
            GUARD_CU(locations_lat	.Allocate(nodes * nodes, target));
            GUARD_CU(locations_lon      .Allocate(nodes * nodes, target));

            GUARD_CU(latitude		.Allocate(nodes, target));
            GUARD_CU(longitude          .Allocate(nodes, target));

	    GUARD_CU(valid_locations	.Allocate(nodes, target));
	    GUARD_CU(active		.Allocate(1, util::HOST|target));
            // </DONE>

            if (target & util::DEVICE) {
                // <DONE> move sub-graph used by the problem onto GPU,
                GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
                // </DONE>
            }
            return retval;
        }

        /**
         * @brief Reset problem function. Must be called prior to each run.
         * @param[in] target      Targeting device location
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Reset(
		ValueT *h_latitude,
		ValueT *h_longitude,
		util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = this -> sub_graph -> nodes;

            // Ensure data are allocated
            // <DONE> ensure size of problem specific data:
            GUARD_CU(locations_lat	.EnsureSize_(nodes * nodes, target));
            GUARD_CU(locations_lon      .EnsureSize_(nodes * nodes, target));

            GUARD_CU(latitude		.EnsureSize_(nodes, target));
	    GUARD_CU(longitude		.EnsureSize_(nodes, target));

            GUARD_CU(valid_locations	.EnsureSize_(nodes, target));
	    GUARD_CU(active		.EnsureSize_(1, util::HOST|target));
            // </DONE>

            // Reset data
            // <DONE> reset problem specific data, e.g.:

	    // Set locations of neighbors to null, this needs to be populated
	    // and using spatial center we can determine the predicted.
	    GUARD_CU(locations_lat.ForEach([]__host__ __device__ (ValueT &x)
	    {
               x = util::PreDefinedValues<ValueT>::InvalidValue;
            }, nodes * nodes, target, this -> stream));

            GUARD_CU(locations_lon.ForEach([]__host__ __device__ (ValueT &x)
            {
               x = util::PreDefinedValues<ValueT>::InvalidValue;
            }, nodes * nodes, target, this -> stream));

            GUARD_CU(valid_locations.ForEach([]__host__ __device__ (SizeT &x)
	    {
               x = (SizeT)0;
            }, nodes, target, this -> stream));

	    GUARD_CU(active.ForAll([]__host__ __device__ (SizeT *x, const VertexT &pos)
	    {
		x[pos] = 0;
            }, 1, target, this -> stream));

	    this-> active_ = 0;

	    // Assumes that all vertices have invalid positions, in reality
	    // a preprocessing step is needed to assign nodes that do have
	    // positions to have proper positions already.
	    GUARD_CU(latitude	.SetPointer(h_latitude, nodes, util::HOST));
            GUARD_CU(latitude	.Move(util::HOST, util::DEVICE));

            GUARD_CU(longitude  .SetPointer(h_longitude, nodes, util::HOST));
            GUARD_CU(longitude  .Move(util::HOST, util::DEVICE));
	    // </DONE>
	
            return retval;
        }
    }; // DataSlice

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // ----------------------------------------------------------------
    // Problem Methods

    /**
     * @brief geolocation default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL) {}

    /**
     * @brief geolocation default destructor
     */
    virtual ~Problem() { Release(); }

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
     * @brief Copy result distancess computed on GPUs back to host-side arrays.
...
     * \return     cudaError_t Error message(s), if any
     */
    cudaError_t Extract(
        // <DONE> problem specific data to extract
        ValueT *h_predicted_lat,
	ValueT *h_predicted_lon,
        // </DONE>
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;

        if (this-> num_gpus == 1) {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE) {
                GUARD_CU(util::SetDevice(this->gpu_idx[0]));

                // <DONE> extract the results from single GPU, e.g.:
                GUARD_CU(data_slice.latitude.SetPointer(h_predicted_lat, nodes, util::HOST));
                GUARD_CU(data_slice.latitude.Move(util::DEVICE, util::HOST));

                GUARD_CU(data_slice.longitude.SetPointer(h_predicted_lon, nodes, util::HOST));
                GUARD_CU(data_slice.longitude.Move(util::DEVICE, util::HOST));
                // </DONE>
            } else if (target == util::HOST) {
                // <DONE> extract the results from single CPU, e.g.:
                GUARD_CU(data_slice.latitude.ForEach(h_predicted_lat,
                   []__host__ __device__ (const ValueT &device_val, ValueT &host_val){
                       host_val = device_val;
                   }, nodes, util::HOST));

                GUARD_CU(data_slice.longitude.ForEach(h_predicted_lon,
                   []__host__ __device__ (const ValueT &device_val, ValueT &host_val){
                       host_val = device_val;
                   }, nodes, util::HOST));
                // </DONE>
            }
        } else { // num_gpus != 1
            
            // ============ INCOMPLETE TEMPLATE - MULTIGPU ============
            
            // // TODO: extract the results from multiple GPUs, e.g.:
            // // util::Array1D<SizeT, ValueT *> th_distances;
            // // th_distances.SetName("bfs::Problem::Extract::th_distances");
            // // GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));

            // for (int gpu = 0; gpu < this->num_gpus; gpu++)
            // {
            //     auto &data_slice = data_slices[gpu][0];
            //     if (target == util::DEVICE)
            //     {
            //         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            //         // GUARD_CU(data_slice.distances.Move(util::DEVICE, util::HOST));
            //     }
            //     // th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
            // } //end for(gpu)

            // for (VertexT v = 0; v < nodes; v++)
            // {
            //     int gpu = this -> org_graph -> GpT::partition_table[v];
            //     VertexT v_ = v;
            //     if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
            //         v_ = this -> org_graph -> GpT::convertion_table[v];

            //     // h_distances[v] = th_distances[gpu][v_];
            // }

            // // GUARD_CU(th_distances.Release());
        }

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

        // <TODO> get problem specific flags from parameters, e.g.:
        // if (this -> parameters.template Get<bool>("mark-pred"))
        //    this -> flag = this -> flag | Mark_Predecessors;
        // </TODO>

        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

            auto &data_slice = data_slices[gpu][0];
            GUARD_CU(data_slice.Init(
                this -> sub_graphs[gpu],
                this -> num_gpus,
                this -> gpu_idx[gpu],
                target,
                this -> flag
            ));
        }

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] src      Source vertex to start.
     * @param[in] location Memory location to work on
     * \return cudaError_t Error message(s), if any
     */
    cudaError_t Reset(
        // <DONE> problem specific data if necessary, eg
        ValueT *h_latitude,
	ValueT *h_longitude,
        // </DONE>
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        // Reset data slices
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            GUARD_CU(data_slices[gpu] -> Reset(h_latitude, h_longitude, target));
            GUARD_CU(data_slices[gpu].Move(util::HOST, target));
        }

        // <TODO> Additional problem specific initialization
        // </TODO>

        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        return retval;
    }
};

} //namespace geo
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
