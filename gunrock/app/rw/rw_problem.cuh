// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hello_problem.cuh
 *
 * @brief GPU Storage management Structure for hello Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace rw {
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

    // <DONE> Add problem specific command-line parameter usages here, e.g.:
    // </DONE>

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
        util::Array1D<SizeT, VertexT> walks;
        util::Array1D<SizeT, float> rand;
        int walk_length;
        curandGenerator_t gen;

        // </DONE>

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            // <DONE> name of the problem specific arrays:
            walks.SetName("walks");
            rand.SetName("rand");
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
            GUARD_CU(walks.Release(target));
            GUARD_CU(rand.Release(target));
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
            ProblemFlag    flag,
            int walk_length_)
        {
            cudaError_t retval  = cudaSuccess;

            GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

            // <DONE> allocate problem specific data here, e.g.:
            walk_length = walk_length_;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

            GUARD_CU(walks.Allocate(sub_graph.nodes * walk_length, target));
            GUARD_CU(rand.Allocate(sub_graph.nodes, target));
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
        cudaError_t Reset(util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = this -> sub_graph -> nodes;

            // Ensure data are allocated
            // <DONE> ensure size of problem specific data:
            GUARD_CU(walks.EnsureSize_(nodes * this -> walk_length, target));
            GUARD_CU(rand.EnsureSize_(nodes, target));
            // </DONE>

            // Reset data
            // <DONE> reset problem specific data, e.g.:
            GUARD_CU(walks.ForEach([]__host__ __device__ (VertexT &x){
               x = (VertexT)0;
            }, nodes * this -> walk_length, target, this -> stream));
            GUARD_CU(rand.ForEach([]__host__ __device__ (float &x){
               x = 0.0;
            }, nodes, target, this -> stream));
            // </DONE>

            return retval;
        }
    }; // DataSlice

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;
    // <ADDITIONAL_TODO>
    int walk_length;
    // </ADDITIONAL_TODO>

    // ----------------------------------------------------------------
    // Problem Methods

    /**
     * @brief hello default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL) {
        // <ADDITIONAL_TODO>
        walk_length = _parameters.Get<int>("walk-length");
        // </ADDITIONAL_TODO>
    }

    /**
     * @brief hello default destructor
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
        VertexT *h_walks,
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
                GUARD_CU(data_slice.walks.SetPointer(h_walks, nodes * this -> walk_length, util::HOST));
                GUARD_CU(data_slice.walks.Move(util::DEVICE, util::HOST));
                // </DONE>
            } else if (target == util::HOST) {
                // <DONE> extract the results from single CPU, e.g.:
                GUARD_CU(data_slice.walks.ForEach(h_walks,
                   []__host__ __device__ (const VertexT &device_val, VertexT &host_val){
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
                this -> flag,
                this -> walk_length
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
        // <TODO> problem specific data if necessary, eg
        // VertexT src,
        // </TODO>
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        // Reset data slices
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            GUARD_CU(data_slices[gpu] -> Reset(target));
            GUARD_CU(data_slices[gpu].Move(util::HOST, target));
        }

        // <TODO> Additional problem specific initialization
        // </TODO>

        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        return retval;
    }
};

} //namespace Template
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
