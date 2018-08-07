// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * template_problem.cuh
 *
 * @brief GPU Storage management Structure for Template Problem Data
 */

#pragma once

#include <iostream>
#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Speciflying parameters for SSSP Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    // TODO: Add problem specific command-line parameter usages here, e.g.:
    // GUARD_CU(parameters.Use<bool>(
    //    "mark-pred",
    //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    false,
    //    "Whether to mark predecessor info.",
    //    __FILE__, __LINE__));

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
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;

    // TODO: Add the graph representation used in the algorithm, e.g.:
    // - DONE
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;

    //Helper structures

    /**
     * @brief Data structure containing SSSP-specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {

        // util::Array1D<SizeT, ValueT>   distances; // distances from source
        util::Array1D<SizeT, ValueT>   bc_values;
        util::Array1D<SizeT, ValueT>   sigmas;
        util::Array1D<SizeT, VertexT>  source_path;

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            // TODO: Set names of the problem specific arrays, for example:
            // - DONE
            bc_values.SetName("bc_values");
            sigmas.SetName("sigmas");
            source_path.SetName("source_path");
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

            GUARD_CU(bc_values.Release(target));
            GUARD_CU(sigmas.Release(target));
            GUARD_CU(source_path.Release(target));

            GUARD_CU(BaseDataSlice::Release(target));
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
            int            gpu_idx = 0,
            util::Location target  = util::DEVICE,
            ProblemFlag    flag    = Problem_None)
        {
            cudaError_t retval  = cudaSuccess;

            GUARD_CU(BaseDataSlice::Init(sub_graph, gpu_idx, target, flag));

            // TODO: allocate problem specific data here, e.g.:
            // - DONE
            GUARD_CU(bc_values.Allocate(sub_graph.nodes, target));
            GUARD_CU(sigmas.Allocate(sub_graph.nodes, target));
            GUARD_CU(source_path.Allocate(sub_graph.nodes, target));

            if (target & util::DEVICE)
            {
                // TODO: move sub-graph used by the problem onto GPU, e.g.:
                // - DONE
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

            // Ensure data are allocated
            // TODO: ensure size of problem specific data, e.g.:
            // - DONE
            GUARD_CU(bc_values.EnsureSize_(nodes, target));
            GUARD_CU(sigmas.EnsureSize_(nodes, target));
            GUARD_CU(source_path.EnsureSize_(nodes, target));

            // Reset data
            // TODO: reset problem specific data, e.g.:
            // - DONE
            GUARD_CU(bc_values.ForEach([]__host__ __device__(ValueT &x){
               x = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(sigmas.ForEach([]__host__ __device__(ValueT &x){
               x = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(source_path.ForEach([]__host__ __device__(VertexT &x){
               x = -1;
            }, nodes, target, this -> stream));

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Methods

    /**
     * @brief BCProblem default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief BCProblem default destructor
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
     * \return     cudaError_t Error message(s), if any
     */
    cudaError_t Extract(
        // TODO: add list of results to extract, e.g.:
        // - DONE
        ValueT *h_bc_values,
        ValueT *h_sigmas,
        VertexT *h_source_path,
        util::Location  target = util::DEVICE)
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

                // TODO: extract the results from single GPU, e.g.:
                // - DONE
                GUARD_CU(data_slice.bc_values.SetPointer(h_bc_values, nodes, util::HOST));
                GUARD_CU(data_slice.bc_values.Move(util::DEVICE, util::HOST));

                GUARD_CU(data_slice.sigmas.SetPointer(h_sigmas, nodes, util::HOST));
                GUARD_CU(data_slice.sigmas.Move(util::DEVICE, util::HOST));

                GUARD_CU(data_slice.source_path.SetPointer(h_source_path, nodes, util::HOST));
                GUARD_CU(data_slice.source_path.Move(util::DEVICE, util::HOST));                
                
            }
            else if (target == util::HOST)
            {
                // TODO: extract the results from single CPU, e.g.:
                // - DONE
                GUARD_CU(data_slice.bc_values.ForEach(h_bc_values,
                    []__host__ __device__ (const ValueT &x, ValueT &h_x){ h_x = x; }, nodes, util::HOST));

                GUARD_CU(data_slice.sigmas.ForEach(h_sigmas,
                    []__host__ __device__ (const ValueT &x, ValueT &h_x){ h_x = x; }, nodes, util::HOST));
                
                GUARD_CU(data_slice.source_path.ForEach(h_source_path,
                   []__host__ __device__ (const VertexT &x, VertexT &h_x){ h_x = x; }, nodes, util::HOST));
            }
        }
        else
        { // num_gpus != 1
            // TODO: extract the results from multiple GPUs, e.g.:
            // - DONE
            util::Array1D<SizeT, ValueT *> th_bc_values;
            util::Array1D<SizeT, ValueT *> th_sigmas;
            util::Array1D<SizeT, VertexT *> th_source_path;
            
            th_bc_values.SetName("bfs::Problem::Extract::th_bc_values");
            th_sigmas.SetName("bfs::Problem::Extract::th_sigmas");
            th_source_path.SetName("bfs::Problem::Extract::th_source_path");
            
            GUARD_CU(th_bc_values.Allocate(this->num_gpus, util::HOST));
            GUARD_CU(th_sigmas.Allocate(this->num_gpus, util::HOST));
            GUARD_CU(th_source_path.Allocate(this->num_gpus, util::HOST));

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE)
                {
                    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                    
                    GUARD_CU(data_slice.bc_values.Move(util::DEVICE, util::HOST));
                    GUARD_CU(data_slice.sigmas.Move(util::DEVICE, util::HOST));
                    GUARD_CU(data_slice.source_path.Move(util::DEVICE, util::HOST));
                    
                }
                th_bc_values[gpu]   = data_slice.bc_values.GetPointer(util::HOST);
                th_sigmas[gpu]      = data_slice.sigmas.GetPointer(util::HOST);
                th_source_path[gpu] = data_slice.source_path.GetPointer(util::HOST);
                
            } //end for(gpu)

            for (VertexT v = 0; v < nodes; v++)
            {
                int gpu = this -> org_graph -> GpT::partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph -> GpT::convertion_table[v];

                h_bc_values[v]   = th_bc_values[gpu][v_];
                h_sigmas[v]      = th_sigmas[gpu][v_];
                h_source_path[v] = th_source_path[gpu][v_];
                
            }

            GUARD_CU(th_bc_values.Release());
            GUARD_CU(th_sigmas.Release());
            GUARD_CU(th_source_path.Release());
        } //end if

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

        // TODO get problem specific flags from parameters, e.g.:
        // if (this -> parameters.template Get<bool>("mark-pred"))
        //    this -> flag = this -> flag | Mark_Predecessors;

        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));
            auto &data_slice = data_slices[gpu][0];
            
            GUARD_CU(data_slice.Init(
                this -> sub_graphs[gpu],
                this -> gpu_idx[gpu], 
                target, 
                this -> flag
            ));
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
        VertexT src,
        util::Location target = util::DEVICE)
    {
        
        std::cout << "Problem->Reset(" << src << ")" << std::endl;
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
           gpu = 0; src_=src;
        } else {
           gpu = this -> org_graph -> partition_table[src];
           if (this -> flag & partitioner::Keep_Node_Num) {
               src_ = src;
           } else {
               src_ = this -> org_graph -> GpT::convertion_table[src];
           }
        }
        GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        
        ValueT _src_sigma = 1.0;
        VertexT _src_source_path = -1;
        if (target & util::HOST)
        {
            data_slices[gpu]->sigmas[src_] = _src_sigma;
            data_slices[gpu]->source_path[src_] = _src_source_path;
        }
        if (target & util::DEVICE)
        {
           GUARD_CU2(cudaMemcpy(
               data_slices[gpu]->sigmas.GetPointer(util::DEVICE) + src_,
               &_src_sigma, sizeof(ValueT),
               cudaMemcpyHostToDevice),
               "SSSPProblem cudaMemcpy distances failed");
           GUARD_CU2(cudaMemcpy(
               data_slices[gpu]->source_path.GetPointer(util::DEVICE) + src_,
               &_src_source_path, sizeof(VertexT),
               cudaMemcpyHostToDevice),
               "SSSPProblem cudaMemcpy distances failed");
        }
        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        
        return retval;
    }

    /** @} */
};

} //namespace Template
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
