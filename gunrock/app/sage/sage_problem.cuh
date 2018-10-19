// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace sage {

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
    //GUARD_CU(parameters.Use<bool>(
    //    "mark-pred",
    //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    false,
    //    "Whether to mark predecessor info.",
    //    __FILE__, __LINE__));

    return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sage
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    //typename _LabelT = typename _GraphT::VertexT,
    typename _ValueT = typename _GraphT::ValueT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
  //  typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;
    //typedef          _LabelT         LabelT;
    typedef          _ValueT         ValueT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;

    //Helper structures

    /**
     * @brief Data structure containing SSSP-specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {
        // sage-specific storage arrays
        util::Array1D<SizeT, ValueT> W_f_1_1D; // w_f_1 1D array. weight matrix for W^1 feature part 
        util::Array1D<SizeT, ValueT> W_a_1_1D; // w_a_1 1D array. weight matrix for W^1 agg part
        util::Array1D<SizeT, ValueT> W_f_2_1D; // w_f_2 1D array. weight matrix for W^2 feature part
        util::Array1D<SizeT, ValueT> W_a_2_1D; // w_a_2 1D array. weight matrix for W^2 agg part 
        util::Array1D<SizeT, ValueT> features_1D; // fature matrix 1D
        util::Array1D<SizeT, ValueT> children_temp;//256 agg(h_B1^1)
        util::Array1D<SizeT, ValueT> source_temp;// 256 h_B2^1
        util::Array1D<SizeT, ValueT> source_result;// 256 h_B2^2
        util::Array1D<SizeT, ValueT> child_temp;// 256 h_B1^1, I feel like this one could be local
        util::Array1D<SizeT, ValueT> sums_child_feat; //64 sum of children's features, I feel like this one could be local as well

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
          
            W_f_1_1D.SetName("W_f_1_1D");
            W_a_1_1D.SetName("W_a_1_1D");
            W_f_2_1D.SetName("W_f_2_1D");
            W_a_2_1D.SetName("W_a_2_1D");
            features_1D.SetName("features_1D");
            children_temp.SetName("children_temp");
            source_temp.SetName("source_temp");
            source_result.SetName("source_result");
            child_temp.SetName("child_temp");
            sums_child_feat.SetName("sums_child_feat");

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

            GUARD_CU(W_f_1_1D      .Release(target));
            GUARD_CU(W_a_1_1D      .Release(target));
            GUARD_CU(W_f_2_1D      .Release(target));
            GUARD_CU(W_a_2_1D      .Release(target));
            GUARD_CU(features_1D   .Release(target));
            GUARD_CU(children_temp .Release(target));
            GUARD_CU(source_temp   .Release(target));
            GUARD_CU(source_result .Release(target));
            GUARD_CU(child_temp    .Release(target));
            GUARD_CU(sums_child_feat.Release(tatget));
            GUARD_CU(BaseDataSlice ::Release(target));

            return retval;
        }

        /**
         * @brief initializing sage-specific data on each gpu
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
            GUARD_CU(W_f_1_1D .Allocate(64*128, target));
            GUARD_CU(W_a_1_1D .Allocate(64*128, target));
            GUARD_CU(W_f_2_1D .Allocate(256*128, target));
            GUARD_CU(W_a_2_1D .Allocate(256*128, target));
            GUARD_CU(fetures_1D.Allocate(sub_graph.nodes,target));
            GUARD_CU(children_temp.Allocate(sub_graph.nodes*256, target));
            GUARD_CU(source_temp.Allocate(sub_graph.nodes*256, target));
            GUARD_CU(source_result.Allocate(sub_graph.nodes*256, target));
            GUARD_CU(child_temp.Allocate(sub_graph.nodes*num_neigh1*256, target));
            GUARD_CU(sums_child_feat.Allocate(sub_graph.nodes*64, target));
                        
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

            // Reset data
            GUARD_CU(children_temp.ForEach([]__host__ __device__
            (ValueT &children_temps){
                children_temps = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(source_temp.ForEach([]__host__ __device__
            (ValueT &source_temps){
                source_temps = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(source_result.ForEach([]__host__ __device__
            (ValueT &source_results){
                source_results = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(child_temp.ForEach([]__host__ __device__
            (ValueT &child_temps){
                child_temps = 0;
            }, nodes, target, this -> stream));

            GUARD_CU(sums_child_feat.ForEach([]__host__ __device__
            (ValueT &sums_child_feats){
                sums_child_feats = 0;
            }, nodes, target, this -> stream));

            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Methods

    /**
     * @brief SSSPProblem default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief SSSPProblem default destructor
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
        //ValueT         *h_distances,
        //VertexT        *h_preds     = NULL,
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

                GUARD_CU(data_slice.source_result.SetPointer(
                    h_distances, nodes, util::HOST));
                GUARD_CU(data_slice.source_result.Move(util::DEVICE, util::HOST));
            }

            else if (target == util::HOST) {
                GUARD_CU(data_slice.source_result.ForEach(h_source_result,
                    []__host__ __device__
                    (const ValueT &source_results, ValueT &h_source_result){
                        h_source_result = source_results;
                    }, nodes, util::HOST));
            }
        }
        else 
        { // num_gpus != 1
            //util::Array1D<SizeT, ValueT *> th_distances;
            //util::Array1D<SizeT, VertexT*> th_preds;
            //th_distances.SetName("bfs::Problem::Extract::th_distances");
            //th_preds    .SetName("bfs::Problem::Extract::th_preds");
            //GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));
            //GUARD_CU(th_preds    .Allocate(this->num_gpus, util::HOST));

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE)
                {
                    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
                //    GUARD_CU(data_slice.distances.Move(util::DEVICE, util::HOST));
                //    if (this -> flag & Mark_Predecessors)
                //        GUARD_CU(data_slice.preds.Move(util::DEVICE, util::HOST));
                }
                //th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
                //th_preds    [gpu] = data_slice.preds    .GetPointer(util::HOST);
            } //end for(gpu)

            for (VertexT v = 0; v < nodes; v++)
            {
                int gpu = this -> org_graph -> GpT::partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph -> GpT::convertion_table[v];

                //h_distances[v] = th_distances[gpu][v_];
                //if (this -> flag & Mark_Predecessors)
                //    h_preds[v] = th_preds    [gpu][v_];
            }

           // GUARD_CU(th_distances.Release());
          //  GUARD_CU(th_preds    .Release());
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

    //    if (this -> parameters.template Get<bool>("mark-pred"))
    //        this -> flag = this -> flag | Mark_Predecessors;

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
       // VertexT    src,
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

} //namespace sage
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
