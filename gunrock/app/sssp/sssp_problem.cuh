// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace sssp {

cudaError_t UseParameters(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    retval = gunrock::app::UseParameters(parameters);
    if (retval) return retval;

    return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure stores device-side vectors for doing SSSP computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of value used for computed values.
 * @tparam _MARK_PREDECESSORS   Whether to mark predecessor value for each node.
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
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, ValueT >    distances  ;     /**< Used for source distance */
        //util::Array1D<SizeT, VertexT    >    visit_lookup;    /**< Used for check duplicate */
        //util::Array1D<SizeT, float       >    delta;
        //util::Array1D<SizeT, int         >    sssp_marker;
        util::Array1D<SizeT, VertexT>    labels; // labels to mark latest iteration the vertex been visited
        util::Array1D<SizeT, VertexT>    preds ; // predecessors of vertices

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            distances       .SetName("distances"       );
            //visit_lookup    .SetName("visit_lookup"    );
            //delta           .SetName("delta"           );
            //sssp_marker     .SetName("sssp_marker"     );
            labels          .SetName("labels"          );
            preds           .SetName("preds"           );
        }

        /*
         * @brief Default destructor
         */
        virtual ~DataSlice()
        {
            Release();
        }

        cudaError_t Release(util::Location target = util::LOCATION_ALL)
        {
            cudaError_t retval = cudaSuccess;
            if (target & util::DEVICE)
                if (retval = util::SetDevice(this->gpu_idx)) return retval;

            if (retval = distances      .Release(target)) return retval;
            //if (retval = visit_lookup  .Release(target)) return retval;
            //if (retval = delta         .Release(target)) return retval;
            //if (retval = sssp_marker   .Release(target)) return retval;
            if (retval = labels         .Release(target)) return retval;
            if (retval = preds          .Release(target)) return retval;
            if (retval = BaseDataSlice ::Release(target)) return retval;
            return retval;
        }

        /*bool HasNegativeValue(Value* vals, size_t len)
        {
            for (int i = 0; i < len; ++i)
                if (vals[i] < 0.0) return true;
            return false;
        }*/

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to the GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] delta_factor Delta factor for delta-stepping.
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         * @param[in] skip_makeout_selection
         * @param[in] keep_node_num
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
            GraphT &sub_graph,
            int     gpu_idx = 0,
            util::Location target = util::DEVICE,
            ProblemFlag flag = Problem_None)
        {
            cudaError_t retval  = cudaSuccess;

            retval = BaseDataSlice::Init(sub_graph, gpu_idx, target, flag);
            if (retval) return retval;

            retval = distances .Allocate(sub_graph.nodes, target);
            if (retval) return retval;

            retval = labels    .Allocate(sub_graph.nodes, target);
            if (retval) return retval;

            if (flag & Mark_Predecessors)
            {
                retval = preds.Allocate(sub_graph.nodes, target);
                if (retval) return retval;
            }

            if (target & util::DEVICE)
            {
                retval = sub_graph.CsrT::row_offsets   .Move(
                    util::HOST, target, -1, 0, this -> stream);
                if (retval) return retval;

                retval = sub_graph.CsrT::column_indices.Move(
                    util::HOST, target, -1, 0, this -> stream);
                if (retval) return retval;

                retval = sub_graph.CsrT::edge_values   .Move(
                    util::HOST, target, -1, 0, this -> stream);
                if (retval) return retval;
            }
            return retval;
        } // Init

        /*
         * @brief Estimate delta factor for delta-stepping.
         *
         * @param[in] graph Reference to the graph we process on.
         *
         * \return float Delta factor.
         */
        /*float EstimatedDelta(const Csr<VertexId, Value, SizeT> &graph) {
            double  avgV = graph.average_edge_value;
            int     avgD = graph.average_degree;
            return avgV * 32 / avgD;
        }*/

        /**
         * @brief Reset problem function. Must be called prior to each run.
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Reset(util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = this -> sub_graph -> nodes;

            // Ensure data are allocated
            retval = distances.EnsureSize_(nodes, target);
            if (retval) return retval;

            retval = labels   .EnsureSize_(nodes, target);
            if (retval) return retval;

            if (this -> flag & Mark_Predecessors)
                retval = preds.EnsureSize_(nodes, target);
            if (retval) return retval;

            //retval = visit_lookup.EnsureSize_(this -> sub_graph -> nodes, target);
            //if (retval) return retval;

            // Reset data
            retval = distances.ForEach([]__host__ __device__ (ValueT &distance){
                    distance = util::PreDefinedValues<ValueT>::MaxValue;
                }, nodes, target, this -> stream);
            if (retval) return retval;

            retval = labels   .ForEach([]__host__ __device__ (VertexT &label){
                    label = util::PreDefinedValues<VertexT>::InvalidValue;
                }, nodes, target, this -> stream);
            if (retval) return retval;

            if (this -> flag & Mark_Predecessors)
                retval = preds.ForAll([]__host__ __device__ (VertexT *preds_, const SizeT &pos){
                    preds_[pos] = pos;
                }, nodes, target, this -> stream);
            if (retval) return retval;

            //retval = visit_lookup.ForEach([]__host__ __device__ (VertexT &lookup){
            //        lookup = util::PreDefinedValues<VertexT>::InvalidValue;
            //    }, nodes, target, this -> stream);
            //if (retval) return retval;

            //retval = sssp_marker.ForEach([]__host__ __device__ (int &marker){
            //        marker = 0;
            //    }, nodes, target, this -> stream);
            //if (retval) return retval;

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

    Problem(ProblemFlag _flag = Problem_None) :
        BaseProblem(_flag),
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

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices == NULL) return retval;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (retval = data_slices[i].Release(target)) return retval;
        }
        if ((target & util::HOST) != 0 && data_slices[0].GetPointer(util::DEVICE) == NULL)
        {
            delete[] data_slices; data_slices=NULL;
        }
        if (retval = BaseProblem::Release(target)) return retval;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result distancess computed on the GPU back to host-side vectors.
     *
     * @param[out] h_distances host-side vector to store computed node distances (distances from the source).
     * @param[out] h_preds host-side vector to store computed node predecessors (used for extracting the actual shortest path).
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(
        ValueT  *h_distances,
        VertexT *h_preds = NULL,
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;

        if (this-> num_gpus == 1)
        {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE)
            {
                retval = util::SetDevice(this->gpu_idx[0]);
                if (retval) return retval;

                retval = data_slice.distances.SetPointer(h_distances, nodes, util::HOST);
                if (retval) return retval;
                retval = data_slice.distances.Move(util::DEVICE, util::HOST);
                if (retval) return retval;

                if ((this -> flag & Mark_Predecessors) == 0)
                    return retval;

                retval = data_slice.preds.SetPointer(h_preds, nodes, util::HOST);
                if (retval) return retval;
                retval = data_slice.preds.Move(util::DEVICE, util::HOST);
                if (retval) return retval;

            } else if (target == util::HOST) {
                retval = data_slice.distances.ForEach(h_distances,
                    []__host__ __device__(const ValueT &distance, ValueT &h_distance){
                        h_distance = distance;
                    }, nodes, util::HOST);
                if (retval) return retval;

                if (this -> flag & Mark_Predecessors)
                    retval = data_slice.preds.ForEach(h_preds,
                    []__host__ __device__(const VertexT &pred, VertexT &h_pred){
                        h_pred = pred;
                    }, nodes, util::HOST);
                if (retval) return retval;
            }

        } else { // num_gpus != 1

            util::Array1D<SizeT, ValueT > th_distances;
            util::Array1D<SizeT, VertexT> th_preds;

            th_distances.SetName("bfs::Problem::Extract::th_distances");
            th_preds    .SetName("bfs::Problem::Extract::th_preds");
            retval = th_distances.Allocate(this->num_gpus, util::HOST);
            if (retval) return retval;
            retval = th_preds    .Allocate(this->num_gpus, util::HOST);
            if (retval) return retval;

            for (int gpu = 0; gpu < this->num_gpus; gpu++)
            {
                auto &data_slice = data_slices[gpu][0];
                if (target == util::DEVICE)
                {
                    retval = util::SetDevice(this->gpu_idx[gpu]);
                    if (retval) return retval;
                    retval = data_slice.distances.Move(util::DEVICE, util::HOST);
                    if (retval) return retval;

                    if (this -> flag & Mark_Predecessors)
                    {
                        retval = data_slice.preds.Move(util::DEVICE, util::HOST);
                        if (retval) return retval;
                    }
                }

                th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
                th_preds    [gpu] = data_slice.preds    .GetPointer(util::HOST);
            } //end for(gpu)

            for (VertexT v = 0; v < nodes; v++)
            {
                int gpu = this -> org_graph.GpT::partition_table[v];
                VertexT v_ = v;
                if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
                    v_ = this -> org_graph.GpT::convertion_table[v];

                h_distances[v] = th_distances[gpu][v_];
                if (this -> flag & Mark_Predecessors)
                    h_preds[v] = th_preds    [gpu][v_];
            }

            retval = th_distances.Release();
            retval = th_preds    .Release();
        } //end if

        return retval;
    }

    /**
     * @brief initialization function.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Pointer to the CSR graph object we process on. @see Csr
     * @param[in] inversegraph Pointer to the inversed CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     * @param[in] gpu_idx GPU index used for testing.
     * @param[in] partition_method Partition method to partition input graph.
     * @param[in] streams CUDA stream.
     * @param[in] delta_factor delta factor for delta-stepping.
     * @param[in] queue_sizing Maximum queue sizing factor.
     * @param[in] in_sizing
     * @param[in] partition_factor Partition factor for partitioner.
     * @param[in] partition_seed Partition seed used for partitioner.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
            util::Parameters &parameters,
            GraphT &graph,
            //partitioner::PartitionFlag partition_flag = partitioner::PARTITION_NONE,
            util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        retval = BaseProblem::Init(parameters, graph, target);
        if (retval) return retval;

        // No data in DataSlice needs to be copied from host

        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                retval = util::SetDevice(this->gpu_idx[gpu]);
            if (retval) return retval;

            retval = data_slices[gpu].Allocate(1, target | util::HOST);
            if (retval) return retval;

            auto &data_slice = data_slices[gpu][0];
            //_data_slice->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);
            retval = data_slice.Init(this -> sub_graphs[gpu], this -> gpu_idx[gpu], target, this -> flag);
            if (retval) return retval;
        } // end for (gpu)

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     *
     * @param[in] src Source node to start.
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            VertexT    src,
            util::Location target = util::DEVICE)
    {

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (target & util::DEVICE)
                retval = util::SetDevice(this->gpu_idx[gpu]);
            if (retval) return retval;
            retval = data_slices[gpu] -> Reset(target);
            if (retval) return retval;
            retval = data_slices[gpu].Move(util::HOST, target);
            if (retval) return retval;
        }

        // Fillin the initial input_queue for SSSP problem
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
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

        ValueT src_distance = 0;
        if (target & util::HOST)
        {
            data_slices[gpu] -> distances[src_] = src_distance;
            if (this -> flag & Mark_Predecessors)
                data_slices[gpu] -> preds[src_] = -1;
        }

        if (target & util::DEVICE)
        {
            retval = util::GRError(cudaMemcpy(
                data_slices[gpu]->distances.GetPointer(util::DEVICE)+ src_,
                &src_distance, sizeof(ValueT),
                cudaMemcpyHostToDevice),
                "SSSPProblem cudaMemcpy distances failed",
                __FILE__, __LINE__);
            if (retval) return retval;

            if (this -> flag & Mark_Predecessors)
            {
                VertexT src_pred = -1;
                retval = util::GRError(cudaMemcpy(
                    data_slices[gpu]->preds.GetPointer(util::DEVICE)+ src_,
                    &src_pred, sizeof(VertexT),
                    cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy preds failed",
                    __FILE__, __LINE__);
                if (retval) return retval;
            }
        }
        return retval;
    }

    /** @} */

};

} //namespace sssp
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
