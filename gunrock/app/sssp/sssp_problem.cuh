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

#include <limits>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief Single-Source Shortest Path Problem structure stores device-side vectors for doing SSSP computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of value used for computed values.
 * @tparam _MARK_PREDECESSORS   Whether to mark predecessor value for each node.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        _MARK_PATHS>
struct SSSPProblem : ProblemBase<VertexId, SizeT, Value,
    true, //MARK_PREDECESSORS
    false, //ENABLE_IDEMPOTENCE
    false, //USE_DOUBLE_BUFFER
    false, //ENABLE_BACKWARD
    false, //KEEP_ORDER
    false> //KEEP_NODE_NUM
{
    static const bool MARK_PATHS            = _MARK_PATHS;

    //Helper structures

    /**
     * @brief Data slice structure which contains SSSP problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        util::Array1D<SizeT, Value       >    labels     ;     /**< Used for source distance */
        util::Array1D<SizeT, Value       >    weights    ;     /**< Used for storing edge weights */
        util::Array1D<SizeT, VertexId    >    visit_lookup;    /**< Used for check duplicate */
        util::Array1D<SizeT, float       >    delta;
        util::Array1D<SizeT, int         >    sssp_marker;

        /*
         * @brief Default constructor
         */
        DataSlice()
        {
            labels          .SetName("labels"          );
            weights         .SetName("weights"         );
            visit_lookup    .SetName("visit_lookup"    );
            delta           .SetName("delta"           );
            sssp_marker     .SetName("sssp_marker"     );
        }

        /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            labels        .Release();
            weights       .Release();
            visit_lookup  .Release();
            delta         .Release();
            sssp_marker   .Release();
        }

        bool HasNegativeValue(Value* vals, size_t len)
        {
            for (int i = 0; i < len; ++i)
                if (vals[i] < 0.0) return true;
            return false;
        }

        /**
         * @brief initialization function.
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] num_vertex_associate Number of vertices associated.
         * @param[in] num_value__associate Number of value associated.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] original_vertex
         * @param[in] delta_factor Delta factor for delta-stepping.
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            VertexId *original_vertex,
            int   delta_factor = 16,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {
            cudaError_t retval  = cudaSuccess;

            // Check if there are negative weights.
            if (HasNegativeValue(graph->edge_values, graph->edges)) {
                GRError(gunrock::util::GR_UNSUPPORTED_INPUT_DATA,
                        "Contains edges with negative weights. Dijkstra's algorithm"
                        "doesn't support the input data.",
                        __FILE__,
                        __LINE__);
                return retval;
            }
            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            if (retval = labels      .Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = weights     .Allocate(graph->edges,util::DEVICE)) return retval;
            if (retval = delta       .Allocate(1           ,util::DEVICE)) return retval;
            if (retval = visit_lookup.Allocate(graph->nodes,util::DEVICE)) return retval;
            if (retval = sssp_marker .Allocate(graph->nodes,util::DEVICE)) return retval;

            weights.SetPointer(graph->edge_values, graph->edges, util::HOST);
            if (retval = weights.Move(util::HOST, util::DEVICE)) return retval;

            float _delta = EstimatedDelta(graph)*delta_factor;
            // printf("estimated delta:%5f\n", _delta);
            delta.SetPointer(&_delta, util::HOST);
            if (retval = delta.Move(util::HOST, util::DEVICE)) return retval;

            if (MARK_PATHS)
            {
                if (retval = this->preds.Allocate(graph->nodes,util::DEVICE)) return retval;
                if (retval = this->temp_preds.Allocate(graph->nodes, util::DEVICE)) return retval;
            } else {
                if (retval = this->preds.Release()) return retval;
                if (retval = this->temp_preds.Release()) return retval;
            }

            if (num_gpus >1)
            {
                this->value__associate_orgs[0] = labels.GetPointer(util::DEVICE);
                if (MARK_PATHS)
                    this->vertex_associate_orgs[0] = this->preds.GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                //if (retval = temp_marker.Allocate(graph->nodes, util::DEVICE)) return retval;
            }

            //util::cpu_mt::PrintMessage("DataSlice Init() end.");
            return retval;
        } // Init

        /*
         * @brief Estimate delta factor for delta-stepping.
         *
         * @param[in] graph Reference to the graph we process on.
         *
         * \return float Delta factor.
         */
        float EstimatedDelta(const Csr<VertexId, Value, SizeT> &graph) {
            double  avgV = graph.average_edge_value;
            int     avgD = graph.average_degree;
            return avgV * 32 / avgD;
        }

        /**
         * @brief Reset problem function. Must be called prior to each run.
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 @TODO
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            VertexId    src,
            FrontierType frontier_type,
            double queue_sizing,
            double queue_sizing1 = -1)
    {

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, queue_sizing1)) return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }

        // Fillin the initial input_queue for SSSP problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1)
        {
            gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables [0][src];
            tsrc= this->convertion_tables[0][src];
        }
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        Value src_label = 0;
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,
                        &src_label,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        if (MARK_PATHS)
        {
            VertexId src_pred = -1;
            if (retval = util::GRError(cudaMemcpy(
                data_slices[gpu]->preds.GetPointer(util::DEVICE)+tsrc,
                &src_pred,
                sizeof(Value),
                cudaMemcpyHostToDevice),
                "SSSPProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
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
