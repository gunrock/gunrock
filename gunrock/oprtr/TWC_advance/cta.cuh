// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------


/**
 * @file
 * cta.cuh
 *
 * @brief CTA tile-processing abstraction for Forward Edge Map
 */

#pragma once
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/io/load_tile.cuh>
#include <gunrock/util/operators.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/track_utils.cuh>

#include <gunrock/util/scan/soa/cooperative_soa_scan.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/advance_base.cuh>

//TODO: use CUB for SOA scan

namespace gunrock {
namespace oprtr {
namespace edge_map_forward {


/**
 * 1D texture setting for efficiently fetch data from graph_row_offsets and graph_column_indices
 */
/*
template <typename SizeT>
struct RowOffsetTex
{
    static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
};

template <typename SizeT>
    texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;

template <typename VertexId>
struct ColumnIndicesTex
{
    static texture<VertexId, cudaTextureType1D, cudaReadModeElementType> ref;
};

template <typename VertexId>
    texture<VertexId, cudaTextureType1D, cudaReadModeElementType> ColumnIndicesTex<VertexId>::ref;
*/


/**
 * @brief CTA tile-processing abstraction for the vertex mapping operator.
 *
 * @tparam KernelPolicy Kernel policy type for the vertex mapping.
 * @tparam ProblemData Problem data type for the vertex mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 */
template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
struct Cta
{
    /**
     * Typedefs
     */

    typedef typename KernelPolicy::VertexId         VertexId;
    typedef typename KernelPolicy::SizeT            SizeT;
    typedef typename KernelPolicy::Value            Value;

    typedef typename KernelPolicy::SmemStorage      SmemStorage;
    typedef typename KernelPolicy::SoaScanOp        SoaScanOp;
    typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
    typedef typename KernelPolicy::TileTuple        TileTuple;

    typedef typename Problem::DataSlice             DataSlice;
    typedef typename Functor::LabelT                LabelT;

    typedef util::Tuple<
        SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
              SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> RankSoa;

    /**
     * Members
     */

    // Input and output device pointers
    VertexId                *d_keys_in;                      // Incoming frontier
    Value                   *d_values_out;
    VertexId                *d_keys_out;                     // Outgoing frontier
    SizeT                   *d_row_offsets;
    SizeT                   *d_inverse_row_offsets;
    VertexId                *d_column_indices;
    VertexId                *d_inverse_column_indices;
    DataSlice               *d_data_slice;                   // Problem Data

    // Work progress
    bool                     queue_reset;
    VertexId                 queue_index;                // Current frontier queue counter index
    util::CtaWorkProgress<SizeT>
                            &work_progress;             // Atomic queueing counters
    SizeT                   max_out_frontier;           // Maximum size (in elements) of outgoing frontier
    LabelT                  label;                      // Current label of the frontier
    SizeT                   input_queue_length;
    //gunrock::oprtr::advance::TYPE           advance_type;
    bool                    input_inverse_graph;
    //gunrock::oprtr::advance::REDUCE_TYPE    r_type;
    //gunrock::oprtr::advance::REDUCE_OP      r_op;
    Value                  *d_value_to_reduce;
    Value                  *d_reduce_frontier;

    // Operational details for raking grid
    RakingSoaDetails        raking_soa_details;

    // Shared memory for the CTA
    SmemStorage             &smem_storage;

    //texture<SizeT, cudaTextureType1D, cudaReadModeElementType> *ts_rowoffset;
    //texture<VertexId, cudaTextureType1D, cudaReadModeElementType> *ts_columnindices;


    /**
     * @brief Tile of incoming frontier to process
     *
     * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
     * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
     */
    template<int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
    struct Tile
    {
        /**
         * Typedefs and Constants
         */

        enum {
            LOADS_PER_TILE      = 1 << LOG_LOADS_PER_TILE,
            LOAD_VEC_SIZE       = 1 << LOG_LOAD_VEC_SIZE
        };

        typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;

        /**
         * Members
         */

        // Dequeued vertex ids
        VertexId                vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

        SizeT                   row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT                   row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

        // Global scatter offsets. Coarse for CTA/warp-based scatters, fine for scan-based scatters
        SizeT                   fine_count;
        SizeT                   coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT                   fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

        // Progress for scan-based forward edge map gather offsets
        SizeT                   row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT                   progress;

        /**
         * @brief Iterate over vertex ids in tile.
         */
        template <int LOAD, int VEC, int dummy = 0>
        struct Iterate
        {
            /**
             * @brief Tile data initialization
             */
            template <typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile)
            {
                tile->row_length[LOAD][VEC] = 0;
                tile->row_progress[LOAD][VEC] = 0;

                Iterate<LOAD, VEC + 1>::Init(tile);
            }

            /**
             * @brief Inspect the neighbor list size of each node in the frontier,
             *        prepare for neighbor list expansion.
             * @tparam Cta CTA tile-processing abstraction type
             * @tparam Tile Tile structure type
             * @param[in] cta Pointer to CTA object
             * @param[in] tile Pointer to Tile object
             */
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
            {
                if (tile->vertex_id[LOAD][VEC] != -1) {

                    // Translate vertex-id into local gpu row-id (currently stride of num_gpu)
                    VertexId row_id = tile->vertex_id[LOAD][VEC]; // / cta->num_gpus;
                    // Load neighbor row range from d_row_offsets
                    Vec2SizeT   row_range;
                    SizeT       row_id1;
                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                    {
                        //row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id);
                        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                            row_range.x,
                            cta->d_row_offsets + row_id);
                        //row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id + 1);
                        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                            row_range.y,
                            cta->d_row_offsets + row_id+1);
                    }

                    if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                        ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                    {
                        row_id1 = (cta->input_inverse_graph)
                            ? cta -> d_inverse_column_indices[row_id]
                            : cta -> d_column_indices[row_id];
                        //row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id1);
                        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                            row_range.x,
                            cta->d_row_offsets + row_id1);
                        //row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id1+1);
                        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                            row_range.y,
                            cta->d_row_offsets + row_id1+1);
                    }

                    // compute row offset and length
                    tile->row_offset[LOAD][VEC] = row_range.x;
                    tile->row_length[LOAD][VEC] = row_range.y - row_range.x;

                }

                tile->fine_row_rank[LOAD][VEC]
                    = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD)
                    ? tile->row_length[LOAD][VEC] : 0;

                tile->coarse_row_rank[LOAD][VEC]
                    = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD)
                    ? 0 : tile->row_length[LOAD][VEC];

                Iterate<LOAD, VEC + 1>::Inspect(cta, tile);

            } // end of Inspect

            /*template <typename Cta, typename Tile>
            static __device__ __forceinline__ void ProcessNeighbor(
                Cta* cta,
                Tile* tile,
                SizeT neighbor_offset,
                VertexId pred_id,
                SizeT edge_id,
                SizeT output_offset)
            {
                VertexId neighbor_id;
                VertexId vertex_out;
                Value value_reduced;

                // Gather
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    neighbor_id,
                    cta->d_column_indices + neighbor_offset);

                // Users can insert a functor call here ProblemData::Apply(pred_id, neighbor_id) (done)
                // if Cond(neighbor_id) returns true
                // if Cond(neighbor_id) returns false or Apply returns false
                // set neighbor_id to -1 for invalid
                if (Functor::CondEdge(pred_id, neighbor_id, cta->problem,
                    neighbor_offset, edge_id))
                {
                    Functor::ApplyEdge(pred_id, neighbor_id, cta->problem,
                        neighbor_offset, edge_id);
                    if (cta->advance_type == gunrock::oprtr::advance::V2V) {
                        vertex_out = neighbor_id;
                    } else if (cta->advance_type == gunrock::oprtr::advance::V2E
                            || cta->advance_type == gunrock::oprtr::advance::E2E)
                    {
                        // TODO: fix this when SizeT and VertexID is not the
                        // same type, othervise will have potential overflow
                        // when neighbor_offset is larger than Max_Value<SizeT>
                        vertex_out = neighbor_offset;
                    }

                    if (cta->d_value_to_reduce != NULL)
                    {
                        if (cta->r_type == gunrock::oprtr::advance::VERTEX)
                        {
                            value_reduced = cta->d_value_to_reduce[neighbor_id];
                        } else if (cta->r_type == gunrock::oprtr::advance::EDGE)
                        {
                            value_reduced = cta->d_value_to_reduce[neighbor_offset];
                        }
                    } else if (cta->r_type != gunrock::oprtr::advance::EMPTY)
                    {
                        // use user-specified function to generate value to reduce
                    }
                }
                else {
                    vertex_out = -1;
                    if (cta->d_value_to_reduce != NULL)
                    {
                        switch (cta->r_op)
                        {
                        case gunrock::oprtr::advance::PLUS :
                            value_reduced = 0;
                            break;
                        case gunrock::oprtr::advance::MULTIPLIES :
                            value_reduced = 1;
                            break;
                        case gunrock::oprtr::advance::MAXIMUM :
                            // TODO: change to adapt various Value type
                            value_reduced = INT_MIN;
                            break;
                        case gunrock::oprtr::advance::MINIMUM :
                            // TODO: change to adapt various Value type
                            value_reduced = INT_MAX;
                            break;
                        case gunrock::oprtr::advance::BIT_OR :
                            value_reduced = 0;
                            break;
                        case gunrock::oprtr::advance::BIT_AND :
                            // TODO: change to adapt various Value type
                            value_reduced = 0xffffffff;
                            break;
                        case gunrock::oprtr::advance::BIT_XOR :
                            value_reduced = 0;
                            break;
                        default:
                            value_reduced = 0;
                            break;
                        }
                    }
                }

                if (cta -> d_out != NULL)
                {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        vertex_out,
                        cta->d_out + output_offset);
                    //util::Store_d_out<VertexId, SizeT, ProblemData>(
                    //    neighbor_id, cta->d_out, 1,
                    //    cta->smem_storage.state.coarse_enqueue_offset,
                    //    coop_rank, cta->problem, cta->queue_index);
                }
                if (ProblemData::ENABLE_IDEMPOTENCE &&
                    ProblemData::MARK_PREDECESSORS &&
                    cta->d_value_out != NULL)
                {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        pred_id,
                        cta->d_value_out + output_offset);
                }
                if (cta->d_value_to_reduce != NULL)
                {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        value_reduced,
                        cta->d_reduce_frontier + output_offset);
                }
            }*/

            /**
             * @brief Expand the node's neighbor list using the whole CTA.
             * @tparam Cta CTA tile-processing abstraction type
             * @tparam Tile Tile structure type
             * @param[in] cta Pointer to CTA object
             * @param[in] tile Pointer to Tile object
             */
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void CtaExpand(Cta *cta, Tile *tile)
            {
                // CTA-based expansion/loading
                while(true)
                {

                    //All threads in block vie for the control of the block
                    if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
                        cta->smem_storage.state.cta_comm = threadIdx.x;
                    }

                    __syncthreads();

                    // Check
                    int owner = cta->smem_storage.state.cta_comm;
                    if (owner == KernelPolicy::THREADS) {
                        // All threads in the block has less neighbor number for CTA Expand
                        break;
                    }

                    __syncthreads();

                    if (owner == threadIdx.x) {
                        // Got control of the CTA: command it
                        cta->smem_storage.state.warp_comm[0][0]
                            = tile -> row_offset[LOAD][VEC];      // start
                        cta->smem_storage.state.warp_comm[0][1]
                            = tile -> coarse_row_rank[LOAD][VEC]; // queue rank
                        cta->smem_storage.state.warp_comm[0][2]
                            = tile -> row_offset[LOAD][VEC]
                            + tile -> row_length[LOAD][VEC];      // oob
                        if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                            ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                        {
                            cta -> smem_storage.state.warp_comm[0][3]
                                = tile -> vertex_id[LOAD][VEC];
                        }
                        if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                            ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                        {
                            cta -> smem_storage.state.warp_comm[0][3]
                                = cta -> input_inverse_graph
                                ? cta -> d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                                : cta -> d_column_indices[tile->vertex_id[LOAD][VEC]];
                        }
                        cta->smem_storage.state.warp_comm[0][4] = tile->vertex_id[LOAD][VEC];

                        // Unset row length
                        tile->row_length[LOAD][VEC] = 0;

                        // Unset my command
                        cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;
                        // So that we won't repeatedly expand this node
                    }
                    __syncthreads();

                    // Read commands
                    SizeT   coop_offset     = cta->smem_storage.state.warp_comm[0][0];
                    SizeT   coop_rank       = cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
                    SizeT   coop_oob        = cta->smem_storage.state.warp_comm[0][2];

                    VertexId pred_id;
                    VertexId input_item = cta->smem_storage.state.warp_comm[0][4];
                    if (Problem::MARK_PREDECESSORS)
                        pred_id = cta->smem_storage.state.warp_comm[0][3];
                    else
                        pred_id = util::InvalidValue<VertexId>();//cta->label;

                    //__syncthreads();

                    while (coop_offset + threadIdx.x < coop_oob)
                    {
                        // Gather
                        VertexId neighbor_id;
                        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                            neighbor_id,
                            cta->d_column_indices + coop_offset + threadIdx.x);

                        ProcessNeighbor
                            <KernelPolicy, Problem, Functor,
                            ADVANCE_TYPE, R_TYPE, R_OP> (
                            pred_id,
                            neighbor_id,
                            cta -> d_data_slice,
                            (SizeT)(coop_offset + threadIdx.x),
                            util::InvalidValue<SizeT>(), // input_pos
                            input_item,
                            cta -> smem_storage.state.coarse_enqueue_offset + coop_rank,
                            cta -> label,
                            cta -> d_keys_out,
                            cta -> d_values_out,
                            cta -> d_value_to_reduce,
                            cta -> d_reduce_frontier);
                        //ProcessNeighbor(
                        //    cta, tile,
                        //    coop_offset + threadIdx.x,
                        //    pred_id, edge_id,
                        //    cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                        coop_offset += KernelPolicy::THREADS;
                        coop_rank += KernelPolicy::THREADS;
                    }
                } // end of while (true)
                __syncthreads();

                // Next vector element
                Iterate<LOAD, VEC + 1>::CtaExpand(cta, tile);
            } // end of CtaExpand

            /**
             * @brief Expand the node's neighbor list using a warp. (Currently disabled in the enactor)
             * @tparam Cta CTA tile-processing abstraction type
             * @tparam Tile Tile structure type
             * @param[in] cta Pointer to CTA object
             * @param[in] tile Pointer to Tile object
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile)
            {
                if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD)
                {
                    // Warp-based expansion/loading
                    int warp_id = threadIdx.x >> GR_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                    int lane_id = util::LaneId();

                    while (::_any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD))
                    {
                        if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)
                        {
                            // All threads inside one warp vie for control of the warp
                            cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                        }

                        if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0])
                        {
                            // Got control of the warp
                            cta -> smem_storage.state.warp_comm[warp_id][0]
                                = tile -> row_offset[LOAD][VEC];       // start
                            cta -> smem_storage.state.warp_comm[warp_id][1]
                                = tile -> coarse_row_rank[LOAD][VEC];  // queue rank
                            cta -> smem_storage.state.warp_comm[warp_id][2]
                                = tile -> row_offset[LOAD][VEC]
                                + tile -> row_length[LOAD][VEC];      // oob
                            if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                                ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                            {
                                cta -> smem_storage.state.warp_comm[warp_id][3]
                                    = tile->vertex_id[LOAD][VEC];
                            }
                            if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                                ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                            {
                                cta -> smem_storage.state.warp_comm[warp_id][3]
                                    = cta -> input_inverse_graph
                                    ? cta -> d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                                    : cta -> d_column_indices[tile->vertex_id[LOAD][VEC]];
                            }
                            cta -> smem_storage.state.warp_comm[warp_id][4]
                                = tile->vertex_id[LOAD][VEC];
                            // Unset row length
                            tile->row_length[LOAD][VEC] = 0; // So that we won't repeatedly expand this node
                        }

                        SizeT coop_offset   = cta->smem_storage.state.warp_comm[warp_id][0];
                        SizeT coop_rank     = cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
                        SizeT coop_oob      = cta->smem_storage.state.warp_comm[warp_id][2];

                        VertexId pred_id;
                        VertexId input_item = cta->smem_storage.state.warp_comm[warp_id][4];
                        if (Problem::MARK_PREDECESSORS)
                            pred_id = cta->smem_storage.state.warp_comm[warp_id][3];
                        else
                            pred_id = util::InvalidValue<VertexId>();//cta->label;

                        while (coop_offset + lane_id < coop_oob)
                        {
                            VertexId neighbor_id;
                            util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                                neighbor_id,
                                cta->d_column_indices + coop_offset + lane_id);

                            ProcessNeighbor
                                <KernelPolicy, Problem, Functor,
                                ADVANCE_TYPE, R_TYPE, R_OP> (
                                pred_id,
                                neighbor_id,
                                cta -> d_data_slice,
                                coop_offset + lane_id,
                                util::InvalidValue<SizeT>(), // input_pos
                                input_item,
                                cta -> smem_storage.state.coarse_enqueue_offset + coop_rank,
                                cta -> label,
                                cta -> d_keys_out,
                                cta -> d_values_out,
                                cta -> d_value_to_reduce,
                                cta -> d_reduce_frontier);
                            //ProcessNeighbor(
                            //    cta, tile,
                            //    coop_offset + lane_id,
                            //    pred_id, edge_id,
                            //    cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

                            coop_offset += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                            coop_rank += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                        }
                    }

                    // Next vector element
                    Iterate<LOAD, VEC + 1>::WarpExpand(cta, tile);
                }
            } // end of WarpExpand

            /**
             * @brief Expand the node's neighbor list using a single thread (Scan).
             * @tparam Cta CTA tile-processing abstraction type
             * @tparam Tile Tile structure type
             * @param[in] cta Pointer to CTA object
             * @param[in] tile Pointer to Tile object
             */
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile)
            {
                //Expand the neighbor list into scratch space
                SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC]
                    + tile->row_progress[LOAD][VEC] - tile->progress;

                while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
                        (scratch_offset < SmemStorage::GATHER_ELEMENTS))
                {
                    // Put gather offset into scratch space
                    cta -> smem_storage.gather_offsets[scratch_offset]
                        = tile -> row_offset  [LOAD][VEC]
                        + tile -> row_progress[LOAD][VEC];
                    cta -> smem_storage.gather_edges  [scratch_offset]
                        = tile -> vertex_id   [LOAD][VEC];
                    if (Problem::MARK_PREDECESSORS)
                    {
                        if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                            ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
                        {
                            cta -> smem_storage.gather_predecessors[scratch_offset]
                                = cta -> input_inverse_graph
                                ? cta -> d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                                : cta -> d_column_indices[tile->vertex_id[LOAD][VEC]];

                            cta -> smem_storage.gather_edges[scratch_offset]
                                = tile -> vertex_id[LOAD][VEC];
                        }
                        if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
                            ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
                            cta -> smem_storage.gather_predecessors[scratch_offset]
                                = tile -> vertex_id[LOAD][VEC];
                    }

                    tile -> row_progress[LOAD][VEC]++;
                    scratch_offset++;
                }

                // Next vector element
                Iterate<LOAD, VEC + 1>::ThreadExpand(cta, tile);
            }
        }; // end of struct Iterate

        /**
         * Iterate next load
         */
        template <int LOAD, int dummy>
        struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
        {
            // Init
            template <typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile)
            {
                Iterate<LOAD + 1, 0>::Init(tile);
            }

            // Inspect
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::Inspect(cta, tile);
            }

            // CTA Expand
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void CtaExpand(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::CtaExpand(cta, tile);
            }

            // Warp Expand
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::WarpExpand(cta, tile);
            }

            // Single Thread Expand
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::ThreadExpand(cta, tile);
            }
        };

        /**
         * Terminate Iterate
         */
        template <int dummy>
        struct Iterate<LOADS_PER_TILE, 0, dummy>
        {
            // Init
            template <typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile) {}

            // Inspect
            template <typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

            // CtaExpand
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void CtaExpand(Cta *cta, Tile *tile) {}

            // WarpExpand
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile) {}

            // SingleThreadExpand
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile) {}
        };

        //Iterate Interface

        // Constructor
        __device__ __forceinline__ Tile()
        {
            Iterate<0, 0>::Init(this);
        }

        // Inspect dequeued nodes
        template <typename Cta>
        __device__ __forceinline__ void Inspect(Cta *cta)
        {
            Iterate<0, 0>::Inspect(cta, this);
        }

        // CTA Expand
        template <typename Cta>
        __device__ __forceinline__ void CtaExpand(Cta *cta)
        {
            Iterate<0, 0>::CtaExpand(cta, this);
        }

        // Warp Expand
        template <typename Cta>
        __device__ __forceinline__ void WarpExpand(Cta *cta)
        {
            Iterate<0, 0>::WarpExpand(cta, this);
        }

        // Single Thread Expand
        template <typename Cta>
        __device__ __forceinline__ void ThreadExpand(Cta *cta)
        {
            Iterate<0, 0>::ThreadExpand(cta, this);
        }

    }; //end of struct Tile

    // Methods

    /**
     * @brief CTA default constructor
     */
    __device__ __forceinline__ Cta(
        bool                          queue_reset,
        VertexId                      queue_index,
        LabelT                        label,
        SizeT                        *d_row_offsets,
        SizeT                        *d_inverse_row_offsets,
        VertexId                     *d_column_indices,
        VertexId                     *d_inverse_column_indices,
        VertexId                     *d_keys_in,
        VertexId                     *d_keys_out,
        Value                        *d_values_out,
        DataSlice                    *d_data_slice,
        SizeT                         input_queue_length,
        SizeT                         max_in_frontier,
        SizeT                         max_out_frontier,
        util::CtaWorkProgress<SizeT> &work_progress,
        SmemStorage                  &smem_storage,
        //gunrock::oprtr::advance::TYPE ADVANCE_TYPE,
        bool                          input_inverse_graph,
        //gunrock::oprtr::advance::REDUCE_TYPE    R_TYPE,
        //gunrock::oprtr::advance::REDUCE_OP      R_OP,
        Value                        *d_value_to_reduce,
        Value                        *d_reduce_frontier) :

        queue_reset             (queue_reset),
        queue_index             (queue_index),
        label                   (label),
        d_row_offsets           (d_row_offsets),
        d_inverse_row_offsets   (d_inverse_row_offsets),
        d_column_indices        (d_column_indices),
        d_inverse_column_indices(d_inverse_column_indices),
        d_keys_in               (d_keys_in),
        d_keys_out              (d_keys_out),
        d_values_out            (d_values_out),
        d_data_slice            (d_data_slice),
        input_queue_length      (input_queue_length),
        max_out_frontier        (max_out_frontier),
        work_progress           (work_progress),
        smem_storage            (smem_storage),
        input_inverse_graph           (input_inverse_graph),
        d_value_to_reduce       (d_value_to_reduce),
        d_reduce_frontier       (d_reduce_frontier),
        raking_soa_details(
            typename RakingSoaDetails::GridStorageSoa(
                smem_storage.coarse_raking_elements,
                smem_storage.fine_raking_elements),
            typename RakingSoaDetails::WarpscanSoa(
                smem_storage.state.coarse_warpscan,
                smem_storage.state.fine_warpscan),
            TileTuple(0,0))
        //advance_type(ADVANCE_TYPE),
        //r_type(R_TYPE),
        //r_op(R_OP),
    {
        if (threadIdx.x == 0) {
            smem_storage.state.cta_comm = KernelPolicy::THREADS;
            smem_storage.state.overflowed = false;
        }
    }

    /**
     * @brief Process a single, full tile.
     *
     * @param[in] cta_offset Offset within the CTA where we want to start the tile processing.
     * @param[in] guarded_elements The guarded elements to prevent the out-of-bound visit.
     */
    __device__ __forceinline__ void ProcessTile(
        SizeT cta_offset,
        SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
    {
        Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

        // Load tile
        util::io::LoadTile<
            KernelPolicy::LOG_LOADS_PER_TILE,
            KernelPolicy::LOG_LOAD_VEC_SIZE,
            KernelPolicy::THREADS,
            Problem::QUEUE_READ_MODIFIER,
            false>::LoadValid(
                    tile.vertex_id,
                    d_keys_in,
                    cta_offset,
                    guarded_elements,
                    (VertexId) -1);

        // Inspect dequeued nodes, updating label and obtaining
        // edge-list details
        tile.Inspect(this);

        // CooperativeSoaTileScan, put result in totals (done)
        SoaScanOp scan_op;
        TileTuple totals;
        gunrock::util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
                totals,
                raking_soa_details,
                RankSoa(tile.coarse_row_rank, tile.fine_row_rank),
                scan_op);

        SizeT coarse_count = totals.t0;
        tile.fine_count = totals.t1;

        // Set input queue length and check for overflow
        if (threadIdx.x == 0)
        {
            SizeT enqueue_amt       = coarse_count + tile.fine_count;
            SizeT enqueue_offset    = work_progress.Enqueue(enqueue_amt, queue_index + 1);

            smem_storage.state.coarse_enqueue_offset = enqueue_offset;
            smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;

            // Check for queue overflow due to redundant expansion
            if (enqueue_offset + enqueue_amt > max_out_frontier)
            {
                smem_storage.state.overflowed = true;
                work_progress.SetOverflow();
            }
        }

        // Protect overflowed flag
        __syncthreads();

        // Quit if overflow
        if (smem_storage.state.overflowed) {
            util::ThreadExit();
        }

        if (coarse_count > 0)
        {
            // Enqueue valid edge lists into outgoing queue by CTA
            tile.CtaExpand(this);

            // Enqueue valid edge lists into outgoing queue by Warp
            tile.WarpExpand(this);
        }

        // Enqueue the adjacency lists of unvisited node-IDs by repeatedly
        // gathering edges into scratch space, and then
        // having the entire CTA copy the scratch pool into the outgoing
        // frontier queue.
        //
        tile.progress = 0;
        while (tile.progress < tile.fine_count) {
            // Fill the scratch space with gather-offsets for neighbor-lists
            tile.ThreadExpand(this);

            __syncthreads();

            // Copy scratch space into queue
            SizeT scratch_remainder = GR_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

            for (SizeT scratch_offset = threadIdx.x;
                    scratch_offset < scratch_remainder;
                    scratch_offset += KernelPolicy::THREADS)
            {
                // Gather a neighbor
                VertexId neighbor_id;
                SizeT    edge_id = smem_storage.gather_offsets[scratch_offset];
                //neighbor_id = tex1Dfetch(ts_columnindices[0], smem_storage.gather_offsets[scratch_offset]);
                util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
                        neighbor_id,
                        d_column_indices + edge_id);// smem_storage.gather_offsets[scratch_offset]);
                VertexId predecessor_id;
                if (Problem::MARK_PREDECESSORS)
                    predecessor_id = smem_storage.gather_predecessors[scratch_offset];
                else
                    predecessor_id = util::InvalidValue<VertexId>();//label;

                // if Cond(neighbor_id) returns true
                // if Cond(neighbor_id) returns false or Apply returns false
                // set neighbor_id to -1 for invalid
                VertexId input_item = smem_storage.gather_edges[scratch_offset];

                ProcessNeighbor
                    <KernelPolicy, Problem, Functor,
                    ADVANCE_TYPE, R_TYPE, R_OP> (
                    predecessor_id,
                    neighbor_id,
                    d_data_slice,
                    edge_id,
                    util::InvalidValue<SizeT>(), // input_pos
                    input_item,
                    smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset,
                    label,
                    d_keys_out,
                    d_values_out,
                    d_value_to_reduce,
                    d_reduce_frontier);

                /*if (Functor::CondEdge(predecessor_id, neighbor_id, problem, smem_storage.gather_offsets[scratch_offset], edge_id)) {
                    Functor::ApplyEdge(predecessor_id, neighbor_id, problem, smem_storage.gather_offsets[scratch_offset], edge_id);
                    if (advance_type == gunrock::oprtr::advance::V2E || advance_type == gunrock::oprtr::advance::E2E)
                        neighbor_id = smem_storage.gather_offsets[scratch_offset];

                    if (d_value_to_reduce != NULL) {
                        if (r_type == gunrock::oprtr::advance::VERTEX) {
                            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                    d_value_to_reduce[neighbor_id],
                                    d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                        } else if (r_type == gunrock::oprtr::advance::EDGE) {
                            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                    d_value_to_reduce[smem_storage.gather_offsets[scratch_offset]],
                                    d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                        }
                    } else if (r_type != gunrock::oprtr::advance::EMPTY) {
                        // use user-specified function to generate value to reduce
                    }
                }
                else {
                    neighbor_id = -1;
                    if (d_value_to_reduce != NULL) {
                        switch (r_op) {
                            case gunrock::oprtr::advance::PLUS :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)0,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::MULTIPLIES :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)1,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::MAXIMUM :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)INT_MIN,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::MINIMUM :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)INT_MAX,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::BIT_OR :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)0,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::BIT_AND :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)0xffffffff,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            case gunrock::oprtr::advance::BIT_XOR :
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)0,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                            default:
                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                        (Value)0,
                                        d_reduce_frontier + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                                break;
                        }
                    }
                }
                // Scatter into out_queue
                if (d_out != NULL) {
                     //util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                     //       neighbor_id,
                     //       d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                     util::Store_d_out<VertexId, SizeT, ProblemData>(
                        neighbor_id, d_out, 9, smem_storage.state.fine_enqueue_offset,
                        tile.progress + scratch_offset, problem, queue_index);
                }

                if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && d_value_out != NULL) {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                            predecessor_id,
                            d_value_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                }*/
            }

            tile.progress += SmemStorage::GATHER_ELEMENTS;

            __syncthreads();
        }
    }
}; // struct cta

} //namespace edge_map_forward
} //namespace oprtr
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
