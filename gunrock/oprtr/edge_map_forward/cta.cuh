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

#include <gunrock/util/scan/soa/cooperative_soa_scan.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>


//TODO: use CUB for SOA scan

namespace gunrock {
namespace oprtr {
namespace edge_map_forward {


    /**
     * 1D texture setting for efficiently fetch data from graph_row_offsets and graph_column_indices
     */
/*    template <typename SizeT>
        struct RowOffsetTex
        {
            static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
        };
    template <typename SizeT>
        texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;
*/
    /*template <typename VertexId>
      struct ColumnIndicesTex
      {
      static texture<VertexId, cudaTextureType1D, cudaReadModeElementType> ref;
      };
      template <typename VertexId>
      texture<VertexId, cudaTextureType1D, cudaReadModeElementType> ColumnIndicesTex<VertexId>::ref;*/


    /**
     * @brief CTA tile-processing abstraction for the vertex mapping operator.
     *
     * @tparam KernelPolicy Kernel policy type for the vertex mapping.
     * @tparam ProblemData Problem data type for the vertex mapping.
     * @tparam Functor Functor type for the specific problem type.
     *
     */
    template <typename KernelPolicy, typename ProblemData, typename Functor>
        struct Cta
        {

            /**
             * Typedefs
             */

            typedef typename KernelPolicy::VertexId         VertexId;
            typedef typename KernelPolicy::SizeT            SizeT;

            typedef typename KernelPolicy::SmemStorage      SmemStorage;
            typedef typename KernelPolicy::SoaScanOp        SoaScanOp;
            typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
            typedef typename KernelPolicy::TileTuple        TileTuple;

            typedef typename ProblemData::DataSlice         DataSlice;

            typedef util::Tuple<
                SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
                      SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> RankSoa;

            /**
             * Members
             */

            // Input and output device pointers
            VertexId                *d_in;                      // Incoming frontier
            VertexId                *d_pred_out;                 // Incoming predecessor frontier
            VertexId                *d_out;                     // Outgoing frontier
            SizeT                   *d_row_offsets;
            VertexId                *d_column_indices;
            VertexId                *d_inverse_column_indices;
            DataSlice               *problem;                   // Problem Data

            // Work progress
            VertexId                queue_index;                // Current frontier queue counter index
            util::CtaWorkProgress   &work_progress;             // Atomic queueing counters
            SizeT                   max_out_frontier;           // Maximum size (in elements) of outgoing frontier
            //int                     num_gpus;                   // Number of GPUs
            int                     label;                      // Current label of the frontier
            gunrock::oprtr::advance::TYPE           advance_type;
            bool                    inverse_graph;

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
                                        if (cta->advance_type == gunrock::oprtr::advance::V2V || cta->advance_type == gunrock::oprtr::advance::V2E) {
                                            //row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                row_range.x,
                                                cta->d_row_offsets + row_id);
                                            //row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id + 1);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                row_range.y,
                                                cta->d_row_offsets + row_id+1);
                                        }

                                        if (cta->advance_type == gunrock::oprtr::advance::E2V || cta->advance_type == gunrock::oprtr::advance::E2E) {
                                            row_id1 = (cta->inverse_graph) ? cta->d_inverse_column_indices[row_id] : cta->d_column_indices[row_id];
                                            //row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id1);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                row_range.x,
                                                cta->d_row_offsets + row_id1);
                                            //row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id1+1);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                row_range.y,
                                                cta->d_row_offsets + row_id1+1);
                                        }

                                        // compute row offset and length
                                        tile->row_offset[LOAD][VEC] = row_range.x;
                                        tile->row_length[LOAD][VEC] = row_range.y - row_range.x;

                                    }

                                    tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
                                        tile->row_length[LOAD][VEC] : 0;

                                    tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
                                        0 : tile->row_length[LOAD][VEC];

                                    Iterate<LOAD, VEC + 1>::Inspect(cta, tile);

                                }

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
                                    while(true) {

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
                                            cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];                                  // start
                                            cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];                             // queue rank
                                            cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];    // oob
                                            if (cta->advance_type == gunrock::oprtr::advance::V2V || cta->advance_type == gunrock::oprtr::advance::V2E) {
                                                cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];
                                                cta->smem_storage.state.warp_comm[0][4] = 0;
                                            }
                                            if (cta->advance_type == gunrock::oprtr::advance::E2V || cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                cta->smem_storage.state.warp_comm[0][3] = cta->inverse_graph ? cta->d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                                                                                                        : cta->d_column_indices[tile->vertex_id[LOAD][VEC]];
                                                cta->smem_storage.state.warp_comm[0][4] = tile->vertex_id[LOAD][VEC];
                                            }

                                            // Unset row length
                                            tile->row_length[LOAD][VEC] = 0;

                                            // Unset my command
                                            cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;   // So that we won't repeatedly expand this node
                                        }
                                        __syncthreads();

                                        // Read commands
                                        SizeT   coop_offset     = cta->smem_storage.state.warp_comm[0][0];
                                        SizeT   coop_rank       = cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
                                        SizeT   coop_oob        = cta->smem_storage.state.warp_comm[0][2];

                                        VertexId pred_id;
                                        VertexId edge_id = cta->smem_storage.state.warp_comm[0][4];
                                        if (ProblemData::MARK_PREDECESSORS)
                                            pred_id = cta->smem_storage.state.warp_comm[0][3];
                                        else
                                            pred_id = cta->label;

                                        VertexId neighbor_id;

                                        while (coop_offset + KernelPolicy::THREADS < coop_oob) {

                                            // Gather
                                            //neighbor_id = tex1Dfetch(ts_columnindices[0], coop_offset+threadIdx.x);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                    neighbor_id,
                                                    cta->d_column_indices + coop_offset+threadIdx.x);

                                            // Users can insert a functor call here ProblemData::Apply(pred_id, neighbor_id) (done)
                                            // if Cond(neighbor_id) returns true
                                            // if Cond(neighbor_id) returns false or Apply returns false
                                            // set neighbor_id to -1 for invalid
                                            if (Functor::CondEdge(pred_id, neighbor_id, cta->problem, coop_offset+threadIdx.x, edge_id)) {
                                                Functor::ApplyEdge(pred_id, neighbor_id, cta->problem, coop_offset+threadIdx.x, edge_id);
                                                if (cta->advance_type == gunrock::oprtr::advance::V2V) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            neighbor_id,
                                                            cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank); 
                                                } else if (cta->advance_type == gunrock::oprtr::advance::V2E
                                                         ||cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            (VertexId)(coop_offset+threadIdx.x),
                                                            cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                }
                                                if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && cta->d_pred_out != NULL) {
                                                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                                pred_id,
                                                                cta->d_pred_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                    }
                                            }
                                            else {
                                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                    -1,
                                                    cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                            }

                                            coop_offset += KernelPolicy::THREADS;
                                            coop_rank += KernelPolicy::THREADS;
                                        }

                                        if (coop_offset + threadIdx.x < coop_oob) {

                                            // Gather
                                            //neighbor_id = tex1Dfetch(ts_columnindices[0], coop_offset+threadIdx.x);
                                            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                    neighbor_id,
                                                    cta->d_column_indices + coop_offset+threadIdx.x);

                                            // Users can insert a functor call here ProblemData::Apply(pred_id, neighbor_id)
                                            // if Cond(neighbor_id) returns true
                                            // if Cond(neighbor_id) returns false or Apply returns false
                                            // set neighbor_id to -1 for invalid                                    
                                            if (Functor::CondEdge(pred_id, neighbor_id, cta->problem, coop_offset+threadIdx.x, edge_id)) {
                                                Functor::ApplyEdge(pred_id, neighbor_id, cta->problem, coop_offset+threadIdx.x, edge_id);
                                                if (cta->advance_type == gunrock::oprtr::advance::V2V) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            neighbor_id,
                                                            cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank); 
                                                } else if (cta->advance_type == gunrock::oprtr::advance::V2E
                                                         ||cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            (VertexId)(coop_offset+threadIdx.x),
                                                            cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                }
                                                if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && cta->d_pred_out != NULL) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            pred_id,
                                                            cta->d_pred_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                }
                                            }
                                            else {
                                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                    -1,
                                                    cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                            }

                                        }
                                    }

                                    // Next vector element
                                    Iterate<LOAD, VEC + 1>::CtaExpand(cta, tile);
                                }

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
                                    if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD) {
                                        // Warp-based expansion/loading
                                        int warp_id = threadIdx.x >> GR_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                                        int lane_id = util::LaneId();

                                        while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)) {
                                            if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD) {
                                                // All threads inside one warp vie for control of the warp
                                                cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                                            }

                                            if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {
                                                // Got control of the warp
                                                cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];                                    // start
                                                cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];                               // queue rank
                                                cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];      // oob
                                            if (cta->advance_type == gunrock::oprtr::advance::V2V || cta->advance_type == gunrock::oprtr::advance::V2E) {
                                                cta->smem_storage.state.warp_comm[warp_id][3] = tile->vertex_id[LOAD][VEC];
                                                cta->smem_storage.state.warp_comm[warp_id][4] = 0;
                                            }
                                            if (cta->advance_type == gunrock::oprtr::advance::E2V || cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                cta->smem_storage.state.warp_comm[warp_id][3] = cta->inverse_graph ? cta->d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]:
                                                cta->d_column_indices[tile->vertex_id[LOAD][VEC]];

                                                cta->smem_storage.state.warp_comm[warp_id][4] = tile->vertex_id[LOAD][VEC];
                                            }
                                                // Unset row length
                                                tile->row_length[LOAD][VEC] = 0; // So that we won't repeatedly expand this node

                                            }

                                            
                                            SizeT coop_offset   = cta->smem_storage.state.warp_comm[warp_id][0];
                                            SizeT coop_rank     = cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
                                            SizeT coop_oob      = cta->smem_storage.state.warp_comm[warp_id][2];

                                            VertexId pred_id;
                                            VertexId edge_id = cta->smem_storage.state.warp_comm[warp_id][4];
                                            if (ProblemData::MARK_PREDECESSORS)
                                                pred_id = cta->smem_storage.state.warp_comm[warp_id][3];
                                            else
                                                pred_id = cta->label;

                                            VertexId neighbor_id;
                                            while (coop_offset + GR_WARP_THREADS(KernelPolicy::CUDA_ARCH) < coop_oob) {

                                                // Gather
                                                //neighbor_id = tex1Dfetch(ts_columnindices[0], coop_offset+lane_id);
                                                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                        neighbor_id,
                                                        cta->d_column_indices + coop_offset+lane_id);

                                                // Users can insert a functor call here ProblemData::Apply(pred_id, neighbor_id)
                                                // if Cond(neighbor_id) returns true
                                                // if Cond(neighbor_id) returns false or Apply returns false
                                                // set neighbor_id to -1 for invalid 
                                                if (Functor::CondEdge(pred_id, neighbor_id, cta->problem, coop_offset+lane_id, edge_id)) {
                                                    Functor::ApplyEdge(pred_id, neighbor_id, cta->problem, coop_offset+lane_id, edge_id);
                                                    if (cta->advance_type == gunrock::oprtr::advance::V2E
                                                      ||cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                        neighbor_id = coop_offset+lane_id;
                                                    }
                                                }
                                                else
                                                    neighbor_id = -1;

                                                if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && cta->d_pred_out != NULL) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            pred_id,
                                                            cta->d_pred_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                }

                                                // Scatter neighbor
                                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                        neighbor_id,
                                                        cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

                                                coop_offset += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                                                coop_rank += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                                            }

                                            if (coop_offset + lane_id < coop_oob) {
                                                // Gather
                                                //neighbor_id = tex1Dfetch(ts_columnindices[0], coop_offset+lane_id);
                                                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                        neighbor_id,
                                                        cta->d_column_indices + coop_offset+lane_id);

                                                // Users can insert a functor call here ProblemData::Apply(pred_id, neighbor_id)
                                                // if Cond(neighbor_id) returns true
                                                // if Cond(neighbor_id) returns false or Apply returns false
                                                // set neighbor_id to -1 for invalid                                            
                                                if (Functor::CondEdge(pred_id, neighbor_id, cta->problem, coop_offset+lane_id, edge_id)) {
                                                    Functor::ApplyEdge(pred_id, neighbor_id, cta->problem, coop_offset+lane_id, edge_id);
                                                    if (cta->advance_type == gunrock::oprtr::advance::V2E
                                                      ||cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                        neighbor_id = coop_offset+lane_id;
                                                    }
                                                }
                                                else
                                                    neighbor_id = -1;

                                                if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && cta->d_pred_out != NULL) {
                                                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                            pred_id,
                                                            cta->d_pred_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                                }

                                                // Scatter neighbor
                                                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                                        neighbor_id,
                                                        cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                                            }
                                        }

                                        // Next vector element
                                        Iterate<LOAD, VEC + 1>::WarpExpand(cta, tile);
                                    }
                                }

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
                                    SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

                                    while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
                                            (scratch_offset < SmemStorage::GATHER_ELEMENTS))
                                    {
                                        // Put gather offset into scratch space
                                        cta->smem_storage.gather_offsets[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];
                                        if (ProblemData::MARK_PREDECESSORS) {
                                            if (cta->advance_type == gunrock::oprtr::advance::E2V || cta->advance_type == gunrock::oprtr::advance::E2E) {
                                                cta->smem_storage.gather_predecessors[scratch_offset] = cta->inverse_graph ? cta->d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]: cta->d_column_indices[tile->vertex_id[LOAD][VEC]];

                                                cta->smem_storage.gather_edges[scratch_offset] = tile->vertex_id[LOAD][VEC];
                                            }
                                            if (cta->advance_type == gunrock::oprtr::advance::V2V || cta->advance_type == gunrock::oprtr::advance::V2E)
                                                cta->smem_storage.gather_predecessors[scratch_offset] = tile->vertex_id[LOAD][VEC];
                                        }

                                        tile->row_progress[LOAD][VEC]++;
                                        scratch_offset++;
                                    }

                                    // Next vector element
                                    Iterate<LOAD, VEC + 1>::ThreadExpand(cta, tile);
                                }
                        };

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

                };

            // Methods

            /**
             * @brief CTA default constructor
             */
            __device__ __forceinline__ Cta(
                    VertexId                    queue_index,
                    //int                         num_gpus,
                    int                         label,
                    SmemStorage                 &smem_storage,
                    VertexId                    *d_in_queue,
                    VertexId                    *d_pred_out,
                    VertexId                    *d_out_queue,
                    SizeT                       *d_row_offsets,
                    VertexId                    *d_column_indices,
                    VertexId                    *d_inverse_column_indices,
                    DataSlice                   *problem,
                    util::CtaWorkProgress       &work_progress,
                    SizeT                       max_out_frontier,
                    //texture<SizeT, cudaTextureType1D, cudaReadModeElementType> *ts_rowoffset,
                    //texture<VertexId, cudaTextureType1D, cudaReadModeElementType> *ts_columnindices, 
                    gunrock::oprtr::advance::TYPE ADVANCE_TYPE,
                    bool                        inverse_graph) :

                queue_index(queue_index),
                //num_gpus(num_gpus),
                label(label),
                smem_storage(smem_storage),
                raking_soa_details(
                        typename RakingSoaDetails::GridStorageSoa(
                            smem_storage.coarse_raking_elements,
                            smem_storage.fine_raking_elements),
                        typename RakingSoaDetails::WarpscanSoa(
                            smem_storage.state.coarse_warpscan,
                            smem_storage.state.fine_warpscan),
                        TileTuple(0,0)),
                d_in(d_in_queue),
                d_pred_out(d_pred_out),
                d_out(d_out_queue),
                d_row_offsets(d_row_offsets),
                d_column_indices(d_column_indices),
                d_inverse_column_indices(d_inverse_column_indices),
                problem(problem),
                work_progress(work_progress),
                max_out_frontier(max_out_frontier),
                //ts_rowoffset(ts_rowoffset),
                //ts_columnindices(ts_columnindices),
                advance_type(ADVANCE_TYPE),
                inverse_graph(inverse_graph)
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
                    ProblemData::QUEUE_READ_MODIFIER,
                    false>::LoadValid(
                            tile.vertex_id,
                            d_in,
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
                if (threadIdx.x == 0) {

                    SizeT enqueue_amt       = coarse_count + tile.fine_count;
                    SizeT enqueue_offset    = work_progress.Enqueue(enqueue_amt, queue_index + 1);

                    smem_storage.state.coarse_enqueue_offset = enqueue_offset;
                    smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;

                    // Check for queue overflow due to redundant expansion
                    if (enqueue_offset + enqueue_amt > max_out_frontier) {
                        smem_storage.state.overflowed = true;
                        work_progress.SetOverflow<SizeT>();
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
                    int scratch_remainder = GR_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

                    for (int scratch_offset = threadIdx.x;
                            scratch_offset < scratch_remainder;
                            scratch_offset += KernelPolicy::THREADS)
                    {
                        // Gather a neighbor
                        VertexId neighbor_id;
                        //neighbor_id = tex1Dfetch(ts_columnindices[0], smem_storage.gather_offsets[scratch_offset]);
                        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                neighbor_id,
                                d_column_indices + smem_storage.gather_offsets[scratch_offset]);
                        VertexId predecessor_id;
                        if (ProblemData::MARK_PREDECESSORS)
                            predecessor_id = smem_storage.gather_predecessors[scratch_offset];
                        else
                            predecessor_id = label;

                        // if Cond(neighbor_id) returns true
                        // if Cond(neighbor_id) returns false or Apply returns false
                        // set neighbor_id to -1 for invalid
                        VertexId edge_id;
                        if (advance_type == gunrock::oprtr::advance::V2E || advance_type == gunrock::oprtr::advance::V2V)
                            edge_id = 0;
                        if (advance_type == gunrock::oprtr::advance::E2E || advance_type == gunrock::oprtr::advance::E2V) {
                            edge_id = smem_storage.gather_offsets[scratch_offset];
                        }
                        if (Functor::CondEdge(predecessor_id, neighbor_id, problem, smem_storage.gather_offsets[scratch_offset], edge_id)) {
                            Functor::ApplyEdge(predecessor_id, neighbor_id, problem, smem_storage.gather_offsets[scratch_offset], edge_id);
                            if (advance_type == gunrock::oprtr::advance::V2E || advance_type == gunrock::oprtr::advance::E2E)
                                neighbor_id = smem_storage.gather_offsets[scratch_offset];
                        }
                        else
                            neighbor_id = -1;
                        // Scatter into out_queue
                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                neighbor_id,
                                d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);

                        if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && d_pred_out != NULL) {
                            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                    predecessor_id,
                                    d_pred_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
                        }
                    }

                    tile.progress += SmemStorage::GATHER_ELEMENTS;

                    __syncthreads();
                }
            }
        };

} //namespace edge_map_forward
} //namespace oprtr
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
