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
 * @brief CTA tile-processing abstraction for Filter
 */

#pragma once

#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/io/initialize_tile.cuh>
#include <gunrock/util/io/load_tile.cuh>
#include <gunrock/util/io/store_tile.cuh>
#include <gunrock/util/io/scatter_tile.cuh>
#include <gunrock/util/cta_work_distribution.cuh>

#include <gunrock/util/operators.cuh>
#include <gunrock/util/reduction/serial_reduce.cuh>
#include <gunrock/util/reduction/tree_reduce.cuh>

#include <gunrock/util/scan/cooperative_scan.cuh>

namespace gunrock {
namespace oprtr {
namespace filter {

/**
* Templated texture reference for visited mask
*/
template <typename VisitedMask>
struct BitmaskTex
{
   static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename VisitedMask>
texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<VisitedMask>::ref;

/**
 * @brief CTA tile-processing abstraction for the filter operator.
 *
 * @tparam KernelPolicy Kernel policy type for filter.
 * @tparam ProblemData Problem data type for filter.
 * @tparam Functor Functor type for the specific problem type.
 *
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Cta
{
    //---------------------------------------------------------------------
    // Typedefs and Constants
    //---------------------------------------------------------------------

    typedef typename KernelPolicy::VertexId         VertexId;
    typedef typename KernelPolicy::SizeT            SizeT;

    typedef typename KernelPolicy::RakingDetails    RakingDetails;
    typedef typename KernelPolicy::SmemStorage      SmemStorage;

    typedef typename ProblemData::DataSlice         DataSlice;
    typedef typename ProblemData::Value             Value;

    /**
     * Members
     */

    // Input and output device pointers
    VertexId                *d_in;                      // Incoming frontier
    VertexId                *d_pred_in;                 // Incoming predecessor frontier (if any)
    VertexId                *d_out;                     // Outgoing frontier
    DataSlice               *problem;                   // Problem data
    unsigned char           *d_visited_mask;            // Mask for detecting visited status

    // Work progress
    VertexId                iteration;                  // Current graph traversal iteration
    VertexId                queue_index;                // Current frontier queue counter index
    util::CtaWorkProgress   &work_progress;             // Atomic workstealing and queueing counters
    SizeT                   max_out_frontier;           // Maximum size (in elements) of outgoing frontier
    //int                     num_gpus;                   // Number of GPUs

    // Operational details for raking_scan_grid
    RakingDetails           raking_details;
    
    // Shared memory for the CTA
    SmemStorage             &smem_storage;

    // Whether or not to perform bitmask culling (incurs extra latency on small frontiers)
    bool                    bitmask_cull;
    //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *t_bitmask;

    //---------------------------------------------------------------------
    // Helper Structures
    //---------------------------------------------------------------------

    /**
     * @brief Tile of incoming frontier to process
     *
     * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
     * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
     */
    template <
        int LOG_LOADS_PER_TILE,
        int LOG_LOAD_VEC_SIZE>
    struct Tile
    {
        //---------------------------------------------------------------------
        // Typedefs and Constants
        //---------------------------------------------------------------------

        enum {
            LOADS_PER_TILE      = 1 << LOG_LOADS_PER_TILE,
            LOAD_VEC_SIZE       = 1 << LOG_LOAD_VEC_SIZE
        };


        //---------------------------------------------------------------------
        // Members
        //---------------------------------------------------------------------

        // Dequeued element ids
        VertexId    element_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
        VertexId    pred_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

        // Whether or not the corresponding element_id is valid for exploring
        unsigned char   flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

        // Global scatter offsets
        SizeT       ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];


        //---------------------------------------------------------------------
        // Helper Structures
        //---------------------------------------------------------------------

        /**
         * @brief Iterate over element ids in tile.
         */
        template <int LOAD, int VEC, int dummy = 0>
        struct Iterate
        {
            /**
             * @brief Initialize flag values for compact new frontier. If vertex id equals to -1, then discard it in the new frontier.
             *
             * @param[in] tile Pointer to Tile object holds the element ids, ranks and flags.
             */
            static __device__ __forceinline__ void InitFlags(Tile *tile)
            {
                // Initially valid if vertex-id is valid
                tile->flags[LOAD][VEC] = (tile->element_id[LOAD][VEC] == -1) ? 0 : 1;
                tile->ranks[LOAD][VEC] = tile->flags[LOAD][VEC];

                // Next
                Iterate<LOAD, VEC + 1>::InitFlags(tile);
            }

            /**
             * @brief Cull redundant vertices using Bitmask
             *
             */
            static __device__ __forceinline__ void BitmaskCull(
                Cta *cta,
                Tile *tile)
            {
                if (tile->element_id[LOAD][VEC] >= 0) {
                    // Location of mask byte to read
                    SizeT mask_byte_offset = (tile->element_id[LOAD][VEC] & KernelPolicy::ELEMENT_ID_MASK) >> 3;

                    // Bit in mask byte corresponding to current vertex id
                    unsigned char mask_bit = 1 << (tile->element_id[LOAD][VEC] & 7);

                    // Read byte from visited mask in tex
                    unsigned char tex_mask_byte = tex1Dfetch(
                        BitmaskTex<unsigned char>::ref,//cta->t_bitmask[0],
                        mask_byte_offset);

                    if (mask_bit & tex_mask_byte) {
                        // Seen it
                        tile->element_id[LOAD][VEC] = -1;
                    } else {
                        unsigned char mask_byte;
                        //util::io::ModifiedLoad<util::io::ld::cg>::Ld(
                        //    mask_byte, cta->d_visited_mask + mask_byte_offset);
                        mask_byte = cta->d_visited_mask[mask_byte_offset];

                        mask_byte |= tex_mask_byte;

                        if (mask_bit & mask_byte) {
                            // Seen it
                            tile->element_id[LOAD][VEC] = -1;
                        } else {
                            // Update with best effort
                            mask_byte |= mask_bit;
                            util::io::ModifiedStore<util::io::st::cg>::St(
                                mask_byte,
                                cta->d_visited_mask + mask_byte_offset);
                        }
                    }
                }

                // Next
                Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
            }

            /**
             * @brief Set vertex id to -1 if we want to cull this vertex from the outgoing frontier.
             * @param[in] cta
             * @param[in] tile
             *
             */
            static __device__ __forceinline__ void VertexCull(
                Cta *cta,
                Tile *tile)
            {
                if (ProblemData::ENABLE_IDEMPOTENCE && cta->iteration != -1) {
                    if (tile->element_id[LOAD][VEC] >= 0) {
                        VertexId row_id = (tile->element_id[LOAD][VEC]&KernelPolicy::ELEMENT_ID_MASK);///cta->num_gpus;

                        VertexId label;
                        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                                                    label,
                                                    cta->problem->labels + row_id);
                        if (label != -1) {
                            // Seen it
                            tile->element_id[LOAD][VEC] = -1;
                        } else {
                            if (ProblemData::MARK_PREDECESSORS) {
                                if (Functor::CondFilter(row_id, cta->problem))
                                Functor::ApplyFilter(row_id, cta->problem, tile->pred_id[LOAD][VEC]);
                            } else {
                                if (Functor::CondFilter(row_id, cta->problem))
                                Functor::ApplyFilter(row_id, cta->problem, cta->iteration);
                            }
                        }
                    }
                } else {
                    if (tile->element_id[LOAD][VEC] >= 0) {
                        // Row index on our GPU (for multi-gpu, element ids are striped across GPUs)
                        VertexId row_id = (tile->element_id[LOAD][VEC]);// / cta->num_gpus;
                        SizeT node_id = threadIdx.x * LOADS_PER_TILE*LOAD_VEC_SIZE + LOAD*LOAD_VEC_SIZE+VEC;
                        if (Functor::CondFilter(row_id, cta->problem, cta->iteration, node_id)) {
                            // ApplyFilter(row_id)
                            Functor::ApplyFilter(row_id, cta->problem, cta->iteration, node_id);
                        }
                        else tile->element_id[LOAD][VEC] = -1;
                    }
                }

                // Next
                Iterate<LOAD, VEC + 1>::VertexCull(cta, tile);
            }

            /**
             * @brief Cull redundant vertices using history hash table
             *
             */
            static __device__ __forceinline__ void HistoryCull(
                Cta *cta,
                Tile *tile)
            {
                if (tile->element_id[LOAD][VEC] >= 0) {
                    int hash = ((unsigned int)tile->element_id[LOAD][VEC]) % SmemStorage::HISTORY_HASH_ELEMENTS;
                    VertexId retrieved = cta->smem_storage.history[hash];

                    if (retrieved == tile->element_id[LOAD][VEC]) {
                        // Seen it
                        tile->element_id[LOAD][VEC] = -1;
                    } else {
                        // Update it
                        cta->smem_storage.history[hash] = tile->element_id[LOAD][VEC];
                    }
                }

                // Next
                Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
            }

            /**
             * @brief Cull redundant vertices using warp hash table
             *
             */
            static __device__ __forceinline__ void WarpCull(
                Cta *cta,
                Tile *tile)
            {
                if (tile->element_id[LOAD][VEC] >= 0) {
                    int warp_id = threadIdx.x >> 5;
                    int hash    = tile->element_id[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

                    cta->smem_storage.state.vid_hashtable[warp_id][hash] = tile->element_id[LOAD][VEC];
                    VertexId retrieved = cta->smem_storage.state.vid_hashtable[warp_id][hash];

                    if (retrieved == tile->element_id[LOAD][VEC]) {
                        cta->smem_storage.state.vid_hashtable[warp_id][hash] = threadIdx.x;
                        VertexId tid = cta->smem_storage.state.vid_hashtable[warp_id][hash];
                        if (tid != threadIdx.x) {
                            tile->element_id[LOAD][VEC] = -1;
                        }
                    }
                }

                // Next
                Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
            }

        };


        /**
         * Iterate next load
         */
        template <int LOAD, int dummy>
        struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
        {
            // InitFlags
            static __device__ __forceinline__ void InitFlags(Tile *tile)
            {
                Iterate<LOAD + 1, 0>::InitFlags(tile);
            }

            // BitmaskCull
            static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
            {
                Iterate<LOAD+1,0>::BitmaskCull(cta, tile);
            }

            // VertexCull
            static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::VertexCull(cta, tile);
            }

            // WarpCull
            static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
            }

            // HistoryCull
            static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile)
            {
                Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
            }
        };



        /**
         * Terminate iteration
         */
        template <int dummy>
        struct Iterate<LOADS_PER_TILE, 0, dummy>
        {
            // InitFlags
            static __device__ __forceinline__ void InitFlags(Tile *tile) {}

            // BitmaskCull
            static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

            // VertexCull
            static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {}
            
            // HistoryCull
            static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {}
            
            // WarpCull
            static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}

        };


        //---------------------------------------------------------------------
        // Interface
        //---------------------------------------------------------------------

        // Initializer
        __device__ __forceinline__ void InitFlags()
        {
            Iterate<0, 0>::InitFlags(this);
        }

        // Culls vertices based on bitmask
        __device__ __forceinline__ void BitmaskCull(Cta *cta)
        {
            Iterate<0, 0>::BitmaskCull(cta, this);
        }

        // Culls vertices
        __device__ __forceinline__ void VertexCull(Cta *cta)
        {
            Iterate<0, 0>::VertexCull(cta, this);
        }

        // Culls redundant vertices within the warp
        __device__ __forceinline__ void WarpCull(Cta *cta)
        {
            Iterate<0, 0>::WarpCull(cta, this);
        }

        // Culls redundant vertices within recent CTA history
        __device__ __forceinline__ void HistoryCull(Cta *cta)
        {
            Iterate<0, 0>::HistoryCull(cta, this);
        }

    };




    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * @brief CTA type default constructor
     */
    __device__ __forceinline__ Cta(
        VertexId                iteration,
        VertexId                queue_index,
        //int                     num_gpus,
        SmemStorage             &smem_storage,
        VertexId                *d_in,
        VertexId                *d_pred_in,
        VertexId                *d_out,
        DataSlice               *problem,
        unsigned char           *d_visited_mask,
        util::CtaWorkProgress   &work_progress,
        SizeT                   max_out_frontier):
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *t_bitmask):
            iteration(iteration),
            queue_index(queue_index),
            //num_gpus(num_gpus),
            raking_details(
                smem_storage.state.raking_elements,
                smem_storage.state.warpscan,
                0),
            smem_storage(smem_storage),
            d_in(d_in),
            d_pred_in(d_pred_in),
            d_out(d_out),
            problem(problem),
            d_visited_mask(d_visited_mask),
            work_progress(work_progress),
            max_out_frontier(max_out_frontier),
	    //t_bitmask(t_bitmask),
            bitmask_cull(
                (KernelPolicy::END_BITMASK_CULL < 0) ?
                    true :
                    (KernelPolicy::END_BITMASK_CULL == 0) ?
                    false :
                    (iteration < KernelPolicy::END_BITMASK_CULL))
    {
        // Initialize history duplicate-filter
        for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
            smem_storage.history[offset] = -1;
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
        const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
    {
        Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

        // Load tile
        util::io::LoadTile<
            KernelPolicy::LOG_LOADS_PER_TILE,
            KernelPolicy::LOG_LOAD_VEC_SIZE,
            KernelPolicy::THREADS,
            ProblemData::QUEUE_READ_MODIFIER,
            false>::LoadValid(
                tile.element_id,
                d_in,
                cta_offset,
                guarded_elements,
                (VertexId) -1);
        
        if (ProblemData::ENABLE_IDEMPOTENCE && ProblemData::MARK_PREDECESSORS && d_pred_in != NULL) {
            util::io::LoadTile<
            KernelPolicy::LOG_LOADS_PER_TILE,
            KernelPolicy::LOG_LOAD_VEC_SIZE,
            KernelPolicy::THREADS,
            ProblemData::QUEUE_READ_MODIFIER,
            false>::LoadValid(
                tile.pred_id,
                d_pred_in,
                cta_offset,
                guarded_elements);
        }

        if (ProblemData::ENABLE_IDEMPOTENCE && bitmask_cull && d_visited_mask != NULL) {
            tile.BitmaskCull(this);
        }
        tile.VertexCull(this);          // using vertex visitation status (update discovered vertices)
        
        if (ProblemData::ENABLE_IDEMPOTENCE && iteration != -1) {
            tile.HistoryCull(this);
            tile.WarpCull(this);
        }

        // Init valid flags and ranks
        tile.InitFlags();

        // Protect repurposable storage that backs both raking lanes and local cull scratch
        __syncthreads();

        // Scan tile of ranks, using an atomic add to reserve
        // space in the contracted queue, seeding ranks
        // Cooperative Scan (done)
        util::Sum<SizeT> scan_op;
        SizeT new_queue_offset = util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
            raking_details,
            tile.ranks,
            work_progress.GetQueueCounter<SizeT>(queue_index + 1),
            scan_op);
        
        // Check updated queue offset for overflow due to redundant expansion
        if (new_queue_offset >= max_out_frontier) {
            //printf(" new_queue_offset >= max_out_frontier, new_queue_offset = %d, max_out_frontier = %d\n", new_queue_offset, max_out_frontier);
            work_progress.SetOverflow<SizeT>();
            util::ThreadExit();
        }

        // Scatter directly (without first contracting in smem scratch), predicated
        // on flags
        if (d_out != NULL) {
            util::io::ScatterTile<
                KernelPolicy::LOG_LOADS_PER_TILE,
                KernelPolicy::LOG_LOAD_VEC_SIZE,
                KernelPolicy::THREADS,
                ProblemData::QUEUE_WRITE_MODIFIER>::Scatter(
                    d_out,
                    tile.element_id,
                    tile.flags,
                    tile.ranks);
        }
    }
};


} // namespace filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
