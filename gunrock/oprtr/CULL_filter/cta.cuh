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
#include <gunrock/util/vector_types.cuh>
#include <gunrock/util/type_limits.cuh>

namespace gunrock {
namespace oprtr {
namespace CULL {

/**
 * Templated texture reference for visited mask
 */
/*template <typename VisitedMask>
struct BitmaskTex
{
   static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename VisitedMask>
texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType>
BitmaskTex<VisitedMask>::ref;

template <typename LabelT>
struct LabelsTex
{
   static texture<LabelT, cudaTextureType1D, cudaReadModeElementType> labels;
};
template <typename LabelT>
texture<LabelT, cudaTextureType1D, cudaReadModeElementType>
LabelsTex<LabelT>::labels;*/

/**
 * @brief CTA tile-processing abstraction for the filter operator.
 *
 * @tparam KernelPolicy Kernel policy type for filter.
 * @tparam Problem Problem data type for filter.
 * @tparam Functor Functor type for the specific problem type.
 *
 */
template <typename KernelPolicyT, typename FilterOpT>
struct Cta {
  //---------------------------------------------------------------------
  // Typedefs and Constants
  //---------------------------------------------------------------------

  static const OprtrFlag FLAG = KernelPolicyT::FLAG;
  // typedef typename KernelPolicyT::VertexId         VertexId;
  typedef typename KernelPolicyT::InKeyT InKeyT;
  typedef typename KernelPolicyT::OutKeyT OutKeyT;
  typedef typename KernelPolicyT::SizeT SizeT;
  typedef typename KernelPolicyT::ValueT ValueT;
  typedef typename KernelPolicyT::LabelT LabelT;

  typedef typename KernelPolicyT::RakingDetails RakingDetails;
  typedef typename KernelPolicyT::SmemStorage SmemStorage;

  /**
   * Members
   */

  // Input and output device pointers
  const InKeyT *keys_in;    // Incoming frontier
  const ValueT *values_in;  // Incoming values
  OutKeyT *keys_out;        // Outgoing frontier
  LabelT *labels;           // Key labels
  const LabelT label;       // Current label
  // DataSlice     *d_data_slice;  // Problem data
  unsigned char *visited_masks;  // Mask for detecting visited status

  // Work progress
  const SizeT queue_index;  // Current frontier queue counter index
  util::CtaWorkProgress<SizeT>
      &work_progress;  // Atomic workstealing and queueing counters
  // const SizeT          max_out_frontier; // Maximum size (in elements) of
  // outgoing frontier
  FilterOpT filter_op;

  // Operational details for raking_scan_grid
  RakingDetails raking_details;

  // Shared memory for the CTA
  SmemStorage &smem_storage;

  // Whether or not to perform bitmask culling (incurs extra latency on small
  // frontiers)
  bool bitmask_cull;
  // texture<unsigned char, cudaTextureType1D, cudaReadModeElementType>
  // *t_bitmask;

  //---------------------------------------------------------------------
  // Helper Structures
  //---------------------------------------------------------------------

  /**
   * @brief Tile of incoming frontier to process
   *
   * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
   * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
   */
  template <int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
  struct Tile {
    //---------------------------------------------------------------------
    // Typedefs and Constants
    //---------------------------------------------------------------------

    enum {
      LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE,
      LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
    };

    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------

    // Dequeued element ids
    InKeyT element_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
    InKeyT pred_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

    // Whether or not the corresponding element_id is valid for exploring
    unsigned char flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

    // Global scatter offsets
    SizeT ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

    //---------------------------------------------------------------------
    // Helper Structures
    //---------------------------------------------------------------------

    /**
     * @brief Iterate over element ids in tile.
     */
    template <int LOAD, int VEC, int dummy = 0>
    struct Iterate {
      /**
       * @brief Initialize flag values for compact new frontier. If vertex id
       * equals to -1, then discard it in the new frontier.
       *
       * @param[in] tile Pointer to Tile object holds the element ids, ranks and
       * flags.
       */
      static __device__ __forceinline__ void InitFlags(Tile *tile) {
        // Initially valid if vertex-id is valid
        tile->flags[LOAD][VEC] = (tile->element_id[LOAD][VEC] ==
                                  util::PreDefinedValues<InKeyT>::InvalidValue)
                                     ? 0
                                     : 1;
        tile->ranks[LOAD][VEC] = tile->flags[LOAD][VEC];

        // Next
        Iterate<LOAD, VEC + 1>::InitFlags(tile);
      }

      /**
       * @brief Cull redundant vertices using Bitmask
       *
       */
      static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {
        if (tile->element_id[LOAD][VEC] >= 0) {
          // Location of mask byte to read
          SizeT mask_byte_offset =
              (tile->element_id[LOAD][VEC]  //& KernelPolicyT::ELEMENT_ID_MASK
               ) >>
              3;

          // Bit in mask byte corresponding to current vertex id
          unsigned char mask_bit = 1 << (tile->element_id[LOAD][VEC] & 7);

          // Read byte from visited mask in tex
          // unsigned char tex_mask_byte =
          // cta->d_visited_mask[mask_byte_offset];
          unsigned char tex_mask_byte =
              _ldg(cta->visited_mask + mask_byte_offset);

          if (mask_bit & tex_mask_byte) {
            // Seen it
            tile->element_id[LOAD][VEC] =
                util::PreDefinedValues<InKeyT>::InvalidValue;
          } else {
            tex_mask_byte |= mask_bit;
            util::io::ModifiedStore<util::io::st::cg>::St(
                tex_mask_byte,  // mask_byte,
                cta->visited_mask + mask_byte_offset);
          }
        }

        // Next
        Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
      }

      template <typename DummyT, bool ENABLE_IDEMPOTENCE>
      struct VertexC {
        static __device__ __forceinline__ void Cull(Cta *cta, Tile *tile) {}
      };

      template <typename DummyT>
      struct VertexC<DummyT, true> {
        static __device__ __forceinline__ void Cull(Cta *cta, Tile *tile) {
          // if (tile -> element_id[LOAD][VEC] >= 0)
          if (util::isValid(tile->element_id[LOAD][VEC])) {
            InKeyT key_in = (tile->element_id[LOAD][VEC]
                             //& KernelPolicyT::ELEMENT_ID_MASK
            );
            // row_id = row_id & KernelPolicyT::ELEMENT_ID_MASK;

            // LabelT label;
            // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            //    label,
            //    cta->d_data_slice ->labels + row_id);
            // if (label != util::MaxValue<LabelT>())
            if (cta->labels[key_in] !=
                util::PreDefinedValues<LabelT>::MaxValue) {
              // Seen it
              tile->element_id[LOAD][VEC] =
                  util::PreDefinedValues<InKeyT>::InvalidValue;
              // row_id = util::InvalidValue<VertexId>();
            } else {
              cta->labels[key_in] = cta->label;
              InKeyT src = (FLAG & OprtrOption_Mark_Predecessors)
                               ? (tile->pred_id[LOAD][VEC])
                               : util::PreDefinedValues<InKeyT>::InvalidValue;
              if (!cta->filter_op(
                      src, key_in, util::PreDefinedValues<SizeT>::InvalidValue,
                      key_in, util::PreDefinedValues<SizeT>::InvalidValue,
                      util::PreDefinedValues<SizeT>::InvalidValue))
                tile->element_id[LOAD][VEC] =
                    util::PreDefinedValues<InKeyT>::InvalidValue;
            }
          }
        }
      };

      template <typename DummyT>
      struct VertexC<DummyT, false> {
        static __device__ __forceinline__ void Cull(Cta *cta, Tile *tile) {
          if (util::isValid(tile->element_id[LOAD][VEC])) {
            SizeT node_id = threadIdx.x * LOADS_PER_TILE * LOAD_VEC_SIZE +
                            LOAD * LOAD_VEC_SIZE + VEC;
            InKeyT src = (FLAG & OprtrOption_Mark_Predecessors)
                             ? (tile->pred_id[LOAD][VEC])
                             : util::PreDefinedValues<InKeyT>::InvalidValue;
            SizeT output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
            if (!cta->filter_op(
                    src, tile->element_id[LOAD][VEC],
                    output_pos,  // util::PreDefinedValues<SizeT>::InvalidValue,
                    tile->element_id[LOAD][VEC], node_id, output_pos))
              tile->element_id[LOAD][VEC] =
                  util::PreDefinedValues<InKeyT>::InvalidValue;
          }
        }
      };

      /**
       * @brief Set vertex id to -1 if we want to cull this vertex from the
       * outgoing frontier.
       * @param[in] cta
       * @param[in] tile
       *
       */
      static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {
        VertexC<SizeT, (FLAG & OprtrOption_Idempotence) != 0>::Cull(cta, tile);

        // Next
        Iterate<LOAD, VEC + 1>::VertexCull(cta, tile);
      }

      /**
       * @brief Cull redundant vertices using history hash table
       *
       */
      static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {
        if (util::isValid(tile->element_id[LOAD][VEC])) {
          int hash = (tile->element_id[LOAD][VEC]) %
                     SmemStorage::HISTORY_HASH_ELEMENTS;
          //(tile -> element_id[LOAD][VEC]) & SmemStorage::HISTORY_HASH_MASK;
          InKeyT retrieved = cta->smem_storage.history[hash];

          if (retrieved == tile->element_id[LOAD][VEC]) {
            // Seen it
            tile->element_id[LOAD][VEC] =
                util::PreDefinedValues<InKeyT>::InvalidValue;
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
      static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {
        if (util::isValid(tile->element_id[LOAD][VEC])) {
          int warp_id = threadIdx.x >> 5;
          int hash = tile->element_id[LOAD][VEC] & SmemStorage::WARP_HASH_MASK;
          //(SmemStorage::WARP_HASH_ELEMENTS - 1);

          /*if (warp_id < 0 || warp_id >= KernelPolicyT::WARPS)
          {
              printf("Invalid warp_id (%d), threadIdx.x == %d, WARPS = %d\n",
              warp_id, threadIdx.x, KernelPolicyT::WARPS);
              return;
          }

          if (hash < 0 || hash >= SmemStorage::WARP_HASH_ELEMENTS)
          {
              printf("Invalid hash (%d), element_id = %lld, WARP_HASH_ELEMENTS =
          %d\n", hash, (long long)tile->element_id[LOAD][VEC],
                  SmemStorage::WARP_HASH_ELEMENTS);
              return;
          }*/

          cta->smem_storage.state.vid_hashtable[warp_id][hash] =
              tile->element_id[LOAD][VEC];
          InKeyT retrieved =
              cta->smem_storage.state.vid_hashtable[warp_id][hash];

          if (retrieved == tile->element_id[LOAD][VEC]) {
            cta->smem_storage.state.vid_hashtable[warp_id][hash] = threadIdx.x;
            InKeyT tid = cta->smem_storage.state.vid_hashtable[warp_id][hash];
            if (tid != threadIdx.x) {
              tile->element_id[LOAD][VEC] =
                  util::PreDefinedValues<InKeyT>::InvalidValue;
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
    struct Iterate<LOAD, LOAD_VEC_SIZE, dummy> {
      // InitFlags
      static __device__ __forceinline__ void InitFlags(Tile *tile) {
        Iterate<LOAD + 1, 0>::InitFlags(tile);
      }

      // BitmaskCull
      static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {
        Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
      }

      // VertexCull
      static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {
        Iterate<LOAD + 1, 0>::VertexCull(cta, tile);
      }

      // WarpCull
      static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {
        Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
      }

      // HistoryCull
      static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {
        Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
      }
    };

    /**
     * Terminate iteration
     */
    template <int dummy>
    struct Iterate<LOADS_PER_TILE, 0, dummy> {
      // InitFlags
      static __device__ __forceinline__ void InitFlags(Tile *tile) {}

      // BitmaskCull
      static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {
      }

      // VertexCull
      static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {}

      // HistoryCull
      static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {
      }

      // WarpCull
      static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}
    };

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    // Initializer
    __device__ __forceinline__ void InitFlags() {
      Iterate<0, 0>::InitFlags(this);
    }

    // Culls vertices based on bitmask
    __device__ __forceinline__ void BitmaskCull(Cta *cta) {
      Iterate<0, 0>::BitmaskCull(cta, this);
    }

    // Culls vertices
    __device__ __forceinline__ void VertexCull(Cta *cta) {
      Iterate<0, 0>::VertexCull(cta, this);
    }

    // Culls redundant vertices within the warp
    __device__ __forceinline__ void WarpCull(Cta *cta) {
      Iterate<0, 0>::WarpCull(cta, this);
    }

    // Culls redundant vertices within recent CTA history
    __device__ __forceinline__ void HistoryCull(Cta *cta) {
      Iterate<0, 0>::HistoryCull(cta, this);
    }
  };

  //---------------------------------------------------------------------
  // Methods
  //---------------------------------------------------------------------

  template <typename LabelT, bool ENABLE_IDEMPOTENCE>
  struct BitMask_Cull {
    static __device__ __forceinline__ bool Eval(LabelT label) { return false; }
  };

  template <typename LabelT>
  struct BitMask_Cull<LabelT, true> {
    static __device__ __forceinline__ bool Eval(LabelT label) {
      if (KernelPolicyT::END_BITMASK_CULL < 0)
        return true;
      else if (KernelPolicyT::END_BITMASK_CULL == 0)
        return false;
      else
        return (label < KernelPolicyT::END_BITMASK_CULL);
    }
  };

  template <typename LabelT>
  struct BitMask_Cull<LabelT, false> {
    static __device__ __forceinline__ bool Eval(LabelT label) { return false; }
  };

  /**
   * @brief CTA type default constructor
   */
  __device__ __forceinline__ Cta(const SizeT queue_index, const InKeyT *keys_in,
                                 const ValueT *values_in, const LabelT label,
                                 LabelT *labels, unsigned char *visited_masks,
                                 OutKeyT *keys_out, SmemStorage &smem_storage,
                                 util::CtaWorkProgress<SizeT> &work_progress,
                                 // const SizeT              max_out_frontier,
                                 const FilterOpT &filter_op)
      : queue_index(queue_index),
        keys_in(keys_in),
        values_in(values_in),
        label(label),
        labels(labels),
        visited_masks(visited_masks),
        keys_out(keys_out),
        raking_details(smem_storage.state.raking_elements,
                       smem_storage.state.warpscan, 0),
        smem_storage(smem_storage),
        work_progress(work_progress),
        // max_out_frontier        (max_out_frontier),
        filter_op(filter_op) {
    bitmask_cull =
        BitMask_Cull<LabelT, (FLAG & OprtrOption_Idempotence) != 0>::Eval(
            label);

    // Initialize history duplicate-filter
    for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS;
         offset += KernelPolicyT::THREADS) {
      smem_storage.history[offset] =
          util::PreDefinedValues<InKeyT>::InvalidValue;
    }
  }

  template <typename T, typename Cta, bool ENABLE_IDEMPOTENCE>
  struct PTile {
    static __device__ __forceinline__ void Process(
        T &tile, Cta *cta, SizeT cta_offset,
        const SizeT &guarded_elements = KernelPolicyT::TILE_ELEMENTS) {}
  };

  template <typename TileT, typename Cta>
  struct PTile<TileT, Cta, true> {
    static __device__ __forceinline__ void Process(
        TileT &tile, Cta *cta, SizeT &cta_offset,
        const SizeT &guarded_elements = KernelPolicyT::TILE_ELEMENTS) {
      if ((FLAG & OprtrOption_Mark_Predecessors) != 0 &&
          cta->values_in != NULL) {
        util::io::LoadTile<KernelPolicyT::LOG_LOADS_PER_TILE,
                           KernelPolicyT::LOG_LOAD_VEC_SIZE,
                           KernelPolicyT::THREADS, QUEUE_READ_MODIFIER,
                           false>::LoadValid(tile.pred_id, cta->values_in,
                                             cta_offset, guarded_elements);
      }

      if (cta->bitmask_cull && cta->visited_masks != NULL) {
        tile.BitmaskCull(cta);
      }
      tile.HistoryCull(cta);
      // tile.WarpCull(cta);
      // using vertex visitation status (update discovered vertices)
      tile.VertexCull(cta);
    }
  };

  template <typename TileT, typename Cta>
  struct PTile<TileT, Cta, false> {
    static __device__ __forceinline__ void Process(
        TileT &tile, Cta *cta, SizeT &cta_offset,
        const SizeT &guarded_elements = KernelPolicyT::TILE_ELEMENTS) {
      // using vertex visitation status (update discovered vertices)
      tile.VertexCull(cta);
    }
  };

  /**
   * @brief Process a single, full tile.
   *
   * @param[in] cta_offset Offset within the CTA where we want to start the tile
   * processing.
   * @param[in] guarded_elements The guarded elements to prevent the
   * out-of-bound visit.
   */
  __device__ __forceinline__ void ProcessTile(
      SizeT cta_offset,
      const SizeT &guarded_elements = KernelPolicyT::TILE_ELEMENTS) {
    typedef Tile<KernelPolicyT::LOG_LOADS_PER_TILE,
                 KernelPolicyT::LOG_LOAD_VEC_SIZE>
        TileT;
    TileT tile;

    // Load tile
    util::io::LoadTile<
        KernelPolicyT::LOG_LOADS_PER_TILE, KernelPolicyT::LOG_LOAD_VEC_SIZE,
        KernelPolicyT::THREADS, QUEUE_READ_MODIFIER,
        false>::LoadValid(tile.element_id, const_cast<InKeyT *>(keys_in),
                          cta_offset, guarded_elements,
                          util::PreDefinedValues<InKeyT>::InvalidValue);

    PTile<TileT, Cta, (FLAG & OprtrOption_Idempotence) != 0>::Process(
        tile, this, cta_offset, guarded_elements);

    // Init valid flags and ranks
    tile.InitFlags();

    // Protect repurposable storage that backs both raking lanes and local cull
    // scratch
    __syncthreads();

    // Scan tile of ranks, using an atomic add to reserve
    // space in the contracted queue, seeding ranks
    // Cooperative Scan (done)
    util::Sum<SizeT> scan_op;
    SizeT new_queue_offset = util::scan::
        CooperativeTileScan<KernelPolicyT::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
            raking_details, tile.ranks,
            work_progress.GetQueueCounter(queue_index + 1), scan_op);

    // Check updated queue offset for overflow due to redundant expansion
    // if (new_queue_offset >= max_out_frontier) {
    // printf(" new_queue_offset >= max_out_frontier, new_queue_offset = %d,
    // max_out_frontier = %d\n", new_queue_offset, max_out_frontier);
    //    work_progress.SetOverflow();
    //    util::ThreadExit();
    //}

    // Scatter directly (without first contracting in smem scratch), predicated
    // on flags
    if (keys_out != NULL) {
      util::io::ScatterTile<KernelPolicyT::LOG_LOADS_PER_TILE,
                            KernelPolicyT::LOG_LOAD_VEC_SIZE,
                            KernelPolicyT::THREADS,
                            QUEUE_WRITE_MODIFIER>::Scatter(keys_out,
                                                           tile.element_id,
                                                           tile.flags,
                                                           tile.ranks);
    }
  }
};

}  // namespace CULL
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
