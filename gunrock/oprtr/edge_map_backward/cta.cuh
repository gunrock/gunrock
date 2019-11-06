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
 * @brief CTA tile-processing abstraction for Backward Edge Map
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

// TODO: use CUB for SOA scan

namespace gunrock {
namespace oprtr {
namespace edge_map_backward {

/**
 * 1D texture setting for efficiently fetch data from graph_row_offsets
 */
/*template <typename SizeT>
    struct RowOffsetTex
    {
        static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
    };
template <typename SizeT>
    texture<SizeT, cudaTextureType1D, cudaReadModeElementType>
RowOffsetTex<SizeT>::ref;*/

/*template <typename VertexId>
    struct ColumnIndicesTex
    {
        static texture<VertexId, cudaTextureType1D, cudaReadModeElementType>
ref;
    };
template <typename VertexId>
    texture<VertexId, cudaTextureType1D, cudaReadModeElementType>
ColumnIndicesTex<VertexId>::ref;*/

/**
 * @brief CTA tile-processing abstraction for backward edge mapping operator.
 *
 * @tparam KernelPolicy Kernel policy type for backward edge mapping.
 * @tparam ProblemData Problem data type for backward edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Cta {
  /**
   * Typedefs
   */

  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;

  typedef typename KernelPolicy::SmemStorage SmemStorage;

  typedef typename KernelPolicy::SoaScanOp SoaScanOp;
  typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
  typedef typename KernelPolicy::TileTuple TileTuple;

  typedef typename ProblemData::DataSlice DataSlice;

  typedef util::Tuple<SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
                      SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]>
      RankSoa;

  /**
   * Members
   */

  // Input and output device pointers
  VertexId *d_queue;   // Incoming and outgoing vertex frontier
  VertexId *d_index;   // Incoming vertex frontier index
  bool *d_bitmap_in;   // Incoming frontier bitmap
  bool *d_bitmap_out;  // Outgoing frontier bitmap
  SizeT *d_row_offsets;
  VertexId *d_column_indices;
  DataSlice *problem;  // Problem Data

  // Work progress
  VertexId queue_index;  // Current frontier queue counter index
  util::CtaWorkProgress<SizeT> &work_progress;  // Atomic queueing counters
  // int                     num_gpus;                   // Number of GPUs

  // Operational details for raking grid
  RakingSoaDetails raking_soa_details;

  // Shared memory for the CTA
  SmemStorage &smem_storage;

  gunrock::oprtr::advance::TYPE ADVANCE_TYPE;

  /**
   * @brief Tile of incoming frontier to process
   *
   * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
   * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
   */
  template <int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
  struct Tile {
    /**
     * Typedefs and Constants
     */

    enum {
      LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE,
      LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
    };

    typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;

    /**
     * Members
     */

    // Dequeued vertex ids
    VertexId vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
    VertexId vertex_idx[LOADS_PER_TILE][LOAD_VEC_SIZE];

    SizeT row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
    SizeT row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

    SizeT fine_count;
    SizeT coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
    SizeT fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

    // Progress for scan-based backward edge map gather offsets
    SizeT row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
    SizeT progress;

    /**
     * @brief Iterate over vertex ids in tile.
     */
    template <typename KernelPolicy, typename ProblemData, typename Functor>
    struct Cta {
      /**
       * Typedefs
       */

      typedef typename KernelPolicy::VertexId VertexId;
      typedef typename KernelPolicy::SizeT SizeT;

      typedef typename KernelPolicy::SmemStorage SmemStorage;

      typedef typename KernelPolicy::SoaScanOp SoaScanOp;
      typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
      typedef typename KernelPolicy::TileTuple TileTuple;

      typedef typename ProblemData::DataSlice DataSlice;

      typedef util::Tuple<SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
                          SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]>
          RankSoa;

      /**
       * Members
       */

      // Input and output device pointers
      VertexId *d_queue;   // Incoming and outgoing vertex frontier
      VertexId *d_index;   // Incoming vertex frontier index
      bool *d_bitmap_in;   // Incoming frontier bitmap
      bool *d_bitmap_out;  // Outgoing frontier bitmap
      SizeT *d_row_offsets;
      VertexId *d_column_indices;
      DataSlice *problem;  // Problem Data

      // Work progress
      VertexId queue_index;  // Current frontier queue counter index
      util::CtaWorkProgress<SizeT> &work_progress;  // Atomic queueing counters
      // int                     num_gpus;                   // Number of GPUs

      // Operational details for raking grid
      RakingSoaDetails raking_soa_details;

      // Shared memory for the CTA
      SmemStorage &smem_storage;

      gunrock::oprtr::advance::TYPE ADVANCE_TYPE;

      /**
       * @brief Tile of incoming frontier to process
       *
       * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
       * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
       */
      template <int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
      struct Tile {
        /**
         * Typedefs and Constants
         */

        enum {
          LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE,
          LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
        };

        typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;

        /**
         * Members
         */

        // Dequeued vertex ids
        VertexId vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
        VertexId vertex_idx[LOADS_PER_TILE][LOAD_VEC_SIZE];

        SizeT row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

        SizeT fine_count;
        SizeT coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

        // Progress for scan-based backward edge map gather offsets
        SizeT row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
        SizeT progress;

        /**
         * @brief Iterate over vertex ids in tile.
         */
        template <int LOAD, int VEC, int dummy = 0>
        struct Iterate {
          /**
           * @brief Tile data initialization
           */
          template <typename Tile>
          static __device__ __forceinline__ void Init(Tile *tile) {
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
          static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {
            if (tile->vertex_id[LOAD][VEC] != -1) {
              // Translate vertex-id into local gpu row-id (currently stride of
              // num_gpu)
              VertexId row_id = tile->vertex_id[LOAD][VEC];  // / cta->num_gpus;

              // Load neighbor row range from d_row_offsets
              Vec2SizeT row_range;
              util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                  row_range.x, cta->d_row_offsets + row_id);
              util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                  row_range.y, cta->d_row_offsets + row_id + 1);
              // row_range.x = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
              // row_range.y = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1);

              // compute row offset and length
              tile->row_offset[LOAD][VEC] = row_range.x;
              tile->row_length[LOAD][VEC] = row_range.y - row_range.x;
            }

            tile->fine_row_rank[LOAD][VEC] =
                (tile->row_length[LOAD][VEC] <
                 KernelPolicy::WARP_GATHER_THRESHOLD)
                    ? tile->row_length[LOAD][VEC]
                    : 0;

            tile->coarse_row_rank[LOAD][VEC] =
                (tile->row_length[LOAD][VEC] <
                 KernelPolicy::WARP_GATHER_THRESHOLD)
                    ? 0
                    : tile->row_length[LOAD][VEC];
            // tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] >
            // 0) ? tile->row_length[LOAD][VEC] : 0;
            // tile->coarse_row_rank[LOAD][VEC] = 0;

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
          static __device__ __forceinline__ void CtaExpand(Cta *cta,
                                                           Tile *tile) {
            // CTA-based expansion/loading
            while (true) {
              // All threads in block vie for the control of the block
              if (tile->row_length[LOAD][VEC] >=
                  KernelPolicy::CTA_GATHER_THRESHOLD) {
                cta->smem_storage.state.cta_comm = threadIdx.x;
              }

              __syncthreads();

              // Check
              int owner = cta->smem_storage.state.cta_comm;
              if (owner == KernelPolicy::THREADS) {
                // All threads in the block has less neighbor number for CTA
                // Expand
                break;
              }

              if (owner == threadIdx.x) {
                // Got control of the CTA: command it
                cta->smem_storage.state.warp_comm[0][0] =
                    tile->row_offset[LOAD][VEC];  // start
                cta->smem_storage.state.warp_comm[0][1] =
                    tile->vertex_idx[LOAD][VEC];  // queue rank
                cta->smem_storage.state.warp_comm[0][2] =
                    tile->row_offset[LOAD][VEC] +
                    tile->row_length[LOAD][VEC];  // oob
                cta->smem_storage.state.warp_comm[0][3] =
                    tile->vertex_id[LOAD][VEC];  // predecessor

                // Unset row length
                tile->row_length[LOAD][VEC] = 0;

                // Unset my command
                cta->smem_storage.state.cta_comm =
                    KernelPolicy::THREADS;  // So that we won't repeatedly
                                            // expand this node
              }
              __syncthreads();

              // Read commands
              SizeT coop_offset = cta->smem_storage.state.warp_comm[0][0];
              SizeT coop_oob = cta->smem_storage.state.warp_comm[0][2];

              VertexId child_id;
              child_id = cta->smem_storage.state.warp_comm[0][3];

              VertexId parent_id;

              while ((coop_offset + KernelPolicy::THREADS < coop_oob) &&
                     (child_id >= 0)) {
                // Gather
                // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
                // coop_offset+threadIdx.x);
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_id,
                    cta->d_column_indices + coop_offset + threadIdx.x);

                // TODO:Users can insert a functor call here
                // ProblemData::Apply(pred_id, neighbor_id) (done)

                bool bitmap_in;
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    bitmap_in, cta->d_bitmap_in + parent_id);
                if (bitmap_in) {
                  if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                        coop_offset + threadIdx.x))
                    Functor::ApplyEdge(parent_id, child_id, cta->problem,
                                       coop_offset + threadIdx.x);
                  child_id = -1;
                }

                if (child_id == -1) {
                  // Mark the node as visited in  d_queue, so that we can cull
                  // it during next vertex_map
                  util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                      St((VertexId)-1,
                         cta->d_queue +
                             cta->smem_storage.state.warp_comm[0][1]);

                  // Set bitmap_out to true
                  util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                      St(true, cta->d_bitmap_out +
                                   cta->smem_storage.state.warp_comm[0][3]);
                }

                coop_offset += KernelPolicy::THREADS;
              }

              if ((coop_offset + threadIdx.x < coop_oob) && (child_id >= 0)) {
                // Gather
                // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
                // coop_offset+threadIdx.x);
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_id,
                    cta->d_column_indices + coop_offset + threadIdx.x);

                bool bitmap_in;
                util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    bitmap_in, cta->d_bitmap_in + parent_id);
                if (bitmap_in) {
                  if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                        coop_offset + threadIdx.x))
                    Functor::ApplyEdge(parent_id, child_id, cta->problem,
                                       coop_offset + threadIdx.x);
                  child_id = -1;
                }

                if (child_id == -1) {
                  // Mark the node as visited in  d_queue, so that we can cull
                  // it during next vertex_map
                  util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                      St((VertexId)-1,
                         cta->d_queue +
                             cta->smem_storage.state.warp_comm[0][1]);

                  // Set bitmap_out to true
                  util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                      St(true, cta->d_bitmap_out +
                                   cta->smem_storage.state.warp_comm[0][3]);
                }
              }
            }

            // Next vector element
            Iterate<LOAD, VEC + 1>::CtaExpand(cta, tile);
          }

          /**
           * @brief Expand the node's neighbor list using a warp.
           * @tparam Cta CTA tile-processing abstraction type
           * @tparam Tile Tile structure type
           * @param[in] cta Pointer to CTA object
           * @param[in] tile Pointer to Tile object
           */
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void WarpExpand(Cta *cta,
                                                            Tile *tile) {
            if (KernelPolicy::WARP_GATHER_THRESHOLD <
                KernelPolicy::CTA_GATHER_THRESHOLD) {
              // Warp-based expansion/loading
              int warp_id =
                  threadIdx.x >> GR_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
              int lane_id = util::LaneId();

              while (::_any(tile->row_length[LOAD][VEC] >=
                            KernelPolicy::WARP_GATHER_THRESHOLD)) {
                if (tile->row_length[LOAD][VEC] >=
                    KernelPolicy::WARP_GATHER_THRESHOLD) {
                  // All threads inside one warp vie for control of the warp
                  cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                }

                if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {
                  // Got control of the warp
                  cta->smem_storage.state.warp_comm[warp_id][0] =
                      tile->row_offset[LOAD][VEC];  // start
                  cta->smem_storage.state.warp_comm[warp_id][1] =
                      tile->vertex_idx[LOAD][VEC];
                  ;  // queue rank
                  cta->smem_storage.state.warp_comm[warp_id][2] =
                      tile->row_offset[LOAD][VEC] +
                      tile->row_length[LOAD][VEC];  // oob
                  cta->smem_storage.state.warp_comm[warp_id][3] =
                      tile->vertex_id[LOAD][VEC];  // predecessor

                  // Unset row length
                  tile->row_length[LOAD][VEC] =
                      0;  // So that we won't repeatedly expand this node
                }

                SizeT coop_offset =
                    cta->smem_storage.state.warp_comm[warp_id][0];
                SizeT coop_oob = cta->smem_storage.state.warp_comm[warp_id][2];

                VertexId child_id;
                child_id = cta->smem_storage.state.warp_comm[warp_id][3];

                VertexId parent_id;
                while ((coop_offset + GR_WARP_THREADS(KernelPolicy::CUDA_ARCH) <
                        coop_oob) &&
                       child_id >= 0) {
                  // Gather

                  // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
                  // coop_offset + lane_id);
                  util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                      parent_id, cta->d_column_indices + coop_offset + lane_id);

                  bool bitmap_in;
                  util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                      bitmap_in, cta->d_bitmap_in + parent_id);
                  if (bitmap_in) {
                    if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                          coop_offset + lane_id))
                      Functor::ApplyEdge(parent_id, child_id, cta->problem,
                                         coop_offset + lane_id);
                    child_id = -1;
                  }

                  if (child_id == -1) {
                    // Mark the node as visited in  d_queue, so that we can cull
                    // it during next vertex_map
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                        St((VertexId)-1,
                           cta->d_queue +
                               cta->smem_storage.state.warp_comm[warp_id][1]);

                    // Set bitmap_out to true
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                        St(true,
                           cta->d_bitmap_out +
                               cta->smem_storage.state.warp_comm[warp_id][3]);
                  }

                  coop_offset += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
                }

                if ((coop_offset + lane_id < coop_oob) && child_id >= 0) {
                  // Gather
                  // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
                  // coop_offset + lane_id);
                  util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                      parent_id, cta->d_column_indices + coop_offset + lane_id);

                  bool bitmap_in;
                  util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                      bitmap_in, cta->d_bitmap_in + parent_id);
                  if (bitmap_in) {
                    if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                          coop_offset + lane_id))
                      Functor::ApplyEdge(parent_id, child_id, cta->problem,
                                         coop_offset + lane_id);
                    child_id = -1;
                  }

                  if (child_id == -1) {
                    // Mark the node as visited in  d_queue, so that we can cull
                    // it during next vertex_map
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                        St((VertexId)-1,
                           cta->d_queue +
                               cta->smem_storage.state.warp_comm[warp_id][1]);

                    // Set bitmap_out to true
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::
                        St(true,
                           cta->d_bitmap_out +
                               cta->smem_storage.state.warp_comm[warp_id][3]);
                  }
                }
              }

              // Next vector element
              Iterate<LOAD, VEC + 1>::WarpExpand(cta, tile);
            }
          }

          /**
           * @brief Expand the node's neighbor list using a single thread.
           * @tparam Cta CTA tile-processing abstraction type
           * @tparam Tile Tile structure type
           * @param[in] cta Pointer to CTA object
           * @param[in] tile Pointer to Tile object
           */
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void ThreadExpand(Cta *cta,
                                                              Tile *tile) {
            // Expand the neighbor list into scratch space
            SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] +
                                   tile->row_progress[LOAD][VEC] -
                                   tile->progress;

            while (
                (tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
                (scratch_offset < SmemStorage::GATHER_ELEMENTS)) {
              // Put gather offset into scratch space
              cta->smem_storage.gather_offsets[scratch_offset] =
                  tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];
              // In edge_map_backward, gather_predecessors actually store
              // vertex_ids as child_id
              cta->smem_storage.gather_predecessors[scratch_offset] =
                  tile->vertex_id[LOAD][VEC];
              cta->smem_storage.gather_offsets2[scratch_offset] =
                  tile->vertex_idx[LOAD][VEC];

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
        struct Iterate<LOAD, LOAD_VEC_SIZE, dummy> {
          // Init
          template <typename Tile>
          static __device__ __forceinline__ void Init(Tile *tile) {
            Iterate<LOAD + 1, 0>::Init(tile);
          }

          // Inspect
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {
            Iterate<LOAD + 1, 0>::Inspect(cta, tile);
          }

          // CTA Expand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void CtaExpand(Cta *cta,
                                                           Tile *tile) {
            Iterate<LOAD + 1, 0>::CtaExpand(cta, tile);
          }

          // Warp Expand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void WarpExpand(Cta *cta,
                                                            Tile *tile) {
            Iterate<LOAD + 1, 0>::WarpExpand(cta, tile);
          }

          // Single Thread Expand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void ThreadExpand(Cta *cta,
                                                              Tile *tile) {
            Iterate<LOAD + 1, 0>::ThreadExpand(cta, tile);
          }
        };

        /**
         * Terminate Iterate
         */
        template <int dummy>
        struct Iterate<LOADS_PER_TILE, 0, dummy> {
          // Init
          template <typename Tile>
          static __device__ __forceinline__ void Init(Tile *tile) {}

          // Inspect
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {
          }

          // CtaExpand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void CtaExpand(Cta *cta,
                                                           Tile *tile) {}

          // WarpExpand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void WarpExpand(Cta *cta,
                                                            Tile *tile) {}

          // SingleThreadExpand
          template <typename Cta, typename Tile>
          static __device__ __forceinline__ void ThreadExpand(Cta *cta,
                                                              Tile *tile) {}
        };

        // Iterate Interface

        // Constructor
        __device__ __forceinline__ Tile() { Iterate<0, 0>::Init(this); }

        // Inspect dequeued nodes
        template <typename Cta>
        __device__ __forceinline__ void Inspect(Cta *cta) {
          Iterate<0, 0>::Inspect(cta, this);
        }

        // CTA Expand
        template <typename Cta>
        __device__ __forceinline__ void CtaExpand(Cta *cta) {
          Iterate<0, 0>::CtaExpand(cta, this);
        }

        // Warp Expand
        template <typename Cta>
        __device__ __forceinline__ void WarpExpand(Cta *cta) {
          Iterate<0, 0>::WarpExpand(cta, this);
        }

        // Single Thread Expand
        template <typename Cta>
        __device__ __forceinline__ void ThreadExpand(Cta *cta) {
          Iterate<0, 0>::ThreadExpand(cta, this);
        }
      };

      // Methods

      /**
       * @brief CTA default constructor
       */
      __device__ __forceinline__
      Cta(VertexId queue_index,
          // int                         num_gpus,
          SmemStorage &smem_storage, VertexId *d_queue, VertexId *d_index,
          bool *d_bitmap_in, bool *d_bitmap_out, SizeT *d_row_offsets,
          VertexId *d_column_indices, DataSlice *problem,
          util::CtaWorkProgress<SizeT> &work_progress,
          gunrock::oprtr::advance::TYPE ADVANCE_TYPE)
          :

            queue_index(queue_index),
            // num_gpus(num_gpus),
            smem_storage(smem_storage),
            raking_soa_details(typename RakingSoaDetails::GridStorageSoa(
                                   smem_storage.coarse_raking_elements,
                                   smem_storage.fine_raking_elements),
                               typename RakingSoaDetails::WarpscanSoa(
                                   smem_storage.state.coarse_warpscan,
                                   smem_storage.state.fine_warpscan),
                               TileTuple(0, 0)),
            d_queue(d_queue),
            d_index(d_index),
            d_bitmap_in(d_bitmap_in),
            d_bitmap_out(d_bitmap_out),
            d_row_offsets(d_row_offsets),
            d_column_indices(d_column_indices),
            problem(problem),
            work_progress(work_progress),
            ADVANCE_TYPE(ADVANCE_TYPE) {
        if (threadIdx.x == 0) {
          smem_storage.state.cta_comm = KernelPolicy::THREADS;
          smem_storage.state.overflowed = false;
        }
      }

      if (child_id == -1) {
        // Mark the node as visited in  d_queue, so that we can cull it
        // during next vertex_map
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            (VertexId)-1,
            cta->d_queue + cta->smem_storage.state.warp_comm[0][1]);

        // Set bitmap_out to true
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, cta->d_bitmap_out + cta->smem_storage.state.warp_comm[0][3]);
      }

      coop_offset += KernelPolicy::THREADS;
    }

    if ((coop_offset + threadIdx.x < coop_oob) && (child_id >= 0)) {
      // Gather
      // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
      // coop_offset+threadIdx.x);
      util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
          parent_id, cta->d_column_indices + coop_offset + threadIdx.x);

      bool bitmap_in;
      util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
          bitmap_in, cta->d_bitmap_in + parent_id);
      if (bitmap_in) {
        if (Functor::CondEdge(parent_id, child_id, cta->problem,
                              coop_offset + threadIdx.x))
          Functor::ApplyEdge(parent_id, child_id, cta->problem,
                             coop_offset + threadIdx.x);
        child_id = -1;
      }

      if (child_id == -1) {
        // Mark the node as visited in  d_queue, so that we can cull it
        // during next vertex_map
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            (VertexId)-1,
            cta->d_queue + cta->smem_storage.state.warp_comm[0][1]);

        // Set bitmap_out to true
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, cta->d_bitmap_out + cta->smem_storage.state.warp_comm[0][3]);
      }
    }
  }

  // Next vector element
  Iterate<LOAD, VEC + 1>::CtaExpand(cta, tile);
}

/**
 * @brief Expand the node's neighbor list using a warp.
 * @tparam Cta CTA tile-processing abstraction type
 * @tparam Tile Tile structure type
 * @param[in] cta Pointer to CTA object
 * @param[in] tile Pointer to Tile object
 */
template <typename Cta, typename Tile>
static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile) {
  if (KernelPolicy::WARP_GATHER_THRESHOLD <
      KernelPolicy::CTA_GATHER_THRESHOLD) {
    // Warp-based expansion/loading
    int warp_id = threadIdx.x >> GR_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
    int lane_id = util::LaneId();

    while (::_any(tile->row_length[LOAD][VEC] >=
                  KernelPolicy::WARP_GATHER_THRESHOLD)) {
      if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD) {
        // All threads inside one warp vie for control of the warp
        cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
      }

      if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {
        // Got control of the warp
        cta->smem_storage.state.warp_comm[warp_id][0] =
            tile->row_offset[LOAD][VEC];  // start
        cta->smem_storage.state.warp_comm[warp_id][1] =
            tile->vertex_idx[LOAD][VEC];
        ;  // queue rank
        cta->smem_storage.state.warp_comm[warp_id][2] =
            tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];  // oob
        cta->smem_storage.state.warp_comm[warp_id][3] =
            tile->vertex_id[LOAD][VEC];  // predecessor

        // Unset row length
        tile->row_length[LOAD][VEC] =
            0;  // So that we won't repeatedly expand this node
      }

      SizeT coop_offset = cta->smem_storage.state.warp_comm[warp_id][0];
      SizeT coop_oob = cta->smem_storage.state.warp_comm[warp_id][2];

      VertexId child_id;
      child_id = cta->smem_storage.state.warp_comm[warp_id][3];

      VertexId parent_id;
      while (
          (coop_offset + GR_WARP_THREADS(KernelPolicy::CUDA_ARCH) < coop_oob) &&
          child_id >= 0) {
        // Gather

        // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
        // coop_offset + lane_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            parent_id, cta->d_column_indices + coop_offset + lane_id);

        bool bitmap_in;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            bitmap_in, cta->d_bitmap_in + parent_id);
        if (bitmap_in) {
          if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                coop_offset + lane_id))
            Functor::ApplyEdge(parent_id, child_id, cta->problem,
                               coop_offset + lane_id);
          child_id = -1;
        }

        if (child_id == -1) {
          // Mark the node as visited in  d_queue, so that we can cull it
          // during next vertex_map
          util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
              (VertexId)-1,
              cta->d_queue + cta->smem_storage.state.warp_comm[warp_id][1]);

          // Set bitmap_out to true
          util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
              true, cta->d_bitmap_out +
                        cta->smem_storage.state.warp_comm[warp_id][3]);
        }

        coop_offset += GR_WARP_THREADS(KernelPolicy::CUDA_ARCH);
      }

      if ((coop_offset + lane_id < coop_oob) && child_id >= 0) {
        // Gather
        // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
        // coop_offset + lane_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            parent_id, cta->d_column_indices + coop_offset + lane_id);

        bool bitmap_in;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            bitmap_in, cta->d_bitmap_in + parent_id);
        if (bitmap_in) {
          if (Functor::CondEdge(parent_id, child_id, cta->problem,
                                coop_offset + lane_id))
            Functor::ApplyEdge(parent_id, child_id, cta->problem,
                               coop_offset + lane_id);
          child_id = -1;
        }

        if (child_id == -1) {
          // Mark the node as visited in  d_queue, so that we can cull it
          // during next vertex_map
          util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
              (VertexId)-1,
              cta->d_queue + cta->smem_storage.state.warp_comm[warp_id][1]);

          // Set bitmap_out to true
          util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
              true, cta->d_bitmap_out +
                        cta->smem_storage.state.warp_comm[warp_id][3]);
        }
      }
    }

    // Next vector element
    Iterate<LOAD, VEC + 1>::WarpExpand(cta, tile);
  }
}

/**
 * @brief Expand the node's neighbor list using a single thread.
 * @tparam Cta CTA tile-processing abstraction type
 * @tparam Tile Tile structure type
 * @param[in] cta Pointer to CTA object
 * @param[in] tile Pointer to Tile object
 */
template <typename Cta, typename Tile>
static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile) {
  // Expand the neighbor list into scratch space
  SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] +
                         tile->row_progress[LOAD][VEC] - tile->progress;

  while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
         (scratch_offset < SmemStorage::GATHER_ELEMENTS)) {
    // Put gather offset into scratch space
    cta->smem_storage.gather_offsets[scratch_offset] =
        tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];
    // In edge_map_backward, gather_predecessors actually store vertex_ids
    // as child_id
    cta->smem_storage.gather_predecessors[scratch_offset] =
        tile->vertex_id[LOAD][VEC];
    cta->smem_storage.gather_offsets2[scratch_offset] =
        tile->vertex_idx[LOAD][VEC];

    tile->row_progress[LOAD][VEC]++;
    scratch_offset++;
  }

  // Next vector element
  Iterate<LOAD, VEC + 1>::ThreadExpand(cta, tile);
}
};  // namespace edge_map_backward

/**
 * Iterate next load
 */
template <int LOAD, int dummy>
struct Iterate<LOAD, LOAD_VEC_SIZE, dummy> {
  // Init
  template <typename Tile>
  static __device__ __forceinline__ void Init(Tile *tile) {
    Iterate<LOAD + 1, 0>::Init(tile);
  }

  // Inspect
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {
    Iterate<LOAD + 1, 0>::Inspect(cta, tile);
  }

  // CTA Expand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void CtaExpand(Cta *cta, Tile *tile) {
    Iterate<LOAD + 1, 0>::CtaExpand(cta, tile);
  }

  // Warp Expand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile) {
    Iterate<LOAD + 1, 0>::WarpExpand(cta, tile);
  }

  // Single Thread Expand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile) {
    Iterate<LOAD + 1, 0>::ThreadExpand(cta, tile);
  }
};

/**
 * Terminate Iterate
 */
template <int dummy>
struct Iterate<LOADS_PER_TILE, 0, dummy> {
  // Init
  template <typename Tile>
  static __device__ __forceinline__ void Init(Tile *tile) {}

  // Inspect
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

  // CtaExpand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void CtaExpand(Cta *cta, Tile *tile) {}

  // WarpExpand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void WarpExpand(Cta *cta, Tile *tile) {}

  // SingleThreadExpand
  template <typename Cta, typename Tile>
  static __device__ __forceinline__ void ThreadExpand(Cta *cta, Tile *tile) {}
};

// Iterate Interface

// Constructor
__device__ __forceinline__ Tile() { Iterate<0, 0>::Init(this); }

// Inspect dequeued nodes
template <typename Cta>
__device__ __forceinline__ void Inspect(Cta *cta) {
  Iterate<0, 0>::Inspect(cta, this);
}

// CTA Expand
template <typename Cta>
__device__ __forceinline__ void CtaExpand(Cta *cta) {
  Iterate<0, 0>::CtaExpand(cta, this);
}

// Warp Expand
template <typename Cta>
__device__ __forceinline__ void WarpExpand(Cta *cta) {
  Iterate<0, 0>::WarpExpand(cta, this);
}

// Single Thread Expand
template <typename Cta>
__device__ __forceinline__ void ThreadExpand(Cta *cta) {
  Iterate<0, 0>::ThreadExpand(cta, this);
}
};  // namespace oprtr

// Methods

/**
 * @brief CTA default constructor
 */
__device__ __forceinline__ Cta(VertexId queue_index,
                               // int                         num_gpus,
                               SmemStorage &smem_storage, VertexId *d_queue,
                               VertexId *d_index, bool *d_bitmap_in,
                               bool *d_bitmap_out, SizeT *d_row_offsets,
                               VertexId *d_column_indices, DataSlice *problem,
                               util::CtaWorkProgress<SizeT> &work_progress,
                               gunrock::oprtr::advance::TYPE ADVANCE_TYPE)
    :

      queue_index(queue_index),
      // num_gpus(num_gpus),
      smem_storage(smem_storage),
      raking_soa_details(typename RakingSoaDetails::GridStorageSoa(
                             smem_storage.coarse_raking_elements,
                             smem_storage.fine_raking_elements),
                         typename RakingSoaDetails::WarpscanSoa(
                             smem_storage.state.coarse_warpscan,
                             smem_storage.state.fine_warpscan),
                         TileTuple(0, 0)),
      d_queue(d_queue),
      d_index(d_index),
      d_bitmap_in(d_bitmap_in),
      d_bitmap_out(d_bitmap_out),
      d_row_offsets(d_row_offsets),
      d_column_indices(d_column_indices),
      problem(problem),
      work_progress(work_progress),
      ADVANCE_TYPE(ADVANCE_TYPE) {
  if (threadIdx.x == 0) {
    smem_storage.state.cta_comm = KernelPolicy::THREADS;
    smem_storage.state.overflowed = false;
  }
}

/**
 * @brief Process a single, full tile.
 *
 * @param[in] cta_offset Offset within the CTA where we want to start the tile
 * processing.
 * @param[in] guarded_elements The guarded elements to prevent the
 * out-of-bound visit.
 */
__device__ __forceinline__ void ProcessTile(
    SizeT cta_offset, SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS) {
  Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

  // Load tile
  util::io::LoadTile<KernelPolicy::LOG_LOADS_PER_TILE,
                     KernelPolicy::LOG_LOAD_VEC_SIZE, KernelPolicy::THREADS,
                     ProblemData::QUEUE_READ_MODIFIER,
                     false>::LoadValid(tile.vertex_id, d_queue, cta_offset,
                                       guarded_elements, (VertexId)-1);

  util::io::LoadTile<KernelPolicy::LOG_LOADS_PER_TILE,
                     KernelPolicy::LOG_LOAD_VEC_SIZE, KernelPolicy::THREADS,
                     ProblemData::QUEUE_READ_MODIFIER,
                     false>::LoadValid(tile.vertex_idx, d_index, cta_offset,
                                       guarded_elements, (VertexId)-1);

  // Inspect dequeued nodes, updating label and obtaining
  // edge-list details
  tile.Inspect(this);

  // CooperativeSoaTileScan, put result in totals (done)
  SoaScanOp scan_op;
  TileTuple totals;
  gunrock::util::scan::soa::CooperativeSoaTileScan<
      KernelPolicy::LOAD_VEC_SIZE>::ScanTile(totals, raking_soa_details,
                                             RankSoa(tile.coarse_row_rank,
                                                     tile.fine_row_rank),
                                             scan_op);

  SizeT coarse_count = totals.t0;
  tile.fine_count = totals.t1;

  // TODO: cub scan to compute fine_row_rank and fine_count

  // Only one work queue, serves both as input and output.
  // So no need to set queue length for index+1 and check
  // overflow.

  if (coarse_count > 0) {
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
    int scratch_remainder =
        GR_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

    for (int scratch_offset = threadIdx.x; scratch_offset < scratch_remainder;
         scratch_offset += KernelPolicy::THREADS) {
      // Gather a incoming-neighbor
      VertexId parent_id;
      // parent_id = tex1Dfetch(ColumnIndicesTex<VertexId>::ref,
      // smem_storage.gather_offsets[scratch_offset]);

      util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
          parent_id,
          d_column_indices + smem_storage.gather_offsets[scratch_offset]);

      VertexId child_id = smem_storage.gather_predecessors[scratch_offset];
      bool bitmap_in;
      util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
          bitmap_in, d_bitmap_in + parent_id);
      if (bitmap_in) {
        if (Functor::CondEdge(parent_id, child_id, problem,
                              smem_storage.gather_offsets[scratch_offset]))
          Functor::ApplyEdge(parent_id, child_id, problem,
                             smem_storage.gather_offsets[scratch_offset]);
        child_id = -1;
      }

      if (child_id == -1) {
        // Mark the node as visited in  d_queue, so that we can cull it
        // during next vertex_map
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            (VertexId)-1,
            d_queue + smem_storage.gather_offsets2[scratch_offset]);
        // Set bitmap_out to true
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true,
            d_bitmap_out + smem_storage.gather_predecessors[scratch_offset]);
      }
    }

    tile.progress += SmemStorage::GATHER_ELEMENTS;

    __syncthreads();
  }
}
};  // namespace gunrock

}  // namespace edge_map_backward
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
