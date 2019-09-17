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

#include <gunrock/oprtr/advance/advance_base.cuh>

// TODO: use CUB for SOA scan

namespace gunrock {
namespace oprtr {
namespace TWC {

/**
 * @brief Tile of incoming frontier to process
 *
 * @tparam LOG_LOADS_PER_TILE   Size of the loads per tile.
 * @tparam LOG_LOAD_VEC_SIZE    Size of the vector size per load.
 */
// template<int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
template <typename _CtaT>
struct Tile {
  /**
   * Typedefs and Constants
   */

  typedef _CtaT CtaT;
  typedef typename CtaT::VertexT VertexT;
  typedef typename CtaT::SizeT SizeT;
  typedef typename CtaT::InKeyT InKeyT;
  typedef typename CtaT::OutKeyT OutKeyT;
  typedef typename CtaT::ValueT ValueT;
  typedef typename CtaT::KernelPolicyT KernelPolicyT;
  typedef typename KernelPolicyT::SmemStorage SmemStorage;
  typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;
  typedef Tile<CtaT> TileT;

  enum {
    FLAG = KernelPolicyT::FLAG,
    LOADS_PER_TILE = 1 << KernelPolicyT::LOG_LOADS_PER_TILE,
    LOAD_VEC_SIZE = 1 << KernelPolicyT::LOG_LOAD_VEC_SIZE
  };

  /**
   * @brief Iterate over vertex ids in tile.
   */
  template <
      // typename TileT,
      int LOAD, int VEC, int dummy = 0>
  struct Iterate {
    // typedef typename TileT::CtaT CtaT;

    /**
     * @brief Tile data initialization
     */
    // template <typename TileT>
    static __device__ __forceinline__ void Init(TileT *tile) {
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
    // template <typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void Inspect(CtaT *cta, TileT *tile) {
      // if (tile->vertex_id[LOAD][VEC] != -1) {
      if (util::isValid(tile->keys_in[LOAD][VEC])) {
        // Translate vertex-id into local gpu row-id (currently stride of
        // num_gpu)
        VertexT row_id = 0;  // / cta->num_gpus;
        // Load neighbor row range from d_row_offsets
        // Vec2SizeT   row_range;
        // SizeT       row_id1;
        // if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
        //    ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
        if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
          row_id = tile->keys_in[LOAD][VEC];
          // row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id);
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    row_range.x,
          //    cta->d_row_offsets + row_id);
          // row_range.x = graph.GetNeighborListOffset(row_id);
          // tile->row_offset[LOAD][VEC] =
          //    cta -> graph.GetNeighborListOffset(row_id);
          // row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id + 1);
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    row_range.y,
          //    cta->d_row_offsets + row_id+1);
          // tile->row_length[LOAD][VEC] =
          //    cta -> graph.GetNeighborListLength(row_id);
        }

        // if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
        //    ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
        if ((FLAG & OprtrType_E2V) != 0 || (FLAG & OprtrType_E2E) != 0) {
          // row_id1 = (cta->input_inverse_graph)
          //    ? cta -> d_inverse_column_indices[row_id]
          //    : cta -> d_column_indices[row_id];
          InKeyT edge_id = tile->keys_in[LOAD][VEC];
          row_id = cta->graph.GetEdgeDest(edge_id);
          // row_range.x = tex1Dfetch(cta->ts_rowoffset[0], row_id1);
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    row_range.x,
          //    cta->d_row_offsets + row_id1);
          // row_range.y = tex1Dfetch(cta->ts_rowoffset[0], row_id1+1);
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    row_range.y,
          //    cta->d_row_offsets + row_id1+1);
        }

        // compute row offset and length
        // tile->row_offset[LOAD][VEC] = row_range.x;
        // tile->row_length[LOAD][VEC] = row_range.y - row_range.x;
        tile->row_offset[LOAD][VEC] = cta->graph.GetNeighborListOffset(row_id);
        tile->row_length[LOAD][VEC] = cta->graph.GetNeighborListLength(row_id);
      }

      tile->fine_row_rank[LOAD][VEC] =
          (tile->row_length[LOAD][VEC] < KernelPolicyT::WARP_GATHER_THRESHOLD)
              ? tile->row_length[LOAD][VEC]
              : 0;

      tile->coarse_row_rank[LOAD][VEC] =
          (tile->row_length[LOAD][VEC] < KernelPolicyT::WARP_GATHER_THRESHOLD)
              ? 0
              : tile->row_length[LOAD][VEC];

      Iterate<LOAD, VEC + 1>::Inspect(cta, tile);

    }  // end of Inspect

    /**
     * @brief Expand the node's neighbor list using the whole CTA.
     * @tparam Cta CTA tile-processing abstraction type
     * @tparam Tile Tile structure type
     * @param[in] cta Pointer to CTA object
     * @param[in] tile Pointer to Tile object
     */
    // template <typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void CtaExpand(CtaT *cta, TileT *tile,
                                                     AdvanceOpT advance_op) {
      // CTA-based expansion/loading
      while (true) {
        // All threads in block vie for the control of the block
        if (tile->row_length[LOAD][VEC] >=
            KernelPolicyT::CTA_GATHER_THRESHOLD) {
          cta->smem_storage.state.cta_comm = threadIdx.x;
        }

        __syncthreads();

        // Check
        int owner = cta->smem_storage.state.cta_comm;
        if (owner == KernelPolicyT::THREADS) {
          // All threads in the block has less neighbor number for CTA Expand
          break;
        }

        __syncthreads();

        if (owner == threadIdx.x) {
          // Got control of the CTA: command it
          cta->smem_storage.state.warp_comm[0][0] =
              tile->row_offset[LOAD][VEC];  // start
          cta->smem_storage.state.warp_comm[0][1] =
              tile->coarse_row_rank[LOAD][VEC];  // queue rank
          cta->smem_storage.state.warp_comm[0][2] =
              tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];  // oob

          // if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
          //    ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
            cta->smem_storage.state.warp_comm[0][3] = tile->keys_in[LOAD][VEC];
          }

          // if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
          //    ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
          if ((FLAG & OprtrType_E2V) != 0 || (FLAG & OprtrType_E2E) != 0) {
            cta->smem_storage.state.warp_comm[0][3]
                //= cta -> input_inverse_graph
                //? cta -> d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                //: cta -> d_column_indices[tile->vertex_id[LOAD][VEC]];
                = cta->graph.GetEdgeDest(tile->keys_in[LOAD][VEC]);
          }
          cta->smem_storage.state.warp_comm[0][4] = tile->keys_in[LOAD][VEC];

          // Unset row length
          tile->row_length[LOAD][VEC] = 0;

          // Unset my command
          cta->smem_storage.state.cta_comm = KernelPolicyT::THREADS;
          // So that we won't repeatedly expand this node
        }
        __syncthreads();

        // Read commands
        SizeT coop_offset = cta->smem_storage.state.warp_comm[0][0];
        SizeT coop_rank = cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
        SizeT coop_oob = cta->smem_storage.state.warp_comm[0][2];

        VertexT pred_id;
        VertexT input_item = cta->smem_storage.state.warp_comm[0][4];
        // if (Problem::MARK_PREDECESSORS)
        pred_id = cta->smem_storage.state.warp_comm[0][3];
        // else
        //    pred_id = util::InvalidValue<VertexId>();//cta->label;

        //__syncthreads();

        while (coop_offset + threadIdx.x < coop_oob) {
          // Gather
          VertexT neighbor_id;
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    neighbor_id,
          //    cta->d_column_indices + coop_offset + threadIdx.x);
          SizeT edge_id = coop_offset + threadIdx.x;
          neighbor_id = cta->graph.GetEdgeDest(edge_id);

          // ProcessNeighbor
          //    <KernelPolicy, Problem, Functor,
          //    ADVANCE_TYPE, R_TYPE, R_OP> (
          //    pred_id,
          //    neighbor_id,
          //    cta -> d_data_slice,
          //    (SizeT)(coop_offset + threadIdx.x),
          //    util::InvalidValue<SizeT>(), // input_pos
          //    input_item,
          //    cta -> smem_storage.state.coarse_enqueue_offset + coop_rank,
          //    cta -> label,
          //    cta -> d_keys_out,
          //    cta -> d_values_out,
          //    cta -> d_value_to_reduce,
          //    cta -> d_reduce_frontier);
          // ProcessNeighbor(
          //    cta, tile,
          //    coop_offset + threadIdx.x,
          //    pred_id, edge_id,
          //    cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

          SizeT output_pos =
              cta->smem_storage.state.coarse_enqueue_offset + coop_rank;
          ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
              pred_id, neighbor_id, edge_id,
              util::PreDefinedValues<SizeT>::InvalidValue, input_item,
              output_pos, cta->keys_out, cta->values_out, NULL,
              NULL,  // cta -> reduce_values_in, cta -> reduce_values_out,
              advance_op);

          coop_offset += KernelPolicyT::THREADS;
          coop_rank += KernelPolicyT::THREADS;
        }
      }  // end of while (true)
      __syncthreads();

      // Next vector element
      Iterate<LOAD, VEC + 1>::CtaExpand(cta, tile, advance_op);
    }  // end of CtaExpand

    /**
     * @brief Expand the node's neighbor list using a warp. (Currently disabled
     * in the enactor)
     * @tparam Cta CTA tile-processing abstraction type
     * @tparam Tile Tile structure type
     * @param[in] cta Pointer to CTA object
     * @param[in] tile Pointer to Tile object
     */
    // template<typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void WarpExpand(CtaT *cta, TileT *tile,
                                                      AdvanceOpT advance_op) {
      if (KernelPolicyT::WARP_GATHER_THRESHOLD <
          KernelPolicyT::CTA_GATHER_THRESHOLD) {
        // Warp-based expansion/loading
        int warp_id = threadIdx.x >> GR_LOG_WARP_THREADS(CUDA_ARCH);
        int lane_id = util::LaneId();

        while (::_any(tile->row_length[LOAD][VEC] >=
                      KernelPolicyT::WARP_GATHER_THRESHOLD)) {
          if (tile->row_length[LOAD][VEC] >=
              KernelPolicyT::WARP_GATHER_THRESHOLD) {
            // All threads inside one warp vie for control of the warp
            cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
          }

          if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {
            // Got control of the warp
            cta->smem_storage.state.warp_comm[warp_id][0] =
                tile->row_offset[LOAD][VEC];  // start
            cta->smem_storage.state.warp_comm[warp_id][1] =
                tile->coarse_row_rank[LOAD][VEC];  // queue rank
            cta->smem_storage.state.warp_comm[warp_id][2] =
                tile->row_offset[LOAD][VEC] +
                tile->row_length[LOAD][VEC];  // oob
            // if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
            //    ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
            if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
              cta->smem_storage.state.warp_comm[warp_id][3] =
                  tile->keys_in[LOAD][VEC];
            }

            // if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
            //    ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
            if ((FLAG & OprtrType_E2V) != 0 || (FLAG & OprtrType_E2E) != 0) {
              cta->smem_storage.state.warp_comm[warp_id][3]
                  //= cta -> input_inverse_graph
                  //? cta ->
                  //d_inverse_column_indices[tile->vertex_id[LOAD][VEC]] : cta
                  //-> d_column_indices[tile->vertex_id[LOAD][VEC]];
                  = cta->graph.GetEdgeDest(tile->keys_in[LOAD][VEC]);
            }
            cta->smem_storage.state.warp_comm[warp_id][4] =
                tile->keys_in[LOAD][VEC];
            // Unset row length
            tile->row_length[LOAD][VEC] =
                0;  // So that we won't repeatedly expand this node
          }

          SizeT coop_offset = cta->smem_storage.state.warp_comm[warp_id][0];
          SizeT coop_rank =
              cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
          SizeT coop_oob = cta->smem_storage.state.warp_comm[warp_id][2];

          VertexT pred_id;
          VertexT input_item = cta->smem_storage.state.warp_comm[warp_id][4];
          // if (Problem::MARK_PREDECESSORS)
          pred_id = cta->smem_storage.state.warp_comm[warp_id][3];
          // else
          //    pred_id = util::InvalidValue<VertexT>();//cta->label;

          while (coop_offset + lane_id < coop_oob) {
            VertexT neighbor_id;
            // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            //    neighbor_id,
            //    cta->d_column_indices + coop_offset + lane_id);
            neighbor_id = cta->graph.GetEdgeDest(coop_offset + lane_id);

            // ProcessNeighbor
            //    <KernelPolicy, Problem, Functor,
            //    ADVANCE_TYPE, R_TYPE, R_OP> (
            //    pred_id,
            //    neighbor_id,
            //    cta -> d_data_slice,
            //    coop_offset + lane_id,
            //    util::InvalidValue<SizeT>(), // input_pos
            //    input_item,
            //    cta -> smem_storage.state.coarse_enqueue_offset + coop_rank,
            //    cta -> label,
            //    cta -> d_keys_out,
            //    cta -> d_values_out,
            //    cta -> d_value_to_reduce,
            //    cta -> d_reduce_frontier);
            // ProcessNeighbor(
            //    cta, tile,
            //    coop_offset + lane_id,
            //    pred_id, edge_id,
            //    cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
            SizeT output_pos =
                cta->smem_storage.state.coarse_enqueue_offset + coop_rank;
            ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
                pred_id, neighbor_id, coop_offset + lane_id,
                util::PreDefinedValues<SizeT>::InvalidValue, input_item,
                output_pos, cta->keys_out, cta->values_out, NULL,
                NULL,  // cta -> reduce_values_in, cta -> reduce_values_out,
                advance_op);

            coop_offset += GR_WARP_THREADS(CUDA_ARCH);
            coop_rank += GR_WARP_THREADS(CUDA_ARCH);
          }
        }

        // Next vector element
        Iterate<LOAD, VEC + 1>::WarpExpand(cta, tile, advance_op);
      }
    }  // end of WarpExpand

    /**
     * @brief Expand the node's neighbor list using a single thread (Scan).
     * @tparam Cta CTA tile-processing abstraction type
     * @tparam Tile Tile structure type
     * @param[in] cta Pointer to CTA object
     * @param[in] tile Pointer to Tile object
     */
    // template <typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void ThreadExpand(CtaT *cta,
                                                        TileT *tile) {
      // Expand the neighbor list into scratch space
      SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] +
                             tile->row_progress[LOAD][VEC] - tile->progress;

      while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
             (scratch_offset < SmemStorage::GATHER_ELEMENTS)) {
        // Put gather offset into scratch space
        cta->smem_storage.gather_offsets[scratch_offset] =
            tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];
        cta->smem_storage.gather_edges[scratch_offset] =
            tile->keys_in[LOAD][VEC];
        // if (Problem::MARK_PREDECESSORS)
        // if ((FLAG & OprtrOption_Mark_Predecessors) != 0)
        {
          // if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
          //    ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
          if ((FLAG & OprtrType_E2V) != 0 || (FLAG & OprtrType_E2E) != 0) {
            cta->smem_storage.gather_predecessors[scratch_offset]
                //= cta -> input_inverse_graph
                //? cta -> d_inverse_column_indices[tile->vertex_id[LOAD][VEC]]
                //: cta -> d_column_indices[tile->vertex_id[LOAD][VEC]];
                = cta->graph.GetEdgeDest(tile->keys_in[LOAD][VEC]);
            cta->smem_storage.gather_edges[scratch_offset] =
                tile->keys_in[LOAD][VEC];
          }

          // if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
          //    ADVANCE_TYPE == gunrock::oprtr::advance::V2E)
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0)
            cta->smem_storage.gather_predecessors[scratch_offset] =
                tile->keys_in[LOAD][VEC];
        }

        tile->row_progress[LOAD][VEC]++;
        scratch_offset++;
      }

      // Next vector element
      Iterate<LOAD, VEC + 1>::ThreadExpand(cta, tile);
    }
  };  // end of struct Iterate

  /**
   * Iterate next load
   */
  template <
      // typename TileT,
      int LOAD, int dummy>
  struct Iterate<LOAD, LOAD_VEC_SIZE, dummy> {
    // typedef typename TileT::CtaT CtaT;

    // Init
    // template <typename TileT>
    static __device__ __forceinline__ void Init(TileT *tile) {
      Iterate<LOAD + 1, 0>::Init(tile);
    }

    // Inspect
    // template <typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void Inspect(CtaT *cta, TileT *tile) {
      Iterate<LOAD + 1, 0>::Inspect(cta, tile);
    }

    // CTA Expand
    // template <typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void CtaExpand(CtaT *cta, TileT *tile,
                                                     AdvanceOpT advance_op) {
      Iterate<LOAD + 1, 0>::CtaExpand(cta, tile, advance_op);
    }

    // Warp Expand
    // template <typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void WarpExpand(CtaT *cta, TileT *tile,
                                                      AdvanceOpT advance_op) {
      Iterate<LOAD + 1, 0>::WarpExpand(cta, tile, advance_op);
    }

    // Single Thread Expand
    // template <typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void ThreadExpand(CtaT *cta,
                                                        TileT *tile) {
      Iterate<LOAD + 1, 0>::ThreadExpand(cta, tile);
    }
  };

  /**
   * Terminate Iterate
   */
  template <int dummy>
  struct Iterate<LOADS_PER_TILE, 0, dummy> {
    // typedef typename TileT::CtaT CtaT;
    // Init
    // template <typename TileT>
    static __device__ __forceinline__ void Init(TileT *tile) {}

    // Inspect
    // template <typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void Inspect(CtaT *cta, TileT *tile) {}

    // CtaExpand
    // template<typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void CtaExpand(CtaT *cta, TileT *tile,
                                                     AdvanceOpT advance_op) {}

    // WarpExpand
    // template<typename CtaT>//, typename TileT>
    template <typename AdvanceOpT>
    static __device__ __forceinline__ void WarpExpand(CtaT *cta, TileT *tile,
                                                      AdvanceOpT advance_op) {}

    // SingleThreadExpand
    // template<typename CtaT>//, typename TileT>
    static __device__ __forceinline__ void ThreadExpand(CtaT *cta,
                                                        TileT *tile) {}
  };

  /**
   * Members
   */

  // Dequeued vertex ids
  InKeyT keys_in[LOADS_PER_TILE][LOAD_VEC_SIZE];

  SizeT row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
  SizeT row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

  // Global scatter offsets. Coarse for CTA/warp-based scatters, fine for
  // scan-based scatters
  SizeT fine_count;
  SizeT coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
  SizeT fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

  // Progress for scan-based forward edge map gather offsets
  SizeT row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
  SizeT progress;

  // Iterate Interface

  // Constructor
  __device__ __forceinline__ Tile() { Iterate<0, 0>::Init(this); }

  // Inspect dequeued nodes
  // template <typename CtaT>
  __device__ __forceinline__ void Inspect(CtaT *cta) {
    Iterate<0, 0>::Inspect(cta, this);
  }

  // CTA Expand
  // template <typename CtaT>
  template <typename AdvanceOpT>
  __device__ __forceinline__ void CtaExpand(CtaT *cta, AdvanceOpT advance_op) {
    Iterate<0, 0>::CtaExpand(cta, this, advance_op);
  }

  // Warp Expand
  // template <typename CtaT>
  template <typename AdvanceOpT>
  __device__ __forceinline__ void WarpExpand(CtaT *cta, AdvanceOpT advance_op) {
    Iterate<0, 0>::WarpExpand(cta, this, advance_op);
  }

  // Single Thread Expand
  // template <typename CtaT>
  __device__ __forceinline__ void ThreadExpand(CtaT *cta) {
    Iterate<0, 0>::ThreadExpand(cta, this);
  }
};  // end of struct Tile

/**
 * @brief CTA tile-processing abstraction for the vertex mapping operator.
 *
 * @tparam KernelPolicy Kernel policy type for the vertex mapping.
 * @tparam ProblemData Problem data type for the vertex mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 */
template <typename _GraphT, typename _KernelPolicyT>
struct Cta {
  /**
   * Typedefs
   */
  typedef _GraphT GraphT;
  typedef _KernelPolicyT KernelPolicyT;
  typedef typename KernelPolicyT::VertexT VertexT;
  typedef typename KernelPolicyT::InKeyT InKeyT;
  typedef typename KernelPolicyT::OutKeyT OutKeyT;
  typedef typename KernelPolicyT::SizeT SizeT;
  typedef typename KernelPolicyT::ValueT ValueT;

  typedef typename KernelPolicyT::SmemStorage SmemStorage;
  typedef typename KernelPolicyT::SoaScanOp SoaScanOp;
  typedef typename KernelPolicyT::RakingSoaDetails RakingSoaDetails;
  typedef typename KernelPolicyT::TileTuple TileTuple;

  typedef util::Tuple<SizeT (*)[KernelPolicyT::LOAD_VEC_SIZE],
                      SizeT (*)[KernelPolicyT::LOAD_VEC_SIZE]>
      RankSoa;

  typedef Cta<GraphT, KernelPolicyT> CtaT;
  typedef Tile<CtaT> TileT;

  /**
   * Members
   */

  // Graph
  const GraphT &graph;

  // Input and output device pointers
  const InKeyT *&keys_in;  // Incoming frontier
  ValueT *&values_out;
  OutKeyT *&keys_out;  // Outgoing frontier

  // Work progress
  const VertexT &queue_index;  // Current frontier queue counter index
  util::CtaWorkProgress<SizeT> &work_progress;  // Atomic queueing counters
  // SizeT                   max_out_frontier;           // Maximum size (in
  // elements) of outgoing frontier LabelT                  label; // Current
  // label of the frontier
  const SizeT &input_queue_length;
  // gunrock::oprtr::advance::TYPE           advance_type;
  // bool                    input_inverse_graph;
  // gunrock::oprtr::advance::REDUCE_TYPE    r_type;
  // gunrock::oprtr::advance::REDUCE_OP      r_op;
  // Value                  *d_value_to_reduce;
  // const ValueT           *&reduce_values_in;
  // ValueT           *&reduce_values_out;
  // Value                  *d_reduce_frontier;

  // Operational details for raking grid
  RakingSoaDetails raking_soa_details;

  // Shared memory for the CTA
  SmemStorage &smem_storage;

  // Methods
  /**
   * @brief CTA default constructor
   */
  __device__ __forceinline__
  Cta(const GraphT &graph,
      // bool                          queue_reset,
      // LabelT                        label,
      // SizeT                        *d_row_offsets,
      // SizeT                        *d_inverse_row_offsets,
      // VertexT                     *d_column_indices,
      // VertexT                     *d_inverse_column_indices,
      const InKeyT *&keys_in, const SizeT &input_queue_length,
      OutKeyT *&keys_out, ValueT *&values_out, const VertexT &queue_index,
      // DataSlice                    *d_data_slice,

      // SizeT                         max_in_frontier,
      // SizeT                         max_out_frontier,
      // const ValueT                      *&reduce_values_in,
      // ValueT                      *&reduce_values_out,
      util::CtaWorkProgress<SizeT> &work_progress, SmemStorage &smem_storage)
      :  // gunrock::oprtr::advance::TYPE ADVANCE_TYPE,
         // bool                          input_inverse_graph,
         // gunrock::oprtr::advance::REDUCE_TYPE    R_TYPE,
         // gunrock::oprtr::advance::REDUCE_OP      R_OP,
         // Value                        *d_value_to_reduce,
         // Value                        *d_reduce_frontier) :

        // queue_reset             (queue_reset),
        graph(graph),
        queue_index(queue_index),
        // label                   (label),
        // d_row_offsets           (d_row_offsets),
        // d_inverse_row_offsets   (d_inverse_row_offsets),
        // d_column_indices        (d_column_indices),
        // d_inverse_column_indices(d_inverse_column_indices),
        keys_in(keys_in),
        keys_out(keys_out),
        values_out(values_out),
        // d_data_slice            (d_data_slice),
        input_queue_length(input_queue_length),
        // max_out_frontier        (max_out_frontier),
        // reduce_values_in        (reduce_values_in),
        // reduce_values_out       (reduce_values_out),
        work_progress(work_progress),
        smem_storage(smem_storage),
        // input_inverse_graph           (input_inverse_graph),
        // d_value_to_reduce       (d_value_to_reduce),
        // d_reduce_frontier       (d_reduce_frontier),
        raking_soa_details(typename RakingSoaDetails::GridStorageSoa(
                               smem_storage.coarse_raking_elements,
                               smem_storage.fine_raking_elements),
                           typename RakingSoaDetails::WarpscanSoa(
                               smem_storage.state.coarse_warpscan,
                               smem_storage.state.fine_warpscan),
                           TileTuple(0, 0))
  // advance_type(ADVANCE_TYPE),
  // r_type(R_TYPE),
  // r_op(R_OP),
  {
    if (threadIdx.x == 0) {
      smem_storage.state.cta_comm = KernelPolicyT::THREADS;
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
  template <typename AdvanceOpT>
  __device__ __forceinline__ void ProcessTile(
      SizeT cta_offset,
      SizeT guarded_elements,  // = KernelPolicyT::TILE_ELEMENTS)
      AdvanceOpT advance_op) {
    TileT tile;

    // Load tile
    util::io::LoadTile<
        KernelPolicyT::LOG_LOADS_PER_TILE, KernelPolicyT::LOG_LOAD_VEC_SIZE,
        KernelPolicyT::THREADS, QUEUE_READ_MODIFIER,
        false>::LoadValid(tile.keys_in, const_cast<InKeyT *>(keys_in),
                          cta_offset, guarded_elements,
                          util::PreDefinedValues<InKeyT>::InvalidValue);

    // Inspect dequeued nodes, updating label and obtaining
    // edge-list details
    tile.Inspect(this);

    // CooperativeSoaTileScan, put result in totals (done)
    SoaScanOp scan_op;
    TileTuple totals;
    gunrock::util::scan::soa::CooperativeSoaTileScan<
        KernelPolicyT::LOAD_VEC_SIZE>::ScanTile(totals, raking_soa_details,
                                                RankSoa(tile.coarse_row_rank,
                                                        tile.fine_row_rank),
                                                scan_op);

    SizeT coarse_count = totals.t0;
    tile.fine_count = totals.t1;

    // Set input queue length and check for overflow
    if (threadIdx.x == 0) {
      SizeT enqueue_amt = coarse_count + tile.fine_count;
      SizeT enqueue_offset =
          work_progress.Enqueue(enqueue_amt, queue_index + 1);

      // printf("(%4d, %4d) outputs = %lld + %lld = %lld, offset = %lld\n",
      //    blockIdx.x, threadIdx.x,
      //    coarse_count, tile.fine_count,
      //    enqueue_amt, enqueue_offset);
      smem_storage.state.coarse_enqueue_offset = enqueue_offset;
      smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;

      // Check for queue overflow due to redundant expansion
      // if (enqueue_offset + enqueue_amt > max_out_frontier)
      //{
      //    smem_storage.state.overflowed = true;
      //    work_progress.SetOverflow();
      //}
    }

    // Protect overflowed flag
    __syncthreads();

    // Quit if overflow
    // if (smem_storage.state.overflowed) {
    //    util::ThreadExit();
    //}

    if (coarse_count > 0) {
      // Enqueue valid edge lists into outgoing queue by CTA
      tile.CtaExpand(this, advance_op);

      // Enqueue valid edge lists into outgoing queue by Warp
      tile.WarpExpand(this, advance_op);
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
      SizeT scratch_remainder =
          GR_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

      for (SizeT scratch_offset = threadIdx.x;
           scratch_offset < scratch_remainder;
           scratch_offset += KernelPolicyT::THREADS) {
        // Gather a neighbor
        VertexT neighbor_id;
        SizeT edge_id = smem_storage.gather_offsets[scratch_offset];
        // neighbor_id = tex1Dfetch(ts_columnindices[0],
        // smem_storage.gather_offsets[scratch_offset]);
        // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //        neighbor_id,
        //        d_column_indices + edge_id);//
        //        smem_storage.gather_offsets[scratch_offset]);
        neighbor_id = graph.GetEdgeDest(edge_id);

        VertexT predecessor_id;
        // if (Problem::MARK_PREDECESSORS)
        // if ((KernelPolicyT::FLAG & OprtrOption_Mark_Predecessors) != 0)
        predecessor_id = smem_storage.gather_predecessors[scratch_offset];
        // else
        //    predecessor_id =
        //    util::PreDefinedValues<VertexT>::InvalidValue;//label;

        // if Cond(neighbor_id) returns true
        // if Cond(neighbor_id) returns false or Apply returns false
        // set neighbor_id to -1 for invalid
        VertexT input_item = smem_storage.gather_edges[scratch_offset];

        // ProcessNeighbor
        //    <KernelPolicy, Problem, Functor,
        //    ADVANCE_TYPE, R_TYPE, R_OP> (
        //    predecessor_id,
        //    neighbor_id,
        //    d_data_slice,
        //    edge_id,
        //    util::InvalidValue<SizeT>(), // input_pos
        //    input_item,
        //    smem_storage.state.fine_enqueue_offset + tile.progress +
        //    scratch_offset, label, d_keys_out, d_values_out,
        //    d_value_to_reduce,
        //    d_reduce_frontier);
        SizeT output_pos = smem_storage.state.fine_enqueue_offset +
                           tile.progress + scratch_offset;
        // printf("(%4d, %4d) output_pos = %lld + %lld + %lld = %lld\n",
        //    blockIdx.x, threadIdx.x,
        //    smem_storage.state.fine_enqueue_offset,
        //    tile.progress, scratch_offset, output_pos);
        ProcessNeighbor<KernelPolicyT::FLAG, VertexT, InKeyT, OutKeyT, SizeT,
                        ValueT>(predecessor_id, neighbor_id, edge_id,
                                util::PreDefinedValues<SizeT>::InvalidValue,
                                input_item, output_pos, keys_out, values_out,
                                NULL,
                                NULL,  // reduce_values_in, reduce_values_out,
                                advance_op);
      }

      tile.progress += SmemStorage::GATHER_ELEMENTS;

      __syncthreads();
    }
  }
};  // struct cta

}  // namespace TWC
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
