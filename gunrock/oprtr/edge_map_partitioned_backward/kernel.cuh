// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Load balanced Edge Map Kernel Entry point
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned_backward {

// GetRowOffsets
//
// RelaxPartitionedEdges

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID
 */
template <typename KernelPolicy, typename ProblemData, typename Functor,
          bool VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch {};

template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true> {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename KernelPolicy::Value Value;
  typedef typename ProblemData::DataSlice DataSlice;
  typedef typename Functor::LabelT LabelT;

  static __device__ __forceinline__ SizeT GetNeighborListLength(
      SizeT *&d_row_offsets, VertexId *&d_column_indices, VertexId &d_vertex_id,
      SizeT &max_vertex, SizeT &max_edge,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE) {
    if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
        ADVANCE_TYPE == gunrock::oprtr::advance::E2E) {
      d_vertex_id = d_column_indices[d_vertex_id];
    }
    SizeT first =
        d_vertex_id >= max_vertex ? max_edge : d_row_offsets[d_vertex_id];
    SizeT second = (d_vertex_id + 1) >= max_vertex
                       ? max_edge
                       : d_row_offsets[d_vertex_id + 1];

    return (second > first) ? second - first : 0;
  }

  static __device__ __forceinline__ void GetEdgeCounts(
      SizeT *&d_row_offsets, VertexId *&d_column_indices, VertexId *&d_queue,
      SizeT *&d_scanned_edges, SizeT &num_elements, SizeT &max_vertex,
      SizeT &max_edge, gunrock::oprtr::advance::TYPE &ADVANCE_TYPE) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int my_id = bid * blockDim.x + tid;

    if (my_id >= num_elements || my_id >= max_edge) return;
    VertexId v_id = d_queue[my_id];
    SizeT num_edges =
        GetNeighborListLength(d_row_offsets, d_column_indices, v_id, max_vertex,
                              max_edge, ADVANCE_TYPE);
    d_scanned_edges[my_id] = num_edges;
  }

  static __device__ __forceinline__ void RelaxPartitionedEdges(
      bool &queue_reset, VertexId &queue_index, int &label,
      SizeT *&d_row_offsets, VertexId *&d_column_indices,
      VertexId *&d_inverse_column_indices, SizeT *&d_scanned_edges,
      unsigned int *&partition_starts, unsigned int &num_partitions,
      // volatile int *&d_done,
      VertexId *&d_queue, bool *&d_bitmap_in, bool *&d_bitmap_out,
      DataSlice *&problem, SizeT &input_queue_len, SizeT *output_queue_len,
      SizeT &partition_size, SizeT &max_vertices, SizeT &max_edges,
      util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE, bool &inverse_graph) {
    // if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0 && blockIdx.x == 0)) {
    //    kernel_stats.MarkStart();
    //}

    // Reset work progress
    // if (queue_reset)
    //{
    //    if (blockIdx.x == 0 && threadIdx.x < util::CtaWorkProgress::COUNTERS)
    //    {
    // Reset all counters
    //        work_progress.template Reset<SizeT>();
    //    }
    //}

    // Determine work decomposition
    if (threadIdx.x == 0) {
      if (!queue_reset)
        input_queue_len = work_progress.LoadQueueLength(queue_index);

      if (blockIdx.x == 0) {
        // obtain problem size
        if (queue_reset) {
          work_progress.StoreQueueLength(input_queue_len, queue_index);
        }

        work_progress.Enqueue(output_queue_len[0], queue_index + 1);

        // Reset our next outgoing queue counter to zero
        work_progress.StoreQueueLength(0, queue_index + 2);
        work_progress.PrepResetSteal(queue_index + 1);
      }
    }

    // Barrier to protect work decomposition
    __syncthreads();

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int my_thread_start, my_thread_end;

    my_thread_start = bid * partition_size;
    my_thread_end = (bid + 1) * partition_size < output_queue_len[0]
                        ? (bid + 1) * partition_size
                        : output_queue_len[0];
    // printf("tid:%d, bid:%d, m_thread_start:%d, m_thread_end:%d\n",tid, bid,
    // my_thread_start, my_thread_end);

    if (my_thread_start >= output_queue_len[0]) return;

    int my_start_partition = partition_starts[bid];
    int my_end_partition = partition_starts[bid + 1] > input_queue_len
                               ? partition_starts[bid + 1]
                               : input_queue_len;
    // if (tid == 0 && bid == 252)
    //    printf("bid(%d) < num_partitions-1(%d)?,
    //    partition_starts[bid+1]+1:%d\n", bid, num_partitions-1,
    //    partition_starts[bid+1]+1);

    __shared__ typename KernelPolicy::SmemStorage smem_storage;
    // smem_storage.s_edges[NT]
    // smem_storage.s_vertices[NT]
    unsigned int *s_edges = (unsigned int *)&smem_storage.s_edges[0];
    unsigned int *s_vertices = (unsigned int *)&smem_storage.s_vertices[0];
    unsigned int *s_edge_ids = (unsigned int *)&smem_storage.s_edge_ids[0];

    int my_work_size = my_thread_end - my_thread_start;
    int out_offset = bid * partition_size;
    int pre_offset =
        my_start_partition > 0 ? d_scanned_edges[my_start_partition - 1] : 0;
    int e_offset = my_thread_start - pre_offset;
    int edges_processed = 0;

    while (edges_processed < my_work_size &&
           my_start_partition < my_end_partition) {
      pre_offset =
          my_start_partition > 0 ? d_scanned_edges[my_start_partition - 1] : 0;

      __syncthreads();

      s_edges[tid] =
          (my_start_partition + tid < my_end_partition
               ? d_scanned_edges[my_start_partition + tid] - pre_offset
               : max_edges);
      // if (bid == 252 && tid == 2)
      //    printf("start_partition+tid:%d < my_end_partition:%d ?,
      //    d_queue[%d]:%d\n", my_start_partition+tid, my_end_partition,
      //    my_start_partition+tid, d_queue[my_start_partition+tid]);
      if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
          ADVANCE_TYPE == gunrock::oprtr::advance::V2E) {
        s_vertices[tid] = my_start_partition + tid < my_end_partition
                              ? d_queue[my_start_partition + tid]
                              : -1;
        s_edge_ids[tid] = 0;
      }
      if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
          ADVANCE_TYPE == gunrock::oprtr::advance::E2E) {
        if (inverse_graph)
          s_vertices[tid] =
              my_start_partition + tid < my_end_partition
                  ? d_inverse_column_indices[d_queue[my_start_partition + tid]]
                  : -1;
        else
          s_vertices[tid] =
              my_start_partition + tid < my_end_partition
                  ? d_column_indices[d_queue[my_start_partition + tid]]
                  : -1;
        s_edge_ids[tid] = my_start_partition + tid < my_end_partition
                              ? d_queue[my_start_partition + tid]
                              : -1;
      }

      int last = my_start_partition + KernelPolicy::THREADS >= my_end_partition
                     ? my_end_partition - my_start_partition - 1
                     : KernelPolicy::THREADS - 1;

      __syncthreads();

      SizeT e_last =
          min(s_edges[last] - e_offset, my_work_size - edges_processed);
      SizeT v_index =
          util::BinarySearch<KernelPolicy::THREADS>(tid + e_offset, s_edges);
      VertexId v = s_vertices[v_index];
      VertexId e_id = s_edge_ids[v_index];
      SizeT end_last =
          (v_index < my_end_partition ? s_edges[v_index] : max_edges);
      SizeT internal_offset = v_index > 0 ? s_edges[v_index - 1] : 0;
      SizeT lookup_offset = d_row_offsets[v];

      for (int i = (tid + e_offset); i < e_last + e_offset;
           i += KernelPolicy::THREADS) {
        if (i >= end_last) {
          v_index = util::BinarySearch<KernelPolicy::THREADS>(i, s_edges);
          if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V ||
              ADVANCE_TYPE == gunrock::oprtr::advance::V2E) {
            v = d_queue[v_index];
            e_id = 0;
          }
          if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
              ADVANCE_TYPE == gunrock::oprtr::advance::E2E) {
            v = inverse_graph ? d_inverse_column_indices[d_queue[v_index]]
                              : d_column_indices[d_queue[v_index]];
            e_id = d_queue[v_index];
          }
          end_last =
              (v_index < KernelPolicy::THREADS ? s_edges[v_index] : max_edges);
          internal_offset = v_index > 0 ? s_edges[v_index - 1] : 0;
          lookup_offset = d_row_offsets[v];
        }

        int e = i - internal_offset;
        int lookup = lookup_offset + e;
        VertexId u = d_column_indices[lookup];
        SizeT out_index = out_offset + edges_processed + (i - e_offset);

        /*{
            if (!ProblemData::MARK_PREDECESSORS) {
                if (Functor::CondEdge(label, u, problem, lookup, e_id)) {
                    Functor::ApplyEdge(label, u, problem, lookup, e_id);
                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V) {
                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                u,
                                d_out + out_index);
                    } else if (ADVANCE_TYPE == gunrock::oprtr::advance::V2E
                             ||ADVANCE_TYPE == gunrock::oprtr::advance::E2E) {
                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                (VertexId)lookup,
                                d_out + out_index);
                    }
                }
                else {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                            -1,
                            d_out + out_index);
                }
            } else {
                if (Functor::CondEdge(v, u, problem, lookup, e_id)) {
                    Functor::ApplyEdge(v, u, problem, lookup, e_id);
                    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V) {
                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                u,
                                d_out + out_index);
                    } else if (ADVANCE_TYPE == gunrock::oprtr::advance::V2E
                             ||ADVANCE_TYPE == gunrock::oprtr::advance::E2E) {
                        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                                (VertexId)lookup,
                                d_out + out_index);
                    }
                }
                else {
                    util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                            -1,
                            d_out + out_index);
                }
            }
        }*/
      }
      edges_processed += e_last;
      my_start_partition += KernelPolicy::THREADS;
      e_offset = 0;
    }

    // if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0 && threadIdx.x == 0)) {
    //    kernel_stats.MarkStop();
    //    kernel_stats.Flush();
    //}
  }

  static __device__ __forceinline__ void RelaxLightEdges(
      bool &queue_reset, VertexId &queue_index, int &label,
      SizeT *&d_row_offsets, VertexId *&d_column_indices,
      VertexId *&d_inverse_column_indices, SizeT *&d_scanned_edges,
      // volatile int *&d_done,
      VertexId *&d_queue, bool *&d_bitmap_in, bool *&d_bitmap_out,
      DataSlice *&problem, SizeT &input_queue_len, SizeT *output_queue_len,
      SizeT &max_vertices, SizeT &max_edges,
      util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE, bool inverse_graph) {
    // if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0 && threadIdx.x == 0)) {
    //    kernel_stats.MarkStart();
    //}

    // Reset work progress
    // if (queue_reset)
    //{
    //    if (blockIdx.x == 0 && threadIdx.x < util::CtaWorkProgress::COUNTERS)
    //    {
    // Reset all counters
    //        work_progress.template Reset<SizeT>();
    //    }
    //}

    // Determine work decomposition
    if (threadIdx.x == 0) {
      // obtain problem size
      if (!queue_reset)
        input_queue_len = work_progress.LoadQueueLength(queue_index);

      if (blockIdx.x == 0) {
        if (queue_reset) {
          work_progress.StoreQueueLength(input_queue_len, queue_index);
        }
        work_progress.Enqueue(output_queue_len[0], queue_index + 1);

        // Reset our next outgoing queue counter to zero
        work_progress.StoreQueueLength(0, queue_index + 2);
        work_progress.PrepResetSteal(queue_index + 1);
      }
    }

    // Barrier to protect work decomposition
    __syncthreads();

    unsigned int range = input_queue_len;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = bid * KernelPolicy::THREADS + tid;

    __shared__ typename KernelPolicy::SmemStorage smem_storage;
    unsigned int *s_edges = (unsigned int *)&smem_storage.s_edges[0];
    unsigned int *s_vertices = (unsigned int *)&smem_storage.s_vertices[0];
    unsigned int *s_edge_ids = (unsigned int *)&smem_storage.s_edge_ids[0];

    int offset = (KernelPolicy::THREADS * bid - 1) > 0
                     ? d_scanned_edges[KernelPolicy::THREADS * bid - 1]
                     : 0;
    int end_id = (KernelPolicy::THREADS * (bid + 1)) >= range
                     ? range - 1
                     : KernelPolicy::THREADS * (bid + 1) - 1;

    end_id = end_id % KernelPolicy::THREADS;
    s_edges[tid] =
        (my_id < range ? d_scanned_edges[my_id] - offset : max_edges);

    if (ADVANCE_TYPE == gunrock::oprtr::advance::V2V) {
      s_vertices[tid] = (my_id < range ? d_queue[my_id] : max_vertices);
      s_edge_ids[tid] = my_id;  // used as input index
    }
    // do not support E2V and E2E for backward BFS now
    /*if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V || ADVANCE_TYPE ==
    gunrock::oprtr::advance::E2E) { if (inverse_graph) s_vertices[tid] = (my_id
    < range ? d_inverse_column_indices[d_queue[my_id]] : max_vertices); else
            s_vertices[tid] = (my_id < range ? d_column_indices[d_queue[my_id]]
    : max_vertices); s_edge_ids[tid] = (my_id < range ? d_queue[my_id] :
    max_vertices);
    }*/

    __syncthreads();
    unsigned int size = s_edges[end_id];

    VertexId v, e, v_id;

    int v_index = util::BinarySearch<KernelPolicy::THREADS>(tid, s_edges);
    v = s_vertices[v_index];
    v_id = s_edge_ids[v_index];
    int end_last =
        (v_index < KernelPolicy::THREADS ? s_edges[v_index] : max_vertices);
    bool found_parent = false;

    for (int i = tid; i < size; i += KernelPolicy::THREADS) {
      if (i >= end_last) {
        v_index = util::BinarySearch<KernelPolicy::THREADS>(i, s_edges);
        v = s_vertices[v_index];
        v_id = s_edge_ids[v_index];
        end_last =
            (v_index < KernelPolicy::THREADS ? s_edges[v_index] : max_vertices);
        found_parent = false;
      }

      if (found_parent) continue;

      int internal_offset = v_index > 0 ? s_edges[v_index - 1] : 0;
      e = i - internal_offset;

      int lookup = d_row_offsets[v] + e;
      VertexId u = d_column_indices[lookup];

      bool parent_in_bitmap = d_bitmap_in[u];

      if (parent_in_bitmap && !found_parent) {
        if (!ProblemData::MARK_PREDECESSORS) {
          if (Functor::CondEdge(label, v, problem))
            Functor::ApplyEdge(label, v, problem);
        } else {
          if (Functor::CondEdge(u, v, problem))
            Functor::ApplyEdge(u, v, problem);
        }

        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            true, d_bitmap_out + v);

        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            (VertexId)-1, d_queue + v_id);

        found_parent = true;
      }
    }

    // if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0 && threadIdx.x == 0)) {
    //    kernel_stats.MarkStop();
    //    kernel_stats.Flush();
    //}
  }
};

/**
 * @brief Kernel entry for relax light edge function
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             label value to use in functor
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices
 * queue
 * @param[in] d_inverse_column_indices  Device pointer of VertexId to the
 * inverse column indices queue
 * @param[in] d_scanned_edges   Device pointer of scanned neighbor list queue of
 * the current frontier
 * @param[in] partition_stats   Device pointer which marks the starting index of
 * each partition
 * @param[in] num_partitions    Number of partitions
 * @param[in] d_done            Pointer of volatile int to the flag to set when
 * we detect incoming frontier is empty
 * @param[in] d_queue           Device pointer of VertexId to the incoming
 * frontier queue
 * @param[out] d_bitmap_in      Device pointer of bool to the input frontier
 * bitmap
 * @param[out] d_bitmap_out     Device pointer of bool to the output frontier
 * bitmap
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_queue_len   Length of the incoming frontier queue
 * @param[in] output_queue_len  Length of the outgoing frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the
 * outgoing frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when
 * KernelPolicy::INSTRUMENT is set)
 * @param[in] ADVANCE_TYPE      enumerator which shows the advance type: V2V,
 * V2E, E2V, or E2E
 * @param[in] inverse_graph     Whether this iteration's advance operator is in
 * the opposite direction to the previous iteration
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void RelaxPartitionedEdges(
        bool queue_reset, typename KernelPolicy::VertexId queue_index,
        int label, typename KernelPolicy::SizeT *d_row_offsets,
        typename KernelPolicy::VertexId *d_column_indices,
        typename KernelPolicy::VertexId *d_inverse_column_indices,
        typename KernelPolicy::VertexId *d_scanned_edges,
        unsigned int *partition_starts, unsigned int num_partitions,
        volatile int *d_done, typename KernelPolicy::VertexId *d_queue,
        bool *d_bitmap_in, bool *d_bitmap_out,
        typename ProblemData::DataSlice *problem,
        typename KernelPolicy::SizeT input_queue_len,
        typename KernelPolicy::SizeT *output_queue_len,
        typename KernelPolicy::SizeT partition_size,
        typename KernelPolicy::SizeT max_vertices,
        typename KernelPolicy::SizeT max_edges,
        util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
        util::KernelRuntimeStats kernel_stats,
        gunrock::oprtr::advance::TYPE ADVANCE_TYPE =
            gunrock::oprtr::advance::V2V,
        bool inverse_graph = false) {
  Dispatch<KernelPolicy, ProblemData, Functor>::RelaxPartitionedEdges(
      queue_reset, queue_index, label, d_row_offsets, d_column_indices,
      d_inverse_column_indices, d_scanned_edges, partition_starts,
      num_partitions,
      // d_done,
      d_queue, d_bitmap_in, d_bitmap_out, problem, input_queue_len,
      output_queue_len, partition_size, max_vertices, max_edges, work_progress,
      kernel_stats, ADVANCE_TYPE, inverse_graph);
}

/**
 * @brief Kernel entry for relax light edge function
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             label value to use in functor
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices
 * queue
 * @param[in] d_inverse_column_indices  Device pointer of VertexId to the
 * inverse column indices queue
 * @param[in] d_scanned_edges   Device pointer of scanned neighbor list queue of
 * the current frontier
 * @param[in] d_done            Pointer of volatile int to the flag to set when
 * we detect incoming frontier is empty
 * @param[in] d_queue           Device pointer of VertexId to the incoming
 * frontier queue
 * @param[out] d_bitmap_in      Device pointer of bool to the input frontier
 * bitmap
 * @param[out] d_bitmap_out     Device pointer of bool to the output frontier
 * bitmap
 * @param[in] problem           Device pointer to the problem object
 * @param[in] input_queue_len   Length of the incoming frontier queue
 * @param[in] output_queue_len  Length of the outgoing frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the
 * outgoing frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when
 * KernelPolicy::INSTRUMENT is set)
 * @param[in] ADVANCE_TYPE      enumerator which shows the advance type: V2V,
 * V2E, E2V, or E2E
 * @param[in] inverse_graph     Whether this iteration's advance operator is in
 * the opposite direction to the previous iteration
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void RelaxLightEdges(
        bool queue_reset, typename KernelPolicy::VertexId queue_index,
        int label, typename KernelPolicy::SizeT *d_row_offsets,
        typename KernelPolicy::VertexId *d_column_indices,
        typename KernelPolicy::VertexId *d_inverse_column_indices,
        typename KernelPolicy::SizeT *d_scanned_edges,
        // volatile int                    *d_done,
        typename KernelPolicy::VertexId *d_queue, bool *d_bitmap_in,
        bool *d_bitmap_out, typename ProblemData::DataSlice *problem,
        typename KernelPolicy::SizeT input_queue_len,
        typename KernelPolicy::SizeT *output_queue_len,
        typename KernelPolicy::SizeT max_vertices,
        typename KernelPolicy::SizeT max_edges,
        util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
        util::KernelRuntimeStats kernel_stats,
        gunrock::oprtr::advance::TYPE ADVANCE_TYPE =
            gunrock::oprtr::advance::V2V,
        bool inverse_graph = false) {
  Dispatch<KernelPolicy, ProblemData, Functor>::RelaxLightEdges(
      queue_reset, queue_index, label, d_row_offsets, d_column_indices,
      d_inverse_column_indices, d_scanned_edges,
      // d_done,
      d_queue, d_bitmap_in, d_bitmap_out, problem, input_queue_len,
      output_queue_len, max_vertices, max_edges, work_progress, kernel_stats,
      ADVANCE_TYPE, inverse_graph);
}

/**
 * @brief Kernel entry for computing neighbor list length for each vertex in the
 * current frontier
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offsets     Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices
 * queue
 * @param[in] d_queue           Device pointer of VertexId to the incoming
 * frontier queue
 * @param[out] d_scanned_edges  Device pointer of scanned neighbor list queue of
 * the current frontier
 * @param[in] num_elements      Length of the current frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the
 * outgoing frontier
 * @param[in] ADVANCE_TYPE      enumerator which shows the advance type: V2V,
 * V2E, E2V, or E2E
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void GetEdgeCounts(typename KernelPolicy::SizeT *d_row_offsets,
                       typename KernelPolicy::VertexId *d_column_indices,
                       typename KernelPolicy::VertexId *d_queue,
                       typename KernelPolicy::SizeT *d_scanned_edges,
                       typename KernelPolicy::SizeT num_elements,
                       typename KernelPolicy::SizeT max_vertex,
                       typename KernelPolicy::SizeT max_edge,
                       gunrock::oprtr::advance::TYPE ADVANCE_TYPE)

{
  Dispatch<KernelPolicy, ProblemData, Functor>::GetEdgeCounts(
      d_row_offsets, d_column_indices, d_queue, d_scanned_edges, num_elements,
      max_vertex, max_edge, ADVANCE_TYPE);
}

}  // namespace edge_map_partitioned_backward
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
