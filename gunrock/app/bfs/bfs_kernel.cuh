// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_kernel.cuh
 *
 * @brief BFS GPU kernels
 */

#pragma once

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief forms unvisited vertices queue, when switching from top-down to
 *    bottom-up visiting direction, with idempotence and for single-GPU
 */
template <typename ProblemT, int LOG_THREADS>
__global__ void From_Unvisited_Queue_IDEM(
    typename ProblemT::SizeT num_nodes, typename ProblemT::SizeT *out_length,
    typename ProblemT::VertexT *keys_out,
    typename ProblemT::MaskT *visited_masks,
    typename ProblemT::LabelT *labels) {
  typedef typename ProblemT::VertexT VertexT;
  typedef typename ProblemT::SizeT SizeT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  VertexT l_keys_out[sizeof(MaskT) * 8];
  SizeT l_counter = 0;
  SizeT output_pos = 0;

  while ((x - threadIdx.x) * sizeof(MaskT) * 8 < num_nodes) {
    MaskT mask_byte = 0;
    bool changed = false;
    l_counter = 0;
    if (x * (sizeof(MaskT) << 3) < num_nodes) {
      mask_byte = _ldg(visited_masks + x);
      if (mask_byte != util::PreDefinedValues<MaskT>::AllOnes)
        for (int i = 0; i < (1 << (2 + sizeof(MaskT))); i++) {
          MaskT mask_bit = 1 << i;
          VertexT key = 0;
          if (mask_byte & mask_bit) continue;

          key = (x << (sizeof(MaskT) + 2)) + i;
          if (key >= num_nodes) break;
          if (_ldg(labels + key) != util::PreDefinedValues<LabelT>::MaxValue) {
            mask_byte |= mask_bit;
            changed = true;
            continue;
          }

          l_keys_out[l_counter] = key;
          l_counter++;
        }
    }
    if (changed) visited_masks[x] = mask_byte;
  }

  BlockScanT::Scan(l_counter, output_pos, scan_space);
  if (threadIdx.x == blockDim.x - 1) {
    block_offset = atomicAdd(out_length, output_pos + l_counter);
  }
  __syncthreads();
  output_pos += block_offset;
  for (int i = 0; i < l_counter; i++) {
    keys_out[output_pos] = l_keys_out[i];
    output_pos++;
  }
  __syncthreads();
  x += STRIDE;
}

/**
 * @brief forms unvisited vertices queue, when switching from top-down to
 *    bottom-up visiting direction, without idempotence, for single-GPU
 */
template <typename ProblemT, int LOG_THREADS>
__global__ void From_Unvisited_Queue(typename ProblemT::SizeT num_nodes,
                                     typename ProblemT::SizeT *out_length,
                                     typename ProblemT::VertexT *vertices_out,
                                     typename ProblemT::LabelT *labels) {
  typedef typename ProblemT::VertexT VertexT;
  typedef typename ProblemT::SizeT SizeT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  VertexT v = (VertexT)blockIdx.x * blockDim.x + threadIdx.x;
  const VertexT STRIDE = (VertexT)blockDim.x * gridDim.x;

  while (v - threadIdx.x < num_nodes) {
    bool to_process = true;
    SizeT output_pos = 0;
    if (v < num_nodes) {
      if (_ldg(labels + v) != util::PreDefinedValues<LabelT>::MaxValue)
        to_process = false;
    } else
      to_process = false;

    BlockScanT::LogicScan(to_process, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      block_offset = atomicAdd(out_length, output_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();
    if (to_process) {
      output_pos += block_offset;
      vertices_out[output_pos] = v;
    }
    __syncthreads();
    v += STRIDE;
  }
}

/**
 * @brief forms unvisited vertices queue, when switching from top-down to
 *    bottom-up visiting direction, without idempotence, for multi-GPUs
 */
template <typename ProblemT, int LOG_THREADS>
__global__ void From_Unvisited_Queue_Local(
    typename ProblemT::SizeT num_local_vertices,
    typename ProblemT::VertexT *local_vertices,
    typename ProblemT::SizeT *out_length,
    typename ProblemT::VertexT *vertices_out,
    typename ProblemT::LabelT *labels) {
  typedef typename ProblemT::VertexT VertexT;
  typedef typename ProblemT::SizeT SizeT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  SizeT input_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while ((input_pos - threadIdx.x) < num_local_vertices) {
    bool to_process = true;
    VertexT v = 0;
    SizeT output_pos = 0;

    if (input_pos < num_local_vertices) {
      v = local_vertices[input_pos];
      if (_ldg(labels + v) != util::PreDefinedValues<LabelT>::MaxValue)
        to_process = false;
    } else
      to_process = false;

    BlockScanT::LogicScan(to_process, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      block_offset = atomicAdd(out_length, output_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();

    if (to_process) {
      output_pos += block_offset;
      vertices_out[output_pos] = v;
    }
    __syncthreads();
    input_pos += STRIDE;
  }
}

/**
 * @brief forms unvisited vertices queue, when switching from top-down to
 *    bottom-up visiting direction, with idempotence and for multi-GPUs
 */
template <typename ProblemT, int LOG_THREADS>
__global__ void From_Unvisited_Queue_Local_IDEM(
    typename ProblemT::SizeT num_local_vertices,
    typename ProblemT::VertexT *local_vertices,
    typename ProblemT::SizeT *out_length,
    typename ProblemT::VertexT *vertices_out,
    typename ProblemT::MaskT *visited_masks,
    typename ProblemT::LabelT *labels) {
  typedef typename ProblemT::VertexT VertexT;
  typedef typename ProblemT::SizeT SizeT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  SizeT input_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while ((input_pos - threadIdx.x) < num_local_vertices) {
    bool to_process = true;
    VertexT v = 0;
    SizeT output_pos = 0;
    if (input_pos < num_local_vertices) {
      v = local_vertices[input_pos];
      SizeT mask_pos = (v  //& KernelPolicy::LOAD_BALANCED_CULL::ELEMENT_ID_MASK
                        ) >>
                       (2 + sizeof(MaskT));
      MaskT mask_byte = _ldg(visited_masks + mask_pos);
      MaskT mask_bit = 1 << (v & ((1 << (2 + sizeof(MaskT))) - 1));
      if (mask_byte & mask_bit)
        to_process = false;
      else {
        if (_ldg(labels + v) != util::PreDefinedValues<LabelT>::MaxValue) {
          mask_byte |= mask_bit;
          visited_masks[mask_pos] = mask_byte;
          to_process = false;
        }
      }
    } else
      to_process = false;

    BlockScanT::LogicScan(to_process, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      if (output_pos != 0 || to_process)
        block_offset =
            atomicAdd(out_length, output_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();
    if (to_process) {
      output_pos += block_offset;
      vertices_out[output_pos] = v;
    }
    __syncthreads();
    input_pos += STRIDE;
  }
}

/**
 * @brief Bottom-up search to find possible parents of unvisited
 *  local vertices; output both newly visited vertices and still-unvisited
 *  vertices
 */
template <typename ProblemT, int LOG_THREADS>
__global__ void Inverse_Expand(
    typename ProblemT::GraphT graph,
    typename ProblemT::SizeT num_unvisited_vertices,
    typename ProblemT::VertexT label, bool idempotence,
    typename ProblemT::VertexT *unvisited_vertices_in,
    typename ProblemT::SizeT *split_lengths,
    typename ProblemT::VertexT *unvisited_vertices_out,
    typename ProblemT::VertexT *visited_vertices_out,
    typename ProblemT::MaskT *visited_masks, typename ProblemT::LabelT *labels,
    typename ProblemT::VertexT *preds) {
  typedef typename ProblemT::SizeT SizeT;
  typedef typename ProblemT::VertexT VertexT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;
  typedef typename ProblemT::GraphT::CscT CscT;
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;
  // static const VertexT VERTEX_MASK = (~(1ULL<<(sizeof(VertexT)*8-2)));

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  SizeT input_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (input_pos - threadIdx.x < num_unvisited_vertices) {
    VertexT v = 0, parent = 0;
    bool discoverable = false;
    bool to_process = true;
    MaskT mask_byte, mask_bit;
    SizeT mask_pos;

    if (input_pos < num_unvisited_vertices) {
      v = _ldg(unvisited_vertices_in + input_pos);
    } else
      to_process = false;

    if (to_process && idempotence) {
      mask_pos = (v  //& VERTEX_MASK
                     // KernelPolicy::LOAD_BALANCED_CULL::ELEMENT_ID_MASK
                  ) >>
                 (2 + sizeof(MaskT));
      mask_byte = _ldg(visited_masks + mask_pos);
      mask_bit = 1 << (v & ((1 << (2 + sizeof(MaskT))) - 1));
      if (mask_byte & mask_bit) to_process = false;
    }

    if (to_process) {
      if (_ldg(labels + v) != util::PreDefinedValues<LabelT>::MaxValue) {
        if (idempotence) {
          mask_byte |= mask_bit;
          visited_masks[mask_pos] = mask_byte;
        }
        to_process = false;
      }
    }

    if (to_process) {
      SizeT edge_start = graph.CscT::GetNeighborListOffset(v);
      SizeT edge_end = edge_start + graph.CscT::GetNeighborListLength(v);
      for (SizeT edge = edge_start; edge < edge_end; edge++) {
        VertexT u = graph.CscT::GetEdgeDest(edge);
        if (_ldg(labels + u) == label - 1) {
          discoverable = true;
          parent = u;
          break;
        }
      }
    }

    if (discoverable) {
      if (idempotence) {
        mask_byte |= mask_bit;
        visited_masks[mask_pos] = mask_byte;
      }
      labels[v] = label;
      if (preds != NULL) preds[v] = parent;
    }

    SizeT output_pos = 0;
    BlockScanT::LogicScan(discoverable, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      if (output_pos != 0 || discoverable)
        block_offset =
            atomicAdd(split_lengths + 1, output_pos + ((discoverable) ? 1 : 0));
    }
    __syncthreads();
    if (discoverable && visited_vertices_out != NULL) {
      output_pos += block_offset;
      visited_vertices_out[output_pos] = v;
    }
    __syncthreads();

    to_process = to_process && (!discoverable);
    BlockScanT::LogicScan(to_process, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      if (output_pos != 0 || to_process)
        block_offset =
            atomicAdd(split_lengths, output_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();
    if (to_process && unvisited_vertices_out != NULL) {
      output_pos += block_offset;
      unvisited_vertices_out[output_pos] = v;
    }
    __syncthreads();

    input_pos += STRIDE;
  }
}

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
