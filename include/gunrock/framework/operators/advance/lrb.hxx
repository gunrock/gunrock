/**
 * @file lrb.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Logarithmic Radix Binning (LRB) load-balancing for advance operator.
 * @date 2026-01-21
 *
 * @copyright Copyright (c) 2026
 *
 * @par Overview
 * LRB (Logarithmic Radix Binning) is a load-balancing technique that bins
 * frontier vertices by their degree (work size) using logarithmic bucketing.
 * Vertices with similar degrees (2^b to 2^(b+1)) are grouped together,
 * enabling efficient load-balanced processing.
 *
 * @par Algorithm
 * 1. Bin Count: For each vertex v with degree d, compute bin = __clz(d)
 * 2. Prefix Sum: Compute prefix sum over bins to determine starting positions
 * 3. Reorder: Reorganize vertices into bins based on their work size
 * 4. Process: Handle each bin with appropriate strategy based on size
 *
 * @par References
 * Fox et al., "Improving Scheduling for Irregular Applications with
 * Logarithmic Radix Binning", IEEE HPEC 2019
 * https://github.com/ogreen/segmented-sort/blob/main/lrb_sort.cuh
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/advance/helpers.hxx>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

// Include appropriate CUB library based on compiler
#if defined(__CUDACC__) && !defined(__HIP__)
  // Pure CUDA (NVCC without HIP): Use CUB
  #include <cub/device/device_scan.cuh>
  namespace cub_namespace = cub;
#else
  // HIP compiler (works for both NVIDIA and AMD platforms): Use hipCUB
  #ifdef __HIP_PLATFORM_AMD__
    #ifdef gfx942
      #pragma push_macro("gfx942")
      #undef gfx942
    #endif
    #ifdef gfx950
      #pragma push_macro("gfx950")
      #undef gfx950
    #endif
    #ifdef gfx90a
      #pragma push_macro("gfx90a")
      #undef gfx90a
    #endif
  #endif
  #include <hipcub/hipcub.hpp>
  #ifdef __HIP_PLATFORM_AMD__
    #ifdef gfx942
      #pragma pop_macro("gfx942")
    #endif
    #ifdef gfx950
      #pragma pop_macro("gfx950")
    #endif
    #ifdef gfx90a
      #pragma pop_macro("gfx90a")
    #endif
  #endif
  namespace cub_namespace = hipcub;
#endif

namespace gunrock {
namespace operators {
namespace advance {
namespace lrb {

// Number of bins for 32-bit integers (0 to 32 leading zeros)
constexpr int NUM_BINS = 33;

// Bin thresholds for different processing strategies
constexpr int BIN_LARGE_START = 20;   // 2^20 = ~1M edges, use CUB
constexpr int BIN_MEDIUM_START = 28;  // 2^28+, use block kernels
constexpr int BIN_SMALL_START = 31;   // 2^31+, use simple kernels

/**
 * @brief Compute bin index for a given degree using count-leading-zeros.
 *
 * @tparam edge_t Edge type (typically int or int64_t)
 * @param degree Number of neighbors for a vertex
 * @return int Bin index (0-32 for 32-bit, 0-64 for 64-bit)
 */
template <typename edge_t>
__device__ __host__ __forceinline__ int compute_bin(edge_t degree) {
  if (degree == 0)
    return NUM_BINS - 1;  // Empty vertices go to last bin
  
  // Use count-leading-zeros intrinsic
  if constexpr (sizeof(edge_t) == 4) {
    return __clz((int)degree);
  } else if constexpr (sizeof(edge_t) == 8) {
    return __clzll((long long)degree);
  } else {
    // Fallback for other sizes
    return __clz((int)degree);
  }
}

/**
 * @brief Kernel to count vertices per bin using shared memory for efficiency.
 *
 * @tparam graph_t Graph type
 * @tparam input_iterator_t Input iterator type
 * @param G Graph object
 * @param input Input frontier or iterator
 * @param bins Output bin counts (size NUM_BINS)
 * @param input_size Number of elements in input frontier
 */
template <typename graph_t, typename input_iterator_t>
__global__ void bin_count_kernel(graph_t G,
                                   input_iterator_t input,
                                   int32_t* bins,
                                   std::size_t input_size) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  // Shared memory for local bin counts to reduce global atomic contention
  __shared__ int32_t local_bins[NUM_BINS];
  
  int tid = threadIdx.x;
  if (tid < NUM_BINS) {
    local_bins[tid] = 0;
  }
  __syncthreads();
  
  // Each thread processes one vertex
  if (i < input_size) {
    vertex_t v = input[i];
    
    // Skip invalid vertices
    if (!gunrock::util::limits::is_valid(v))
      return;
    
    edge_t degree = G.get_number_of_neighbors(v);
    int bin = compute_bin(degree);
    
    // Atomically increment local bin count
    atomicAdd(&local_bins[bin], 1);
  }
  
  __syncthreads();
  
  // Reduce local bins to global bins
  if (tid < NUM_BINS) {
    atomicAdd(&bins[tid], local_bins[tid]);
  }
}

/**
 * @brief Kernel to compute prefix sum over bins.
 * 
 * This is a simple serial kernel since we only have 33 bins.
 * For larger bin counts, consider using CUB's device-level scan.
 *
 * @param bins Input bin counts (size NUM_BINS)
 * @param prefix_bins Output prefix sum (size NUM_BINS+1)
 */
__global__ void bin_prefix_kernel(const int32_t* bins, int32_t* prefix_bins) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  
  prefix_bins[0] = 0;
  for (int b = 0; b < NUM_BINS; b++) {
    prefix_bins[b + 1] = prefix_bins[b] + bins[b];
  }
}

/**
 * @brief Kernel to compute bin membership for each vertex (Phase 1 optimization).
 * Instead of reordering vertices, we just record which bin each belongs to.
 *
 * @tparam graph_t Graph type
 * @tparam input_iterator_t Input iterator type
 * @param G Graph object
 * @param input Input frontier or iterator
 * @param bin_membership Output array: bin_membership[i] = bin index for vertex i
 * @param input_size Number of elements in input frontier
 */
template <typename graph_t, typename input_iterator_t>
__global__ void compute_bin_membership_kernel(graph_t G,
                                                input_iterator_t input,
                                                int32_t* bin_membership,
                                                std::size_t input_size) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i >= input_size)
    return;
  
  vertex_t v = input[i];
  
  if (!gunrock::util::limits::is_valid(v)) {
    bin_membership[i] = NUM_BINS - 1;  // Invalid vertices go to last bin
    return;
  }
  
  edge_t degree = G.get_number_of_neighbors(v);
  bin_membership[i] = compute_bin(degree);
}

/**
 * @brief Kernel to count vertices per bin after sorting (Phase 1 optimization).
 *
 * @param sorted_bin_membership Sorted bin membership array
 * @param bins Output bin counts
 * @param input_size Number of elements
 */
__global__ void count_sorted_bins_kernel(int32_t* sorted_bin_membership,
                                           int32_t* bins,
                                           std::size_t input_size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i >= input_size)
    return;
  
  int my_bin = sorted_bin_membership[i];
  atomicAdd(&bins[my_bin], 1);
}

/**
 * @brief Kernel to process small bins using indirect indexing (Phase 1 optimized).
 * Uses simple per-thread processing without vertex reordering.
 *
 * @tparam graph_t Graph type
 * @tparam operator_t Operator type
 * @tparam frontier_t Frontier type (or iterator)
 * @param G Graph object
 * @param op User-defined operator
 * @param input Original input frontier
 * @param sorted_indices Indices sorted by bin
 * @param output Output frontier
 * @param segments Segment offsets (output positions for original vertices)
 * @param start_idx Starting index in sorted_indices array
 * @param end_idx Ending index in sorted_indices array
 */
template <advance_io_type_t output_type,
          advance_io_type_t input_type,
          typename graph_t,
          typename operator_t,
          typename input_iterator_t,
          typename output_iterator_t,
          typename work_tiles_t>
__global__ void process_small_bins_kernel(graph_t G,
                                            operator_t op,
                                            input_iterator_t input,
                                            int32_t* sorted_indices,
                                            output_iterator_t output,
                                            work_tiles_t* segments,
                                            int start_idx,
                                            int end_idx) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  
  int sorted_pos = threadIdx.x + blockIdx.x * blockDim.x + start_idx;
  
  if (sorted_pos >= end_idx)
    return;
  
  // Get the original index of this vertex
  int orig_idx = sorted_indices[sorted_pos];
  
  // Get the vertex from original input
  vertex_t v;
  if constexpr (input_type == advance_io_type_t::graph) {
    v = static_cast<vertex_t>(orig_idx);
  } else {
    v = input[orig_idx];
  }
  
  if (!gunrock::util::limits::is_valid(v))
    return;
  
  // Use original segments (no recomputation needed!)
  edge_t output_start = segments[orig_idx];
  edge_t output_end = segments[orig_idx + 1];
  edge_t degree = output_end - output_start;
  
  // Get the starting edge in the graph for this vertex
  edge_t graph_edge_start = G.get_starting_edge(v);
  
  // Process all edges for this vertex
  for (edge_t local_e = 0; local_e < degree; local_e++) {
    edge_t graph_e = graph_edge_start + local_e;
    edge_t output_pos = output_start + local_e;
    
    vertex_t n = G.get_destination_vertex(graph_e);
    weight_t w = G.get_edge_weight(graph_e);
    
    bool cond = op(v, n, graph_e, w);
    
    if constexpr (output_type != advance_io_type_t::none) {
      output[output_pos] = cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
    }
  }
}

/**
 * @brief Kernel to process medium bins using indirect indexing (Phase 1 optimized).
 * Uses block-level processing with binary search, no vertex reordering.
 *
 * @tparam THREADS_PER_BLOCK Block size
 * @tparam graph_t Graph type
 * @tparam operator_t Operator type
 * @tparam frontier_t Frontier type
 * @param G Graph object
 * @param op User-defined operator
 * @param input Original input frontier
 * @param sorted_indices Indices sorted by bin
 * @param output Output frontier
 * @param segments Segment offsets (for original vertices)
 * @param start_idx Starting index in sorted_indices array
 * @param end_idx Ending index in sorted_indices array
 */
template <unsigned int THREADS_PER_BLOCK,
          advance_io_type_t output_type,
          advance_io_type_t input_type,
          typename graph_t,
          typename operator_t,
          typename input_iterator_t,
          typename output_iterator_t,
          typename work_tiles_t>
__global__ void process_medium_bins_kernel(graph_t G,
                                             operator_t op,
                                             input_iterator_t input,
                                             int32_t* sorted_indices,
                                             output_iterator_t output,
                                             work_tiles_t* segments,
                                             int start_idx,
                                             int end_idx) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  
  // Shared memory for vertices, graph edge starts, and output positions
  __shared__ vertex_t vertices[THREADS_PER_BLOCK];
  __shared__ edge_t graph_edge_starts[THREADS_PER_BLOCK];
  __shared__ edge_t output_starts[THREADS_PER_BLOCK];
  __shared__ edge_t degrees[THREADS_PER_BLOCK];
  
  int sorted_pos = blockIdx.x * blockDim.x + start_idx;
  int local_idx = threadIdx.x;
  
  // Load vertices using indirect indexing
  if (sorted_pos + local_idx < end_idx) {
    int orig_idx = sorted_indices[sorted_pos + local_idx];
    
    vertex_t v;
    if constexpr (input_type == advance_io_type_t::graph) {
      v = static_cast<vertex_t>(orig_idx);
    } else {
      v = input[orig_idx];
    }
    
    vertices[local_idx] = v;
    
    if (gunrock::util::limits::is_valid(v)) {
      // Use original segments - no recomputation!
      output_starts[local_idx] = segments[orig_idx];
      degrees[local_idx] = segments[orig_idx + 1] - segments[orig_idx];
      graph_edge_starts[local_idx] = G.get_starting_edge(v);
    } else {
      degrees[local_idx] = 0;
    }
  } else {
    degrees[local_idx] = 0;
  }
  __syncthreads();
  
  // Compute total work for this block
  edge_t total_work = 0;
  for (int i = 0; i < THREADS_PER_BLOCK; i++) {
    total_work += degrees[i];
  }
  
  // Each thread processes multiple edges
  for (edge_t work_idx = local_idx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
    // Binary search to find which vertex this work belongs to
    edge_t cumulative = 0;
    int vertex_idx = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
      if (work_idx >= cumulative && work_idx < cumulative + degrees[i]) {
        vertex_idx = i;
        break;
      }
      cumulative += degrees[i];
    }
    
    vertex_t v = vertices[vertex_idx];
    if (!gunrock::util::limits::is_valid(v))
      continue;
    
    // Compute edge offset within this vertex's edges
    edge_t cumulative_before = 0;
    for (int i = 0; i < vertex_idx; i++) {
      cumulative_before += degrees[i];
    }
    edge_t edge_offset = work_idx - cumulative_before;
    
    // Get graph edge and output position
    edge_t graph_e = graph_edge_starts[vertex_idx] + edge_offset;
    edge_t output_pos = output_starts[vertex_idx] + edge_offset;
    
    vertex_t n = G.get_destination_vertex(graph_e);
    weight_t w = G.get_edge_weight(graph_e);
    
    bool cond = op(v, n, graph_e, w);
    
    if constexpr (output_type != advance_io_type_t::none) {
      output[output_pos] = cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
    }
  }
}

/**
 * @brief Main execution function for LRB advance operator.
 *
 * @tparam direction Advance direction (forward/backward)
 * @tparam input_type Input frontier type
 * @tparam output_type Output frontier type
 * @tparam graph_t Graph type
 * @tparam operator_t Operator type
 * @tparam frontier_t Frontier type
 * @tparam work_tiles_t Segment offsets type
 * @param G Graph object
 * @param op User-defined operator
 * @param input Input frontier
 * @param output Output frontier
 * @param segments Segment offsets (reused for LRB)
 * @param context CUDA context
 */
template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t& input,
             frontier_t& output,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  
  // Get input size
  std::size_t input_size = (input_type == advance_io_type_t::graph)
                               ? G.get_number_of_vertices()
                               : input.get_number_of_elements();
  
  if (input_size == 0) {
    if constexpr (output_type != advance_io_type_t::none) {
      output.set_number_of_elements(0);
    }
    return;
  }
  
  // Compute output size and allocate output frontier
  std::size_t output_size = 0;
  if constexpr (output_type != advance_io_type_t::none) {
    output_size = compute_output_length(G, input, context,
                                        (input_type == advance_io_type_t::graph));
    
    if (output_size == 0) {
      output.set_number_of_elements(0);
      return;
    }
    
    if (output.get_capacity() < output_size)
      output.reserve(output_size);
    output.set_number_of_elements(output_size);
    output.fill(gunrock::numeric_limits<vertex_t>::invalid(), context.stream());
  }
  
  // Phase 1 Optimization: Use indirect indexing instead of reordering
  // Allocate LRB data structures (reduced memory footprint)
  thrust::device_vector<int32_t> bins(NUM_BINS);
  thrust::device_vector<int32_t> prefix_bins(NUM_BINS + 1);
  thrust::device_vector<int32_t> bin_membership(input_size);  // Which bin each vertex belongs to
  thrust::device_vector<int32_t> sorted_indices(input_size);  // Indices sorted by bin
  
  // Compute output offsets for the original frontier (used directly, no recomputation needed)
  std::size_t computed_output_size = compute_output_offsets(
      G, &input, segments, context,
      (input_type == advance_io_type_t::graph));
  
  // Verify output size matches
  if constexpr (output_type != advance_io_type_t::none) {
    if (computed_output_size != output_size) {
      output_size = computed_output_size;
      if (output.get_capacity() < output_size)
        output.reserve(output_size);
      output.set_number_of_elements(output_size);
      output.fill(gunrock::numeric_limits<vertex_t>::invalid(), context.stream());
    }
  }
  
  // Step 1: Compute bin membership for each vertex (Phase 1 optimization)
  int membership_blocks = (input_size + 255) / 256;
  if (membership_blocks > 0) {
    auto input_data = (input_type == advance_io_type_t::graph)
                          ? nullptr
                          : input.data();
    
    if (input_type == advance_io_type_t::graph) {
      compute_bin_membership_kernel<<<membership_blocks, 256, 0, context.stream()>>>(
          G, thrust::make_counting_iterator<vertex_t>(0),
          bin_membership.data().get(), input_size);
    } else {
      compute_bin_membership_kernel<<<membership_blocks, 256, 0, context.stream()>>>(
          G, input_data, bin_membership.data().get(), input_size);
    }
  }
  
  // Step 2: Initialize sorted_indices as [0, 1, 2, ..., input_size-1]
  thrust::sequence(context.execution_policy(), 
                   sorted_indices.begin(), sorted_indices.end());
  
  // Step 3: Sort indices by bin membership (stable sort preserves order within bins)
  // This gives us indices grouped by bin, but we don't move the actual vertices
  thrust::stable_sort_by_key(
      context.execution_policy(),
      bin_membership.begin(), bin_membership.end(),
      sorted_indices.begin()
  );
  
  // Step 4: Count bins and compute prefix sum
  thrust::fill(context.execution_policy(), bins.begin(), bins.end(), 0);
  
  // Count how many vertices in each bin (after sorting)
  int count_blocks = (input_size + 255) / 256;
  if (count_blocks > 0) {
    count_sorted_bins_kernel<<<count_blocks, 256, 0, context.stream()>>>(
        bin_membership.data().get(), bins.data().get(), input_size);
  }
  
  // Compute prefix sum over bins
  bin_prefix_kernel<<<1, 1, 0, context.stream()>>>(bins.data().get(),
                                                     prefix_bins.data().get());
  
  // Copy prefix bins to host to determine bin boundaries
  thrust::host_vector<int32_t> h_prefix_bins(prefix_bins);
  
  // Step 5: Process bins with appropriate strategies (Phase 1 optimized)
  // Note: Bins are indexed from 0 (largest degrees) to NUM_BINS-1 (smallest)
  // We use sorted_indices to access vertices, and original segments for output positions
  
  auto input_data = (input_type == advance_io_type_t::graph)
                        ? nullptr
                        : input.data();
  
  // Process large bins (bins 0 to BIN_LARGE_START)
  // These have degrees from 2^(31-BIN_LARGE_START) to 2^31
  // For now, use medium bin strategy (Phase 3 will add CUB)
  
  // Process medium bins (bins BIN_LARGE_START to BIN_MEDIUM_START)
  for (int bin = 0; bin <= BIN_MEDIUM_START; bin++) {
    int start = h_prefix_bins[bin];
    int end = h_prefix_bins[bin + 1];
    
    if (end <= start)
      continue;
    
    int num_vertices = end - start;
    int blocks = (num_vertices + 255) / 256;
    
    if (input_type == advance_io_type_t::graph) {
      process_medium_bins_kernel<256, output_type, advance_io_type_t::graph>
          <<<blocks, 256, 0, context.stream()>>>(
              G, op, thrust::make_counting_iterator<vertex_t>(0),
              sorted_indices.data().get(), output.data(),
              segments.data().get(), start, end);
    } else {
      process_medium_bins_kernel<256, output_type, advance_io_type_t::vertices>
          <<<blocks, 256, 0, context.stream()>>>(
              G, op, input_data, sorted_indices.data().get(),
              output.data(), segments.data().get(), start, end);
    }
  }
  
  // Process small bins (bins BIN_MEDIUM_START+1 to NUM_BINS-1)
  for (int bin = BIN_MEDIUM_START + 1; bin < NUM_BINS; bin++) {
    int start = h_prefix_bins[bin];
    int end = h_prefix_bins[bin + 1];
    
    if (end <= start)
      continue;
    
    int num_vertices = end - start;
    int blocks = (num_vertices + 255) / 256;
    
    if (input_type == advance_io_type_t::graph) {
      process_small_bins_kernel<output_type, advance_io_type_t::graph>
          <<<blocks, 256, 0, context.stream()>>>(
              G, op, thrust::make_counting_iterator<vertex_t>(0),
              sorted_indices.data().get(), output.data(),
              segments.data().get(), start, end);
    } else {
      process_small_bins_kernel<output_type, advance_io_type_t::vertices>
          <<<blocks, 256, 0, context.stream()>>>(
              G, op, input_data, sorted_indices.data().get(),
              output.data(), segments.data().get(), start, end);
    }
  }
  
  context.synchronize();
}

}  // namespace lrb
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
