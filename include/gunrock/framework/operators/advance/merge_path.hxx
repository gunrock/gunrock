/**
 * @file merge_path.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Merge-Path load-balancing based Advance.
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 * Merge-path partitions work across BOTH segments (vertices) and atoms (edges)
 * using a diagonal search. Each tile processes a fixed amount of total work
 * (segments + atoms), and the merge-path search determines how that work is
 * split between segments and atoms.
 * 
 * This is fundamentally different from block_mapped which assigns one block
 * per fixed range of vertices.
 * 
 * ALGORITHM:
 * 1. Compute prefix sum of segment sizes (compute_output_offsets)
 * 2. For each tile, use merge-path search to find tile boundaries
 * 3. Load segment offsets into shared memory
 * 4. Each thread uses merge-path to find its starting position
 * 5. Serial merge: walk the merge path processing items
 * 
 * TRADE-OFFS vs block_mapped:
 * - Requires O(n) prefix scan per iteration (vs O(n) reduce for block_mapped)
 * - Better load balancing for power-law graphs with hub vertices
 * - Worse performance for uniform-degree graphs (road networks, meshes)
 * - Each tile processes exactly merge_tile_size work items
 * - Hub vertices spanning multiple tiles are handled gracefully
 * 
 * USE CASES:
 * - Power-law social networks (e.g., Twitter, LiveJournal)
 * - Web graphs with hub pages
 * - Any graph with highly skewed degree distribution
 * 
 * For uniform-degree graphs, use block_mapped instead.
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>
#include <gunrock/cuda/cuda.hxx>
#include <gunrock/cuda/launch_box.hxx>
#include <gunrock/error.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/advance/helpers.hxx>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <gunrock/memory.hxx>

namespace gunrock {
namespace operators {
namespace advance {
namespace merge_path {

/**
 * @brief Coordinate for merge path - represents position on the merge path
 */
template <typename offset_t>
struct coordinate_t {
  offset_t x;  // Position in A (segments/rows)
  offset_t y;  // Position in B (atoms/edges)
};

/**
 * @brief Merge path binary search
 * 
 * Given a diagonal index, finds the coordinate (x, y) where x + y = diagonal
 * such that A[x-1] <= B[y-1] and A[x] > B[y-1] (merge semantics).
 * 
 * A = segment end offsets (row_offsets + 1)
 * B = counting iterator (0, 1, 2, ...)
 */
template <typename AIteratorT, typename BIteratorT, typename OffsetT>
__device__ __forceinline__ void merge_path_search(
    OffsetT diagonal,
    AIteratorT a,
    BIteratorT b,
    OffsetT a_len,
    OffsetT b_len,
    coordinate_t<OffsetT>& path_coordinate) {
  
  OffsetT split_min = max(diagonal - b_len, OffsetT(0));
  OffsetT split_max = min(diagonal, a_len);

  while (split_min < split_max) {
    OffsetT mid = (split_min + split_max) >> 1;
    if (a[mid] <= b[diagonal - mid - 1]) {
      split_min = mid + 1;
    } else {
      split_max = mid;
    }
  }

  path_coordinate.x = min(split_min, a_len);
  path_coordinate.y = diagonal - split_min;
}

/**
 * @brief True merge-path kernel
 * 
 * Key differences from block_mapped:
 * 1. Tiles are based on total work (segments + atoms), not just segments
 * 2. Two-level merge-path search (CTA level + thread level)
 * 3. Serial merge within each thread
 * 4. Can handle vertices with very high degree spanning multiple tiles
 */
template <int threads_per_block,
          int items_per_thread,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename segments_t>
__global__ void merge_path_kernel(graph_t G,
                                  operator_t op,
                                  frontier_t input,
                                  frontier_t output,
                                  const segments_t* segments,
                                  int num_segments,
                                  int num_atoms,
                                  int num_merge_tiles) {
  using type_t = typename frontier_t::type_t;
  using offset_t = int;
  
  constexpr int merge_tile_size = threads_per_block * items_per_thread;
  
  // Shared memory for tile coordinates and cached segment offsets
  // Size needs to accommodate tile_num_rows + items_per_thread entries
  // In worst case, tile_num_rows could be merge_tile_size (if all atoms = 0)
  __shared__ coordinate_t<offset_t> tile_coords[2];
  __shared__ offset_t tile_row_end_offsets[merge_tile_size + items_per_thread + 1];
  
  // segments is the exclusive prefix scan of degrees
  // row_end_offsets = segments + 1 gives us the end offset of each row
  const segments_t* row_end_offsets = segments + 1;
  
  int tile_idx = blockIdx.x;
  if (tile_idx >= num_merge_tiles)
    return;
  
  // =========================================================================
  // Phase 1: CTA-level merge-path search
  // Two threads find the tile's start and end coordinates on the merge path
  // =========================================================================
  if (threadIdx.x < 2) {
    offset_t diagonal = (tile_idx + threadIdx.x) * merge_tile_size;
    thrust::counting_iterator<offset_t> atom_indices(0);
    
    coordinate_t<offset_t> coords;
    merge_path_search(
        diagonal,
        row_end_offsets,
        atom_indices,
        offset_t(num_segments),
        offset_t(num_atoms),
        coords);
    
    tile_coords[threadIdx.x] = coords;
  }
  __syncthreads();
  
  coordinate_t<offset_t> tile_start = tile_coords[0];
  coordinate_t<offset_t> tile_end = tile_coords[1];
  
  // Number of rows and atoms in this tile
  offset_t tile_num_rows = tile_end.x - tile_start.x;
  offset_t tile_num_atoms = tile_end.y - tile_start.y;
  
  // Early exit if no work
  if (tile_num_rows + tile_num_atoms <= 0)
    return;
  
  // =========================================================================
  // Phase 2: Load row end-offsets into shared memory
  // We need offsets for all rows in this tile plus some extra for thread search
  // =========================================================================
  #pragma unroll 1
  for (int item = threadIdx.x; 
       item < tile_num_rows + items_per_thread; 
       item += threads_per_block) {
    offset_t row = tile_start.x + item;
    if (row < num_segments) {
      tile_row_end_offsets[item] = row_end_offsets[row];
    } else {
      tile_row_end_offsets[item] = num_atoms;
    }
  }
  __syncthreads();
  
  // =========================================================================
  // Phase 3: Thread-level merge-path search
  // Each thread finds its starting coordinate within the tile
  // =========================================================================
  thrust::counting_iterator<offset_t> tile_atom_indices(tile_start.y);
  
  coordinate_t<offset_t> thread_start;
  merge_path_search(
      offset_t(threadIdx.x * items_per_thread),
      tile_row_end_offsets,
      tile_atom_indices,
      tile_num_rows,
      tile_num_atoms,
      thread_start);
  
  __syncthreads();
  
  // =========================================================================
  // Phase 4: Serial merge
  // Each thread processes items_per_thread items by walking the merge path
  // =========================================================================
  
  #pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    // Global atom index for this iteration
    offset_t global_atom = tile_start.y + thread_start.y;
    
    // Bounds check
    if (global_atom >= num_atoms)
      break;
    if (thread_start.x >= tile_num_rows + items_per_thread)
      break;
    
    // Compare: is current atom within current row?
    // tile_atom_indices[thread_start.y] = tile_start.y + thread_start.y = global_atom
    // tile_row_end_offsets[thread_start.x] = end offset of row (tile_start.x + thread_start.x)
    if (global_atom < tile_row_end_offsets[thread_start.x]) {
      // Process this atom (edge)
      offset_t global_row = tile_start.x + thread_start.x;
      
      if (global_row < num_segments) {
        // Get vertex from input frontier
        auto v = (input_type == advance_io_type_t::graph)
                     ? type_t(global_row)
                     : input.get_element_at(global_row);
        
        if (gunrock::util::limits::is_valid(v)) {
          // Compute edge index
          auto starting_edge = G.get_starting_edge(v);
          
          // Get the start offset of this row to compute rank within row
          offset_t row_start_offset;
          if (thread_start.x == 0) {
            // First row in tile - read from global memory
            row_start_offset = (global_row == 0) ? 0 : segments[global_row];
          } else {
            // Use previous row's end offset from shared memory
            row_start_offset = tile_row_end_offsets[thread_start.x - 1];
          }
          
          offset_t rank = global_atom - row_start_offset;
          
          if (rank >= 0) {
            auto e = starting_edge + rank;
            auto n = G.get_destination_vertex(e);
            auto w = G.get_edge_weight(e);
            
            bool cond = op(v, n, e, w);
            
            if constexpr (output_type != advance_io_type_t::none) {
              type_t element = cond ? n : gunrock::numeric_limits<type_t>::invalid();
              output.set_element_at(element, global_atom);
            }
          }
        }
      }
      
      // Move to next atom (advance in B)
      ++thread_start.y;
    } else {
      // Move to next row (advance in A)
      ++thread_start.x;
    }
  }
}

template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input_ptr,
             frontier_t* output_ptr,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  if (input_ptr == nullptr || output_ptr == nullptr) {
    error::throw_if_exception(hipErrorInvalidValue, 
        "merge_path::execute: null frontier pointer");
  }

  frontier_t& input = *input_ptr;
  frontier_t& output = *output_ptr;

  // Merge-path requires the prefix scan (segments) for the merge-path search
  auto size_of_output = compute_output_offsets(
      G, input_ptr, segments, context,
      (input_type == advance_io_type_t::graph) ? true : false);

  if (size_of_output <= 0) {
    output.set_number_of_elements(0);
    return;
  }

  if constexpr (output_type != advance_io_type_t::none) {
    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);
    output.set_number_of_elements(size_of_output);
  }

  int num_segments = (input_type == advance_io_type_t::graph)
                         ? G.get_number_of_vertices()
                         : input.get_number_of_elements();
  
  int num_atoms = (size_of_output > (std::size_t)INT_MAX) 
                      ? INT_MAX 
                      : (int)size_of_output;

  if (num_atoms <= 0 || num_segments <= 0)
    return;

  // Set-up launch box for merge-path advance
  // items_per_thread=11 provides good balance between occupancy and work per thread
  using namespace gcuda::launch_box;
  using launch_t = launch_box_t<
      launch_params_dynamic_grid_t<fallback, dim3_t<256>, 11>>;  // 256 threads, 11 items/thread

  launch_t launch_box;
  
  constexpr int merge_tile_size = launch_t::block_dimensions.x * launch_t::items_per_thread;

  // Total merge path length = num_segments + num_atoms
  // This is different from block_mapped which only considers num_segments
  int num_merge_items = num_segments + num_atoms;
  int num_merge_tiles = (num_merge_items + merge_tile_size - 1) / merge_tile_size;

  launch_box.grid_dimensions = dimensions_t(num_merge_tiles, 1, 1);

  // Launch kernel
  auto kernel = merge_path_kernel<
      launch_t::block_dimensions.x,
      launch_t::items_per_thread,
      input_type, output_type,
      graph_t, operator_t, frontier_t,
      typename work_tiles_t::value_type>;

  launch_box.launch(context, kernel,
          G, op, input, output, 
          segments.data().get(), num_segments, num_atoms, num_merge_tiles);

  context.synchronize();
}

}  // namespace merge_path
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
