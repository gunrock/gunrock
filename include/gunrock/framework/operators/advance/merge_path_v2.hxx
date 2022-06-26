/**
 * @file merge_path_v2.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-01-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace merge_path_v2 {

template <typename index_t>
struct coordinate_t {
  index_t x, y;
};

namespace search {
/**
 * @brief Thrust based 2D binary-search.
 *
 */
template <typename offset_t, typename a_iterator_t, typename b_iterator_t>
__host__ __device__ __forceinline__ void merge_path(
    const offset_t diagonal,
    const a_iterator_t a,
    const b_iterator_t b,
    const offset_t a_len,
    const offset_t b_len,
    coordinate_t<offset_t>& coordinate) {
  // Diagonal search range (in x-coordinate space)
  offset_t x_min = max(diagonal - b_len, 0);
  offset_t x_max = min(diagonal, a_len);

  auto it = thrust::lower_bound(
      thrust::seq,                                 // Sequential impl
      thrust::counting_iterator<offset_t>(x_min),  // Start iterator @ x_min
      thrust::counting_iterator<offset_t>(x_max),  // End   iterator @ x_max
      diagonal,                                    // ...
      [=] __host__ __device__(const offset_t& idx, const offset_t& diagonal) {
        return a[idx] <= b[diagonal - idx - 1];
      });

  coordinate.x = min(*it, a_len);
  coordinate.y = diagonal - *it;
}
}  // namespace search

template <std::size_t items_per_thread,
          std::size_t num_threads,
          std::size_t merge_tile_size,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
__global__ void merge_path_v2_kernel(graph_t G,
                                     operator_t op,
                                     frontier_t input,
                                     frontier_t output,
                                     work_tiles_t* segments,
                                     std::size_t num_merge_tiles) {
  using type_t = typename frontier_t::type_t;
  using offset_t = typename frontier_t::offset_t;

  __shared__ coordinate_t<type_t> tile_coords[2];
  __shared__ offset_t
      _tile_row_end_offsets[items_per_thread + merge_tile_size + 1];

  auto elements = (input_type == advance_io_type_t::graph)
                      ? G.get_number_of_vertices()
                      : input.get_number_of_elements();

  auto atoms = (input_type == advance_io_type_t::graph)
                   ? G.get_number_of_edges()
                   : output.get_number_of_elements();

  auto row_end_offsets = (input_type == advance_io_type_t::graph)
                             ? G.get_row_offsets() + 1
                             : segments + 1;

  int tile_idx = blockIdx.x * gridDim.y + blockIdx.y;  // Tile Index
  if (tile_idx >= num_merge_tiles)
    return;

  /// For each block, generate the starting and ending coordinates of a tile.
  /// This corresponds to the starting and ending vertices and the edges.
  if (threadIdx.x < 2) {
    coordinate_t<offset_t> coords;
    offset_t diagonal = (tile_idx + threadIdx.x) * merge_tile_size;
    thrust::counting_iterator<offset_t> indices(0);

    search::merge_path((offset_t)diagonal,  // diagonal
                       row_end_offsets,     // list A
                       indices,             // list B
                       (offset_t)elements,  // input elements
                       (offset_t)atoms,     // work items
                       coords               // coordinates
    );

    tile_coords[threadIdx.x] = coords;
  }
  __syncthreads();

  coordinate_t<type_t> tile_idx_start = tile_coords[0];
  coordinate_t<type_t> tile_idx_end = tile_coords[1];

  // Consume Tile
  type_t tile_num_rows = tile_idx_end.x - tile_idx_start.x;
  type_t tile_num_nonzeros = tile_idx_end.y - tile_idx_start.y;

// Gather the row end-offsets for the merge tile into shared memory
#pragma unroll 1
  for (int item = threadIdx.x; item < tile_num_rows + items_per_thread;
       item += num_threads) {
    const offset_t offset =
        min((offset_t)(tile_idx_start.x + item), (offset_t)(elements - 1));
    _tile_row_end_offsets[item] = row_end_offsets[offset];
  }

  __syncthreads();

  // Identify starting and ending diagonals and find starting and ending
  // Merge-Path coordinates (row-idx, nonzero-idx) for each thread.
  thrust::counting_iterator<offset_t> tile_nonzeros(
      tile_idx_start.y);  // Per Thread Merge
                          // list B: Non-zero indices (ℕ)
  coordinate_t<offset_t> thread_idx_start;

  search::merge_path((offset_t)(threadIdx.x * items_per_thread),  // Diagonal
                     _tile_row_end_offsets,             // List A: SHMEM
                     tile_nonzeros,                     // List B
                     tile_num_rows, tile_num_nonzeros,  // NZR, NNZ
                     thread_idx_start                   // Output: Coords
  );
  __syncthreads();

  // reset back to initial M,NZ indices for the new column
  auto source = thread_idx_start.x;
  auto nz_idx = thread_idx_start.y;

#pragma unroll
  for (int ITEM = 0; ITEM < items_per_thread; ++ITEM) {
    offset_t edge = min((offset_t)tile_nonzeros[nz_idx], (offset_t)(atoms - 1));
    auto row_end_offset = _tile_row_end_offsets[source];
    auto neighbor = G.get_destination_vertex(edge);
    auto weight = G.get_edge_weight(edge);

    if (tile_nonzeros[nz_idx] < row_end_offset) {
      // Move down (accumulate)
      bool cond = op(source, neighbor, edge, weight);
      printf("%d %d (%d, %d) %f\n", (int)source, (int)neighbor, (int)edge,
             (int)nz_idx, (float)weight);

      if (output_type != advance_io_type_t::none) {
        // std::size_t out_idx = ;
        // type_t element = cond ? neighbor
        //                  : gunrock::numeric_limits<type_t>::invalid();
        // output.set_element_at(element, out_idx);
      }

      ++nz_idx;
    } else {
      // Move right (reset)
      ++source;
    }
  }
  //   __syncthreads();
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
             frontier_t& input,
             frontier_t& output,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  auto size_of_output = compute_output_offsets(
      G, &input, segments, context,
      (input_type == advance_io_type_t::graph) ? true : false);

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input.get_number_of_elements();

  // Kernel configuration.
  constexpr std::size_t num_threads = 128;
  constexpr std::size_t items_per_thread = 1;

  // Tile size of merge path.
  constexpr std::size_t merge_tile_size = num_threads * items_per_thread;

  // Total number of work items to process.
  std::size_t num_merge_items = num_elements + size_of_output;

  // Number of tiles for the kernel.
  std::size_t num_merge_tiles =
      math::divide_round_up(num_merge_items, merge_tile_size);

  dim3 grid(num_merge_tiles, num_merge_tiles, 1);

  // Launch kernel.
  merge_path_v2_kernel<items_per_thread, num_threads, merge_tile_size,
                       input_type, output_type>
      <<<grid, num_threads, 0, context.stream()>>>(
          G, op, input, output, segments.data().get(), num_merge_tiles);

  context.synchronize();
}
}  // namespace merge_path_v2
}  // namespace advance
}  // namespace operators
}  // namespace gunrock