
/**
 * @file merge_path_v2.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
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
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <typename AIteratorT,
          typename BIteratorT,
          typename OffsetT,
          typename CoordinateT>
__host__ __device__ __forceinline__ void merge_path(
    OffsetT diagonal,
    AIteratorT a,
    BIteratorT b,
    OffsetT a_len,
    OffsetT b_len,
    CoordinateT& path_coordinate) {
  OffsetT split_min = std::max(diagonal - b_len, OffsetT(0));
  OffsetT split_max = std::min(diagonal, a_len);

  while (split_min < split_max) {
    OffsetT split_pivot = (split_min + split_max) >> 1;
    if (a[split_pivot] <= b[diagonal - split_pivot - 1]) {
      // Move candidate split range up A, down B
      split_min = split_pivot + 1;
    } else {
      // Move candidate split range up B, down A
      split_max = split_pivot;
    }
  }

  path_coordinate.x = std::min(split_min, a_len);
  path_coordinate.y = diagonal - split_min;
}
}  // namespace search

template <std::size_t items_per_thread,
          std::size_t threads_per_block,
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

  __shared__ coordinate_t<offset_t> tile_coords[2];
  __shared__ offset_t
      tile_row_offsets_end[items_per_thread + merge_tile_size + 1];

  auto elements = (input_type == advance_io_type_t::graph)
                      ? G.get_number_of_vertices()
                      : input.get_number_of_elements();

  auto atoms = (input_type == advance_io_type_t::graph)
                   ? G.get_number_of_edges()
                   : output.get_number_of_elements();

  auto row_end_offsets = (input_type == advance_io_type_t::graph)
                             ? G.get_row_offsets() + 1
                             : segments + 1;

  int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y;  // Tile Index
  if (tile_idx >= num_merge_tiles)
    return;

  /// For each block, generate the starting and ending coordinates of a tile.
  /// This corresponds to the starting and ending vertices and the edges.
  if (threadIdx.x < 2) {
    coordinate_t<offset_t> coords;
    offset_t diagonal = (tile_idx + threadIdx.x) * merge_tile_size;
    thrust::counting_iterator<offset_t> indices(0);

    search::merge_path(static_cast<offset_t>(diagonal),  // diagonal
                       row_end_offsets,                  // list A
                       indices,                          // list B
                       static_cast<offset_t>(elements),  // input elements
                       static_cast<offset_t>(atoms),     // work items
                       coords                            // coordinates
    );

    tile_coords[threadIdx.x] = coords;
  }
  __syncthreads();

  coordinate_t<offset_t> tile_idx_start = tile_coords[0];
  coordinate_t<offset_t> tile_idx_end = tile_coords[1];

  // Consume Tile
  offset_t tile_num_rows = tile_idx_end.x - tile_idx_start.x;
  offset_t tile_num_nonzeros = tile_idx_end.y - tile_idx_start.y;

// Gather the row end-offsets for the merge tile into shared memory
#pragma unroll 1
  for (unsigned int item = threadIdx.x;          //
       item < tile_num_rows + items_per_thread;  //
       item += threads_per_block) {
    const int offset = std::min(static_cast<int>(tile_idx_start.x + item),
                                static_cast<int>(elements - 1));
    tile_row_offsets_end[item] = row_end_offsets[offset];
  }

  __syncthreads();

  // Identify starting and ending diagonals and find starting and ending
  // Merge-Path coordinates (row-idx, nonzero-idx) for each thread.
  thrust::counting_iterator<offset_t> tile_nonzeros(
      tile_idx_start.y);  // Per Thread Merge
                          // list B: Non-zero indices (ℕ)

  coordinate_t<offset_t> thread_idx_start;
  // coordinate_t<offset_t> thread_idx_end;

  search::merge_path(
      static_cast<offset_t>(threadIdx.x * items_per_thread),  // Diagonal
      tile_row_offsets_end,                                   // List A: SHMEM
      tile_nonzeros,                                          // List B
      tile_num_rows, tile_num_nonzeros,                       // NZR, NNZ
      thread_idx_start                                        // Output: Coords
  );

  // Perf-sync, not required.
  __syncthreads();

  // Early exit if out-of-bound.
  if (thread_idx_start.x >= elements) {
#if 0
    printf("REACHED: (row,elements,nz) %d,%d,%d\n",  //
           thread_idx_start.x,                       //
           static_cast<offset_t>(elements),          //
           tile_nonzeros[thread_idx_start.y]         //
    );
#endif
    return;
  }

#pragma unroll
  for (int ITEM = 0; ITEM < items_per_thread; ++ITEM) {
    if (tile_nonzeros[thread_idx_start.y] <
        tile_row_offsets_end[thread_idx_start.x]) {
      // Find the actual vertex by queuing the frontier/graph.
      auto v = (input_type == advance_io_type_t::graph)
                   ? type_t(thread_idx_start.x)
                   : input.get_element_at(thread_idx_start.x);

      // If vertex is invalid, exit. If we do this exit outside this
      // if-condition, we skip a very important aspect of incrementing the
      // thread_idx_start.x (in the else statement). This will cause parts of
      // the frontier to be unexplored.
      if (!gunrock::util::limits::is_valid(v))
        continue;

      // Contiguous index that goes from 0..num_edges_for_frontier, unique among
      // all threads across all blocks.
      offset_t global_contiguous_edge_index_processed =
          std::min(static_cast<offset_t>(tile_nonzeros[thread_idx_start.y]),
                   static_cast<offset_t>(atoms - 1));

      // Some weird computation to figure out which edge we are
      // processing in the frontier. This is simpler when being done
      // for the whole graph, but for frontier with non-sorted subset of the
      // graph vertices, we have to do this.
      auto starting_edge = G.get_starting_edge(v);
      offset_t offset_for_correct_edge_for_this_vertex =
          (thread_idx_start.x == 0)
              ? 0
              : tile_row_offsets_end[thread_idx_start.x - 1];

      // We compute the edge in the Graph by finding the offset of a
      // previous vertex in the segments array, and subtracting that
      // from the global contiguous edge count + starting edge.
      offset_t e = starting_edge + (global_contiguous_edge_index_processed -
                                    offset_for_correct_edge_for_this_vertex);

#if 0
      printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",        //
             gcuda::thread::global::id::x(),           //
             thread_idx_start.x,                       //
             v,                                        //
             thread_idx_start.y,                       //
             e,                                        //
             global_contiguous_edge_index_processed,   //
             starting_edge,                            //
             offset_for_correct_edge_for_this_vertex,  //
             tile_nonzeros[thread_idx_start.y],        //
             tile_row_offsets_end[thread_idx_start.x]);
#endif

      // Fetch neighbor, weight, and run the op.
      auto n = G.get_destination_vertex(e);
      auto w = G.get_edge_weight(e);
      bool cond = op(v, n, e, w);

      // Output to an output frontier if it exists.
      if (output_type != advance_io_type_t::none) {
        type_t element = cond ? n : gunrock::numeric_limits<type_t>::invalid();
        output.set_element_at(element, global_contiguous_edge_index_processed);
      }

      // Next edge.
      ++thread_idx_start.y;
    } else {
      // Next source vertex.
      ++thread_idx_start.x;
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
             frontier_t& input,
             frontier_t& output,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;
  // Merge-path works on the "prefix sum", so even if there is no output
  // required for this; we still need the scan to balance the work.
  std::size_t size_of_output = compute_output_offsets(
      G, &input, segments, context,
      (input_type == advance_io_type_t::graph) ? true : false);

  // If output frontier is empty, resize and return.
  if (size_of_output <= 0) {
    output.set_number_of_elements(0);
    return;
  }

  if constexpr (output_type != advance_io_type_t::none) {
    /// @todo Resize the output (inactive) buffer to the new size.
    /// Can be hidden within the frontier struct.
    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);
    output.set_number_of_elements(size_of_output);
    output.fill(gunrock::numeric_limits<type_t>::invalid(), context.stream());
  }

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input.get_number_of_elements();

  // Calculated number of blocks, if they do not fit into the x-dimension
  // of dim3, overflow the rest to y-dimension (for larger problems)
  int max_dim_x =
      gcuda::properties::get_max_grid_dimension_x(context.ordinal());

  // Kernel configuration.
  constexpr std::size_t num_threads = 128;
  constexpr std::size_t items_per_thread = 5;

  // Tile size of merge path.
  constexpr std::size_t merge_tile_size = num_threads * items_per_thread;

  // Total number of work items to process.
  std::size_t num_merge_items = num_elements + size_of_output;

  // Number of tiles for the kernel.
  std::size_t num_merge_tiles =
      math::divide_round_up(num_merge_items, merge_tile_size);

  std::size_t within_bounds =
      std::min(num_merge_tiles, static_cast<std::size_t>(max_dim_x));
  std::size_t overflow = math::divide_round_up(
      num_merge_tiles, static_cast<std::size_t>(max_dim_x));

  dim3 grid(within_bounds, overflow, 1);

  std::cout << "Input ";
  input.sort();
  input.print();

  // Launch kernel.
  merge_path_v2_kernel<items_per_thread, num_threads, merge_tile_size,
                       input_type, output_type>
      <<<grid, num_threads, 0, context.stream()>>>(
          G, op, input, output, segments.data().get(), num_merge_tiles);

  context.synchronize();

  std::cout << "Output ";
  output.sort();
  output.print();
}
}  // namespace merge_path_v2
}  // namespace advance
}  // namespace operators
}  // namespace gunrock