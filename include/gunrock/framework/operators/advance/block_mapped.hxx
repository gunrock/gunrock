/**
 * @file block_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/transform_scan.h>
#include <thrust/iterator/discard_iterator.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

namespace gunrock {
namespace operators {
namespace advance {
namespace block_mapped {

template <unsigned int THREADS_PER_BLOCK,
          unsigned int ITEMS_PER_THREAD,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename frontier_t,
          typename offset_counter_t,
          typename operator_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
    block_mapped_kernel(graph_t const G,
                        operator_t op,
                        frontier_t* input,
                        frontier_t* output,
                        std::size_t input_size,
                        offset_counter_t* block_offsets) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  // TODO: accept gunrock::frontier_t instead of typename frontier_t::type_t.
  using type_t = frontier_t;

  // Specialize Block Scan for 1D block of THREADS_PER_BLOCK.
  using block_scan_t = cub::BlockScan<edge_t, THREADS_PER_BLOCK>;

  auto global_idx = gcuda::thread::global::id::x();
  auto local_idx = gcuda::thread::local::id::x();

  thrust::counting_iterator<type_t> all_vertices(0);
  __shared__ typename block_scan_t::TempStorage scan;
  __shared__ uint64_t offset[1];

  /// 1. Load input data to shared/register memory.
  __shared__ vertex_t vertices[THREADS_PER_BLOCK];
  __shared__ edge_t degrees[THREADS_PER_BLOCK];
  __shared__ edge_t sedges[THREADS_PER_BLOCK];
  edge_t th_deg[ITEMS_PER_THREAD];

  if (global_idx < input_size) {
    vertex_t v = (input_type == advance_io_type_t::graph)
                     ? all_vertices[global_idx]
                     : input[global_idx];
    vertices[local_idx] = v;
    if (gunrock::util::limits::is_valid(v)) {
      sedges[local_idx] = G.get_starting_edge(v);
      th_deg[0] = G.get_number_of_neighbors(v);
    } else {
      th_deg[0] = 0;
    }
  } else {
    vertices[local_idx] = gunrock::numeric_limits<vertex_t>::invalid();
    th_deg[0] = 0;
  }
  __syncthreads();

  /// 2. Exclusive sum of degrees to find total work items per block.
  edge_t aggregate_degree_per_block;
  block_scan_t(scan).ExclusiveSum(th_deg, th_deg, aggregate_degree_per_block);
  __syncthreads();

  // Store back to shared memory (to later use in the binary search).
  degrees[local_idx] = th_deg[0];

  /// 3. Compute block offsets if there's an output frontier.
  if constexpr (output_type != advance_io_type_t::none) {
    // Accumulate the output size to global memory, only done once per block by
    // threadIdx.x == 0, and retrieve the previously stored value from the
    // global memory. The previous value is now current block's starting offset.
    // All writes from this block will be after this offset. Note: this does not
    // guarantee that the data written will be in any specific order.
    if (local_idx == 0)
      offset[0] = math::atomic::add(
          &block_offsets[0], (offset_counter_t)aggregate_degree_per_block);
  }
  __syncthreads();

  auto length = global_idx - local_idx + gcuda::block::size::x();

  if (input_size < length)
    length = input_size;

  length -= global_idx - local_idx;

  /// 4. Compute. Using binary search, find the source vertex each thread is
  /// processing, and the corresponding edge, neighbor and weight tuple. Passed
  /// to the user-defined lambda operator to process. If there's an output, the
  /// resultant neighbor or invalid vertex is written to the output frontier.
  for (edge_t i = local_idx;            // threadIdx.x
       i < aggregate_degree_per_block;  // total degree to process
       i += gcuda::block::size::x()     // increment by blockDim.x
  ) {
    // Binary search to find which vertex id to work on.
    int id = search::binary::rightmost(degrees, i, length);

    // If the id is greater than the width of the block or the input size, we
    // exit.
    if (id >= length)
      continue;

    // Fetch the vertex corresponding to the id.
    vertex_t v = vertices[id];
    if (!gunrock::util::limits::is_valid(v))
      continue;

    // If the vertex is valid, get its edge, neighbor and edge weight.
    auto e = sedges[id] + i - degrees[id];
    auto n = G.get_destination_vertex(e);
    auto w = G.get_edge_weight(e);

    // Use-defined advance condition.
    bool cond = op(v, n, e, w);

    // Store [neighbor] into the output frontier.
    if constexpr (output_type != advance_io_type_t::none) {
      output[offset[0] + i] =
          cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
    }
  }
}

template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t& input,
             frontier_t& output,
             gcuda::standard_context_t& context) {
  if constexpr (output_type != advance_io_type_t::none) {
    auto size_of_output = compute_output_length(G, input, context);

    // If output frontier is empty, resize and return.
    if (size_of_output <= 0) {
      output.set_number_of_elements(0);
      return;
    }

    /// @todo Resize the output (inactive) buffer to the new size.
    /// Can be hidden within the frontier struct.
    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);
    output.set_number_of_elements(size_of_output);
  }

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input.get_number_of_elements();

  // Set-up and launch block-mapped advance.
  using namespace gcuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>>>;

  launch_t launch_box;

  launch_box.calculate_grid_dimensions_strided(num_elements);
  auto kernel = block_mapped_kernel<  // kernel
      launch_box.block_dimensions.x,  // threas per block
      1,                              // items per thread
      input_type, output_type,        // i/o parameters
      graph_t,                        // graph type
      typename frontier_t::type_t,    // frontier value type
      typename frontier_t::offset_t,  // segments value type
      operator_t                      // lambda type
      >;

  /// @todo Is there a better place to create block_offsets? This is always a
  /// one element array.
  thrust::device_vector<typename frontier_t::offset_t> block_offsets(1);
  launch_box.calculate_grid_dimensions_strided(num_elements);
  launch_box.launch(context, kernel, G, op, input.data(), output.data(),
                    num_elements, block_offsets.data().get());
  context.synchronize();
}

}  // namespace block_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
