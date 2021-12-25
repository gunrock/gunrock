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
          typename work_tiles_t,
          typename operator_t>
void __global__ block_mapped_kernel(graph_t const G,
                                    operator_t op,
                                    frontier_t* input,
                                    frontier_t* output,
                                    std::size_t input_size,
                                    work_tiles_t* offsets) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  // TODO: accept gunrock::frontier_t instead of typename frontier_t::type_t.
  using type_t = frontier_t;

  // Specialize Block Scan for 1D block of THREADS_PER_BLOCK.
  using block_scan_t = cub::BlockScan<edge_t, THREADS_PER_BLOCK>;
  // using block_load_t =
  //     cub::BlockLoad<edge_t, THREADS_PER_BLOCK, ITEMS_PER_THREAD>;

  auto global_idx = cuda::thread::global::id::x();
  auto local_idx = cuda::thread::local::id::x();

  thrust::counting_iterator<type_t> all_vertices(0);

  __shared__ union TempStorage {
    typename block_scan_t::TempStorage scan;
    // typename block_load_t::TempStorage load;
  } storage;

  // Prepare data to process (to shmem/registers).
  __shared__ vertex_t offset[1];
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

  // Exclusive sum of degrees.
  edge_t aggregate_degree_per_block;
  block_scan_t(storage.scan)
      .ExclusiveSum(th_deg, th_deg, aggregate_degree_per_block);
  __syncthreads();

  // Store back to shared memory.
  degrees[local_idx] = th_deg[0];

  if (output_type != advance_io_type_t::none) {
    // Accumulate the output size to global memory, only done once per block.
    if (local_idx == 0)
      if ((cuda::block::id::x() * cuda::block::size::x()) < input_size)
        offset[0] = offsets[cuda::block::id::x() * cuda::block::size::x()];
  }

  __syncthreads();

  // To search for which vertex id we are computing on.
  auto length = global_idx - local_idx + cuda::block::size::x();

  // Bound check.
  if (input_size < length)
    length = input_size;

  length -= global_idx - local_idx;

  for (edge_t i = local_idx;            // threadIdx.x
       i < aggregate_degree_per_block;  // total degree to process
       i += cuda::block::size::x()      // increment by blockDim.x
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
    if (output_type != advance_io_type_t::none)
      output[offset[0] + i] =
          (cond && n != v) ? n : gunrock::numeric_limits<vertex_t>::invalid();
  }
}  // namespace block_mapped

template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  if constexpr (output_type != advance_io_type_t::none) {
    // This shouldn't be required for block-mapped.
    auto size_of_output = compute_output_length(G, input, segments, context);

    // If output frontier is empty, resize and return.
    if (size_of_output <= 0) {
      output->set_number_of_elements(0);
      return;
    }

    // <todo> Resize the output (inactive) buffer to the new size.
    // Can be hidden within the frontier struct.
    if (output->get_capacity() < size_of_output)
      output->reserve(size_of_output);
    output->set_number_of_elements(size_of_output);
  }

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input->get_number_of_elements();

  // Set-up and launch block-mapped advance.
  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<128>>>;

  launch_t launch_box;

  launch_box.calculate_grid_dimensions_strided(num_elements);
  auto __bm = block_mapped_kernel<        // kernel
      launch_box.block_dimensions.x,      // threas per block
      1,                                  // items per thread
      input_type, output_type,            // i/o parameters
      graph_t,                            // graph type
      typename frontier_t::type_t,        // frontier value type
      typename work_tiles_t::value_type,  // segments value type
      operator_t                          // lambda type
      >;
  auto __args = std::make_tuple(G, op, input->data(), output->data(),
                                num_elements, segments.data().get());
  launch_box.launch(__bm, __args, context);
  context.synchronize();
}

}  // namespace block_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
