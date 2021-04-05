/**
 * @file thread_mapped.hxx
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
#include <gunrock/cuda/context.hxx>
#include <gunrock/cuda/launch_box.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/transform_scan.h>
#include <thrust/iterator/discard_iterator.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

namespace gunrock {
namespace operators {
namespace advance {
namespace block_mapped {

template <int THREADS_PER_BLOCK,
          int ITEMS_PER_THREAD,
          typename graph_t,
          typename frontier_t,
          typename operator_t>
void __global__ block_mapped_kernel(graph_t const G,
                                    operator_t op,
                                    frontier_t* input,
                                    frontier_t* output,
                                    std::size_t input_size,
                                    std::size_t* output_size) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  // Specialize Block Scan for 1D block of THREADS_PER_BLOCK.
  using block_scan_t = cub::BlockScan<edge_t, THREADS_PER_BLOCK>;
  // using block_load_t =
  //     cub::BlockLoad<edge_t, THREADS_PER_BLOCK, ITEMS_PER_THREAD>;

  __shared__ union TempStorage {
    typename block_scan_t::TempStorage scan;
    // typename block_load_t::TempStorage load;
  } storage;

  __shared__ vertex_t shared_vertices[THREADS_PER_BLOCK];
  __shared__ edge_t shared_degrees[THREADS_PER_BLOCK];
  __shared__ edge_t shared_starting_edges[THREADS_PER_BLOCK];

  auto idx = threadIdx.x + (blockDim.x * blockIdx.x);

  if (idx < input_size) {
    vertex_t v = input[idx];
    shared_vertices[threadIdx.x] = v;
    if (gunrock::util::limits::is_valid(v)) {
      shared_starting_edges[threadIdx.x] = G.get_starting_edge(v);
      shared_degrees[threadIdx.x] = G.get_number_of_neighbors(v);
    } else {
      shared_degrees[threadIdx.x] = 0;
    }
  } else {
    shared_vertices[threadIdx.x] = gunrock::numeric_limits<vertex_t>::invalid();
    shared_degrees[threadIdx.x] = 0;
  }

  // Per-thread tile data
  __syncthreads();
  edge_t deg[ITEMS_PER_THREAD];
  deg[0] = shared_degrees[threadIdx.x];

  // if (idx < input_size)
  // printf("\t# input: shared_degrees = %i\n", shared_degrees[threadIdx.x]);

  __syncthreads();
  vertex_t aggregate_degree_per_block;
  block_scan_t(storage.scan).ExclusiveSum(deg, deg, aggregate_degree_per_block);
  __syncthreads();

  // Store back to shared memory
  shared_degrees[threadIdx.x] = deg[0];
  __syncthreads();

  // if (idx < input_size)
  //   printf("\t# output: shared_degrees = %i\n", shared_degrees[threadIdx.x]);

  if (threadIdx.x == 0)
    math::atomic::add((vertex_t*)output_size, aggregate_degree_per_block);

  auto length = idx - threadIdx.x + blockDim.x;

  // Bound check.
  if (input_size < length)
    length = input_size;
  length -= idx - threadIdx.x;

  for (int i = threadIdx.x; i < aggregate_degree_per_block; i += blockDim.x) {
    // printf("global_edge_processing = %i\n", i);
    int id = algo::search::binary::rightmost(shared_degrees, i, length);

    // printf("id, length = %i, %i\n", id, length);
    if (id >= length)
      continue;

    vertex_t v = shared_vertices[id];

    // printf("vertex = %i\n", v);
    if (!gunrock::util::limits::is_valid(v))
      continue;

    auto e = shared_starting_edges[id] + i - shared_degrees[id];
    auto n = G.get_destination_vertex(e);
    auto w = G.get_edge_weight(e);
    bool cond = op(v, n, e, w);

    /* printf("output[%i] = v, n, e, w, cond = %i, %i, %i, %f, %s\n", i, v, n,
       e, w, cond ? "true" : "false"); */

    output[i] =
        (cond && n != v) ? n : gunrock::numeric_limits<vertex_t>::invalid();
  }
}

template <advance_type_t type,
          advance_direction_t direction,
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
  // TODO: Use launch_box_t instead.
  constexpr int block_size = 128;
  int grid_size =
      (input->get_number_of_elements() + block_size - 1) / block_size;

  thrust::device_vector<std::size_t> output_size(1, 0);

  // Launch blocked-mapped advance kernel.
  block_mapped_kernel<block_size, 1>
      <<<grid_size, block_size, 0, context.stream()>>>(
          G, op, input->data(), output->data(), input->get_number_of_elements(),
          output_size.data().get());
  context.synchronize();

  thrust::host_vector<std::size_t> h_output_size = output_size;
  std::cout << "Output Size = " << h_output_size[0] << std::endl;
  output->set_number_of_elements(h_output_size[0]);
}

}  // namespace block_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock