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

#include <moderngpu/operators.hxx>

namespace gunrock {
namespace operators {
namespace advance {
namespace block_mapped {

int32_t __device__ upperbound(int32_t* array, int32_t key, int32_t len) {
  int32_t s = 0;
  while (len > 0) {
    int32_t half = len >> 1;
    int32_t mid = s + half;
    if (array[mid] > key) {
      len = half;
    } else {
      s = mid + 1;
      len = len - half - 1;
    }
  }
  return s;
}

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
  using block_load_t =
      cub::BlockLoad<edge_t, THREADS_PER_BLOCK, ITEMS_PER_THREAD>;

  __shared__ union TempStorage {
    typename block_scan_t::TempStorage load;
    typename block_load_t::TempStorage scan;
  } storage;

  __shared__ vertex_t shared_vertices[THREADS_PER_BLOCK];
  __shared__ edge_t shared_degrees[THREADS_PER_BLOCK];
  __shared__ edge_t shared_starting_edges[THREADS_PER_BLOCK];

  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  auto load_degree = [=] __device__(int index) {
    if (index < input_size) {
      vertex_t v = input[index];
      shared_vertices[index] = v;
      if (gunrock::util::limits::is_valid(v)) {
        shared_starting_edges[threadIdx.x] = G.get_starting_edge(v);
        return G.get_number_of_neighbors(v);
      } else
        return 0;
    } else {
      shared_vertices[index] = gunrock::numeric_limits<vertex_t>::invalid();
      return 0;
    }
  };

  edge_t degrees[ITEMS_PER_THREAD];

  block_load_t(storage.load)
      .Load(mgpu::make_load_iterator<edge_t>(load_degree), degrees);

  __syncthreads();

  // if (idx < input_size) {
  //   v = input[idx];
  //   shared_vertices[threadIdx.x] = v;
  //   printf("v = %i\n", v);

  //   if (gunrock::util::limits::is_valid(v)) {
  //     shared_degrees[threadIdx.x] = G.get_number_of_neighbors(v);
  //     shared_starting_edges[threadIdx.x] = G.get_starting_edge(v);
  //   } else {
  //     shared_degrees[threadIdx.x] = 0;
  //     shared_starting_edges[threadIdx.x] =
  //         gunrock::numeric_limits<vertex_t>::invalid();
  //   }
  // } else {
  //   shared_vertices[threadIdx.x] =
  //   gunrock::numeric_limits<vertex_t>::invalid(); shared_degrees[threadIdx.x]
  //   = 0;
  // }

  printf("shared degree = %i\n", degrees[0]);

  __syncthreads();
  vertex_t aggregate_degree_per_block;
  block_scan_t(storage.scan)
      .ExclusiveSum(degrees, degrees, aggregate_degree_per_block);
  __syncthreads();

  printf("#### shared degree = %i\n", degrees[0]);

  // auto width = idx - threadIdx.x + blockDim.x;
  // if (aggregate_degree_per_block < width)
  //   width = aggregate_degree_per_block;

  // width -= idx - threadIdx.x;
  // for (int i = threadIdx.x; i < aggregate_degree_per_block; i += blockDim.x)
  // {
  //   int id = /* algo::search::binary::rightmost */ upperbound(shared_degrees,
  //   i,
  //                                                             width) -
  //            1;
  //   if (id >= width)
  //     continue;

  //   vertex_t v = shared_vertices[id];
  //   if (!gunrock::util::limits::is_valid(v))
  //     continue;

  //   auto e = shared_starting_edges[id] + i - shared_degrees[id];
  //   auto n = G.get_destination_vertex(e);
  //   auto w = G.get_edge_weight(e);

  //   // printf("(v, n, e) = (%i, %i, %i), %i\n", v, n, e,
  //   //        aggregate_degree_per_block);
  //   bool cond = op(v, e, n, w);
  //   output[shared_degrees[id]] =
  //       cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
  // }
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

  thrust::host_vector<std::size_t> h_output_size = output_size;
  output->set_number_of_elements(h_output_size[0]);
}

}  // namespace block_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock