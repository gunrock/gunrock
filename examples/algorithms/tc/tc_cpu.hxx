/**
 * @file tc_cpu.hxx
 * @author Muhammad A. Awad (mawad@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-06-24
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <chrono>
#include <vector>
#include <algorithm>

namespace tc_cpu {

using namespace std::chrono;

template <typename csr_t, typename count_t>
float run(const csr_t& csr,
          std::vector<count_t>& triangles_count,
          std::size_t& total_triangles) {
  using edge_t = typename csr_t::offset_type;
  using vertex_t = typename csr_t::index_type;

  thrust::host_vector<edge_t> row_offsets(csr.row_offsets);
  thrust::host_vector<vertex_t> column_indices(csr.column_indices);
  vertex_t n_vertices = csr.number_of_rows;
  vertex_t n_edges = csr.number_of_nonzeros;

  auto t_start = high_resolution_clock::now();

  for (vertex_t source = 0; source < n_vertices; source++) {
    auto source_offset_start = row_offsets[source];
    auto source_offset_end = row_offsets[source + 1];
    auto source_neighbors_count = source_offset_end - source_offset_start;
    if (source_neighbors_count == 0)
      continue;
    auto source_neighbors_ptr = column_indices.data() + source_offset_start;

    auto needle = source_neighbors_ptr[0];

    for (vertex_t i = 0; i < source_neighbors_count; i++) {
      auto destination = source_neighbors_ptr[i];
      if (destination >= source)
        break;
      auto destination_offset_start = row_offsets[destination];
      auto destination_offset_end = row_offsets[destination + 1];
      auto destination_neighbors_count =
          destination_offset_end - destination_offset_start;
      if (destination_neighbors_count == 0)
        continue;
      auto destination_neighbors_ptr =
          column_indices.data() + destination_offset_start;

      auto destination_search_end =
          destination_neighbors_ptr + destination_neighbors_count;
      auto destination_search_begin = std::lower_bound(
          destination_neighbors_ptr, destination_search_end, needle);

      auto source_search_begin = source_neighbors_ptr;
      auto source_search_end = source_neighbors_ptr + source_neighbors_count;

      while (source_search_begin < source_search_end &&
             destination_search_begin < destination_search_end) {
        auto source_neighbor = *source_search_begin;
        auto destination_neighbor = *destination_search_begin;
        if (source_neighbor == destination_neighbor) {
          if (source_neighbor != source && source_neighbor != destination) {
            triangles_count[source_neighbor]++;
            total_triangles++;
          }
          destination_search_begin++;
          source_search_begin++;
        } else if (source_neighbor > destination_neighbor) {
          destination_search_begin++;
        } else {
          source_search_begin++;
        }
      }
    }
  }
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace tc_cpu