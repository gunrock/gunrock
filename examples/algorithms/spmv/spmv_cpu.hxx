/**
 * @file spmv_cpu.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-03-14
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <chrono>
#include <vector>
#include <queue>

#include <gunrock/memory.hxx>
#include <gunrock/formats/formats.hxx>

namespace spmv_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename dvector_t, typename vector_t>
float run(csr_t& csr, dvector_t& x, vector_t& y) {
  using vertex_t = typename csr_t::index_type;
  using edge_t = typename csr_t::offset_type;
  using weight_t = typename csr_t::value_type;

  thrust::host_vector<edge_t> row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> column_indices = csr.column_indices;
  thrust::host_vector<weight_t> values = csr.nonzero_values;

  thrust::host_vector<weight_t> x_h = x;

  auto t_start = high_resolution_clock::now();

  for (auto row = 0; row < csr.number_of_rows; ++row) {
    weight_t sum = 0;
    for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
      sum += values[nz] * x_h[column_indices[nz]];
    }
    y[row] = sum;
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace spmv_cpu