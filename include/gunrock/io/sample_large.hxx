/**
 * @file sample.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-23
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/formats/formats.hxx>
#include <gunrock/graph/graph.hxx>

namespace gunrock {
namespace io {
namespace sample_large {

using namespace memory;

/**
 * @brief Returns a large sample CSR matrix of size 10000 x 10000,
 * filled with ones.
 *
 * @tparam space Memory space of the CSR matrix.
 * @tparam vertex_t Type of vertex.
 * @tparam edge_t Type of edge.
 * @tparam weight_t Type of weight.
 * @return format::csr_t<space, vertex_t, edge_t, weight_t> CSR matrix.
 */
template <memory_space_t space = memory_space_t::device,
          typename vertex_t = int,
          typename edge_t = int,
          typename weight_t = float>
format::csr_t<space, vertex_t, edge_t, weight_t> csr() {
  using csr_t = format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;

  int dim = 10000;
  csr_t matrix(dim, dim, dim * dim);

  // Row Offsets
  thrust::host_vector<int> rowSeq(dim + 1);
  thrust::host_vector<int> multVector(dim + 1);
  thrust::sequence(rowSeq.begin(), rowSeq.end());
  thrust::fill(multVector.begin(), multVector.end(), dim);
  thrust::transform(rowSeq.begin(), rowSeq.end(), multVector.begin(),
                    matrix.row_offsets.begin(), thrust::multiplies<int>());

  // Column Indices
  thrust::host_vector<int> colSeq(dim * dim);
  thrust::host_vector<int> modVector(dim * dim);
  thrust::sequence(colSeq.begin(), colSeq.end());
  thrust::fill(modVector.begin(), modVector.end(), dim);
  thrust::transform(colSeq.begin(), colSeq.end(), modVector.begin(),
                    matrix.column_indices.begin(), thrust::modulus<int>());

  // Non-zero values
  thrust::fill(matrix.nonzero_values.begin(), matrix.nonzero_values.end(), 1);

  if (space == memory_space_t::host) {
    return matrix;
  } else {
    using d_csr_t =
        format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
    d_csr_t d_matrix(matrix);
    return d_matrix;
  }
}

}  // namespace sample_large
}  // namespace io
}  // namespace gunrock