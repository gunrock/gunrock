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

namespace gunrock {
namespace sample {

using namespace memory;

template <memory_space_t space = memory_space_t::device,
          typename vertex_t = int,
          typename edge_t = int,
          typename weight_t = float>
format::csr_t<space, vertex_t, edge_t, weight_t> csr() {
  // Logical Matrix Representation
  // r/c  0 1 2 3
  //  0 [ 0 0 0 0 ]
  //  1 [ 5 8 0 0 ]
  //  2 [ 0 0 3 0 ]
  //  3 [ 0 6 0 0 ]

  // Logical Graph Representation
  // (i, j) [w]
  // (1, 0) [5]
  // (1, 1) [8]
  // (2, 2) [3]
  // (3, 1) [6]

  // CSR Matrix Representation
  // V            = [ 5 8 3 6 ]
  // COL_INDEX    = [ 0 1 2 1 ]
  // ROW_OFFSETS  = [ 0 0 2 3 4 ]
  using csr_t = format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  csr_t matrix(4, 4, 4);

  // Row Offsets
  matrix.row_offsets[0] = 0;
  matrix.row_offsets[1] = 0;
  matrix.row_offsets[2] = 2;
  matrix.row_offsets[3] = 3;
  matrix.row_offsets[4] = 4;

  // Column Indices
  matrix.column_indices[0] = 0;
  matrix.column_indices[1] = 1;
  matrix.column_indices[2] = 2;
  matrix.column_indices[3] = 3;

  // Non-zero values
  matrix.nonzero_values[0] = 5;
  matrix.nonzero_values[1] = 8;
  matrix.nonzero_values[2] = 3;
  matrix.nonzero_values[3] = 6;

  if (space == memory_space_t::host) {
    return matrix;
  } else {
    using d_csr_t =
        format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
    d_csr_t d_matrix(matrix);
    return d_matrix;
  }
}

}  // namespace sample
}  // namespace gunrock