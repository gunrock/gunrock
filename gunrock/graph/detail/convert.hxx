/**
 * @file convert.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-12-04
 *
 * @copyright Copyright (c) 2020
 *
 */

namespace gunrock {
namespace graph {
namespace convert {

template <typename type_t, typename index_t, typename op_t>
__global__ void for_all_kernel(type_t* array, op_t apply, index_t length) {
  const index_t STRIDE = (index_t)blockDim.x * gridDim.x;
  index_t i = (index_t)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    apply(array + 0, i);
    i += STRIDE;
  }
}

template <memory_space_t space,
          typename type_t,
          typename index_t,
          typename op_t>
void for_all(type_t* elements,
             op_t apply,
             index_t length,
             cudaStream_t stream = 0) {
  if constexpr (space == memory_space_t::host) {
#pragma omp parallel for
    for (index_t i = 0; i < length; i++)
      apply(elements, i);
  }

  if constexpr (space == memory_space_t::device) {
    for_all_kernel<<<256, 256, 0, stream>>>(elements, apply, length);
  }
}

template <memory_space_t space, typename vertex_t, typename edge_t>
void generate_row_indices(vertex_t const& r,
                          edge_t const& nnz,
                          edge_t* Ap,
                          vertex_t* I) {
  for_all<space>(Ap,
                 [I] __device__(edge_t * row_offsets, const vertex_t& row) {
                   edge_t e_end = row_offsets[row + 1];
                   for (edge_t e = row_offsets[row]; e < e_end; e++) {
                     printf("%u\n", row);
                     I[e] = row;
                   }
                 },
                 r, 0);
}

void generate_column_indices() {}

void generate_column_offsets() {}
void generate_row_offsets() {}

}  // namespace convert
}  // namespace graph
}  // namespace gunrock
