/**
 * @file tc.cuh
 * @author Muhammad A. Awad (mawad@ucdavis.edu)
 * @brief Unit test for the triangle counting algorithm.
 * @version 0.1
 * @date 2022-17-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <gunrock/graph/graph.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/algorithms/tc.hxx>

using namespace gunrock;
using namespace memory;

TEST(algorithm, tc) {
  // CSR Matrix Representation
  // V            = [ 1 1 1 1 ]
  // ROW_OFFSETS  = [ 0 3 5 8 10 ]
  // COL_INDEX    = [ 1 2 3 | 0 2 | 0 1 3 | 0 2]

  using vertex_t = int;
  using edge_t = int;
  using weight_t = int;

  vertex_t number_of_rows = 4, number_of_columns = 4;
  edge_t number_of_nonzeros = 10;
  thrust::device_vector<edge_t> Ap = std::vector{0, 3, 5, 8, 10};
  thrust::device_vector<vertex_t> Aj =
      std::vector{1, 2, 3, 0, 2, 0, 1, 3, 0, 2};
  thrust::device_vector<weight_t> Ax(number_of_nonzeros, 0);

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      number_of_rows, number_of_columns, number_of_nonzeros, Ap.data().get(),
      Aj.data().get(), Ax.data().get());

  std::size_t total_triangles = 0;
  thrust::device_vector<vertex_t> d_triangles_count(number_of_rows, 0);
  tc::run(G, true, d_triangles_count.data().get(), &total_triangles);

  thrust::host_vector<vertex_t> h_triangles_count(d_triangles_count);

  std::size_t reference_total_triangles = 6;
  thrust::host_vector<vertex_t> reference_traingles_count =
      std::vector{2, 1, 2, 1};

  for (std::size_t v = 0; v < number_of_rows; v++) {
    EXPECT_EQ(h_triangles_count[v], reference_traingles_count[v]);
  }

  EXPECT_EQ(total_triangles, reference_total_triangles);
}

TEST(algorithm, tc_self_loop_vertex) {
  // CSR Matrix Representation
  // V            = [ 1 1 1 1 ]
  // ROW_OFFSETS  = [ 0 4 7 10 12 ]
  // COL_INDEX    = [ 0 1 2 3 | 0 1 2 | 0 1 3 | 0 2]

  using vertex_t = int;
  using edge_t = int;
  using weight_t = int;

  vertex_t number_of_rows = 4, number_of_columns = 4;
  edge_t number_of_nonzeros = 12;
  thrust::device_vector<edge_t> Ap = std::vector{0, 4, 7, 10, 12};
  thrust::device_vector<vertex_t> Aj =
      std::vector{0, 1, 2, 3, 0, 1, 2, 0, 1, 3, 0, 2};
  thrust::device_vector<weight_t> Ax(number_of_nonzeros, 0);

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      number_of_rows, number_of_columns, number_of_nonzeros, Ap.data().get(),
      Aj.data().get(), Ax.data().get());

  std::size_t total_triangles = 0;
  thrust::device_vector<vertex_t> d_triangles_count(number_of_rows, 0);
  tc::run(G, true, d_triangles_count.data().get(), &total_triangles);

  thrust::host_vector<vertex_t> h_triangles_count(d_triangles_count);

  std::size_t reference_total_triangles = 6;
  thrust::host_vector<vertex_t> reference_traingles_count =
      std::vector{2, 1, 2, 1};

  for (std::size_t v = 0; v < number_of_rows; v++) {
    EXPECT_EQ(h_triangles_count[v], reference_traingles_count[v]);
  }

  EXPECT_EQ(total_triangles, reference_total_triangles);
}
