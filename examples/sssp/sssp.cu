#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp/sssp.hxx>

using namespace gunrock;

void test_sssp() {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  constexpr memory::memory_space_t space = memory::memory_space_t::device;

  // Logical Graph Representation
  // (i, j) [w]
  // (1, 0) [5]
  // (1, 1) [8] // Self-loop
  // (2, 2) [3]
  // (3, 1) [6]
  vertex_t r = 4, c = 4;
  edge_t nnz = 4;

  // let's use thrust vector<type_t> for initial arrays
  thrust::host_vector<edge_t> h_Ap(r + 1);
  thrust::host_vector<vertex_t> h_Aj(nnz);
  thrust::host_vector<weight_t> h_Ax(nnz);

  auto Ap = h_Ap.data();
  auto Aj = h_Aj.data();
  auto Ax = h_Ax.data();

  // row offsets
  Ap[0] = 0;
  Ap[1] = 0;
  Ap[2] = 2;
  Ap[3] = 3;
  Ap[4] = 4;

  // column indices
  Aj[0] = 0;
  Aj[1] = 1;
  Aj[2] = 2;
  Aj[3] = 3;

  // nonzero values (weights)
  Ax[0] = 5;
  Ax[1] = 8;
  Ax[2] = 3;
  Ax[3] = 6;

  thrust::device_vector<edge_t> d_Ap = h_Ap;
  thrust::device_vector<vertex_t> d_Aj = h_Aj;
  thrust::device_vector<weight_t> d_Ax = h_Ax;

  vertex_t source = 1;
  thrust::device_vector<weight_t> d_distances(nnz);

  // calling sssp
  float elapsed = sssp::execute<space>(r,           // number of vertices
                                       c,           // number of columns
                                       nnz,         // number of edges
                                       d_Ap,        // row_offsets
                                       d_Aj,        // column_indices
                                       d_Ax,        // nonzero values
                                       source,      // single source
                                       d_distances  // output distances
  );

  std::cout << "Distances = ";
  thrust::copy(d_distances.begin(), d_distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "SSSP Elapsed Time: " << elapsed << std::endl;
}

int main(int argc, char** argv) {
  test_sssp();
  return EXIT_SUCCESS;
}