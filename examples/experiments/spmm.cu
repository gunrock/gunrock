/**
 * @file spmm.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse matrix-matrix multiplication
 * @version 0.1
 * @date 2022-01-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <chrono>
#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gunrock/formats/formats.hxx>
#include <gunrock/io/matrix_market.hxx>
#include <gunrock/algorithms/generate/random.hxx>
#include <gunrock/util/print.hxx>
#include <gunrock/util/timer.hxx>
#include <gunrock/util/compare.hxx>
#include <gunrock/util/load_store.hxx>

using namespace gunrock;

enum access_mode_t { row_major, col_major };

template <typename type_t, access_mode_t mode = row_major>
struct matrix_t {
  __host__ __device__ constexpr matrix_t()
      : height(0), width(0), data(nullptr) {}
  __host__ __device__ constexpr matrix_t(std::size_t _height,
                                         std::size_t _width,
                                         type_t* _data)
      : height(_height), width(_width), data(_data) {}

  __host__ __device__ constexpr matrix_t(const matrix_t& other)
      : height(other.height), width(other.width), data(other.data) {}

  __host__ __device__ __forceinline__ constexpr std::size_t size()
      const noexcept {
    return height * width;
  }

  __host__ __device__ __forceinline__ constexpr type_t& operator()(
      std::size_t row,
      std::size_t col) noexcept {
    // return data[mode == row_major ? row * width + col : row + col * height];
    return data[row * width + col];
  }

  __host__ __device__ __forceinline__ constexpr const type_t& operator()(
      std::size_t row,
      std::size_t col) const noexcept {
    return data[row * width + col];
  }

  void print() {
    thrust::device_vector<type_t> d_data(data, data + size());
    thrust::host_vector<type_t> h_data = d_data;

    std::cout << "Matrix: " << height << " x " << width << std::endl;
    std::cout << "==========================" << std::endl;
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        std::cout << h_data[row * width + col] << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << "==========================" << std::endl;
  }

  std::size_t height;
  std::size_t width;
  type_t* data;
};

template <typename type_t, typename offset_t, typename index_t>
void cpu_spmm(std::size_t const m,
              std::size_t const n,
              std::size_t const nnz,
              const offset_t* offsets,
              const index_t* indices,
              const type_t* values,
              matrix_t<type_t> const B,
              matrix_t<type_t> C) {
  for (index_t row = 0; row < m; ++row) {
    for (index_t col = 0; col < n; ++col) {
      type_t sum = 0.0f;
      for (auto nz = offsets[row]; nz < offsets[row + 1]; ++nz) {
        sum += values[nz] * B(indices[nz], col);
      }
      C(row, col) = sum;
    }
  }
}

template <std::size_t ATOM_X = 1,
          std::size_t ATOM_Y = 1,
          std::size_t TILE_X = 16,
          std::size_t TILE_Y = 16,
          typename type_t,
          typename offset_t,
          typename index_t>
__global__ void spmm(std::size_t nnz,
                     offset_t* offsets,
                     index_t* indices,
                     type_t* values,
                     matrix_t<type_t> const B,
                     matrix_t<type_t> C) {
  //   constexpr std::size_t ATOM_SIZE = ATOM_X * ATOM_Y;
  constexpr std::size_t TILE_SIZE = TILE_X * TILE_Y;
  __shared__ offset_t sh_offsets[TILE_SIZE + 1];

  // For all rows of sparse-matrix A.
  for (index_t row = threadIdx.x + (blockIdx.x * blockDim.x); row < C.height;
       row += blockDim.x * gridDim.x) {
    // Load the row offsets of A into shared memory.
    sh_offsets[threadIdx.x] = offsets[row];
    if (threadIdx.x == (blockDim.x - 1) || row == (C.height - 1)) {
      sh_offsets[threadIdx.x + 1] = offsets[row + 1];
    }
    __syncthreads();

    offset_t offset = sh_offsets[threadIdx.x];
    offset_t end = sh_offsets[threadIdx.x + 1];

    // For all columns of sparse-matrix B.
    for (index_t col = 0; col < C.width; ++col) {
      type_t sum = 0.0f;
      for (offset_t nz = offset; nz < end; ++nz) {
        index_t k = thread::load(&indices[nz]);
        type_t val = thread::load(&values[nz]);
        type_t b_val = B(k, col);
        sum += val * b_val;
      }
      thread::store(&C(row, col), sum);
    }
  }
}

int main(int argc, char** argv) {
  using type_t = float;
  using vertex_t = int;
  using edge_t = int;
  using weight_t = type_t;

  using namespace gunrock;
  using namespace memory;
  using namespace format;

  std::string filename = argv[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> A;
  A.from_coo(mm.load(filename));

  std::size_t m = A.number_of_rows;
  std::size_t k = A.number_of_columns;
  std::size_t n = 4;
  std::size_t nnz = A.number_of_nonzeros;

  thrust::device_vector<type_t> B_vec(k * n);
  thrust::device_vector<type_t> C_vec(m * n);

  generate::random::uniform_distribution(B_vec, 1.0f, 2.0f);

  matrix_t<type_t> B(k, n, B_vec.data().get());
  matrix_t<type_t> C(m, n, C_vec.data().get());

  constexpr std::size_t BLK_X = 16;
  constexpr std::size_t BLK_Y = 16;
  constexpr std::size_t num_threads = BLK_X * BLK_Y;
  std::size_t num_blocks = (m + num_threads - 1) / num_threads;

  util::timer_t timer;
  timer.start();
  spmm<<<num_blocks, num_threads>>>(nnz, A.row_offsets.data().get(),
                                    A.column_indices.data().get(),
                                    A.nonzero_values.data().get(), B, C);
  cudaDeviceSynchronize();
  auto gpu_elapsed = timer.end();

  std::cout << "GPU Elapsed (ms): " << gpu_elapsed << std::endl;

  csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> A_host(A);
  thrust::host_vector<type_t> B_host_vec = B_vec;
  thrust::host_vector<type_t> C_host_vec(m * n);

  matrix_t<type_t> B_host(k, n, B_host_vec.data());
  matrix_t<type_t> C_host(m, n, C_host_vec.data());

  auto start = std::chrono::high_resolution_clock::now();
  cpu_spmm(m, n, nnz,                     // Sizes
           A_host.row_offsets.data(),     // A offsets
           A_host.column_indices.data(),  // A indices
           A_host.nonzero_values.data(),  // A non-zero values
           B_host,                        // B tensor
           C_host                         // C tensor
  );
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  std::cout << "CPU Elapsed (ms): " << (float)(elapsed / 1000) << std::endl;

  int n_errors = util::compare(
      C_vec.data().get(), C_host_vec.data(), m * n,
      [](const weight_t a, const weight_t b) { return std::abs(a - b) > 1e-6; },
      true);

  std::cout << "Number of errors: " << n_errors << std::endl;
}