/**
 * @file matrix_market.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-09
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <externals/mtx/mmio.hxx>

#include <gunrock/util/filepath.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/memory.hxx>

namespace gunrock {
namespace io {

using namespace memory;

using matrix_market_code_t = uint32_t;
enum : matrix_market_code_t {
  matrix = 1 << 0,      // 0x01
  sparse = 1 << 1,      // 0x02
  coordinate = 1 << 2,  // 0x03
  dense = 1 << 3,       // 0x04
  array = 1 << 4,       // 0x05
  complex = 1 << 5,     // 0x06
  real = 1 << 6,        // 0x07
  pattern = 1 << 7,     // 0x08
  integer = 1 << 8,     // 0x09
  symmetric = 1 << 9,   // 0x10
  general = 1 << 10,    // 0x11
  skew = 1 << 11,       // 0x12
  hermitian = 1 << 12,  // 0x13
  none = 0 << 0,        // 0x00
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct matrix_market_t {
  typedef FILE* file_t;
  typedef MM_typecode mm_type_t;

  std::string dataset;
  mm_code_t types;

  matrix_market_t() {}

  /**
   * @brief If format was specified as coordinate,
   *        then the size line has the form:
   *          m n nonzeros
   *        where
   *          m is the number of rows in the matrix;
   *          n is the number of columns in the matrix;
   *          nonzeros is the number of nonzero entries
   *          in the matrix (for general symmetry), or
   *          the number of nonzero entries on or below
   *          the diagonal (for symmetric or Hermitian
   *          symmetry), or the number of nonzero entries
   *          below the diagonal (for skew-symmetric symmetry).
   *
   * @param filename
   */
  void load(std::string filename) {
    dataset = util::extract_dataset(util::extract_filename(filename));

    file_t file;
    mm_type_t code;

    // Load MTX information
    if ((file = fopen(filename.c_str(), "r")) == NULL) {
      std::cerr << "File could not be opened: " << filename << std::endl;
      exit(1);
    }

    if (mm_read_banner(file, &code) != 0) {
      std::cerr << "Could not process Matrix Market banner" << std::endl;
      exit(1);
    }

    vertex_t num_rows, num_columns, num_nonzeros;
    if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=
        0) {
      std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
      exit(1);
    }

    // mtx are generally written as coordinate formaat
    coo_t<memory_space_t::host, vertex_t, edge_t, weight_t> coo(
        num_rows, num_columns, num_nonzeros);

    if (mm_is_pattern(code)) {
      types |= pattern;

      // pattern matrix defines sparsity pattern, but not values
      for (vertex_t i = 0; i < num_nonzeros; i++) {
        assert(fscanf(file, " %d %d \n", &(coo.row_indices[i]),
                      &(coo.column_indices[i])) == 2);
        coo.row_indices[i]--;  // adjust from 1-based to 0-based indexing
        coo.column_indices[i]--;
        coo.nonzero_values[i] =
            (weight_t)1.0;  // use value 1.0 for all nonzero entries
      }
    } else if (mm_is_real(code) || mm_is_integer(code)) {
      types |= real;

      for (vertex_t i = 0; i < coo.number_of_nonzeros; i++) {
        vertex_t I, J;
        double V;  // always read in a double and convert later if necessary

        assert(fscanf(file, " %d %d %lf \n", &I, &J, &V) == 3);

        coo.row_indices[i] = (vertex_t)I - 1;
        coo.column_indices[i] = (vertex_t)J - 1;
        coo.nonzero_values[i] = (weight_t)V;
      }
    } else {
      std::cerr << "Unrecognized matrix market format type" << std::endl;
      exit(1);
    }

    if (mm_is_symmetric(code)) {  // duplicate off diagonal entries
      types |= symmetric;
      vertex_t off_diagonals = 0;
      for (vertex_t i = 0; i < coo.number_of_nonzeros; i++) {
        if (coo.row_indices[i] != coo.column_indices[i])
          off_diagonals++;
      }

      vertex_t _nonzeros =
          2 * off_diagonals + (coo.number_of_nonzeros - off_diagonals);

      thrust::host_vector<vertex_t> new_I(_nonzeros);
      thrust::host_vector<vertex_t> new_J(_nonzeros);
      thrust::host_vector<weight_t> new_V(_nonzeros);

      vertex_t* _I = new_I.data();
      vertex_t* _J = new_J.data();
      weight_t* _V = new_V.data();

      vertex_t ptr = 0;
      for (vertex_t i = 0; i < coo.number_of_nonzeros; i++) {
        if (coo.row_indices[i] != coo.column_indices[i]) {
          _I[ptr] = coo.row_indices[i];
          _J[ptr] = coo.column_indices[i];
          _V[ptr] = coo.nonzero_values[i];
          ptr++;
          _J[ptr] = coo.row_indices[i];
          _I[ptr] = coo.column_indices[i];
          _V[ptr] = coo.nonzero_values[i];
          ptr++;
        } else {
          _I[ptr] = coo.row_indices[i];
          _J[ptr] = coo.column_indices[i];
          _V[ptr] = coo.nonzero_values[i];
          ptr++;
        }
      }
      coo.row_indices = new_I;
      coo.column_indices = new_J;
      coo.nonzero_values = new_V;
      coo.number_of_nonzeros = _nonzeros;
    }  // end symmetric case

    fclose(file);
  }
};

}  // namespace io
}  // namespace gunrock