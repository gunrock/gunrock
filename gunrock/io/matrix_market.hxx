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

#include <string>

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

/**
 * @brief Reads a MARKET graph from an input-stream
 * into a specified sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comments
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct matrix_market_t {
  typedef FILE* file_t;
  typedef MM_typecode mm_type_t;

  std::string filename;
  std::string dataset;
  matrix_market_code_t types;

  matrix_market_t() {}
  ~matrix_market_t() {}

  /**
   * @brief Loads the given .mtx file into a coordinate format, and returns the
   * coordinate array. This needs to be further extended to support dense
   * arrays, those are the only two formats mtx are written in.
   *
   * @param _filename input file name (.mtx)
   * @return coordinate sparse format
   */
  auto load(std::string _filename) {
    filename = _filename;
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

    int num_rows, num_columns, num_nonzeros;  // XXX: requires all ints intially
    if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=
        0) {
      std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
      exit(1);
    }

    // mtx are generally written as coordinate formaat
    format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t> coo(
        (vertex_t)num_rows, (vertex_t)num_columns, (edge_t)num_nonzeros);

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

    return coo;
  }
};

}  // namespace io
}  // namespace gunrock