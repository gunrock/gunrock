/**
 * @file smtx.hxx
 * @author Jonathan Wapman (jdwapman@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <string>
#include <limits>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gunrock/io/detail/mmio.hxx>

#include <gunrock/util/filepath.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/memory.hxx>
#include <gunrock/error.hxx>
#include <gunrock/algorithms/generate/random.hxx>

namespace gunrock {
namespace io {

using namespace memory;

std::string leading_trim(std::string s) {
  size_t start = s.find_first_not_of(" ");
  return (start == std::string::npos) ? "" : s.substr(start);
}

/**
 * @brief Reads a smtx graph from an input-stream
 * into a specified sparse format
 *
 * Here is an example of the smtx format
 * +----------------------------------------------+
 * |% Sparse matrix file format .smtx             | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comments
 * |%                                             | <--+
 * |  M K NNZ                                     | <--- rows, columns, entries
 * |  row_offsets                                 | <--+
 * |  column_indices                              | <--+-- 2 lines
 * |                                              |
 * +----------------------------------------------+
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct smtx_t {
  std::string filename;
  std::string dataset;

  smtx_t() {}
  ~smtx_t() {}

  /**
   * @brief Loads the given .smtx file into a csr format.
   *
   * @param _filename input file name (.smtx)
   * @return csr sparse format
   */
  auto load(std::string _filename, bool first_line_csv = false) {
    filename = _filename;
    dataset = util::extract_dataset(util::extract_filename(filename));

    std::ifstream smtx_file(filename);
    unsigned int row_ptrs_buf;
    vertex_t col_idxs_buf;

    // smtx is written in CSR format
    format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr(
        (vertex_t)0, (vertex_t)0, (edge_t)0);
    csr.row_offsets.resize(0);
    csr.column_indices.resize(0);
    csr.nonzero_values.resize(0);

    if (smtx_file.is_open()) {
      std::size_t num_rows, num_columns, num_nonzeros;

      std::string line;  // Buffer for storing file lines

      for (int line_num = 0; line_num < 3; line_num++) {
        // Skip over comment lines
        do {
          std::getline(smtx_file, line);
        } while (line[0] == '%');

        std::istringstream line_stream(line);

        if (line_num == 0) {  // First Line has dimensions and nnz
          if (first_line_csv) {
            std::string buf;
            std::getline(line_stream, buf, ',');
            leading_trim(buf);
            num_rows = std::stoi(buf);
            std::getline(line_stream, buf, ',');
            leading_trim(buf);
            num_columns = std::stoi(buf);
            std::getline(line_stream, buf, ',');
            leading_trim(buf);
            num_nonzeros = std::stoi(buf);
          } else {
            line_stream >> num_rows;
            line_stream >> num_columns;
            line_stream >> num_nonzeros;
          }

          error::throw_if_exception(
              num_rows >= std::numeric_limits<vertex_t>::max() ||
                  num_columns >= std::numeric_limits<vertex_t>::max(),
              "vertex_t overflow");
          error::throw_if_exception(
              num_nonzeros >= std::numeric_limits<edge_t>::max(),
              "edge_t overflow");

          csr.number_of_rows = num_rows;
          csr.number_of_columns = num_columns;
          csr.number_of_nonzeros = num_nonzeros;

          csr.row_offsets.reserve(csr.number_of_rows + 1);
          csr.column_indices.reserve(csr.number_of_nonzeros);
          csr.nonzero_values.reserve(csr.number_of_nonzeros);
        } else if (line_num == 1) {  // Second line has row pointers
          int count = 0;
          while (line_stream >> row_ptrs_buf) {
            csr.row_offsets.push_back(row_ptrs_buf);
            count++;
          }
        } else if (line_num == 2) {  // Third line has column indices
          while (line_stream >> col_idxs_buf) {
            csr.column_indices.push_back(col_idxs_buf);
            csr.nonzero_values.push_back(
                gunrock::generate::random::get_random<weight_t>(1.0f, 10.0f));
          }
        }
      }

      smtx_file.close();
    } else {
      throw(std::runtime_error("Unable to open file"));
    }

    if (csr.row_offsets.size() - 1 != csr.number_of_rows) {
      std::ostringstream ss;
      ss << "Number of rows in " << filename << " ("
         << csr.row_offsets.size() - 1
         << ") does not match the count in the first line ("
         << csr.number_of_rows << ")";
      throw(std::invalid_argument(ss.str()));
    }

    if (csr.nonzero_values.size() != csr.number_of_nonzeros) {
      std::ostringstream ss;
      ss << "Number of non-zeros in " << filename << " ("
         << csr.nonzero_values.size()
         << ") does not match the count in the first line ("
         << csr.number_of_nonzeros << ")";
      throw(std::invalid_argument(ss.str()));
    }

    return csr;
  }
};

}  // namespace io
}  // namespace gunrock